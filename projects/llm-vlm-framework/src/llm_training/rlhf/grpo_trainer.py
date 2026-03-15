"""
GRPO (Group Relative Policy Optimization) 训练器
DeepSeek-R1 使用的训练算法，通过组内相对优势估计进行策略优化
"""

import os
import sys
import json
import math
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """GRPO训练配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    ref_model_name: Optional[str] = None  # 参考模型，默认与model_name相同
    
    # 数据配置
    train_data_path: str = "./data/grpo/train.jsonl"
    eval_data_path: Optional[str] = None
    max_length: int = 2048
    max_prompt_length: int = 512
    max_completion_length: int = 1024
    
    # GRPO特定配置
    group_size: int = 4  # 每组采样的数量
    kl_coeff: float = 0.1  # KL散度系数
    clip_eps: float = 0.2  # PPO裁剪参数
    
    # 奖励配置
    reward_model_path: Optional[str] = None  # 奖励模型路径
    reward_functions: List[str] = None  # 奖励函数列表 ['accuracy', 'format', 'length']
    
    # 训练配置
    output_dir: str = "./outputs/llm_grpo"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    
    # 优化器配置
    learning_rate: float = 1e-6
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # 生成配置
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    
    # 日志配置
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    
    def __post_init__(self):
        if self.reward_functions is None:
            self.reward_functions = ['accuracy']


class GRPODataset:
    """GRPO数据集 - 包含问题和参考答案"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_prompt_length: int = 512,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """加载数据"""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.data[idx]
        
        # 构建prompt
        prompt = item.get('prompt', item.get('question', ''))
        answer = item.get('answer', item.get('reference', ''))
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'prompt': prompt,
            'answer': answer,
            'prompt_input_ids': prompt_tokens['input_ids'].squeeze(0),
            'prompt_attention_mask': prompt_tokens['attention_mask'].squeeze(0),
            'metadata': item.get('metadata', {}),
        }


class GRPOTrainer:
    """
    GRPO训练器
    
    核心思想：
    1. 对每个prompt采样group_size个completion
    2. 计算每个completion的奖励
    3. 使用组内相对优势（reward - mean）作为优势估计
    4. 通过PPO式裁剪目标进行策略优化
    """
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        
        # 设置随机种子
        if hasattr(config, 'seed'):
            torch.manual_seed(config.seed)
        
        # 初始化模型和tokenizer
        self._setup_model()
        self._setup_reference_model()
        self._setup_reward_model()
        self._setup_dataloader()
        self._setup_optimizer()
        
        # 准备分布式训练
        self.model, self.ref_model, self.optimizer, self.train_dataloader = \
            self.accelerator.prepare(
                self.model, self.ref_model, self.optimizer, self.train_dataloader
            )
        
        self.global_step = 0
        
    def _setup_model(self):
        """加载策略模型"""
        logger.info(f"加载策略模型: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side='left',
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
        )
        
        logger.info(f"模型加载完成，参数量: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
        
    def _setup_reference_model(self):
        """加载参考模型（固定参数）"""
        ref_model_name = self.config.ref_model_name or self.config.model_name
        logger.info(f"加载参考模型: {ref_model_name}")
        
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
        )
        
        # 冻结参考模型参数
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.ref_model.eval()
        
    def _setup_reward_model(self):
        """加载或初始化奖励模型"""
        if self.config.reward_model_path:
            logger.info(f"加载奖励模型: {self.config.reward_model_path}")
            # TODO: 加载外部奖励模型
            self.reward_model = None
        else:
            logger.info("使用规则奖励函数")
            self.reward_model = None
            
    def _setup_dataloader(self):
        """设置数据加载器"""
        logger.info("设置数据加载器...")
        
        dataset = GRPODataset(
            data_path=self.config.train_data_path,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.max_prompt_length,
        )
        
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        
    def _collate_fn(self, batch):
        """批次处理"""
        return {
            'prompts': [item['prompt'] for item in batch],
            'answers': [item['answer'] for item in batch],
            'prompt_input_ids': torch.nn.utils.rnn.pad_sequence(
                [item['prompt_input_ids'] for item in batch],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            'prompt_attention_mask': torch.nn.utils.rnn.pad_sequence(
                [item['prompt_attention_mask'] for item in batch],
                batch_first=True,
                padding_value=0,
            ),
            'metadata': [item['metadata'] for item in batch],
        }
    
    def _setup_optimizer(self):
        """设置优化器"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        total_steps = len(self.train_dataloader) * self.config.num_train_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
    def generate_completions(self, prompt_input_ids, prompt_attention_mask, num_return_sequences: int = 1):
        """
        生成多个completion
        
        Args:
            prompt_input_ids: [batch_size, prompt_len]
            prompt_attention_mask: [batch_size, prompt_len]
            num_return_sequences: 每个prompt生成的序列数
            
        Returns:
            completions: [batch_size * num_return_sequences, total_len]
            log_probs: [batch_size * num_return_sequences]
        """
        batch_size = prompt_input_ids.shape[0]
        
        # 扩展prompt以生成多个completion
        if num_return_sequences > 1:
            prompt_input_ids = prompt_input_ids.repeat_interleave(num_return_sequences, dim=0)
            prompt_attention_mask = prompt_attention_mask.repeat_interleave(num_return_sequences, dim=0)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=self.config.max_completion_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=self.config.do_sample,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        completions = outputs.sequences
        
        # 计算log概率
        log_probs = self._compute_log_probs(completions, prompt_input_ids.shape[1])
        
        return completions, log_probs
    
    def _compute_log_probs(self, sequences, prompt_len):
        """计算序列的log概率"""
        with torch.no_grad():
            outputs = self.model(input_ids=sequences)
            logits = outputs.logits[:, :-1, :]  # [batch, seq_len-1, vocab]
            targets = sequences[:, 1:]  # [batch, seq_len-1]
            
            # 只计算completion部分的log prob
            completion_mask = torch.zeros_like(targets, dtype=torch.bool)
            completion_mask[:, prompt_len-1:] = True
            
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            
            # 对每个序列求和
            seq_log_probs = (token_log_probs * completion_mask.float()).sum(dim=-1)
            
        return seq_log_probs
    
    def compute_rewards(self, prompts: List[str], completions: List[str], answers: List[str]) -> torch.Tensor:
        """
        计算奖励
        
        简化版本：使用规则奖励函数
        - accuracy: 答案匹配度
        - format: 格式正确性
        - length: 长度奖励
        """
        rewards = []
        
        for prompt, completion, answer in zip(prompts, completions, answers):
            reward = 0.0
            
            # 准确率奖励
            if 'accuracy' in self.config.reward_functions:
                # 简单匹配，实际应该使用更复杂的评估
                if answer.lower() in completion.lower():
                    reward += 1.0
                else:
                    reward += 0.0
            
            # 格式奖励
            if 'format' in self.config.reward_functions:
                # 检查是否有特定格式（如\boxed{}）
                if '\\boxed{' in completion or 'Answer:' in completion:
                    reward += 0.5
            
            # 长度奖励（惩罚过长）
            if 'length' in self.config.reward_functions:
                comp_len = len(completion.split())
                if comp_len < 100:
                    reward += 0.3
                elif comp_len < 200:
                    reward += 0.1
            
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def compute_grpo_loss(
        self,
        prompt_input_ids,
        completions,
        old_log_probs,
        rewards,
    ):
        """
        计算GRPO损失
        
        Args:
            prompt_input_ids: [batch_size, prompt_len]
            completions: [batch_size * group_size, seq_len]
            old_log_probs: [batch_size * group_size]
            rewards: [batch_size * group_size]
            
        Returns:
            loss, metrics
        """
        # 获取当前策略的log prob
        current_log_probs = self._compute_log_probs(completions, prompt_input_ids.shape[1])
        
        # 获取参考模型的log prob（用于KL惩罚）
        with torch.no_grad():
            ref_log_probs = self._compute_ref_log_probs(completions, prompt_input_ids.shape[1])
        
        # 计算KL散度
        kl_div = old_log_probs - ref_log_probs
        
        # 组内相对优势估计
        batch_size = prompt_input_ids.shape[0]
        group_size = self.config.group_size
        
        # 重塑为 [batch_size, group_size]
        rewards_grouped = rewards.view(batch_size, group_size)
        advantages = rewards_grouped - rewards_grouped.mean(dim=1, keepdim=True)
        advantages = advantages.view(-1)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算重要性采样比率
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # PPO裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL惩罚
        kl_loss = kl_div.mean()
        
        # 总损失
        loss = policy_loss + self.config.kl_coeff * kl_loss
        
        metrics = {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            'mean_reward': rewards.mean().item(),
            'mean_advantage': advantages.mean().item(),
            'mean_ratio': ratio.mean().item(),
        }
        
        return loss, metrics
    
    def _compute_ref_log_probs(self, sequences, prompt_len):
        """计算参考模型的log概率"""
        with torch.no_grad():
            outputs = self.ref_model(input_ids=sequences)
            logits = outputs.logits[:, :-1, :]
            targets = sequences[:, 1:]
            
            completion_mask = torch.zeros_like(targets, dtype=torch.bool)
            completion_mask[:, prompt_len-1:] = True
            
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            seq_log_probs = (token_log_probs * completion_mask.float()).sum(dim=-1)
            
        return seq_log_probs
    
    def train(self):
        """训练循环"""
        logger.info("开始GRPO训练...")
        
        self.model.train()
        self.global_step = 0
        
        for epoch in range(self.config.num_train_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_train_epochs}")
            
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    # 生成多个completion
                    completions, old_log_probs = self.generate_completions(
                        batch['prompt_input_ids'],
                        batch['prompt_attention_mask'],
                        num_return_sequences=self.config.group_size,
                    )
                    
                    # 解码completion
                    completion_texts = self.tokenizer.batch_decode(
                        completions[:, batch['prompt_input_ids'].shape[1]:],
                        skip_special_tokens=True,
                    )
                    
                    # 扩展prompts和answers以匹配completion数量
                    expanded_prompts = []
                    expanded_answers = []
                    for prompt, answer in zip(batch['prompts'], batch['answers']):
                        expanded_prompts.extend([prompt] * self.config.group_size)
                        expanded_answers.extend([answer] * self.config.group_size)
                    
                    # 计算奖励
                    rewards = self.compute_rewards(
                        expanded_prompts,
                        completion_texts,
                        expanded_answers,
                    ).to(self.accelerator.device)
                    
                    # 计算GRPO损失
                    loss, metrics = self.compute_grpo_loss(
                        batch['prompt_input_ids'],
                        completions,
                        old_log_probs,
                        rewards,
                    )
                    
                    # 反向传播
                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # 日志
                    if self.global_step % self.config.logging_steps == 0:
                        logger.info(
                            f"Step {self.global_step} | "
                            f"Loss: {metrics['loss']:.4f} | "
                            f"Reward: {metrics['mean_reward']:.4f} | "
                            f"Advantage: {metrics['mean_advantage']:.4f}"
                        )
        
        logger.info("训练完成!")
        self._save_model()
    
    def _save_model(self):
        """保存模型"""
        output_dir = Path(self.config.output_dir) / "final"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"保存模型到: {output_dir}")
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存配置
        with open(output_dir / "grpo_config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)


if __name__ == "__main__":
    # 测试配置
    config = GRPOConfig(
        model_name="gpt2",
        train_data_path="./data/grpo/train.jsonl",
        output_dir="./outputs/grpo_test",
        num_train_epochs=1,
        group_size=4,
        per_device_train_batch_size=1,
        logging_steps=1,
    )
    
    trainer = GRPOTrainer(config)
    trainer.train()
