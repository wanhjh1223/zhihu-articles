"""
PPO (Proximal Policy Optimization) RLHF 训练器
经典RLHF算法，使用奖励模型提供的奖励信号进行策略优化
"""

import os
import sys
import json
import math
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
class PPOConfig:
    """PPO训练配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    reward_model_path: str = "./outputs/reward_model/final"
    ref_model_name: Optional[str] = None
    
    # 数据配置
    train_data_path: str = "./data/ppo/train.jsonl"
    max_length: int = 512
    max_prompt_length: int = 256
    max_completion_length: int = 256
    
    # PPO超参数
    ppo_epochs: int = 4  # 每个batch的PPO更新次数
    mini_batch_size: int = 1
    clip_eps: float = 0.2
    kl_coeff: float = 0.1  # KL惩罚系数
    vf_coeff: float = 0.1  # Value函数损失系数
    entropy_coeff: float = 0.01  # 熵奖励系数
    gamma: float = 0.99  # 折扣因子
    lam: float = 0.95  # GAE lambda
    
    # 训练配置
    output_dir: str = "./outputs/llm_ppo"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    
    # 优化器配置
    learning_rate: float = 1e-6
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # 生成配置
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    
    # 日志配置
    logging_steps: int = 10
    save_steps: int = 500
    
    seed: int = 42


class PPODataset(Dataset):
    """PPO数据集 - 包含prompt"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_prompt_length: int = 256,
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
        
        prompt = item.get('prompt', item.get('question', item.get('instruction', '')))
        
        # Tokenize
        prompt_tokens = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        return {
            'prompt': prompt,
            'prompt_input_ids': prompt_tokens['input_ids'].squeeze(0),
            'prompt_attention_mask': prompt_tokens['attention_mask'].squeeze(0),
            'metadata': item.get('metadata', {}),
        }


class ValueHead(nn.Module):
    """价值函数头部 - 估计状态价值"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            values: [batch, seq_len]
        """
        value = self.dense(hidden_states).squeeze(-1)
        return value


class PPOTrainer:
    """
    PPO RLHF 训练器
    
    训练流程：
    1. 使用当前策略模型生成completion
    2. 使用奖励模型计算奖励
    3. 计算优势函数（使用GAE）
    4. 使用PPO目标函数更新策略
    5. KL散度惩罚防止策略偏离太远
    """
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        
        torch.manual_seed(config.seed)
        
        # 初始化模型
        self._setup_policy_model()
        self._setup_reference_model()
        self._setup_reward_model()
        self._setup_value_head()
        self._setup_dataloader()
        self._setup_optimizer()
        
        # 准备分布式训练
        self.model, self.value_head, self.optimizer, self.train_dataloader = \
            self.accelerator.prepare(
                self.model, self.value_head, self.optimizer, self.train_dataloader
            )
        
        self.global_step = 0
        self.stats_buffer = deque(maxlen=100)  # 用于记录训练统计
        
    def _setup_policy_model(self):
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
            trust_remote_code=True,
        )
        
        logger.info(f"策略模型参数量: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
        
    def _setup_reference_model(self):
        """加载参考模型（固定）"""
        ref_name = self.config.ref_model_name or self.config.model_name
        logger.info(f"加载参考模型: {ref_name}")
        
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            ref_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        
    def _setup_reward_model(self):
        """加载奖励模型"""
        logger.info(f"加载奖励模型: {self.config.reward_model_path}")
        
        try:
            from transformers import AutoModelForSequenceClassification
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.reward_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            for param in self.reward_model.parameters():
                param.requires_grad = False
            self.reward_model.eval()
        except Exception as e:
            logger.warning(f"加载奖励模型失败: {e}，将使用随机奖励")
            self.reward_model = None
            
    def _setup_value_head(self):
        """初始化价值函数头部"""
        hidden_size = self.model.config.hidden_size
        self.value_head = ValueHead(hidden_size)
        
    def _setup_dataloader(self):
        """设置数据加载器"""
        dataset = PPODataset(
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
        return {
            'prompts': [item['prompt'] for item in batch],
            'prompt_input_ids': torch.stack([item['prompt_input_ids'] for item in batch]),
            'prompt_attention_mask': torch.stack([item['prompt_attention_mask'] for item in batch]),
            'metadata': [item['metadata'] for item in batch],
        }
        
    def _setup_optimizer(self):
        """设置优化器 - 同时优化策略和价值函数"""
        # 合并策略模型和价值函数的参数
        params = list(self.model.parameters()) + list(self.value_head.parameters())
        
        self.optimizer = torch.optim.AdamW(
            params,
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
    
    def generate_completions(self, prompt_input_ids, prompt_attention_mask):
        """生成completion并计算log prob"""
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
        
        sequences = outputs.sequences
        prompt_len = prompt_input_ids.shape[1]
        
        # 计算策略模型的log prob
        with torch.no_grad():
            policy_outputs = self.model(input_ids=sequences)
            policy_logits = policy_outputs.logits[:, :-1, :]
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            
            # 获取实际token的log prob
            targets = sequences[:, 1:]
            token_log_probs = torch.gather(
                policy_log_probs, 
                dim=-1, 
                index=targets.unsqueeze(-1)
            ).squeeze(-1)
            
            # 只计算completion部分
            completion_mask = torch.zeros_like(targets, dtype=torch.bool)
            completion_mask[:, prompt_len-1:] = True
            
            seq_log_probs = (token_log_probs * completion_mask.float()).sum(dim=-1)
        
        return sequences, seq_log_probs
    
    def compute_rewards(self, sequences, prompt_len):
        """使用奖励模型计算奖励"""
        if self.reward_model is None:
            # 随机奖励（用于测试）
            return torch.randn(sequences.shape[0], device=sequences.device)
        
        # 提取文本并计算奖励
        with torch.no_grad():
            outputs = self.reward_model(input_ids=sequences)
            rewards = outputs.logits.squeeze(-1)
        
        return rewards
    
    def compute_kl_penalty(self, sequences, prompt_len):
        """计算KL散度惩罚"""
        with torch.no_grad():
            # 策略模型log prob
            policy_outputs = self.model(input_ids=sequences)
            policy_logits = policy_outputs.logits[:, :-1, :]
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            
            # 参考模型log prob
            ref_outputs = self.ref_model(input_ids=sequences)
            ref_logits = ref_outputs.logits[:, :-1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            
            # 计算KL
            targets = sequences[:, 1:]
            policy_token_logprobs = torch.gather(
                policy_log_probs, dim=-1, index=targets.unsqueeze(-1)
            ).squeeze(-1)
            ref_token_logprobs = torch.gather(
                ref_log_probs, dim=-1, index=targets.unsqueeze(-1)
            ).squeeze(-1)
            
            completion_mask = torch.zeros_like(targets, dtype=torch.bool)
            completion_mask[:, prompt_len-1:] = True
            
            kl_div = (policy_token_logprobs - ref_token_logprobs) * completion_mask.float()
            kl_per_seq = kl_div.sum(dim=-1)
        
        return kl_per_seq
    
    def compute_advantages(self, rewards, values, kl_penalty):
        """
        计算优势函数（使用GAE）
        
        简化版本：直接返回 (reward - value)
        """
        # 减去KL惩罚
        adjusted_rewards = rewards - self.config.kl_coeff * kl_penalty
        
        # 简化的优势估计
        advantages = adjusted_rewards - values.detach()
        
        return advantages
    
    def ppo_update(self, sequences, old_log_probs, rewards, advantages):
        """PPO更新"""
        # 计算新的log probs和values
        outputs = self.model(input_ids=sequences, output_hidden_states=True)
        logits = outputs.logits[:, :-1, :]
        hidden_states = outputs.hidden_states[-1][:, :-1, :]
        
        # 新的log probs
        new_log_probs = F.log_softmax(logits, dim=-1)
        targets = sequences[:, 1:]
        token_log_probs = torch.gather(
            new_log_probs, dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)
        
        completion_mask = torch.zeros_like(targets, dtype=torch.bool)
        prompt_len = sequences.shape[1] - self.config.max_completion_length
        completion_mask[:, max(0, prompt_len-1):] = True
        
        new_seq_log_probs = (token_log_probs * completion_mask.float()).sum(dim=-1)
        
        # 价值估计
        values = self.value_head(hidden_states).squeeze(-1)
        values = (values * completion_mask.float()).sum(dim=-1)
        
        # PPO比率
        ratio = torch.exp(new_seq_log_probs - old_log_probs)
        
        # PPO损失
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value损失
        value_loss = F.mse_loss(values, rewards)
        
        # 熵奖励
        entropy = -(new_log_probs * torch.exp(new_log_probs)).sum(dim=-1)
        entropy = (entropy * completion_mask.float()).sum(dim=-1).mean()
        
        # 总损失
        loss = (
            policy_loss 
            + self.config.vf_coeff * value_loss 
            - self.config.entropy_coeff * entropy
        )
        
        metrics = {
            'ppo_loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'mean_ratio': ratio.mean().item(),
            'mean_reward': rewards.mean().item(),
            'mean_advantage': advantages.mean().item(),
        }
        
        return loss, metrics
    
    def train(self):
        """训练循环"""
        logger.info("开始PPO训练...")
        
        self.model.train()
        self.value_head.train()
        self.global_step = 0
        
        for epoch in range(self.config.num_train_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_train_epochs}")
            
            for batch in self.train_dataloader:
                # 1. 生成completion（旧策略）
                sequences, old_log_probs = self.generate_completions(
                    batch['prompt_input_ids'],
                    batch['prompt_attention_mask'],
                )
                
                prompt_len = batch['prompt_input_ids'].shape[1]
                
                # 2. 计算奖励
                rewards = self.compute_rewards(sequences, prompt_len)
                
                # 3. 计算KL惩罚
                kl_penalty = self.compute_kl_penalty(sequences, prompt_len)
                
                # 4. 估计价值（用于计算优势）
                with torch.no_grad():
                    outputs = self.model(input_ids=sequences, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1][:, :-1, :]
                    values = self.value_head(hidden_states).squeeze(-1)
                    completion_mask = torch.zeros(values.shape, dtype=torch.bool, device=values.device)
                    completion_mask[:, prompt_len-1:] = True
                    values = (values * completion_mask.float()).sum(dim=-1)
                
                # 5. 计算优势
                advantages = self.compute_advantages(rewards, values, kl_penalty)
                
                # 6. PPO更新（多个epoch）
                for ppo_epoch in range(self.config.ppo_epochs):
                    with self.accelerator.accumulate(self.model):
                        loss, metrics = self.ppo_update(
                            sequences, old_log_probs, rewards, advantages
                        )
                        
                        self.accelerator.backward(loss)
                        self.accelerator.clip_grad_norm_(
                            list(self.model.parameters()) + list(self.value_head.parameters()),
                            self.config.max_grad_norm,
                        )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        
                        self.global_step += 1
                        
                        # 记录统计
                        self.stats_buffer.append(metrics)
                        
                        # 日志
                        if self.global_step % self.config.logging_steps == 0:
                            avg_metrics = {
                                k: sum(s[k] for s in self.stats_buffer) / len(self.stats_buffer)
                                for k in self.stats_buffer[0].keys()
                            }
                            logger.info(
                                f"Step {self.global_step} | "
                                f"PPO Loss: {avg_metrics['ppo_loss']:.4f} | "
                                f"Reward: {avg_metrics['mean_reward']:.4f} | "
                                f"Advantage: {avg_metrics['mean_advantage']:.4f}"
                            )
                
                # 保存检查点
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()
        
        logger.info("训练完成!")
        self._save_model()
    
    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # 保存value head
        unwrapped_value_head = self.accelerator.unwrap_model(self.value_head)
        torch.save(unwrapped_value_head.state_dict(), checkpoint_dir / "value_head.pt")
    
    def _save_model(self):
        """保存最终模型"""
        output_dir = Path(self.config.output_dir) / "final"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"保存模型到: {output_dir}")
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        unwrapped_value_head = self.accelerator.unwrap_model(self.value_head)
        torch.save(unwrapped_value_head.state_dict(), output_dir / "value_head.pt")
        
        with open(output_dir / "ppo_config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)


if __name__ == "__main__":
    config = PPOConfig(
        model_name="gpt2",
        reward_model_path="./outputs/reward_model/final",
        train_data_path="./data/ppo/train.jsonl",
        output_dir="./outputs/ppo_test",
        num_train_epochs=1,
        logging_steps=1,
    )
    
    trainer = PPOTrainer(config)
    trainer.train()
