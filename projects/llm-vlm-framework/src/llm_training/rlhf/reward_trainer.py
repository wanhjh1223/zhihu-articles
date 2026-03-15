"""
Reward Model 奖励模型训练器
用于训练偏好预测模型，为RLHF提供奖励信号
"""

import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RewardModelConfig:
    """奖励模型训练配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-7B"
    num_labels: int = 1  # 奖励是标量值
    
    # 数据配置
    train_data_path: str = "./data/reward/train.jsonl"
    eval_data_path: Optional[str] = None
    max_length: int = 512
    
    # 训练配置
    output_dir: str = "./outputs/reward_model"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # 优化器配置
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # 损失配置
    loss_type: str = "ranking"  # 'ranking', 'regression', 'btl'
    margin: float = 0.5  # ranking loss的margin
    
    # 评估配置
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10
    
    # 其他
    seed: int = 42


class RewardModelDataset(Dataset):
    """
    奖励模型数据集
    
    数据格式：
    {
        "prompt": "问题...",
        "chosen": "更好的回答...",
        "rejected": "较差的回答...",
        "chosen_score": 1.0,  # 可选
        "rejected_score": 0.0  # 可选
    }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """加载数据"""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # 验证必要字段
                if 'prompt' in item and 'chosen' in item and 'rejected' in item:
                    data.append(item)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.data[idx]
        
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        # 构建完整文本
        chosen_text = f"{prompt}\n\n{chosen}"
        rejected_text = f"{prompt}\n\n{rejected}"
        
        # Tokenize
        chosen_tokens = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        rejected_tokens = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        return {
            'chosen_input_ids': chosen_tokens['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_tokens['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_tokens['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_tokens['attention_mask'].squeeze(0),
            'chosen_score': item.get('chosen_score', 1.0),
            'rejected_score': item.get('rejected_score', 0.0),
        }


class RewardModelTrainer:
    """奖励模型训练器"""
    
    def __init__(self, config: RewardModelConfig):
        self.config = config
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        
        # 初始化
        self._setup_model()
        self._setup_dataloader()
        self._setup_optimizer()
        
        # 准备分布式训练
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.eval_dataloader
            )
        
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
    def _setup_model(self):
        """加载奖励模型"""
        logger.info(f"加载奖励模型: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 使用SequenceClassification模型，输出标量奖励
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # 修改模型输出为标量奖励（如果不是已经配置好的）
        if hasattr(self.model, 'score'):
            # 对于GPT类模型，score层输出奖励
            pass
        else:
            # 添加奖励头
            hidden_size = self.model.config.hidden_size
            self.reward_head = nn.Linear(hidden_size, 1)
        
        logger.info(f"模型加载完成，参数量: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
        
    def _setup_dataloader(self):
        """设置数据加载器"""
        logger.info("设置数据加载器...")
        
        # 训练数据
        train_dataset = RewardModelDataset(
            data_path=self.config.train_data_path,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        
        # 验证数据
        if self.config.eval_data_path:
            eval_dataset = RewardModelDataset(
                data_path=self.config.eval_data_path,
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
            )
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.config.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=self._collate_fn,
            )
        else:
            self.eval_dataloader = None
    
    def _collate_fn(self, batch):
        """批次处理"""
        return {
            'chosen_input_ids': torch.stack([item['chosen_input_ids'] for item in batch]),
            'chosen_attention_mask': torch.stack([item['chosen_attention_mask'] for item in batch]),
            'rejected_input_ids': torch.stack([item['rejected_input_ids'] for item in batch]),
            'rejected_attention_mask': torch.stack([item['rejected_attention_mask'] for item in batch]),
            'chosen_score': torch.tensor([item['chosen_score'] for item in batch], dtype=torch.float32),
            'rejected_score': torch.tensor([item['rejected_score'] for item in batch], dtype=torch.float32),
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
    
    def compute_rewards(self, input_ids, attention_mask):
        """计算奖励分数"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # 对于SequenceClassification，logits就是奖励值
        rewards = outputs.logits.squeeze(-1)
        return rewards
    
    def compute_loss(self, batch):
        """
        计算损失
        
        支持多种损失类型：
        - ranking: 排序损失（chosen > rejected）
        - regression: 回归损失（拟合给定分数）
        - btl: Bradley-Terry-Luce 模型损失
        """
        # 计算chosen和rejected的奖励
        chosen_rewards = self.compute_rewards(
            batch['chosen_input_ids'],
            batch['chosen_attention_mask'],
        )
        
        rejected_rewards = self.compute_rewards(
            batch['rejected_input_ids'],
            batch['rejected_attention_mask'],
        )
        
        if self.config.loss_type == 'ranking':
            # 排序损失：chosen的奖励应该比rejected高
            loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
            
        elif self.config.loss_type == 'regression':
            # 回归损失：拟合给定分数
            chosen_loss = F.mse_loss(chosen_rewards, batch['chosen_score'])
            rejected_loss = F.mse_loss(rejected_rewards, batch['rejected_score'])
            loss = chosen_loss + rejected_loss
            
        elif self.config.loss_type == 'btl':
            # BTL模型损失
            diff = chosen_rewards - rejected_rewards
            loss = -torch.log(torch.sigmoid(diff)).mean()
            
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        # 计算准确率（chosen > rejected的比例）
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        # 奖励差距
        reward_margin = (chosen_rewards - rejected_rewards).mean()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'chosen_reward': chosen_rewards.mean().item(),
            'rejected_reward': rejected_rewards.mean().item(),
            'reward_margin': reward_margin.item(),
        }
        
        return loss, metrics
    
    def train(self):
        """训练循环"""
        logger.info("开始奖励模型训练...")
        
        self.model.train()
        self.global_step = 0
        
        for epoch in range(self.config.num_train_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_train_epochs}")
            
            epoch_losses = []
            epoch_accuracies = []
            
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    loss, metrics = self.compute_loss(batch)
                    
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
                    epoch_losses.append(metrics['loss'])
                    epoch_accuracies.append(metrics['accuracy'])
                    
                    # 日志
                    if self.global_step % self.config.logging_steps == 0:
                        logger.info(
                            f"Step {self.global_step} | "
                            f"Loss: {metrics['loss']:.4f} | "
                            f"Accuracy: {metrics['accuracy']:.4f} | "
                            f"Chosen: {metrics['chosen_reward']:.4f} | "
                            f"Rejected: {metrics['rejected_reward']:.4f}"
                        )
                    
                    # 评估
                    if self.eval_dataloader and self.global_step % self.config.eval_steps == 0:
                        self.evaluate()
                    
                    # 保存
                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint()
            
            # Epoch结束统计
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_acc = sum(epoch_accuracies) / len(epoch_accuracies)
            logger.info(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.4f}")
        
        logger.info("训练完成!")
        self._save_model()
    
    @torch.no_grad()
    def evaluate(self):
        """评估"""
        if self.eval_dataloader is None:
            return
        
        logger.info("开始评估...")
        self.model.eval()
        
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        for batch in self.eval_dataloader:
            loss, metrics = self.compute_loss(batch)
            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        logger.info(f"Eval - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        
        # 保存最佳模型
        if avg_loss < self.best_eval_loss:
            self.best_eval_loss = avg_loss
            self._save_model(suffix='best')
            logger.info(f"保存最佳模型，Eval Loss: {avg_loss:.4f}")
        
        self.model.train()
    
    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
    
    def _save_model(self, suffix='final'):
        """保存模型"""
        output_dir = Path(self.config.output_dir) / suffix
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"保存模型到: {output_dir}")
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存配置
        with open(output_dir / "reward_config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)


if __name__ == "__main__":
    # 测试
    config = RewardModelConfig(
        model_name="gpt2",
        train_data_path="./data/reward/train.jsonl",
        output_dir="./outputs/reward_test",
        num_train_epochs=1,
        logging_steps=1,
    )
    
    trainer = RewardModelTrainer(config)
    trainer.train()
