"""
LLM 预训练完整实现
支持：
- 混合精度训练 (BF16/FP16)
- 梯度累积
- 梯度检查点
- Flash Attention
- 分布式训练 (DDP/FSDP/DeepSpeed)
- Wandb/Tensorboard 日志
"""

import os
import sys
import json
import math
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
from accelerate.utils import set_seed

# 导入数据加载器
from .pretrain_dataloader import PretrainDataset, PretrainDataConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class PretrainTrainingConfig:
    """预训练配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-7B"
    model_revision: Optional[str] = None
    trust_remote_code: bool = True
    torch_dtype: str = "bf16"  # bf16, fp16, fp32
    use_flash_attention: bool = True
    
    # 数据配置
    train_data_path: str = "./data/pretrain/train.jsonl"
    eval_data_path: Optional[str] = None
    max_length: int = 2048
    text_column: str = "text"
    
    # 训练配置
    output_dir: str = "./outputs/llm_pretrain"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # 优化器配置
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.01
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # 保存配置
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 10
    save_total_limit: int = 3
    
    # 性能配置
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0  # 流式数据建议用 0
    
    # 其他
    seed: int = 42
    report_to: str = "tensorboard"  # tensorboard, wandb, none


class PretrainTrainer:
    """预训练器"""
    
    def __init__(self, config: PretrainTrainingConfig):
        self.config = config
        self.accelerator = None
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.writer = None
        
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
    def setup(self):
        """初始化训练环境"""
        logger.info("=" * 60)
        logger.info("开始初始化预训练环境")
        logger.info("=" * 60)
        
        # 初始化 Accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.torch_dtype,
            log_with=self.config.report_to if self.config.report_to != "none" else None,
        )
        
        # 设置随机种子
        set_seed(self.config.seed)
        
        # 只在主进程创建输出目录
        if self.accelerator.is_main_process:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            # 保存配置
            with open(f"{self.config.output_dir}/training_config.json", "w") as f:
                json.dump(asdict(self.config), f, indent=2, default=str)
        
        # 加载 tokenizer
        self._setup_tokenizer()
        
        # 加载模型
        self._setup_model()
        
        # 设置数据加载器
        self._setup_dataloaders()
        
        # 设置优化器和学习率调度器
        self._setup_optimizer()
        
        # 设置日志
        if self.accelerator.is_main_process:
            if self.config.report_to == "tensorboard":
                self.writer = SummaryWriter(log_dir=f"{self.config.output_dir}/runs")
        
        # 使用 accelerator 准备所有组件
        self.model, self.optimizer, self.train_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.scheduler
        )
        if self.eval_dataloader:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
        
        logger.info("初始化完成!")
        logger.info(f"训练样本数 (估算): {len(self.train_dataloader.dataset)}")
        logger.info(f"每步 batch size: {self.config.per_device_train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps}")
        logger.info(f"总训练步数 (估算): {len(self.train_dataloader) * self.config.num_train_epochs}")
    
    def _setup_tokenizer(self):
        """设置分词器"""
        logger.info(f"加载 tokenizer: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            revision=self.config.model_revision,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right",
        )
        
        # 设置 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info(f"Vocab size: {len(self.tokenizer)}")
    
    def _setup_model(self):
        """设置模型"""
        logger.info(f"加载模型: {self.config.model_name}")
        
        # 确定 torch dtype
        if self.config.torch_dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif self.config.torch_dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        # 模型加载参数
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": torch_dtype,
        }
        
        # Flash Attention
        if self.config.use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("使用 Flash Attention 2")
            except Exception as e:
                logger.warning(f"Flash Attention 不可用: {e}")
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # 梯度检查点
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("启用梯度检查点")
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"总参数量: {total_params / 1e9:.2f}B")
        logger.info(f"可训练参数量: {trainable_params / 1e9:.2f}B")
    
    def _setup_dataloaders(self):
        """设置数据加载器"""
        logger.info("设置数据加载器...")
        
        # 训练数据
        train_data_config = PretrainDataConfig(
            data_path=self.config.train_data_path,
            max_length=self.config.max_length,
            text_column=self.config.text_column,
            concat_samples=True,
        )
        
        train_dataset = PretrainDataset(
            config=train_data_config,
            tokenizer=self.tokenizer,
            world_size=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
        )
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True,
        )
        
        # 验证数据
        self.eval_dataloader = None
        if self.config.eval_data_path:
            eval_data_config = PretrainDataConfig(
                data_path=self.config.eval_data_path,
                max_length=self.config.max_length,
                text_column=self.config.text_column,
                concat_samples=True,
            )
            
            eval_dataset = PretrainDataset(
                config=eval_data_config,
                tokenizer=self.tokenizer,
                world_size=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
            )
            
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.config.per_device_eval_batch_size,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=True,
            )
    
    def _setup_optimizer(self):
        """设置优化器和学习率调度器"""
        logger.info("设置优化器...")
        
        # AdamW 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        
        # 计算总训练步数
        total_steps = len(self.train_dataloader) * self.config.num_train_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        logger.info(f"总训练步数: {total_steps}")
        logger.info(f"Warmup 步数: {warmup_steps}")
        
        # 学习率调度器
        if self.config.lr_scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
    
    def train(self):
        """训练循环"""
        logger.info("=" * 60)
        logger.info("开始训练")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.config.num_train_epochs):
            self.epoch = epoch
            self._train_epoch()
            
            # 每个 epoch 结束评估
            if self.eval_dataloader:
                eval_loss = self._evaluate()
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self._save_checkpoint(is_best=True)
        
        # 保存最终模型
        self._save_checkpoint(is_final=True)
        
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"训练完成! 总耗时: {total_time / 3600:.2f} 小时")
        logger.info("=" * 60)
    
    def _train_epoch(self):
        """单个 epoch 的训练"""
        self.model.train()
        
        epoch_loss = 0.0
        step_in_epoch = 0
        
        progress_bar = self.accelerator.prepare(
            torch.utils.data.DataLoader(
                range(len(self.train_dataloader)),
                batch_size=1,
            )
        )
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.model):
                # 前向传播
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                
                # 反向传播
                self.accelerator.backward(loss)
                
                # 梯度裁剪
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                
                # 优化器步进
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # 只在梯度同步时更新统计
            if self.accelerator.sync_gradients:
                self.global_step += 1
                step_in_epoch += 1
                epoch_loss += loss.item()
                
                # 日志记录
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / step_in_epoch
                    current_lr = self.scheduler.get_last_lr()[0]
                    
                    if self.accelerator.is_main_process:
                        logger.info(
                            f"Epoch {self.epoch} | Step {self.global_step} | "
                            f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e}"
                        )
                        
                        if self.writer:
                            self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                            self.writer.add_scalar("train/lr", current_lr, self.global_step)
                
                # 保存检查点
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()
                
                # 评估
                if self.eval_dataloader and self.global_step % self.config.eval_steps == 0:
                    eval_loss = self._evaluate()
                    self.model.train()
    
    @torch.no_grad()
    def _evaluate(self) -> float:
        """评估"""
        if self.eval_dataloader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        logger.info("开始评估...")
        
        for batch in self.eval_dataloader:
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            total_loss += outputs.loss.item()
            num_batches += 1
            
            # 限制评估步数
            if num_batches >= 100:
                break
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        logger.info(f"评估结果 - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        if self.accelerator.is_main_process and self.writer:
            self.writer.add_scalar("eval/loss", avg_loss, self.global_step)
            self.writer.add_scalar("eval/perplexity", perplexity, self.global_step)
        
        return avg_loss
    
    def _save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """保存检查点"""
        if not self.accelerator.is_main_process:
            return
        
        if is_final:
            output_dir = f"{self.config.output_dir}/final"
        elif is_best:
            output_dir = f"{self.config.output_dir}/best"
        else:
            output_dir = f"{self.config.output_dir}/checkpoint-{self.global_step}"
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"保存模型到: {output_dir}")
        
        # 保存模型
        self.accelerator.unwrap_model(self.model).save_pretrained(
            output_dir,
            save_function=self.accelerator.save,
        )
        
        # 保存 tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存训练状态
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
        }
        torch.save(state, f"{output_dir}/training_state.pt")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM 预训练")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/llm_pretrain")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=2048)
    
    args = parser.parse_args()
    
    # 创建配置
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        config = PretrainTrainingConfig(**config_dict)
    else:
        config = PretrainTrainingConfig(
            model_name=args.model,
            train_data_path=args.train_data,
            eval_data_path=args.eval_data,
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            max_length=args.max_length,
        )
    
    # 开始训练
    trainer = PretrainTrainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
