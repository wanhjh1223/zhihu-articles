"""
LLM 训练流程
支持预训练、SFT 微调
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import Dataset

from ..models.base_model import LLMModel
from ...common.data_loader.llm_dataloader import LLMDataset, SFTDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    model_name: str = "qwen2.5-7b"
    
    # 数据配置
    train_data_path: str = "./data/train.jsonl"
    eval_data_path: Optional[str] = None
    max_length: int = 2048
    
    # 训练配置
    output_dir: str = "./outputs/llm"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # 保存配置
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    
    # 精度配置
    fp16: bool = False
    bf16: bool = True
    
    # LoRA 配置
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # 其他配置
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    seed: int = 42


class LLMTrainer:
    """LLM 训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        
        self._setup_model()
        self._setup_data()
    
    def _setup_model(self):
        """设置模型"""
        logger.info("正在设置模型...")
        
        # 加载模型
        self.model_wrapper = LLMModel(
            model_name_or_path=self.config.model_name,
            load_in_4bit=False,  # 训练时不使用量化
            torch_dtype="bf16" if self.config.bf16 else "fp16",
            use_flash_attention=True,
        )
        
        self.model = self.model_wrapper.model
        self.tokenizer = self.model_wrapper.tokenizer
        
        # 添加 LoRA
        if self.config.use_lora:
            self.model_wrapper.add_lora(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
            )
            self.model = self.model_wrapper.model
        
        # 启用梯度检查点
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        
        logger.info("模型设置完成")
    
    def _setup_data(self):
        """设置数据"""
        logger.info("正在设置数据...")
        
        # 加载训练数据
        self.train_dataset = SFTDataset(
            data_path=self.config.train_data_path,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )
        
        # 加载验证数据
        self.eval_dataset = None
        if self.config.eval_data_path:
            self.eval_dataset = SFTDataset(
                data_path=self.config.eval_data_path,
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
            )
        
        logger.info(f"训练数据: {len(self.train_dataset)} 条")
        if self.eval_dataset:
            logger.info(f"验证数据: {len(self.eval_dataset)} 条")
    
    def train(self):
        """开始训练"""
        logger.info("开始训练...")
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps" if self.eval_dataset else "no",
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=self.config.remove_unused_columns,
            seed=self.config.seed,
            report_to=["tensorboard"],
        )
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=-100,
        )
        
        # 创建 Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if self.eval_dataset else None,
        )
        
        # 训练
        self.trainer.train()
        
        # 保存最终模型
        self.save_model()
        
        logger.info("训练完成!")
    
    def save_model(self):
        """保存模型"""
        output_dir = Path(self.config.output_dir) / "final"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        self.trainer.save_model(output_dir)
        
        # 保存分词器
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存配置
        config_dict = {k: str(v) for k, v in vars(self.config).items()}
        with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型已保存到: {output_dir}")


class Pretrainer:
    """预训练器（继续预训练）"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
        self._setup_model()
    
    def _setup_model(self):
        """设置模型"""
        self.model_wrapper = LLMModel(
            model_name_or_path=self.config.model_name,
            load_in_4bit=False,
            torch_dtype="bf16" if self.config.bf16 else "fp16",
            use_flash_attention=True,
        )
        
        self.model = self.model_wrapper.model
        self.tokenizer = self.model_wrapper.tokenizer
        
        if self.config.use_lora:
            self.model_wrapper.add_lora(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
            )
            self.model = self.model_wrapper.model
    
    def train(self):
        """开始预训练"""
        logger.info("开始预训练...")
        
        # 加载预训练数据集（纯文本）
        from ...common.data_loader.llm_dataloader import PretrainDataset
        
        train_dataset = PretrainDataset(
            data_path=self.config.train_data_path,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            seed=self.config.seed,
        )
        
        # 使用 MLM 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # 因果语言建模
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        trainer.train()
        trainer.save_model(Path(self.config.output_dir) / "final")
        
        logger.info("预训练完成!")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM 训练")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--mode", choices=['pretrain', 'sft'], default='sft',
                       help="训练模式")
    
    # 基本参数
    parser.add_argument("--model", default="qwen2.5-7b", help="模型名称")
    parser.add_argument("--train-data", required=True, help="训练数据路径")
    parser.add_argument("--eval-data", help="验证数据路径")
    parser.add_argument("--output", default="./outputs/llm", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建配置
    config = TrainingConfig(
        model_name=args.model,
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        output_dir=args.output,
    )
    
    # 开始训练
    if args.mode == 'pretrain':
        trainer = Pretrainer(config)
    else:
        trainer = LLMTrainer(config)
    
    trainer.train()


if __name__ == "__main__":
    main()
