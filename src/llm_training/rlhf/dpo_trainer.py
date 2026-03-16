"""
强化学习训练模块
支持 DPO、GRPO、RLHF (PPO)
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import torch
from transformers import TrainingArguments
try:
    from trl import DPOTrainer
except ImportError:
    DPOTrainer = None
from peft import LoraConfig

from llm_training.models.base_model import LLMModel
from common.data_loader.llm_dataloader import PreferenceDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RLHFConfig:
    """RLHF 训练配置"""
    # 模型配置
    model_path: str = "./outputs/llm/sft"
    ref_model_path: Optional[str] = None
    
    # 数据配置
    train_data_path: str = "./data/preference_train.jsonl"
    eval_data_path: Optional[str] = None
    max_length: int = 2048
    max_prompt_length: int = 1024
    
    # 训练配置
    output_dir: str = "./outputs/llm_dpo"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    
    # DPO 配置
    beta: float = 0.1  # DPO temperature parameter
    label_smoothing: float = 0.0
    
    # 保存配置
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    
    # 精度配置
    fp16: bool = False
    bf16: bool = True
    
    # LoRA 配置
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    seed: int = 42


class DPOTrainer:
    """DPO (Direct Preference Optimization) 训练器"""
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        
        self._setup_models()
        self._setup_data()
    
    def _setup_models(self):
        """设置模型"""
        logger.info("正在加载模型...")
        
        # 加载策略模型
        policy_wrapper = LLMModel(
            model_name_or_path=self.config.model_path,
            load_in_4bit=False,
            torch_dtype="bf16" if self.config.bf16 else "fp16",
        )
        
        self.model = policy_wrapper.model
        self.tokenizer = policy_wrapper.tokenizer
        
        # 添加 LoRA
        if self.config.use_lora:
            policy_wrapper.add_lora(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
            )
            self.model = policy_wrapper.model
        
        # 加载参考模型
        if self.config.ref_model_path:
            ref_wrapper = LLMModel(
                model_name_or_path=self.config.ref_model_path,
                load_in_4bit=False,
                torch_dtype="bf16" if self.config.bf16 else "fp16",
            )
            self.ref_model = ref_wrapper.model
        else:
            # 使用相同模型，但不进行梯度更新
            self.ref_model = None
        
        logger.info("模型加载完成")
    
    def _setup_data(self):
        """设置数据"""
        logger.info("正在加载数据...")
        
        self.train_dataset = PreferenceDataset(
            data_path=self.config.train_data_path,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
        )
        
        self.eval_dataset = None
        if self.config.eval_data_path:
            self.eval_dataset = PreferenceDataset(
                data_path=self.config.eval_data_path,
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
                max_prompt_length=self.config.max_prompt_length,
            )
        
        logger.info(f"训练数据: {len(self.train_dataset)} 条")
    
    def train(self):
        """开始 DPO 训练"""
        logger.info("开始 DPO 训练...")
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps" if self.eval_dataset else "no",
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            remove_unused_columns=False,
            seed=self.config.seed,
        )
        
        # 创建 DPO Trainer
        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            beta=self.config.beta,
            label_smoothing=self.config.label_smoothing,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
        )
        
        # 训练
        trainer.train()
        
        # 保存模型
        output_path = Path(self.config.output_dir) / "final"
        trainer.save_model(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"DPO 训练完成，模型已保存到: {output_path}")


class GRPOTrainer:
    """
    GRPO (Generalized Reward-Penalty Optimization) 训练器
    简化版实现，基于 PPO
    """
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.reward_model = None
        
        self._setup_models()
    
    def _setup_models(self):
        """设置模型"""
        logger.info("正在加载模型...")
        
        # 加载策略模型（带价值头）
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
        )
        
        self.tokenizer = LLMModel(
            model_name_or_path=self.config.model_path
        ).tokenizer
        
        # 加载参考模型
        if self.config.ref_model_path:
            self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.config.ref_model_path,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            )
        
        logger.info("模型加载完成")
    
    def train(self):
        """开始 GRPO 训练"""
        logger.info("GRPO 训练尚未完全实现，建议使用 DPO")
        # TODO: 实现 GRPO 训练逻辑
        pass


class RLHFTrainer:
    """
    RLHF (PPO) 训练器
    需要奖励模型
    """
    
    def __init__(self, 
                 config: RLHFConfig,
                 reward_model_path: Optional[str] = None):
        self.config = config
        self.reward_model_path = reward_model_path
        self.ppo_trainer = None
        
    def train(self):
        """开始 RLHF 训练"""
        logger.info("RLHF (PPO) 训练需要奖励模型，建议使用 DPO 作为替代")
        # TODO: 实现 PPO 训练逻辑
        pass


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RLHF 训练")
    parser.add_argument("--mode", choices=['dpo', 'grpo', 'rlhf'], default='dpo',
                       help="训练模式")
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--ref-model", help="参考模型路径")
    parser.add_argument("--train-data", required=True, help="训练数据路径")
    parser.add_argument("--eval-data", help="验证数据路径")
    parser.add_argument("--output", default="./outputs/llm_dpo", help="输出目录")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta 参数")
    
    args = parser.parse_args()
    
    config = RLHFConfig(
        model_path=args.model,
        ref_model_path=args.ref_model,
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        output_dir=args.output,
        beta=args.beta,
    )
    
    if args.mode == 'dpo':
        trainer = DPOTrainer(config)
    elif args.mode == 'grpo':
        trainer = GRPOTrainer(config)
    else:
        trainer = RLHFTrainer(config)
    
    trainer.train()


if __name__ == "__main__":
    main()
