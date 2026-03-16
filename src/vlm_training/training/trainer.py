"""
VLM 训练流程
支持预训练和指令微调
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
from transformers import (
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from .vlm_model import VisionLanguageModel, VLMConfig
from common.data_loader.vlm_dataloader import VLMDataset, VLMCollator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VLMTrainingConfig:
    """VLM 训练配置"""
    # 模型配置
    vlm_config: VLMConfig = None
    
    # 数据配置
    train_data_path: str = "./data/vlm_train.jsonl"
    eval_data_path: Optional[str] = None
    image_folder: str = "./data/images"
    
    # 训练配置
    output_dir: str = "./outputs/vlm"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # 保存配置
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    
    # 精度配置
    fp16: bool = False
    bf16: bool = True
    
    # 阶段
    training_stage: str = "pretrain"  # 'pretrain' | 'sft'
    
    seed: int = 42


class VLMTrainer(Trainer):
    """自定义 VLM Trainer"""
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """保存 checkpoint"""
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial)
        output_dir = Path(run_dir) / checkpoint_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(output_dir)
        
        # 保存 trainer state
        self.state.save_to_json(str(output_dir / "trainer_state.json"))
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """保存模型"""
        if output_dir is None:
            output_dir = self.args.output_dir
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)


class VLMTrainingPipeline:
    """VLM 训练流程"""
    
    def __init__(self, config: VLMTrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
        self._setup_model()
        self._setup_data()
    
    def _setup_model(self):
        """设置模型"""
        logger.info("正在设置 VLM 模型...")
        
        vlm_config = self.config.vlm_config or VLMConfig()
        self.model = VisionLanguageModel(vlm_config)
        self.tokenizer = self.model.tokenizer
        
        logger.info("VLM 模型设置完成")
    
    def _setup_data(self):
        """设置数据"""
        logger.info("正在加载数据...")
        
        self.train_dataset = VLMDataset(
            data_path=self.config.train_data_path,
            image_folder=self.config.image_folder,
            tokenizer=self.tokenizer,
            image_token=self.model.config.image_token,
        )
        
        self.eval_dataset = None
        if self.config.eval_data_path:
            self.eval_dataset = VLMDataset(
                data_path=self.config.eval_data_path,
                image_folder=self.config.image_folder,
                tokenizer=self.tokenizer,
                image_token=self.model.config.image_token,
            )
        
        logger.info(f"训练数据: {len(self.train_dataset)} 条")
    
    def train(self):
        """开始训练"""
        logger.info(f"开始 {self.config.training_stage} 训练...")
        
        # 训练参数
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
            save_total_limit=3,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            remove_unused_columns=False,
            seed=self.config.seed,
            report_to=["tensorboard"],
        )
        
        # 数据整理器
        data_collator = VLMCollator(tokenizer=self.tokenizer)
        
        # 创建 Trainer
        trainer = VLMTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )
        
        # 训练
        trainer.train()
        
        # 保存最终模型
        final_dir = Path(self.config.output_dir) / "final"
        self.model.save_pretrained(final_dir)
        
        logger.info("训练完成!")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VLM 训练")
    parser.add_argument("--stage", choices=['pretrain', 'sft'], default='sft',
                       help="训练阶段")
    parser.add_argument("--llm", default="Qwen/Qwen2.5-7B", help="语言模型")
    parser.add_argument("--vision", default="clip-vit-large", help="视觉编码器")
    parser.add_argument("--train-data", required=True, help="训练数据")
    parser.add_argument("--image-folder", required=True, help="图像文件夹")
    parser.add_argument("--output", default="./outputs/vlm", help="输出目录")
    
    args = parser.parse_args()
    
    vlm_config = VLMConfig(
        llm_model_name=args.llm,
        vision_model_name=args.vision,
        freeze_llm=(args.stage == 'pretrain'),
    )
    
    config = VLMTrainingConfig(
        vlm_config=vlm_config,
        train_data_path=args.train_data,
        image_folder=args.image_folder,
        output_dir=args.output,
        training_stage=args.stage,
    )
    
    pipeline = VLMTrainingPipeline(config)
    pipeline.train()


if __name__ == "__main__":
    main()
