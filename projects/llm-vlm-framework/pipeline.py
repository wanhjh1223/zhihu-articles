#!/usr/bin/env python3
"""
LLM-VLM 训练 Pipeline
整合所有训练阶段的统一入口

支持：
- 预训练 (Pretrain)
- 监督微调 (SFT)
- DPO 对齐
- GRPO 对齐
- PPO RLHF
- VLM 训练
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pretrain(args):
    """运行预训练"""
    logger.info("=" * 60)
    logger.info("阶段 1: LLM 预训练")
    logger.info("=" * 60)
    
    from llm_training.training.pretrain_trainer import PretrainTrainer, PretrainTrainingConfig
    
    config = PretrainTrainingConfig(
        model_name=args.model_name,
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir or "./outputs/pretrain",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )
    
    trainer = PretrainTrainer(config)
    trainer.train()
    logger.info("预训练完成!")


def run_sft(args):
    """运行SFT"""
    logger.info("=" * 60)
    logger.info("阶段 2: 监督微调 (SFT)")
    logger.info("=" * 60)
    
    from llm_training.training.sft_trainer import SFTTrainer, SFTTrainingConfig
    
    config = SFTTrainingConfig(
        model_name=args.model_name,
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir or "./outputs/sft",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    trainer = SFTTrainer(config)
    trainer.train()
    logger.info("SFT完成!")


def run_dpo(args):
    """运行DPO"""
    logger.info("=" * 60)
    logger.info("阶段 3: DPO 对齐")
    logger.info("=" * 60)
    
    from llm_training.rlhf.dpo_trainer import DPOTrainer, DPOConfig
    
    config = DPOConfig(
        model_name=args.model_name,
        ref_model_name=args.ref_model,
        train_data_path=args.train_data,
        output_dir=args.output_dir or "./outputs/dpo",
        beta=args.beta,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    
    trainer = DPOTrainer(config)
    trainer.train()
    logger.info("DPO完成!")


def run_grpo(args):
    """运行GRPO"""
    logger.info("=" * 60)
    logger.info("阶段 3b: GRPO 对齐")
    logger.info("=" * 60)
    
    from llm_training.rlhf.grpo_trainer import GRPOTrainer, GRPOConfig
    
    config = GRPOConfig(
        model_name=args.model_name,
        ref_model_name=args.ref_model,
        train_data_path=args.train_data,
        output_dir=args.output_dir or "./outputs/grpo",
        group_size=args.group_size,
        kl_coeff=args.kl_coeff,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    
    trainer = GRPOTrainer(config)
    trainer.train()
    logger.info("GRPO完成!")


def run_reward_model(args):
    """训练奖励模型"""
    logger.info("=" * 60)
    logger.info("阶段 3c: 奖励模型训练")
    logger.info("=" * 60)
    
    from llm_training.rlhf.reward_trainer import RewardModelTrainer, RewardModelConfig
    
    config = RewardModelConfig(
        model_name=args.model_name,
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir or "./outputs/reward_model",
        loss_type=args.loss_type,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    
    trainer = RewardModelTrainer(config)
    trainer.train()
    logger.info("奖励模型训练完成!")


def run_ppo(args):
    """运行PPO"""
    logger.info("=" * 60)
    logger.info("阶段 3d: PPO RLHF")
    logger.info("=" * 60)
    
    from llm_training.rlhf.ppo_trainer import PPOTrainer, PPOConfig
    
    config = PPOConfig(
        model_name=args.model_name,
        reward_model_path=args.reward_model,
        ref_model_name=args.ref_model,
        train_data_path=args.train_data,
        output_dir=args.output_dir or "./outputs/ppo",
        kl_coeff=args.kl_coeff,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    
    trainer = PPOTrainer(config)
    trainer.train()
    logger.info("PPO完成!")


def run_vlm_pretrain(args):
    """运行VLM预训练"""
    logger.info("=" * 60)
    logger.info("阶段 4a: VLM 预训练")
    logger.info("=" * 60)
    
    from vlm_training.training.trainer import VLMTrainer, VLMConfig
    
    config = VLMConfig(
        llm_model_name=args.model_name,
        vision_model_name=args.vision_model,
        train_data_path=args.train_data,
        output_dir=args.output_dir or "./outputs/vlm_pretrain",
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    
    trainer = VLMTrainer(config)
    trainer.train()
    logger.info("VLM预训练完成!")


def run_vlm_sft(args):
    """运行VLM SFT"""
    logger.info("=" * 60)
    logger.info("阶段 4b: VLM 监督微调")
    logger.info("=" * 60)
    
    from vlm_training.training.trainer import VLMTrainer, VLMConfig
    
    config = VLMConfig(
        llm_model_name=args.model_name,
        vision_model_name=args.vision_model,
        train_data_path=args.train_data,
        output_dir=args.output_dir or "./outputs/vlm_sft",
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    
    trainer = VLMTrainer(config)
    trainer.train()
    logger.info("VLM SFT完成!")


def run_eval(args):
    """运行评估"""
    logger.info("=" * 60)
    logger.info("模型评估")
    logger.info("=" * 60)
    
    from common.evaluation.evaluator import Evaluator, EvalConfig
    
    config = EvalConfig(
        model_path=args.model_path,
        eval_tasks=args.eval_tasks.split(','),
        output_dir=args.output_dir or "./eval_results",
    )
    
    evaluator = Evaluator(config)
    
    # 加载评估数据
    import json
    with open(args.eval_data, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    results = evaluator.run_full_eval(eval_data)
    logger.info("评估完成!")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="LLM-VLM 训练 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 预训练
  python pipeline.py pretrain --model_name gpt2 --train_data ./data/train.jsonl
  
  # SFT
  python pipeline.py sft --model_name ./outputs/pretrain/final --train_data ./data/sft.jsonl
  
  # DPO
  python pipeline.py dpo --model_name ./outputs/sft/final --train_data ./data/dpo.jsonl
  
  # GRPO
  python pipeline.py grpo --model_name ./outputs/sft/final --train_data ./data/grpo.jsonl
  
  # 训练奖励模型
  python pipeline.py reward --model_name gpt2 --train_data ./data/reward.jsonl
  
  # PPO
  python pipeline.py ppo --model_name ./outputs/sft/final --reward_model ./outputs/reward/final --train_data ./data/ppo.jsonl
  
  # VLM预训练
  python pipeline.py vlm_pretrain --model_name ./outputs/sft/final --vision_model clip-vit-base --train_data ./data/vlm.jsonl
  
  # 评估
  python pipeline.py eval --model_path ./outputs/sft/final --eval_data ./data/eval.json --eval_tasks perplexity,generation
"""
    )
    
    subparsers = parser.add_subparsers(dest='stage', help='训练阶段')
    
    # 预训练
    pretrain_parser = subparsers.add_parser('pretrain', help='LLM预训练')
    pretrain_parser.add_argument('--model_name', required=True, help='模型名称或路径')
    pretrain_parser.add_argument('--train_data', required=True, help='训练数据路径')
    pretrain_parser.add_argument('--eval_data', help='验证数据路径')
    pretrain_parser.add_argument('--output_dir', help='输出目录')
    pretrain_parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    pretrain_parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    pretrain_parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    pretrain_parser.add_argument('--max_length', type=int, default=2048, help='最大长度')
    
    # SFT
    sft_parser = subparsers.add_parser('sft', help='监督微调')
    sft_parser.add_argument('--model_name', required=True, help='模型名称或路径')
    sft_parser.add_argument('--train_data', required=True, help='训练数据路径')
    sft_parser.add_argument('--eval_data', help='验证数据路径')
    sft_parser.add_argument('--output_dir', help='输出目录')
    sft_parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    sft_parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    sft_parser.add_argument('--learning_rate', type=float, default=2e-4, help='学习率')
    
    # DPO
    dpo_parser = subparsers.add_parser('dpo', help='DPO对齐')
    dpo_parser.add_argument('--model_name', required=True, help='模型名称或路径')
    dpo_parser.add_argument('--ref_model', help='参考模型路径')
    dpo_parser.add_argument('--train_data', required=True, help='训练数据路径')
    dpo_parser.add_argument('--output_dir', help='输出目录')
    dpo_parser.add_argument('--beta', type=float, default=0.1, help='DPO温度参数')
    dpo_parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    dpo_parser.add_argument('--learning_rate', type=float, default=1e-6, help='学习率')
    
    # GRPO
    grpo_parser = subparsers.add_parser('grpo', help='GRPO对齐')
    grpo_parser.add_argument('--model_name', required=True, help='模型名称或路径')
    grpo_parser.add_argument('--ref_model', help='参考模型路径')
    grpo_parser.add_argument('--train_data', required=True, help='训练数据路径')
    grpo_parser.add_argument('--output_dir', help='输出目录')
    grpo_parser.add_argument('--group_size', type=int, default=4, help='采样组大小')
    grpo_parser.add_argument('--kl_coeff', type=float, default=0.1, help='KL系数')
    grpo_parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    grpo_parser.add_argument('--learning_rate', type=float, default=1e-6, help='学习率')
    
    # 奖励模型
    reward_parser = subparsers.add_parser('reward', help='训练奖励模型')
    reward_parser.add_argument('--model_name', required=True, help='模型名称或路径')
    reward_parser.add_argument('--train_data', required=True, help='训练数据路径')
    reward_parser.add_argument('--eval_data', help='验证数据路径')
    reward_parser.add_argument('--output_dir', help='输出目录')
    reward_parser.add_argument('--loss_type', default='ranking', help='损失类型')
    reward_parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    reward_parser.add_argument('--learning_rate', type=float, default=1e-5, help='学习率')
    
    # PPO
    ppo_parser = subparsers.add_parser('ppo', help='PPO RLHF')
    ppo_parser.add_argument('--model_name', required=True, help='模型名称或路径')
    ppo_parser.add_argument('--reward_model', required=True, help='奖励模型路径')
    ppo_parser.add_argument('--ref_model', help='参考模型路径')
    ppo_parser.add_argument('--train_data', required=True, help='训练数据路径')
    ppo_parser.add_argument('--output_dir', help='输出目录')
    ppo_parser.add_argument('--kl_coeff', type=float, default=0.1, help='KL系数')
    ppo_parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    ppo_parser.add_argument('--learning_rate', type=float, default=1e-6, help='学习率')
    
    # VLM预训练
    vlm_pretrain_parser = subparsers.add_parser('vlm_pretrain', help='VLM预训练')
    vlm_pretrain_parser.add_argument('--model_name', required=True, help='LLM模型路径')
    vlm_pretrain_parser.add_argument('--vision_model', default='openai/clip-vit-base-patch32', help='视觉模型')
    vlm_pretrain_parser.add_argument('--train_data', required=True, help='训练数据路径')
    vlm_pretrain_parser.add_argument('--output_dir', help='输出目录')
    vlm_pretrain_parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    vlm_pretrain_parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    
    # VLM SFT
    vlm_sft_parser = subparsers.add_parser('vlm_sft', help='VLM监督微调')
    vlm_sft_parser.add_argument('--model_name', required=True, help='VLM模型路径')
    vlm_sft_parser.add_argument('--vision_model', default='openai/clip-vit-base-patch32', help='视觉模型')
    vlm_sft_parser.add_argument('--train_data', required=True, help='训练数据路径')
    vlm_sft_parser.add_argument('--output_dir', help='输出目录')
    vlm_sft_parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    vlm_sft_parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    
    # 评估
    eval_parser = subparsers.add_parser('eval', help='模型评估')
    eval_parser.add_argument('--model_path', required=True, help='模型路径')
    eval_parser.add_argument('--eval_data', required=True, help='评估数据路径')
    eval_parser.add_argument('--eval_tasks', default='perplexity', help='评估任务，逗号分隔')
    eval_parser.add_argument('--output_dir', help='输出目录')
    
    args = parser.parse_args()
    
    if args.stage is None:
        parser.print_help()
        sys.exit(1)
    
    # 执行对应阶段
    stage_functions = {
        'pretrain': run_pretrain,
        'sft': run_sft,
        'dpo': run_dpo,
        'grpo': run_grpo,
        'reward': run_reward_model,
        'ppo': run_ppo,
        'vlm_pretrain': run_vlm_pretrain,
        'vlm_sft': run_vlm_sft,
        'eval': run_eval,
    }
    
    if args.stage in stage_functions:
        stage_functions[args.stage](args)
    else:
        logger.error(f"未知阶段: {args.stage}")
        sys.exit(1)


if __name__ == "__main__":
    main()
