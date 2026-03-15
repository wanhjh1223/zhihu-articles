"""
RLHF (Reinforcement Learning from Human Feedback) 模块
提供多种对齐算法实现
"""

from .dpo_trainer import DPOTrainer, DPOConfig
from .grpo_trainer import GRPOTrainer, GRPOConfig
from .reward_trainer import RewardModelTrainer, RewardModelConfig
from .ppo_trainer import PPOTrainer, PPOConfig

__all__ = [
    'DPOTrainer',
    'DPOConfig',
    'GRPOTrainer', 
    'GRPOConfig',
    'RewardModelTrainer',
    'RewardModelConfig',
    'PPOTrainer',
    'PPOConfig',
]
