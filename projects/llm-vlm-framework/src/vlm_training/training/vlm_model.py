"""
VLM (Vision Language Model) 主模型
整合视觉编码器、连接器、语言模型
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
import logging

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)

from ..vision_encoder import create_vision_encoder, VisionTower
from ..multimodal_fusion import create_multimodal_connector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VLMConfig:
    """VLM 配置"""
    # 语言模型配置
    llm_model_name: str = "Qwen/Qwen2.5-7B"
    
    # 视觉编码器配置
    vision_encoder_type: str = "clip"
    vision_model_name: str = "clip-vit-large"
    freeze_vision_encoder: bool = True
    
    # 连接器配置
    projector_type: str = "mlp"
    projector_hidden_dim: int = 4096
    num_projector_layers: int = 2
    
    # 图像配置
    image_token_index: int = 32000  # 图像 token 的索引
    image_start_token: str = "<image>"
    image_end_token: str = "</image>"
    image_token: str = "<image_patch>"
    
    # 训练配置
    freeze_llm: bool = False
    tune_mm_mlp_adapter: bool = True


class VisionLanguageModel(nn.Module):
    """视觉语言模型"""
    
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config
        
        # 加载语言模型
        logger.info(f"正在加载语言模型: {config.llm_model_name}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model_name,
            trust_remote_code=True,
            padding_side="right",
        )
        
        # 设置特殊 token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 添加图像相关 special tokens
        special_tokens = {
            'additional_special_tokens': [
                config.image_start_token,
                config.image_end_token,
                config.image_token,
            ]
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))
        
        # 获取图像 token ID
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(config.image_token)
        
        # 加载视觉编码器
        logger.info(f"正在加载视觉编码器: {config.vision_encoder_type}")
        vision_encoder = create_vision_encoder(
            encoder_type=config.vision_encoder_type,
            model_name=config.vision_model_name,
            freeze=config.freeze_vision_encoder,
        )
        self.vision_tower = VisionTower(vision_encoder)
        
        # 创建多模态连接器
        logger.info(f"正在创建连接器: {config.projector_type}")
        self.mm_connector = create_multimodal_connector(
            vision_encoder=self.vision_tower,
            projector_type=config.projector_type,
            llm_hidden_size=self.llm.config.hidden_size,
            hidden_dim=config.projector_hidden_dim,
            num_layers=config.num_projector_layers,
        )
        
        # 冻结设置
        self._setup_freezing()
        
        logger.info("VLM 模型初始化完成")
    
    def _setup_freezing(self):
        """设置参数冻结"""
        if self.config.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            logger.info("语言模型已冻结")
        
        # 连接器默认可训练
        for param in self.mm_connector.parameters():
            param.requires_grad = self.config.tune_mm_mlp_adapter
    
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        编码图像
        
        Args:
            pixel_values: 图像张量 [B, C, H, W]
            
        Returns:
            视觉特征 [B, num_tokens, hidden_size]
        """
        return self.mm_connector(pixel_values)
    
    def prepare_inputs_for_multimodal(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        准备多模态输入
        
        Args:
            input_ids: 输入 token IDs
            pixel_values: 图像像素值
            
        Returns:
            处理后的输入
        """
        if pixel_values is None:
            return {'input_ids': input_ids}
        
        # 编码图像
        image_features = self.encode_images(pixel_values)
        
        # 找到图像 token 的位置
        image_token_mask = (input_ids == self.image_token_id)
        
        # 创建新的输入嵌入
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 替换图像 token 的嵌入
        batch_size = input_ids.size(0)
        for b in range(batch_size):
            image_positions = image_token_mask[b].nonzero(as_tuple=True)[0]
            if len(image_positions) > 0:
                # 假设每个样本只有一张图片
                num_image_tokens = image_positions.numel()
                if num_image_tokens == image_features.size(1):
                    inputs_embeds[b, image_positions] = image_features[b]
                else:
                    # 需要调整图像特征数量
                    # 简单处理：重复或截断
                    adjusted_features = self._adjust_image_features(
                        image_features[b], num_image_tokens
                    )
                    inputs_embeds[b, image_positions] = adjusted_features
        
        return {'inputs_embeds': inputs_embeds}
    
    def _adjust_image_features(
        self,
        image_features: torch.Tensor,
        num_tokens: int
    ) -> torch.Tensor:
        """调整图像特征数量以匹配 token 数量"""
        current_tokens = image_features.size(0)
        
        if current_tokens == num_tokens:
            return image_features
        elif current_tokens > num_tokens:
            # 截断
            return image_features[:num_tokens]
        else:
            # 重复最后一个特征
            repeat_times = num_tokens - current_tokens
            padding = image_features[-1:].repeat(repeat_times, 1)
            return torch.cat([image_features, padding], dim=0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs
            pixel_values: 图像像素值
            attention_mask: 注意力掩码
            labels: 标签
            
        Returns:
            模型输出
        """
        # 准备多模态输入
        if pixel_values is not None:
            inputs = self.prepare_inputs_for_multimodal(input_ids, pixel_values)
        else:
            inputs = {'input_ids': input_ids}
        
        # 前向传播
        outputs = self.llm(
            attention_mask=attention_mask,
            labels=labels,
            **inputs,
            **kwargs
        )
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> torch.Tensor:
        """
        生成文本
        
        Args:
            input_ids: 输入 token IDs
            pixel_values: 图像像素值
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: 核采样参数
            
        Returns:
            生成的 token IDs
        """
        # 准备输入
        if pixel_values is not None:
            inputs = self.prepare_inputs_for_multimodal(input_ids, pixel_values)
        else:
            inputs = {'input_ids': input_ids}
        
        # 生成
        with torch.no_grad():
            output_ids = self.llm.generate(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **inputs,
                **kwargs
            )
        
        return output_ids
    
    def save_pretrained(self, output_dir: str):
        """保存模型"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存语言模型
        self.llm.save_pretrained(output_dir)
        
        # 保存分词器
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存连接器
        connector_path = os.path.join(output_dir, "mm_connector.pt")
        torch.save(self.mm_connector.state_dict(), connector_path)
        
        # 保存视觉编码器配置
        vision_config_path = os.path.join(output_dir, "vision_config.json")
        import json
        with open(vision_config_path, 'w') as f:
            json.dump({
                'vision_encoder_type': self.config.vision_encoder_type,
                'vision_model_name': self.config.vision_model_name,
                'projector_type': self.config.projector_type,
            }, f)
        
        logger.info(f"模型已保存到: {output_dir}")
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """从预训练模型加载"""
        # 加载配置
        import json
        config_path = f"{model_path}/vision_config.json"
        with open(config_path, 'r') as f:
            vision_config = json.load(f)
        
        # 创建配置
        config = VLMConfig(
            llm_model_name=model_path,
            vision_encoder_type=vision_config['vision_encoder_type'],
            vision_model_name=vision_config['vision_model_name'],
            projector_type=vision_config['projector_type'],
        )
        
        # 创建模型
        model = cls(config)
        
        # 加载连接器权重
        connector_path = f"{model_path}/mm_connector.pt"
        if os.path.exists(connector_path):
            state_dict = torch.load(connector_path, map_location='cpu')
            model.mm_connector.load_state_dict(state_dict)
        
        return model


def create_vlm(config: VLMConfig) -> VisionLanguageModel:
    """创建 VLM 工厂函数"""
    return VisionLanguageModel(config)
