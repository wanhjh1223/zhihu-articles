"""
视觉编码器模块
支持 CLIP, SigLIP, EVA-CLIP, InternViT
"""

import torch
import torch.nn as nn
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoModel,
    AutoImageProcessor,
)
from typing import Optional, List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionEncoder(nn.Module):
    """视觉编码器基类"""
    
    def __init__(self, model_name: str, freeze: bool = False):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.vision_model = None
        self.image_processor = None
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        raise NotImplementedError
    
    def preprocess_images(self, images: List) -> torch.Tensor:
        """预处理图像"""
        raise NotImplementedError
    
    @property
    def hidden_size(self) -> int:
        """获取隐藏层大小"""
        raise NotImplementedError
    
    @property
    def num_patches(self) -> int:
        """获取 patch 数量"""
        raise NotImplementedError


class CLIPVisionEncoder(VisionEncoder):
    """CLIP 视觉编码器"""
    
    MODEL_NAMES = {
        'clip-vit-base': 'openai/clip-vit-base-patch32',
        'clip-vit-large': 'openai/clip-vit-large-patch14',
        'clip-vit-large-336': 'openai/clip-vit-large-patch14-336',
    }
    
    def __init__(self, model_name: str = 'clip-vit-large', freeze: bool = True):
        super().__init__(model_name, freeze)
        
        model_path = self.MODEL_NAMES.get(model_name, model_name)
        
        logger.info(f"正在加载 CLIP 视觉编码器: {model_path}")
        
        self.vision_model = CLIPVisionModel.from_pretrained(model_path)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_path)
        
        if freeze:
            self._freeze_model()
        
        logger.info(f"CLIP 视觉编码器加载完成，hidden_size={self.hidden_size}")
    
    def _freeze_model(self):
        """冻结模型参数"""
        for param in self.vision_model.parameters():
            param.requires_grad = False
        logger.info("CLIP 视觉编码器已冻结")
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            pixel_values: 图像张量 [B, C, H, W]
            
        Returns:
            图像特征 [B, num_patches, hidden_size]
        """
        outputs = self.vision_model(pixel_values, output_hidden_states=True)
        # 返回最后一层隐藏状态（去掉 CLS token）
        return outputs.hidden_states[-1][:, 1:, :]
    
    def preprocess_images(self, images: List) -> torch.Tensor:
        """预处理图像"""
        return self.image_processor(images, return_tensors="pt")["pixel_values"]
    
    @property
    def hidden_size(self) -> int:
        return self.vision_model.config.hidden_size
    
    @property
    def num_patches(self) -> int:
        # 计算 patch 数量 (假设 14x14 patch)
        image_size = self.vision_model.config.image_size
        patch_size = self.vision_model.config.patch_size
        return (image_size // patch_size) ** 2


class SigLIPVisionEncoder(VisionEncoder):
    """SigLIP 视觉编码器 (Google)"""
    
    MODEL_NAMES = {
        'siglip-base': 'google/siglip-base-patch16-224',
        'siglip-large': 'google/siglip-large-patch16-256',
        'siglip-so400m': 'google/siglip-so400m-patch14-384',
    }
    
    def __init__(self, model_name: str = 'siglip-so400m', freeze: bool = True):
        super().__init__(model_name, freeze)
        
        model_path = self.MODEL_NAMES.get(model_name, model_name)
        
        logger.info(f"正在加载 SigLIP 视觉编码器: {model_path}")
        
        self.vision_model = AutoModel.from_pretrained(model_path)
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)
        
        if freeze:
            self._freeze_model()
        
        logger.info(f"SigLIP 视觉编码器加载完成")
    
    def _freeze_model(self):
        """冻结模型参数"""
        for param in self.vision_model.parameters():
            param.requires_grad = False
        logger.info("SigLIP 视觉编码器已冻结")
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        outputs = self.vision_model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )
        # 返回 patch embeddings
        return outputs.last_hidden_state[:, 1:, :]
    
    def preprocess_images(self, images: List) -> torch.Tensor:
        """预处理图像"""
        return self.image_processor(images, return_tensors="pt")["pixel_values"]
    
    @property
    def hidden_size(self) -> int:
        return self.vision_model.config.vision_config.hidden_size
    
    @property
    def num_patches(self) -> int:
        config = self.vision_model.config.vision_config
        return (config.image_size // config.patch_size) ** 2


class InternViTEncoder(VisionEncoder):
    """InternViT 视觉编码器"""
    
    MODEL_NAMES = {
        'internvit-6b': 'OpenGVLab/InternViT-6B-448px-V1-5',
        'internvit-300m': 'OpenGVLab/InternViT-300M-448px',
    }
    
    def __init__(self, model_name: str = 'internvit-6b', freeze: bool = True):
        super().__init__(model_name, freeze)
        
        model_path = self.MODEL_NAMES.get(model_name, model_name)
        
        logger.info(f"正在加载 InternViT 视觉编码器: {model_path}")
        
        self.vision_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        if freeze:
            self._freeze_model()
        
        logger.info(f"InternViT 视觉编码器加载完成")
    
    def _freeze_model(self):
        """冻结模型参数"""
        for param in self.vision_model.parameters():
            param.requires_grad = False
        logger.info("InternViT 视觉编码器已冻结")
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        outputs = self.vision_model(pixel_values, output_hidden_states=True)
        return outputs.last_hidden_state
    
    def preprocess_images(self, images: List) -> torch.Tensor:
        """预处理图像"""
        return self.image_processor(images, return_tensors="pt")["pixel_values"]
    
    @property
    def hidden_size(self) -> int:
        return self.vision_model.config.hidden_size
    
    @property
    def num_patches(self) -> int:
        config = self.vision_model.config
        return (config.image_size // config.patch_size) ** 2


def create_vision_encoder(
    encoder_type: str = 'clip',
    model_name: Optional[str] = None,
    freeze: bool = True
) -> VisionEncoder:
    """
    创建视觉编码器工厂函数
    
    Args:
        encoder_type: 编码器类型 'clip' | 'siglip' | 'internvit'
        model_name: 具体模型名称
        freeze: 是否冻结参数
        
    Returns:
        VisionEncoder 实例
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type == 'clip':
        model_name = model_name or 'clip-vit-large'
        return CLIPVisionEncoder(model_name, freeze)
    
    elif encoder_type == 'siglip':
        model_name = model_name or 'siglip-so400m'
        return SigLIPVisionEncoder(model_name, freeze)
    
    elif encoder_type in ['internvit', 'intern_vit']:
        model_name = model_name or 'internvit-6b'
        return InternViTEncoder(model_name, freeze)
    
    else:
        raise ValueError(f"不支持的视觉编码器类型: {encoder_type}")


class VisionTower(nn.Module):
    """视觉塔（带可选的层选择）"""
    
    def __init__(self,
                 vision_encoder: VisionEncoder,
                 select_layer: int = -1,
                select_feature: str = 'patch'):
        """
        初始化视觉塔
        
        Args:
            vision_encoder: 视觉编码器
            select_layer: 选择的层索引，-1 表示最后一层
            select_feature: 特征选择 'patch' | 'cls_patch'
        """
        super().__init__()
        self.vision_encoder = vision_encoder
        self.select_layer = select_layer
        self.select_feature = select_feature
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            pixel_values: 图像张量 [B, C, H, W]
            
        Returns:
            选定的视觉特征
        """
        if hasattr(self.vision_encoder.vision_model, 'vision_model'):
            # CLIP/SigLIP 结构
            outputs = self.vision_encoder.vision_model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
        else:
            outputs = self.vision_encoder.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
        
        # 选择特定层
        if self.select_layer == -1:
            image_features = outputs.last_hidden_state
        else:
            image_features = outputs.hidden_states[self.select_layer]
        
        # 选择特征类型
        if self.select_feature == 'patch':
            # 去掉 CLS token
            image_features = image_features[:, 1:]
        
        return image_features
    
    @property
    def hidden_size(self) -> int:
        return self.vision_encoder.hidden_size
    
    @property
    def num_patches(self) -> int:
        return self.vision_encoder.num_patches
