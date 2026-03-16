"""
多模态融合模块
实现视觉-语言投影层和 Connector
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class MultimodalProjector(nn.Module):
    """多模态投影层基类"""
    
    def __init__(self, vision_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size


class LinearProjector(MultimodalProjector):
    """简单的线性投影"""
    
    def __init__(self, vision_hidden_size: int, llm_hidden_size: int):
        super().__init__(vision_hidden_size, llm_hidden_size)
        self.projection = nn.Linear(vision_hidden_size, llm_hidden_size)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        投影视觉特征
        
        Args:
            vision_features: [B, num_patches, vision_hidden_size]
            
        Returns:
            [B, num_patches, llm_hidden_size]
        """
        return self.projection(vision_features)


class MLPProjector(MultimodalProjector):
    """MLP 投影层"""
    
    def __init__(self, 
                 vision_hidden_size: int, 
                 llm_hidden_size: int,
                 hidden_dim: Optional[int] = None,
                 num_layers: int = 2):
        super().__init__(vision_hidden_size, llm_hidden_size)
        
        hidden_dim = hidden_dim or llm_hidden_size
        
        layers = []
        in_dim = vision_hidden_size
        
        for i in range(num_layers):
            if i == num_layers - 1:
                layers.append(nn.Linear(in_dim, llm_hidden_size))
            else:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                ])
                in_dim = hidden_dim
        
        self.projection = nn.Sequential(*layers)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.projection(vision_features)


class QFormerProjector(MultimodalProjector):
    """Q-Former 风格的投影（简化版）"""
    
    def __init__(self,
                 vision_hidden_size: int,
                 llm_hidden_size: int,
                 num_query_tokens: int = 32,
                 num_layers: int = 2,
                 num_heads: int = 8):
        super().__init__(vision_hidden_size, llm_hidden_size)
        
        self.num_query_tokens = num_query_tokens
        
        # 可学习的查询 token
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, vision_hidden_size)
        )
        
        # Transformer 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vision_hidden_size,
            nhead=num_heads,
            dim_feedforward=vision_hidden_size * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 最终投影
        self.projection = nn.Linear(vision_hidden_size, llm_hidden_size)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        使用 Q-Former 压缩视觉特征
        
        Args:
            vision_features: [B, num_patches, vision_hidden_size]
            
        Returns:
            [B, num_query_tokens, llm_hidden_size]
        """
        batch_size = vision_features.size(0)
        
        # 扩展查询 token
        queries = self.query_tokens.expand(batch_size, -1, -1)
        
        # 拼接查询和视觉特征
        combined = torch.cat([queries, vision_features], dim=1)
        
        # Transformer 处理
        output = self.transformer(combined)
        
        # 只取查询 token 的输出
        query_output = output[:, :self.num_query_tokens, :]
        
        # 投影到 LLM 维度
        return self.projection(query_output)


class PerceiverProjector(MultimodalProjector):
    """Perceiver Resampler 投影"""
    
    def __init__(self,
                 vision_hidden_size: int,
                 llm_hidden_size: int,
                 num_latents: int = 64,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dim_head: int = 64):
        super().__init__(vision_hidden_size, llm_hidden_size)
        
        self.num_latents = num_latents
        
        # Latent 数组
        self.latents = nn.Parameter(torch.randn(1, num_latents, vision_hidden_size))
        
        # Cross-attention 层
        self.layers = nn.ModuleList([
            PerceiverLayer(vision_hidden_size, num_heads, dim_head)
            for _ in range(num_layers)
        ])
        
        # 最终投影
        self.projection = nn.Linear(vision_hidden_size, llm_hidden_size)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        batch_size = vision_features.size(0)
        latents = self.latents.expand(batch_size, -1, -1)
        
        for layer in self.layers:
            latents = layer(latents, vision_features)
        
        return self.projection(latents)


class PerceiverLayer(nn.Module):
    """Perceiver 层"""
    
    def __init__(self, dim: int, num_heads: int, dim_head: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Cross-attention
        latents_norm = self.ln1(latents)
        attn_out, _ = self.cross_attn(latents_norm, context, context)
        latents = latents + attn_out
        
        # FFN
        latents = latents + self.ffn(self.ln2(latents))
        
        return latents


class MultimodalConnector(nn.Module):
    """多模态连接器（整合视觉编码器和投影层）"""
    
    def __init__(self,
                 vision_encoder,
                 projector_type: str = 'mlp',
                 projector_config: Optional[dict] = None):
        """
        初始化多模态连接器
        
        Args:
            vision_encoder: 视觉编码器
            projector_type: 投影类型 'linear' | 'mlp' | 'qformer' | 'perceiver'
            projector_config: 投影层配置
        """
        super().__init__()
        
        self.vision_encoder = vision_encoder
        projector_config = projector_config or {}
        
        # 创建投影层
        vision_hidden = vision_encoder.hidden_size
        
        if projector_type == 'linear':
            self.projector = LinearProjector(vision_hidden, projector_config.get('llm_hidden_size', 4096))
        elif projector_type == 'mlp':
            self.projector = MLPProjector(
                vision_hidden,
                projector_config.get('llm_hidden_size', 4096),
                hidden_dim=projector_config.get('hidden_dim'),
                num_layers=projector_config.get('num_layers', 2),
            )
        elif projector_type == 'qformer':
            self.projector = QFormerProjector(
                vision_hidden,
                projector_config.get('llm_hidden_size', 4096),
                num_query_tokens=projector_config.get('num_query_tokens', 32),
                num_layers=projector_config.get('num_layers', 2),
                num_heads=projector_config.get('num_heads', 8),
            )
        elif projector_type == 'perceiver':
            self.projector = PerceiverProjector(
                vision_hidden,
                projector_config.get('llm_hidden_size', 4096),
                num_latents=projector_config.get('num_latents', 64),
                num_layers=projector_config.get('num_layers', 6),
                num_heads=projector_config.get('num_heads', 8),
            )
        else:
            raise ValueError(f"不支持的投影类型: {projector_type}")
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        处理图像并返回视觉 token
        
        Args:
            pixel_values: 图像张量
            
        Returns:
            投影后的视觉 token
        """
        # 编码视觉特征
        vision_features = self.vision_encoder(pixel_values)
        
        # 投影
        projected_features = self.projector(vision_features)
        
        return projected_features
    
    @property
    def num_vision_tokens(self) -> int:
        """获取视觉 token 数量"""
        if hasattr(self.projector, 'num_query_tokens'):
            return self.projector.num_query_tokens
        if hasattr(self.projector, 'num_latents'):
            return self.projector.num_latents
        return self.vision_encoder.num_patches


def create_multimodal_connector(
    vision_encoder,
    projector_type: str = 'mlp',
    llm_hidden_size: int = 4096,
    **kwargs
) -> MultimodalConnector:
    """
    创建多模态连接器工厂函数
    
    Args:
        vision_encoder: 视觉编码器
        projector_type: 投影类型
        llm_hidden_size: LLM 隐藏层大小
        **kwargs: 其他投影层参数
        
    Returns:
        MultimodalConnector 实例
    """
    projector_config = {
        'llm_hidden_size': llm_hidden_size,
        **kwargs
    }
    
    return MultimodalConnector(
        vision_encoder=vision_encoder,
        projector_type=projector_type,
        projector_config=projector_config,
    )
