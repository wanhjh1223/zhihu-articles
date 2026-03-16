"""
VLM 训练示例
演示如何使用 VLM 训练模块
"""

import sys
sys.path.insert(0, '../../src')

from vlm_training.vision_encoder import create_vision_encoder
from vlm_training.multimodal_fusion import create_multimodal_connector
from vlm_training.training.vlm_model import VisionLanguageModel, VLMConfig


def example_vision_encoder():
    """示例：视觉编码器"""
    print("=" * 50)
    print("示例：视觉编码器")
    print("=" * 50)
    
    # 创建 CLIP 编码器
    encoder = create_vision_encoder(
        encoder_type='clip',
        model_name='clip-vit-large',
        freeze=True
    )
    
    print(f"视觉编码器创建完成")
    print(f"  - Hidden Size: {encoder.hidden_size}")
    print(f"  - Num Patches: {encoder.num_patches}")


def example_multimodal_connector():
    """示例：多模态连接器"""
    print("\n" + "=" * 50)
    print("示例：多模态连接器")
    print("=" * 50)
    
    # 创建视觉编码器
    vision_encoder = create_vision_encoder(
        encoder_type='clip',
        freeze=True
    )
    
    # 创建连接器
    connector = create_multimodal_connector(
        vision_encoder=vision_encoder,
        projector_type='mlp',
        llm_hidden_size=4096,
        num_layers=2,
    )
    
    print(f"多模态连接器创建完成")
    print(f"  - 类型: MLP")
    print(f"  - 输入维度: {vision_encoder.hidden_size}")
    print(f"  - 输出维度: 4096")


def example_vlm_model():
    """示例：VLM 模型配置"""
    print("\n" + "=" * 50)
    print("示例：VLM 模型配置")
    print("=" * 50)
    
    config = VLMConfig(
        llm_model_name="Qwen/Qwen2.5-7B",
        vision_encoder_type="clip",
        vision_model_name="clip-vit-large",
        projector_type="mlp",
        freeze_vision_encoder=True,
        freeze_llm=False,
    )
    
    print("VLM 配置已创建")
    print(f"  - LLM: {config.llm_model_name}")
    print(f"  - Vision Encoder: {config.vision_encoder_type}")
    print(f"  - Projector: {config.projector_type}")
    print(f"  - Freeze Vision: {config.freeze_vision_encoder}")
    print(f"  - Freeze LLM: {config.freeze_llm}")


if __name__ == "__main__":
    print("注意：运行完整示例需要 GPU 和模型下载")
    print("以下代码展示了如何使用 API\n")
    
    # 示例代码
    example_vision_encoder()
    example_multimodal_connector()
    example_vlm_model()
    
    print("\n" + "=" * 50)
    print("示例代码查看完成！")
    print("=" * 50)
