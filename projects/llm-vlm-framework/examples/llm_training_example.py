"""
LLM 训练示例
演示如何使用 LLM 训练模块
"""

import sys
sys.path.insert(0, '../../src')

from llm_training.models.base_model import LLMModel
from llm_training.training.sft_trainer import LLMTrainer, TrainingConfig


def example_model_loading():
    """示例：加载模型"""
    print("=" * 50)
    print("示例：加载模型")
    print("=" * 50)
    
    # 加载模型（使用量化）
    model = LLMModel(
        model_name_or_path="qwen2.5-7b",
        load_in_4bit=True,
        device_map="auto",
    )
    
    print(f"模型加载完成: {model.get_model_size()}")
    
    # 添加 LoRA
    model.add_lora(r=8, lora_alpha=32)
    print(f"添加 LoRA 后: {model.get_model_size()}")
    
    return model


def example_text_generation():
    """示例：文本生成"""
    print("\n" + "=" * 50)
    print("示例：文本生成")
    print("=" * 50)
    
    # 加载模型
    model = LLMModel(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        load_in_4bit=True,
    )
    
    # 生成文本
    prompt = "介绍一下人工智能的发展历程："
    response = model.generate(
        prompt=prompt,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")


def example_training_config():
    """示例：训练配置"""
    print("\n" + "=" * 50)
    print("示例：训练配置")
    print("=" * 50)
    
    config = TrainingConfig(
        model_name="Qwen/Qwen2.5-7B",
        train_data_path="./data/train.jsonl",
        eval_data_path="./data/val.jsonl",
        output_dir="./outputs/demo",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        use_lora=True,
        lora_r=8,
        lora_alpha=32,
    )
    
    print("训练配置已创建")
    print(f"  - 模型: {config.model_name}")
    print(f"  - Epochs: {config.num_train_epochs}")
    print(f"  - Batch Size: {config.per_device_train_batch_size}")
    print(f"  - Learning Rate: {config.learning_rate}")
    print(f"  - Use LoRA: {config.use_lora}")


if __name__ == "__main__":
    print("注意：运行完整示例需要 GPU 和模型下载")
    print("以下代码展示了如何使用 API\n")
    
    # 示例代码（不实际运行，避免下载模型）
    example_training_config()
    
    print("\n" + "=" * 50)
    print("示例代码查看完成！")
    print("=" * 50)
