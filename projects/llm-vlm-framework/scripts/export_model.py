#!/usr/bin/env python3
"""
模型导出工具
支持导出为多种格式：PyTorch、GGUF、ONNX
"""

import sys
sys.path.insert(0, './src')

import os
import json
import torch
import argparse
from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer


def export_pytorch(model_path: str, output_dir: str):
    """
    导出为标准PyTorch格式
    
    Args:
        model_path: 模型路径
        output_dir: 输出目录
    """
    print(f"📦 导出PyTorch格式...")
    print(f"   输入: {model_path}")
    print(f"   输出: {output_dir}")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ PyTorch格式导出完成")
    print(f"   模型文件: {output_dir}/pytorch_model.bin")
    print(f"   配置文件: {output_dir}/config.json")


def export_gguf(model_path: str, output_path: str, quantize: str = "q4_0"):
    """
    导出为GGUF格式（用于llama.cpp）
    
    Args:
        model_path: 模型路径
        output_path: 输出文件路径
        quantize: 量化类型 (q4_0, q4_1, q5_0, q5_1, q8_0, f16, f32)
    """
    print(f"📦 导出GGUF格式...")
    print(f"   输入: {model_path}")
    print(f"   输出: {output_path}")
    print(f"   量化: {quantize}")
    
    try:
        # 尝试使用llama-cpp-python转换
        from llama_cpp import Llama
        
        # 注意：这需要llama.cpp的convert脚本
        # 这里提供命令行提示
        print("\n⚠️  请手动运行以下命令:")
        print(f"""
# 1. 克隆llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# 2. 安装依赖
pip install -r requirements.txt

# 3. 转换模型
python convert.py {model_path} --outfile {output_path} --outtype {quantize}

# 或使用llama-quantize量化
./llama-quantize {output_path.replace('.gguf', '_f16.gguf')} {output_path} {quantize}
""")
        
    except ImportError:
        print("❌ 未安装llama-cpp-python")
        print("   安装命令: pip install llama-cpp-python")


def export_onnx(model_path: str, output_dir: str, opset_version: int = 14):
    """
    导出为ONNX格式
    
    Args:
        model_path: 模型路径
        output_dir: 输出目录
        opset_version: ONNX opset版本
    """
    print(f"📦 导出ONNX格式...")
    print(f"   输入: {model_path}")
    print(f"   输出: {output_dir}")
    print(f"   Opset: {opset_version}")
    
    try:
        from transformers import AutoModelForCausalLM
        import torch.onnx
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        
        # 创建示例输入
        dummy_input = tokenizer("Hello, world!", return_tensors="pt")
        input_ids = dummy_input["input_ids"]
        attention_mask = dummy_input["attention_mask"]
        
        # 导出ONNX
        os.makedirs(output_dir, exist_ok=True)
        onnx_path = os.path.join(output_dir, "model.onnx")
        
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=opset_version,
        )
        
        print(f"✅ ONNX格式导出完成")
        print(f"   模型文件: {onnx_path}")
        
        # 验证ONNX模型
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"   ✅ ONNX模型验证通过")
        except ImportError:
            print(f"   ⚠️  未安装onnx，跳过验证")
        
    except ImportError as e:
        print(f"❌ 导出失败: {e}")
        print("   安装依赖: pip install onnx onnxruntime")


def merge_lora(base_model_path: str, lora_path: str, output_dir: str):
    """
    合并LoRA权重到基础模型
    
    Args:
        base_model_path: 基础模型路径
        lora_path: LoRA权重路径
        output_dir: 输出目录
    """
    print(f"📦 合并LoRA权重...")
    print(f"   基础模型: {base_model_path}")
    print(f"   LoRA: {lora_path}")
    print(f"   输出: {output_dir}")
    
    try:
        from peft import PeftModel, AutoPeftModelForCausalLM
        
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # 加载LoRA
        model = PeftModel.from_pretrained(model, lora_path)
        
        # 合并权重
        model = model.merge_and_unload()
        
        # 保存
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"✅ LoRA合并完成")
        print(f"   输出目录: {output_dir}")
        
    except ImportError:
        print("❌ 未安装peft")
        print("   安装命令: pip install peft")


def main():
    parser = argparse.ArgumentParser(description="模型导出工具")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出路径")
    parser.add_argument("--format", type=str, choices=["pytorch", "gguf", "onnx"], 
                       default="pytorch", help="导出格式")
    parser.add_argument("--quantize", type=str, default="q4_0", 
                       help="GGUF量化类型 (q4_0, q4_1, q5_0, q5_1, q8_0, f16, f32)")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset版本")
    
    # LoRA合并参数
    parser.add_argument("--merge_lora", action="store_true", help="合并LoRA权重")
    parser.add_argument("--lora_path", type=str, help="LoRA权重路径")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 LLM-VLM 模型导出工具")
    print("=" * 60)
    
    # 合并LoRA（如果需要）
    if args.merge_lora and args.lora_path:
        merge_lora(args.model_path, args.lora_path, args.output_path)
        model_path = args.output_path
    else:
        model_path = args.model_path
    
    # 导出
    if args.format == "pytorch":
        export_pytorch(model_path, args.output_path)
    elif args.format == "gguf":
        export_gguf(model_path, args.output_path, args.quantize)
    elif args.format == "onnx":
        export_onnx(model_path, args.output_path, args.opset)
    
    print("\n" + "=" * 60)
    print("✅ 导出完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
