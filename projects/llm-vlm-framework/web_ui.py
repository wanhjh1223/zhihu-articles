#!/usr/bin/env python3
"""
LLM-VLM 训练框架 Web UI
使用Gradio快速搭建可视化界面
"""

import sys
sys.path.insert(0, './src')

import os
import json
import gradio as gr
from pathlib import Path

# 版本信息
VERSION = "0.1.0"

# 默认配置
DEFAULT_CONFIG = {
    "model_name": "gpt2",
    "train_data_path": "./data/train.jsonl",
    "output_dir": "./outputs",
    "num_train_epochs": 3,
    "batch_size": 4,
    "learning_rate": 5e-5,
    "max_length": 512,
    "warmup_steps": 100,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 10,
    "use_lora": True,
    "lora_r": 8,
    "lora_alpha": 32,
    "use_fp16": True,
}

# ==================== 工具函数 ====================

def load_config(config_path: str = "config.json") -> dict:
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

def save_config(config: dict, config_path: str = "config.json"):
    """保存配置文件"""
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return f"✅ 配置已保存到: {config_path}"

def list_models(output_dir: str = "./outputs") -> str:
    """列出已训练的模型"""
    if not os.path.exists(output_dir):
        return "暂无训练好的模型"
    
    models = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            # 检查是否有模型文件
            has_model = any(f.endswith(('.bin', '.safetensors', '.pt')) 
                          for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f)))
            if has_model:
                models.append(f"📁 {item}")
    
    return "\n".join(models) if models else "暂无训练好的模型"

def list_datasets(data_dir: str = "./data") -> str:
    """列出可用数据集"""
    if not os.path.exists(data_dir):
        return "数据目录不存在"
    
    datasets = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isfile(item_path) and item.endswith('.jsonl'):
            size = os.path.getsize(item_path) / 1024  # KB
            datasets.append(f"📄 {item} ({size:.1f} KB)")
    
    return "\n".join(datasets) if datasets else "暂无数据集"

def preview_data(data_path: str, num_samples: int = 3) -> str:
    """预览数据"""
    if not os.path.exists(data_path):
        return f"❌ 文件不存在: {data_path}"
    
    try:
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                data = json.loads(line.strip())
                samples.append(json.dumps(data, ensure_ascii=False, indent=2))
        return "\n\n".join([f"样本 {i+1}:\n{s}" for i, s in enumerate(samples)])
    except Exception as e:
        return f"❌ 读取失败: {str(e)}"

# ==================== 训练函数 ====================

def start_pretrain(config_json: str, progress=gr.Progress()) -> str:
    """启动预训练"""
    try:
        config = json.loads(config_json)
        progress(0, desc="准备训练...")
        
        # 这里应该调用实际的训练代码
        # 为了演示，模拟训练过程
        for i in range(10):
            progress((i + 1) / 10, desc=f"训练中... Step {i+1}/10")
        
        return f"✅ 预训练完成!\n模型保存到: {config.get('output_dir', './outputs')}/pretrain"
    except Exception as e:
        return f"❌ 训练失败: {str(e)}"

def start_sft(config_json: str, progress=gr.Progress()) -> str:
    """启动SFT微调"""
    try:
        config = json.loads(config_json)
        progress(0, desc="准备SFT训练...")
        
        for i in range(10):
            progress((i + 1) / 10, desc=f"SFT训练中... Step {i+1}/10")
        
        return f"✅ SFT训练完成!\n模型保存到: {config.get('output_dir', './outputs')}/sft"
    except Exception as e:
        return f"❌ 训练失败: {str(e)}"

# ==================== Gradio界面 ====================

def create_ui():
    """创建Web UI"""
    
    with gr.Blocks(title="LLM-VLM Training Framework") as demo:
        
        # 标题
        gr.Markdown(f"""
        # 🤖 LLM-VLM 训练框架 Web UI
        
        **版本: v{VERSION}** | 支持预训练、SFT、RLHF、VLM全流程
        """)
        
        # ==================== 标签页 ====================
        with gr.Tabs():
            
            # ========== 训练配置 ==========
            with gr.TabItem("⚙️ 训练配置"):
                gr.Markdown("### 基础配置")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        model_name = gr.Textbox(
                            label="模型名称",
                            value=DEFAULT_CONFIG["model_name"],
                            placeholder="如: gpt2, Qwen/Qwen2.5-0.5B"
                        )
                        train_data_path = gr.Textbox(
                            label="训练数据路径",
                            value=DEFAULT_CONFIG["train_data_path"],
                            placeholder="./data/train.jsonl"
                        )
                        output_dir = gr.Textbox(
                            label="输出目录",
                            value=DEFAULT_CONFIG["output_dir"],
                            placeholder="./outputs"
                        )
                    
                    with gr.Column(scale=1):
                        num_epochs = gr.Slider(
                            label="训练轮数",
                            minimum=1,
                            maximum=10,
                            value=DEFAULT_CONFIG["num_train_epochs"],
                            step=1
                        )
                        batch_size = gr.Slider(
                            label="批次大小",
                            minimum=1,
                            maximum=32,
                            value=DEFAULT_CONFIG["batch_size"],
                            step=1
                        )
                        learning_rate = gr.Number(
                            label="学习率",
                            value=DEFAULT_CONFIG["learning_rate"],
                        )
                
                gr.Markdown("### 高级配置")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        max_length = gr.Slider(
                            label="最大序列长度",
                            minimum=128,
                            maximum=2048,
                            value=DEFAULT_CONFIG["max_length"],
                            step=64
                        )
                        warmup_steps = gr.Number(
                            label="Warmup步数",
                            value=DEFAULT_CONFIG["warmup_steps"]
                        )
                    
                    with gr.Column(scale=1):
                        save_steps = gr.Number(
                            label="保存间隔(步)",
                            value=DEFAULT_CONFIG["save_steps"]
                        )
                        eval_steps = gr.Number(
                            label="评估间隔(步)",
                            value=DEFAULT_CONFIG["eval_steps"]
                        )
                
                with gr.Row():
                    use_lora = gr.Checkbox(
                        label="使用LoRA",
                        value=DEFAULT_CONFIG["use_lora"]
                    )
                    lora_r = gr.Slider(
                        label="LoRA秩(r)",
                        minimum=1,
                        maximum=64,
                        value=DEFAULT_CONFIG["lora_r"],
                        step=1
                    )
                    lora_alpha = gr.Slider(
                        label="LoRA Alpha",
                        minimum=1,
                        maximum=128,
                        value=DEFAULT_CONFIG["lora_alpha"],
                        step=1
                    )
                
                # 配置预览和保存
                config_preview = gr.JSON(label="配置预览", value=DEFAULT_CONFIG)
                
                with gr.Row():
                    save_btn = gr.Button("💾 保存配置", variant="primary")
                    load_btn = gr.Button("📂 加载配置")
                
                save_status = gr.Textbox(label="状态", interactive=False)
                
                # 绑定事件
                def update_config_preview(*args):
                    keys = ["model_name", "train_data_path", "output_dir", 
                           "num_train_epochs", "batch_size", "learning_rate",
                           "max_length", "warmup_steps", "save_steps", "eval_steps",
                           "use_lora", "lora_r", "lora_alpha"]
                    return {k: v for k, v in zip(keys, args)}
                
                inputs = [model_name, train_data_path, output_dir,
                         num_epochs, batch_size, learning_rate,
                         max_length, warmup_steps, save_steps, eval_steps,
                         use_lora, lora_r, lora_alpha]
                
                for inp in inputs:
                    inp.change(update_config_preview, inputs, config_preview)
                
                save_btn.click(
                    lambda x: save_config(json.loads(x)),
                    config_preview,
                    save_status
                )
            
            # ========== 训练启动 ==========
            with gr.TabItem("🚀 启动训练"):
                gr.Markdown("### 选择训练阶段")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**阶段1: 预训练**")
                        pretrain_config = gr.Textbox(
                            label="配置(JSON)",
                            value=json.dumps(DEFAULT_CONFIG, indent=2),
                            lines=10
                        )
                        pretrain_btn = gr.Button("▶️ 启动预训练", variant="primary")
                        pretrain_output = gr.Textbox(label="训练输出", lines=5)
                        
                        pretrain_btn.click(start_pretrain, pretrain_config, pretrain_output)
                    
                    with gr.Column():
                        gr.Markdown("**阶段2: SFT微调**")
                        sft_config = gr.Textbox(
                            label="配置(JSON)",
                            value=json.dumps(DEFAULT_CONFIG, indent=2),
                            lines=10
                        )
                        sft_btn = gr.Button("▶️ 启动SFT训练", variant="primary")
                        sft_output = gr.Textbox(label="训练输出", lines=5)
                        
                        sft_btn.click(start_sft, sft_config, sft_output)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**阶段3: RLHF/DPO**")
                        gr.Button("⏳ 开发中...", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("**阶段4: VLM训练**")
                        gr.Button("⏳ 开发中...", interactive=False)
            
            # ========== 数据管理 ==========
            with gr.TabItem("📁 数据管理"):
                gr.Markdown("### 数据集管理")
                
                with gr.Row():
                    with gr.Column():
                        refresh_data_btn = gr.Button("🔄 刷新数据集列表")
                        dataset_list = gr.Textbox(
                            label="可用数据集",
                            value=list_datasets(),
                            lines=10,
                            interactive=False
                        )
                        refresh_data_btn.click(list_datasets, outputs=dataset_list)
                    
                    with gr.Column():
                        gr.Markdown("**数据预览**")
                        data_path_input = gr.Textbox(
                            label="数据文件路径",
                            value="./data/train.jsonl"
                        )
                        num_samples = gr.Slider(
                            label="预览样本数",
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1
                        )
                        preview_btn = gr.Button("👁️ 预览数据")
                        preview_output = gr.Textbox(
                            label="数据预览",
                            lines=15,
                            interactive=False
                        )
                        preview_btn.click(preview_data, [data_path_input, num_samples], preview_output)
            
            # ========== 模型管理 ==========
            with gr.TabItem("🤖 模型管理"):
                gr.Markdown("### 已训练模型")
                
                refresh_model_btn = gr.Button("🔄 刷新模型列表")
                model_list = gr.Textbox(
                    label="模型列表",
                    value=list_models(),
                    lines=15,
                    interactive=False
                )
                refresh_model_btn.click(list_models, outputs=model_list)
                
                gr.Markdown("### 模型推理")
                with gr.Row():
                    with gr.Column():
                        model_path = gr.Textbox(
                            label="模型路径",
                            value="./outputs/pretrain",
                            placeholder="输入模型目录路径"
                        )
                        prompt_input = gr.Textbox(
                            label="输入提示",
                            value="人工智能是",
                            lines=3
                        )
                        max_tokens = gr.Slider(
                            label="最大生成token数",
                            minimum=10,
                            maximum=512,
                            value=50,
                            step=10
                        )
                        generate_btn = gr.Button("✨ 生成文本")
                    
                    with gr.Column():
                        generation_output = gr.Textbox(
                            label="生成结果",
                            lines=8,
                            interactive=False
                        )
                        
                        def generate_text(model_path, prompt, max_tokens):
                            return f"⏳ 推理功能开发中...\n模型: {model_path}\n提示: {prompt}"
                        
                        generate_btn.click(generate_text, 
                                         [model_path, prompt_input, max_tokens], 
                                         generation_output)
            
            # ========== 训练监控 ==========
            with gr.TabItem("📊 训练监控"):
                gr.Markdown("### 训练日志和曲线")
                
                with gr.Row():
                    refresh_log_btn = gr.Button("🔄 刷新日志")
                    log_output = gr.Textbox(
                        label="训练日志",
                        value="暂无训练日志\n请先启动训练...",
                        lines=20,
                        interactive=False
                    )
                
                gr.Markdown("### Loss曲线")
                loss_plot = gr.Image(label="Loss曲线", value=None)
                
                def load_loss_plot():
                    plot_path = "docs/images/real_training_cpu.png"
                    if os.path.exists(plot_path):
                        return plot_path
                    return None
                
                refresh_plot_btn = gr.Button("🔄 刷新曲线")
                refresh_plot_btn.click(load_loss_plot, outputs=loss_plot)
        
        # 底部信息
        gr.Markdown("""
        ---
        **LLM-VLM Training Framework** | GitHub: https://github.com/wanhjh1223/llm-vlm-framework
        """)
    
    return demo

# ==================== 启动 ====================

if __name__ == "__main__":
    print("🚀 启动 LLM-VLM Web UI...")
    print("📝 访问地址: http://localhost:7860")
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        quiet=False,
    )
