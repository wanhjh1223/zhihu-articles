"""
LLM 模型定义
支持 Qwen, LLaMA, Baichuan, ChatGLM 等主流模型
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from typing import Optional, Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMModel:
    """大语言模型封装"""
    
    # 支持的模型映射
    MODEL_MAPPING = {
        'qwen': 'Qwen/Qwen2.5-7B',
        'qwen2.5-7b': 'Qwen/Qwen2.5-7B',
        'qwen2.5-14b': 'Qwen/Qwen2.5-14B',
        'llama3-8b': 'meta-llama/Meta-Llama-3-8B',
        'llama3-70b': 'meta-llama/Meta-Llama-3-70B',
        'baichuan2-7b': 'baichuan-inc/Baichuan2-7B-Base',
        'baichuan2-13b': 'baichuan-inc/Baichuan2-13B-Base',
        'chatglm3-6b': 'THUDM/chatglm3-6b',
        'glm4-9b': 'THUDM/glm-4-9b',
        'internlm2-7b': 'internlm/internlm2-7b',
        'internlm2-20b': 'internlm/internlm2-20b',
    }
    
    def __init__(self,
                 model_name_or_path: str,
                 load_in_8bit: bool = False,
                 load_in_4bit: bool = False,
                 device_map: str = "auto",
                 trust_remote_code: bool = True,
                 torch_dtype: str = "auto",
                 use_flash_attention: bool = False):
        """
        初始化模型
        
        Args:
            model_name_or_path: 模型名称或路径
            load_in_8bit: 是否使用 8bit 量化
            load_in_4bit: 是否使用 4bit 量化
            device_map: 设备映射策略
            trust_remote_code: 是否信任远程代码
            torch_dtype: 数据类型
            use_flash_attention: 是否使用 Flash Attention
        """
        self.model_name = self._resolve_model_name(model_name_or_path)
        self.tokenizer = None
        self.model = None
        self.peft_config = None
        
        # 加载参数
        self.load_kwargs = {
            'trust_remote_code': trust_remote_code,
            'device_map': device_map,
        }
        
        # 量化设置
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            self.load_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            self.load_kwargs['load_in_8bit'] = True
        
        # 数据类型
        if torch_dtype == "bf16":
            self.load_kwargs['torch_dtype'] = torch.bfloat16
        elif torch_dtype == "fp16":
            self.load_kwargs['torch_dtype'] = torch.float16
        
        # Flash Attention
        if use_flash_attention:
            self.load_kwargs['attn_implementation'] = "flash_attention_2"
        
        self._load_model()
    
    def _resolve_model_name(self, name: str) -> str:
        """解析模型名称"""
        name_lower = name.lower()
        if name_lower in self.MODEL_MAPPING:
            return self.MODEL_MAPPING[name_lower]
        return name
    
    def _load_model(self):
        """加载模型和分词器"""
        logger.info(f"正在加载模型: {self.model_name}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right",
        )
        
        # 设置填充符
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **self.load_kwargs
        )
        
        logger.info(f"模型加载完成，参数量: {self.get_model_size()}")
    
    def get_model_size(self) -> str:
        """获取模型大小"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        total_size = total_params / 1e9
        trainable_size = trainable_params / 1e9
        
        return f"{total_size:.2f}B (可训练: {trainable_size:.2f}B)"
    
    def add_lora(self,
                 r: int = 8,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1,
                 target_modules: Optional[List[str]] = None,
                 bias: str = "none",
                 task_type: str = "CAUSAL_LM"):
        """
        添加 LoRA 适配器
        
        Args:
            r: LoRA 秩
            lora_alpha: LoRA alpha 参数
            lora_dropout: Dropout 率
            target_modules: 目标模块列表
            bias: 偏置训练策略
            task_type: 任务类型
        """
        if target_modules is None:
            # 自动推断目标模块
            target_modules = self._get_default_target_modules()
        
        self.peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            task_type=task_type,
        )
        
        self.model = get_peft_model(self.model, self.peft_config)
        
        logger.info(f"LoRA 适配器已添加:")
        logger.info(f"  r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
        logger.info(f"  目标模块: {target_modules}")
        logger.info(f"  当前可训练参数量: {self.get_model_size()}")
    
    def _get_default_target_modules(self) -> List[str]:
        """获取默认目标模块"""
        model_type = self.model.config.model_type
        
        if model_type in ['qwen', 'qwen2']:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif model_type == 'llama':
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif 'baichuan' in model_type:
            return ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif 'chatglm' in model_type:
            return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif 'internlm' in model_type:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            # 默认设置
            return ["q_proj", "v_proj"]
    
    def merge_and_unload(self):
        """合并 LoRA 权重并卸载"""
        if isinstance(self.model, PeftModel):
            self.model = self.model.merge_and_unload()
            logger.info("LoRA 权重已合并到基础模型")
    
    def save_pretrained(self, output_dir: str):
        """保存模型"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(output_dir)
        
        # 保存分词器
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"模型已保存到: {output_dir}")
    
    def generate(self,
                 prompt: str,
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 repetition_penalty: float = 1.1,
                 do_sample: bool = True,
                 **kwargs) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: 核采样参数
            top_k: Top-K 采样参数
            repetition_penalty: 重复惩罚
            do_sample: 是否采样
            
        Returns:
            生成的文本
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text


def load_model_for_inference(model_path: str,
                             load_in_4bit: bool = True) -> LLMModel:
    """
    加载模型用于推理
    
    Args:
        model_path: 模型路径
        load_in_4bit: 是否使用 4bit 量化
        
    Returns:
        LLMModel 实例
    """
    return LLMModel(
        model_name_or_path=model_path,
        load_in_4bit=load_in_4bit,
        device_map="auto",
    )
