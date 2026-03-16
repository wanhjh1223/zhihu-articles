#!/usr/bin/env python3
"""
SFT多轮对话数据加载器
支持多种对话格式：ShareGPT、Alpaca、OpenAI等
"""

import sys
sys.path.insert(0, './src')

import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from dataclasses import dataclass
from transformers import PreTrainedTokenizer


@dataclass
class Conversation:
    """单轮对话"""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class SFTDataConfig:
    """SFT数据配置"""
    data_path: str
    max_length: int = 2048
    conversation_template: str = "default"  # default, chatml, llama2, qwen
    system_prompt: Optional[str] = None


class SFTConversationDataset(Dataset):
    """
    SFT多轮对话数据集
    
    支持格式：
    1. ShareGPT格式: {"conversations": [{"from": "human", "value": "..."}, ...]}
    2. OpenAI格式: {"messages": [{"role": "user", "content": "..."}, ...]}
    3. Alpaca格式: {"instruction": "...", "input": "...", "output": "..."}
    """
    
    def __init__(self, config: SFTDataConfig, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """加载并解析数据"""
        data = []
        with open(self.config.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def _parse_conversation(self, item: Dict) -> List[Conversation]:
        """解析对话格式"""
        conversations = []
        
        # ShareGPT格式
        if 'conversations' in item:
            for conv in item['conversations']:
                role_map = {
                    'human': 'user',
                    'gpt': 'assistant',
                    'system': 'system'
                }
                role = role_map.get(conv.get('from', 'human'), 'user')
                conversations.append(Conversation(role=role, content=conv['value']))
        
        # OpenAI格式
        elif 'messages' in item:
            for msg in item['messages']:
                conversations.append(Conversation(
                    role=msg.get('role', 'user'),
                    content=msg.get('content', '')
                ))
        
        # Alpaca格式 (单轮)
        elif 'instruction' in item:
            instruction = item['instruction']
            input_text = item.get('input', '')
            output = item.get('output', '')
            
            if input_text:
                user_content = f"{instruction}\n\n{input_text}"
            else:
                user_content = instruction
            
            conversations.append(Conversation(role='user', content=user_content))
            conversations.append(Conversation(role='assistant', content=output))
        
        return conversations
    
    def _apply_template(self, conversations: List[Conversation]) -> str:
        """应用对话模板"""
        template = self.config.conversation_template
        
        if template == "chatml":
            # ChatML格式
            formatted = []
            if self.config.system_prompt:
                formatted.append(f"<|im_start|>system\n{self.config.system_prompt}<|im_end|>")
            for conv in conversations:
                formatted.append(f"<|im_start|>{conv.role}\n{conv.content}<|im_end|>")
            return "\n".join(formatted) + "<|im_start|>assistant\n"
        
        elif template == "llama2":
            # LLaMA2格式
            formatted = []
            if self.config.system_prompt:
                formatted.append(f"[INST] <<SYS>>\n{self.config.system_prompt}\n<</SYS>>")
            for i, conv in enumerate(conversations):
                if conv.role == 'user':
                    formatted.append(f"[INST] {conv.content} [/INST]")
                elif conv.role == 'assistant':
                    formatted.append(conv.content)
            return "\n".join(formatted)
        
        elif template == "qwen":
            # Qwen格式
            formatted = []
            if self.config.system_prompt:
                formatted.append(f"<|im_start|>system\n{self.config.system_prompt}<|im_end|>")
            for conv in conversations:
                formatted.append(f"<|im_start|>{conv.role}\n{conv.content}<|im_end|>")
            return "\n".join(formatted) + "<|im_start|>assistant\n"
        
        else:  # default
            # 简单格式
            formatted = []
            for conv in conversations:
                if conv.role == 'system':
                    formatted.append(f"System: {conv.content}")
                elif conv.role == 'user':
                    formatted.append(f"User: {conv.content}")
                elif conv.role == 'assistant':
                    formatted.append(f"Assistant: {conv.content}")
            return "\n".join(formatted) + "\nAssistant: "
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        item = self.data[idx]
        
        # 解析对话
        conversations = self._parse_conversation(item)
        
        # 应用模板
        text = self._apply_template(conversations)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
        }


def create_sample_sft_data():
    """创建示例SFT数据"""
    samples = [
        # ShareGPT格式
        {
            "conversations": [
                {"from": "human", "value": "你好，请介绍一下机器学习"},
                {"from": "gpt", "value": "机器学习是人工智能的一个分支..."},
                {"from": "human", "value": "那深度学习呢？"},
                {"from": "gpt", "value": "深度学习是机器学习的子领域..."}
            ]
        },
        # OpenAI格式
        {
            "messages": [
                {"role": "system", "content": "你是一个有用的AI助手"},
                {"role": "user", "content": "什么是神经网络？"},
                {"role": "assistant", "content": "神经网络是受人脑启发的计算模型..."}
            ]
        },
        # Alpaca格式
        {
            "instruction": "解释什么是自然语言处理",
            "input": "",
            "output": "自然语言处理(NLP)是人工智能的一个领域..."
        }
    ]
    
    import os
    os.makedirs('./data/sft', exist_ok=True)
    
    with open('./data/sft/train.jsonl', 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✓ 示例SFT数据已创建: ./data/sft/train.jsonl")
    print(f"  样本数: {len(samples)}")


if __name__ == "__main__":
    # 创建示例数据
    create_sample_sft_data()
    
    # 测试加载
    from transformers import GPT2Tokenizer
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    config = SFTDataConfig(
        data_path='./data/sft/train.jsonl',
        max_length=512,
        conversation_template='default'
    )
    
    dataset = SFTConversationDataset(config, tokenizer)
    print(f"\n✓ 数据集加载成功")
    print(f"  样本数: {len(dataset)}")
    
    # 测试第一个样本
    sample = dataset[0]
    print(f"\n第一个样本:")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  decoded: {tokenizer.decode(sample['input_ids'][:50])}...")
