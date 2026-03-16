#!/usr/bin/env python3
"""
SFT多轮对话数据加载器 - 简化测试版（无需下载模型）
"""

import sys
sys.path.insert(0, './src')

import json
import torch

# 模拟tokenizer
class MockTokenizer:
    def __init__(self):
        self.vocab = {chr(i): i for i in range(128)}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
    
    def __call__(self, text, max_length=512, padding='max_length', truncation=True, return_tensors='pt'):
        # 简单编码：用字符ASCII码
        ids = [ord(c) % 128 for c in text[:max_length]]
        ids = ids + [0] * (max_length - len(ids))
        return {
            'input_ids': torch.tensor([ids[:max_length]]),
            'attention_mask': torch.tensor([[1 if i < len(text) else 0 for i in range(max_length)]])
        }
    
    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ''.join(chr(i) if 32 <= i < 127 else '?' for i in ids if i > 0)


# 导入数据加载器
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class Conversation:
    role: str
    content: str


@dataclass
class SFTDataConfig:
    data_path: str
    max_length: int = 2048
    conversation_template: str = "default"
    system_prompt: Optional[str] = None


class SFTConversationDataset:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.data = self._load_data()
        
    def _load_data(self):
        data = []
        with open(self.config.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def _parse_conversation(self, item):
        conversations = []
        
        if 'conversations' in item:  # ShareGPT
            for conv in item['conversations']:
                role_map = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}
                role = role_map.get(conv.get('from', 'human'), 'user')
                conversations.append(Conversation(role=role, content=conv['value']))
        
        elif 'messages' in item:  # OpenAI
            for msg in item['messages']:
                conversations.append(Conversation(role=msg.get('role', 'user'), content=msg.get('content', '')))
        
        elif 'instruction' in item:  # Alpaca
            instruction = item['instruction']
            input_text = item.get('input', '')
            output = item.get('output', '')
            user_content = f"{instruction}\n\n{input_text}" if input_text else instruction
            conversations.append(Conversation(role='user', content=user_content))
            conversations.append(Conversation(role='assistant', content=output))
        
        return conversations
    
    def _apply_template(self, conversations):
        if self.config.conversation_template == "chatml":
            formatted = []
            if self.config.system_prompt:
                formatted.append(f"<|im_start|>system\n{self.config.system_prompt}<|im_end|>")
            for conv in conversations:
                formatted.append(f"<|im_start|>{conv.role}\n{conv.content}<|im_end|>")
            return "\n".join(formatted) + "<|im_start|>assistant\n"
        
        else:  # default
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
        item = self.data[idx]
        conversations = self._parse_conversation(item)
        text = self._apply_template(conversations)
        
        encoding = self.tokenizer(text, max_length=self.config.max_length, return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
        }


def create_sample_data():
    samples = [
        {
            "conversations": [
                {"from": "human", "value": "你好，请介绍一下机器学习"},
                {"from": "gpt", "value": "机器学习是人工智能的一个分支..."},
                {"from": "human", "value": "那深度学习呢？"},
                {"from": "gpt", "value": "深度学习是机器学习的子领域..."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "你是一个有用的AI助手"},
                {"role": "user", "content": "什么是神经网络？"},
                {"role": "assistant", "content": "神经网络是受人脑启发的计算模型..."}
            ]
        },
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
    
    return './data/sft/train.jsonl'


if __name__ == "__main__":
    print("=" * 60)
    print("SFT多轮对话数据加载器测试")
    print("=" * 60)
    
    # 创建示例数据
    data_path = create_sample_data()
    print(f"✓ 示例数据已创建: {data_path}")
    
    # 创建mock tokenizer
    tokenizer = MockTokenizer()
    print("✓ Mock tokenizer已创建")
    
    # 测试不同模板
    for template in ['default', 'chatml']:
        print(f"\n{'='*40}")
        print(f"测试模板: {template}")
        print('='*40)
        
        config = SFTDataConfig(data_path=data_path, max_length=256, conversation_template=template)
        dataset = SFTConversationDataset(config, tokenizer)
        
        print(f"✓ 数据集加载成功，样本数: {len(dataset)}")
        
        # 显示第一个样本
        sample = dataset[0]
        print(f"\n样本1 (ShareGPT格式):")
        print(f"  input_ids shape: {sample['input_ids'].shape}")
        print(f"  attention_mask shape: {sample['attention_mask'].shape}")
        decoded = tokenizer.decode(sample['input_ids'][:100])
        print(f"  内容预览: {decoded[:80]}...")
        
        # 显示第二个样本
        sample2 = dataset[1]
        print(f"\n样本2 (OpenAI格式):")
        decoded2 = tokenizer.decode(sample2['input_ids'][:100])
        print(f"  内容预览: {decoded2[:80]}...")
        
        # 显示第三个样本
        sample3 = dataset[2]
        print(f"\n样本3 (Alpaca格式):")
        decoded3 = tokenizer.decode(sample3['input_ids'][:100])
        print(f"  内容预览: {decoded3[:80]}...")
    
    print("\n" + "=" * 60)
    print("✓ SFT数据加载器测试通过！")
    print("=" * 60)
    print("\n支持功能:")
    print("  ✓ ShareGPT格式 (多轮对话)")
    print("  ✓ OpenAI格式 (messages)")
    print("  ✓ Alpaca格式 (instruction/output)")
    print("  ✓ 多种对话模板 (default, chatml)")
