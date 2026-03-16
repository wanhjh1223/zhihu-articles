"""
LLM 数据加载器
支持预训练、SFT、偏好数据
"""

import json
import copy
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class PretrainDataset(Dataset):
    """预训练数据集（纯文本）"""
    
    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    text = item.get('text', '')
                    if text:
                        self.data.append(text)
                except json.JSONDecodeError:
                    continue
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.data[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
        }


class SFTDataset(Dataset):
    """监督微调数据集"""
    
    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 2048,
                 system_prompt: Optional[str] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        
        # 加载数据
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载并解析数据"""
        data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    
                    # 支持多种格式
                    if 'messages' in item:
                        # conversation 格式
                        formatted = self._format_conversation(item['messages'])
                    elif 'conversations' in item:
                        # sharegpt 格式
                        formatted = self._format_sharegpt(item['conversations'])
                    elif 'instruction' in item and 'output' in item:
                        # alpaca 格式
                        formatted = self._format_alpaca(item)
                    elif 'prompt' in item and 'completion' in item:
                        # instruction 格式
                        formatted = self._format_instruction(item)
                    else:
                        continue
                    
                    data.append(formatted)
                    
                except json.JSONDecodeError:
                    continue
        
        return data
    
    def _format_conversation(self, messages: List[Dict]) -> Dict:
        """格式化 conversation 格式"""
        text = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                text += f"<|system|>\n{content}\n"
            elif role == 'user':
                text += f"<|user|>\n{content}\n"
            elif role == 'assistant':
                text += f"<|assistant|>\n{content}\n"
        
        text += "<|assistant|>"
        return {'text': text}
    
    def _format_sharegpt(self, conversations: List[Dict]) -> Dict:
        """格式化 sharegpt 格式"""
        text = ""
        for conv in conversations:
            from_role = conv.get('from', 'human')
            value = conv.get('value', '')
            
            if from_role == 'human':
                text += f"<|user|>\n{value}\n"
            else:
                text += f"<|assistant|>\n{value}\n"
        
        return {'text': text}
    
    def _format_alpaca(self, item: Dict) -> Dict:
        """格式化 alpaca 格式"""
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')
        
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction
        
        text = f"<|user|>\n{prompt}\n<|assistant|>\n{output}"
        return {'text': text}
    
    def _format_instruction(self, item: Dict) -> Dict:
        """格式化 instruction 格式"""
        prompt = item.get('prompt', '')
        completion = item.get('completion', '')
        
        text = f"<|user|>\n{prompt}\n<|assistant|>\n{completion}"
        return {'text': text}
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item['text']
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].squeeze()
        labels = input_ids.clone()
        
        # 只对 assistant 部分计算损失
        # 这里需要更复杂的逻辑来识别 assistant 部分
        # 简化处理：假设所有 token 都参与计算
        
        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels,
        }


class PreferenceDataset(Dataset):
    """偏好数据集（用于 DPO）"""
    
    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 2048,
                 max_prompt_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        
        # 加载数据
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载偏好数据"""
        data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    
                    # 需要包含 prompt, chosen, rejected
                    if 'prompt' in item and 'chosen' in item and 'rejected' in item:
                        data.append({
                            'prompt': item['prompt'],
                            'chosen': item['chosen'],
                            'rejected': item['rejected'],
                        })
                except json.JSONDecodeError:
                    continue
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        # 编码 prompt
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            truncation=True,
            return_tensors='pt',
        )
        
        # 编码 chosen
        chosen_text = f"{prompt}{chosen}"
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )
        
        # 编码 rejected
        rejected_text = f"{prompt}{rejected}"
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'prompt_input_ids': prompt_encoding['input_ids'].squeeze(),
            'prompt_attention_mask': prompt_encoding['attention_mask'].squeeze(),
            'chosen_input_ids': chosen_encoding['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_encoding['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_encoding['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_encoding['attention_mask'].squeeze(),
        }
