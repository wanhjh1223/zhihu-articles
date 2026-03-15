"""
VLM 数据加载器
支持图文多模态数据
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class VLMDataset(Dataset):
    """VLM 数据集"""
    
    def __init__(self,
                 data_path: str,
                 image_folder: str,
                 tokenizer: PreTrainedTokenizer,
                 image_token: str = "<image_patch>",
                 max_length: int = 2048):
        self.image_folder = Path(image_folder)
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.max_length = max_length
        
        # 加载数据
        self.data = self._load_data(data_path)
        
        # 图像 token ID
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载多模态数据"""
        data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    
                    # 需要包含图像路径和对话
                    if 'image' in item and 'conversations' in item:
                        data.append(item)
                    elif 'images' in item and 'messages' in item:
                        data.append({
                            'image': item['images'][0],
                            'conversations': item['messages'],
                        })
                        
                except json.JSONDecodeError:
                    continue
        
        return data
    
    def _format_conversations(self, conversations: List[Dict]) -> str:
        """格式化对话"""
        text = ""
        
        for conv in conversations:
            role = conv.get('from', conv.get('role', 'user'))
            value = conv.get('value', conv.get('content', ''))
            
            # 替换图像占位符
            if '<image>' in value or '<ImageHere>' in value:
                value = value.replace('<image>', self.image_token * 256)  # 假设 256 个图像 token
                value = value.replace('<ImageHere>', self.image_token * 256)
            
            if role in ['human', 'user']:
                text += f"<|user|>\n{value}\n"
            elif role in ['gpt', 'assistant']:
                text += f"<|assistant|>\n{value}\n"
        
        return text
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # 加载图像
        image_path = self.image_folder / item['image']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # 如果图像加载失败，使用空白图像
            image = Image.new('RGB', (224, 224), color='white')
        
        # 格式化文本
        text = self._format_conversations(item['conversations'])
        
        # 编码文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
            'images': image,
        }


class VLMCollator:
    """VLM 数据整理器"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, image_processor=None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """整理批次数据"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        # 处理图像
        images = [item['images'] for item in batch]
        
        # 如果有 image processor，进行预处理
        if self.image_processor:
            pixel_values = self.image_processor(images, return_tensors="pt")["pixel_values"]
        else:
            # 如果没有 processor，返回原始图像列表
            pixel_values = images
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pixel_values': pixel_values,
        }
