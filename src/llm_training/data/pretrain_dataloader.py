"""
LLM 预训练数据加载器
支持多种数据源和格式
"""

import os
import json
import random
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, List, Callable
from dataclasses import dataclass
import logging

import torch
from torch.utils.data import IterableDataset, Dataset
from transformers import PreTrainedTokenizer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PretrainDataConfig:
    """预训练数据配置"""
    # 数据路径
    data_path: str
    
    # 字段配置
    text_column: str = "text"
    
    # 长度配置
    max_length: int = 2048
    min_length: int = 50
    
    # 拼接配置
    concat_samples: bool = True
    concat_window: int = 1000
    
    # 数据清洗
    apply_filter: bool = True
    filter_repetition: bool = True
    max_repetition_ratio: float = 0.3
    
    # 采样配置
    shuffle_buffer_size: int = 10000
    seed: int = 42


class PretrainDataset(IterableDataset):
    """
    流式预训练数据集
    支持大文件流式读取，内存友好
    """
    
    def __init__(
        self,
        config: PretrainDataConfig,
        tokenizer: PreTrainedTokenizer,
        world_size: int = 1,
        rank: int = 0,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.world_size = world_size
        self.rank = rank
        
        # 检查数据文件
        self.data_path = Path(config.data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {config.data_path}")
        
        # 获取文件总大小用于估算长度
        self.file_size = self.data_path.stat().st_size
        
        # 设置随机种子
        random.seed(config.seed + rank)
        
        logger.info(f"初始化预训练数据集: {config.data_path}")
        logger.info(f"文件大小: {self.file_size / 1024 / 1024:.2f} MB")
        logger.info(f"Max length: {config.max_length}")
        logger.info(f"Concat samples: {config.concat_samples}")
    
    def _read_jsonl_stream(self) -> Iterator[Dict[str, Any]]:
        """流式读取 JSONL 文件"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                # 分布式采样：只处理属于当前 rank 的样本
                if line_num % self.world_size != self.rank:
                    continue
                
                try:
                    item = json.loads(line.strip())
                    yield item
                except json.JSONDecodeError:
                    logger.warning(f"解析失败，跳过第 {line_num} 行")
                    continue
    
    def _filter_text(self, text: str) -> bool:
        """过滤低质量文本"""
        if not text or not isinstance(text, str):
            return False
        
        # 长度过滤
        if len(text) < self.config.min_length:
            return False
        
        # 重复过滤
        if self.config.filter_repetition:
            # 检查字符重复率
            unique_chars = len(set(text))
            if unique_chars / len(text) < (1 - self.config.max_repetition_ratio):
                return False
            
            # 检查是否有长重复子串
            for length in [10, 20, 50]:
                if length < len(text):
                    substr = text[:length]
                    if text.count(substr) > len(text) / length * 0.5:
                        return False
        
        return True
    
    def _tokenize_text(self, text: str) -> List[int]:
        """将文本转换为 token IDs"""
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
    
    def _create_sample(self, token_ids: List[int]) -> Dict[str, torch.Tensor]:
        """创建训练样本"""
        # 截断到 max_length
        if len(token_ids) > self.config.max_length:
            token_ids = token_ids[:self.config.max_length]
        
        # 创建 attention mask
        attention_mask = [1] * len(token_ids)
        
        # Padding
        pad_length = self.config.max_length - len(token_ids)
        if pad_length > 0:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(token_ids, dtype=torch.long),
        }
    
    def _concat_samples_stream(self, samples: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """
        流式拼接样本以充分利用序列长度
        将多个短文本拼接成一个长序列
        """
        if not self.config.concat_samples:
            yield from samples
            return
        
        buffer = []
        buffer_tokens = []
        current_length = 0
        
        for sample in samples:
            text = sample.get(self.config.text_column, "")
            
            if not self._filter_text(text):
                continue
            
            token_ids = self._tokenize_text(text)
            
            # 如果单个样本就超过 max_length，直接截断输出
            if len(token_ids) >= self.config.max_length:
                yield self._create_sample(token_ids)
                continue
            
            # 尝试将当前样本加入 buffer
            if current_length + len(token_ids) <= self.config.max_length:
                buffer.append(sample)
                buffer_tokens.extend(token_ids)
                current_length += len(token_ids)
            else:
                # buffer 已满，输出拼接后的样本
                if buffer_tokens:
                    yield self._create_sample(buffer_tokens)
                
                # 开始新的 buffer
                buffer = [sample]
                buffer_tokens = token_ids[:]
                current_length = len(token_ids)
        
        # 处理剩余样本
        if buffer_tokens:
            yield self._create_sample(buffer_tokens)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """迭代器"""
        # 流式读取
        samples = self._read_jsonl_stream()
        
        # 拼接样本
        for sample in self._concat_samples_stream(samples):
            yield sample
    
    def __len__(self) -> int:
        """估算长度"""
        # 粗略估算：假设平均每行 1KB，每个样本平均 500 tokens
        avg_line_size = 1024
        estimated_lines = self.file_size / avg_line_size
        estimated_samples = estimated_lines / self.world_size
        return int(estimated_samples)


class HuggingFacePretrainDataset(IterableDataset):
    """
    基于 HuggingFace datasets 的预训练数据加载器
    支持从 HuggingFace Hub 流式加载数据
    """
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        text_column: str = "text",
        streaming: bool = True,
        split: str = "train",
        config_name: Optional[str] = None,
        **kwargs
    ):
        from datasets import load_dataset
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        
        # 加载数据集
        logger.info(f"加载 HuggingFace 数据集: {dataset_name}")
        
        load_kwargs = {
            "path": dataset_name,
            "split": split,
            "streaming": streaming,
        }
        if config_name:
            load_kwargs["name"] = config_name
        load_kwargs.update(kwargs)
        
        try:
            self.dataset = load_dataset(**load_kwargs)
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise
        
        logger.info(f"数据集加载完成")
    
    def __iter__(self):
        """迭代器"""
        buffer_tokens = []
        
        for item in self.dataset:
            text = item.get(self.text_column, "")
            
            if not text or len(text) < 50:
                continue
            
            # Tokenize
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
            )
            
            # 拼接策略
            if len(buffer_tokens) + len(token_ids) <= self.max_length:
                buffer_tokens.extend(token_ids)
            else:
                # 输出当前 buffer
                if buffer_tokens:
                    yield self._create_sample(buffer_tokens)
                buffer_tokens = token_ids[:self.max_length]
        
        # 处理剩余
        if buffer_tokens:
            yield self._create_sample(buffer_tokens)
    
    def _create_sample(self, token_ids: List[int]) -> Dict[str, torch.Tensor]:
        """创建训练样本"""
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        attention_mask = [1] * len(token_ids)
        
        # Padding
        pad_length = self.max_length - len(token_ids)
        if pad_length > 0:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(token_ids, dtype=torch.long),
        }


def create_pretrain_dataloader(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 2048,
    num_workers: int = 0,
    world_size: int = 1,
    rank: int = 0,
) -> torch.utils.data.DataLoader:
    """
    创建预训练数据加载器
    
    Args:
        data_path: 数据文件路径或 HuggingFace 数据集名称
        tokenizer: 分词器
        batch_size: batch 大小
        max_length: 最大序列长度
        num_workers: 数据加载线程数 (流式数据建议用 0)
        world_size: 分布式世界大小
        rank: 当前进程 rank
    
    Returns:
        DataLoader 实例
    """
    # 判断是本地文件还是 HuggingFace 数据集
    if Path(data_path).exists():
        # 本地文件
        config = PretrainDataConfig(
            data_path=data_path,
            max_length=max_length,
        )
        dataset = PretrainDataset(
            config=config,
            tokenizer=tokenizer,
            world_size=world_size,
            rank=rank,
        )
    else:
        # 尝试从 HuggingFace 加载
        dataset = HuggingFacePretrainDataset(
            dataset_name=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
        )
    
    # 创建 DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


# ==================== 数据加载测试 ====================

def test_pretrain_dataloader():
    """测试预训练数据加载器"""
    from transformers import AutoTokenizer
    
    # 创建测试数据
    test_data_path = "/tmp/test_pretrain_data.jsonl"
    with open(test_data_path, "w", encoding="utf-8") as f:
        for i in range(100):
            text = f"这是第{i}个测试样本。" * 50  # 制造一些长度
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")
    
    logger.info(f"创建测试数据: {test_data_path}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据加载器
    config = PretrainDataConfig(
        data_path=test_data_path,
        max_length=512,
        concat_samples=True,
    )
    
    dataset = PretrainDataset(
        config=config,
        tokenizer=tokenizer,
    )
    
    # 测试迭代
    logger.info("测试数据迭代...")
    for i, batch in enumerate(dataset):
        if i >= 3:
            break
        
        logger.info(f"\nBatch {i}:")
        logger.info(f"  input_ids shape: {batch['input_ids'].shape}")
        logger.info(f"  attention_mask shape: {batch['attention_mask'].shape}")
        
        # 解码查看
        text = tokenizer.decode(batch['input_ids'][:50], skip_special_tokens=True)
        logger.info(f"  前50个token解码: {text[:100]}...")
    
    logger.info("\n测试通过!")
    
    # 清理
    os.remove(test_data_path)


if __name__ == "__main__":
    test_pretrain_dataloader()
