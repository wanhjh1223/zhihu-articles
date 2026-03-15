"""
训练数据集构建模块
支持多种格式的训练数据构建
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetFormat(Enum):
    """数据集格式"""
    ALPACA = "alpaca"  # Alpaca 格式
    SHAREGPT = "sharegpt"  # ShareGPT 格式
    CONVERSATION = "conversation"  # 多轮对话格式
    INSTRUCTION = "instruction"  # 指令格式
    PREFERENCE = "preference"  # 偏好对格式 (for DPO/RLHF)


@dataclass
class DatasetConfig:
    """数据集配置"""
    format: DatasetFormat = DatasetFormat.ALPACA
    system_prompt: Optional[str] = None
    max_length: int = 8192
    split_ratio: float = 0.1  # 验证集比例
    shuffle: bool = True
    seed: int = 42


class DatasetBuilder:
    """数据集构建器"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        random.seed(config.seed)
        
    def build_from_raw(self,
                       input_files: List[str],
                       output_dir: str,
                       text_key: str = 'text'):
        """
        从原始文本构建数据集
        
        Args:
            input_files: 输入文件列表
            output_dir: 输出目录
            text_key: 文本字段名
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取所有数据
        all_data = []
        for file_path in input_files:
            data = self._load_jsonl(file_path)
            all_data.extend(data)
        
        logger.info(f"共加载 {len(all_data)} 条原始数据")
        
        # 根据格式构建数据集
        if self.config.format == DatasetFormat.ALPACA:
            dataset = self._build_alpaca_format(all_data, text_key)
        elif self.config.format == DatasetFormat.SHAREGPT:
            dataset = self._build_sharegpt_format(all_data, text_key)
        elif self.config.format == DatasetFormat.CONVERSATION:
            dataset = self._build_conversation_format(all_data, text_key)
        elif self.config.format == DatasetFormat.INSTRUCTION:
            dataset = self._build_instruction_format(all_data, text_key)
        else:
            raise ValueError(f"不支持的格式: {self.config.format}")
        
        # 划分训练集和验证集
        train_data, val_data = self._split_dataset(dataset)
        
        # 保存
        self._save_dataset(train_data, output_dir / "train.jsonl")
        self._save_dataset(val_data, output_dir / "val.jsonl")
        
        # 保存配置
        self._save_config(output_dir)
        
        logger.info(f"数据集构建完成:")
        logger.info(f"  训练集: {len(train_data)}")
        logger.info(f"  验证集: {len(val_data)}")
        logger.info(f"  输出目录: {output_dir}")
    
    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """加载 JSONL 文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return data
    
    def _save_dataset(self, data: List[Dict], output_path: Path):
        """保存数据集"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def _save_config(self, output_dir: Path):
        """保存配置"""
        config_path = output_dir / "dataset_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'format': self.config.format.value,
                'system_prompt': self.config.system_prompt,
                'max_length': self.config.max_length,
                'split_ratio': self.config.split_ratio,
                'shuffle': self.config.shuffle,
                'seed': self.config.seed,
            }, f, ensure_ascii=False, indent=2)
    
    def _split_dataset(self, dataset: List[Dict]) -> tuple:
        """划分数据集"""
        if self.config.shuffle:
            random.shuffle(dataset)
        
        split_idx = int(len(dataset) * (1 - self.config.split_ratio))
        return dataset[:split_idx], dataset[split_idx:]
    
    def _build_alpaca_format(self, 
                            data: List[Dict], 
                            text_key: str) -> List[Dict]:
        """
        构建 Alpaca 格式数据
        {
            "instruction": "...",
            "input": "...",
            "output": "..."
        }
        """
        alpaca_data = []
        
        for item in data:
            text = item.get(text_key, '')
            
            # 简单策略：将文本分为 instruction 和 output
            # 实际使用时可根据具体需求调整
            lines = text.split('\n', 1)
            instruction = lines[0][:200] if lines else ''
            output = lines[1] if len(lines) > 1 else text[200:]
            
            alpaca_item = {
                'instruction': instruction,
                'input': '',
                'output': output[:self.config.max_length],
            }
            
            if self.config.system_prompt:
                alpaca_item['system'] = self.config.system_prompt
            
            alpaca_data.append(alpaca_item)
        
        return alpaca_data
    
    def _build_sharegpt_format(self, 
                               data: List[Dict], 
                               text_key: str) -> List[Dict]:
        """
        构建 ShareGPT 格式数据
        {
            "conversations": [
                {"from": "human", "value": "..."},
                {"from": "gpt", "value": "..."}
            ]
        }
        """
        sharegpt_data = []
        
        for item in data:
            text = item.get(text_key, '')
            
            # 简单策略：模拟单轮对话
            lines = text.split('\n', 1)
            human_msg = lines[0][:500] if lines else ''
            gpt_msg = lines[1] if len(lines) > 1 else text[500:]
            
            conversations = [
                {'from': 'human', 'value': human_msg},
                {'from': 'gpt', 'value': gpt_msg[:self.config.max_length - 500]},
            ]
            
            sharegpt_item = {'conversations': conversations}
            sharegpt_data.append(sharegpt_item)
        
        return sharegpt_data
    
    def _build_conversation_format(self, 
                                   data: List[Dict], 
                                   text_key: str) -> List[Dict]:
        """
        构建多轮对话格式
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }
        """
        conversation_data = []
        
        for item in data:
            text = item.get(text_key, '')
            messages = []
            
            # 系统消息
            if self.config.system_prompt:
                messages.append({
                    'role': 'system',
                    'content': self.config.system_prompt
                })
            
            # 用户和助手消息
            lines = text.split('\n', 1)
            user_msg = lines[0][:500] if lines else ''
            assistant_msg = lines[1] if len(lines) > 1 else text[500:]
            
            messages.append({'role': 'user', 'content': user_msg})
            messages.append({
                'role': 'assistant', 
                'content': assistant_msg[:self.config.max_length - 500]
            })
            
            conversation_data.append({'messages': messages})
        
        return conversation_data
    
    def _build_instruction_format(self, 
                                   data: List[Dict], 
                                   text_key: str) -> List[Dict]:
        """
        构建指令格式
        {
            "prompt": "...",
            "completion": "..."
        }
        """
        instruction_data = []
        
        for item in data:
            text = item.get(text_key, '')
            lines = text.split('\n', 1)
            
            prompt = lines[0][:500] if lines else ''
            completion = lines[1] if len(lines) > 1 else text[500:]
            
            if self.config.system_prompt:
                prompt = f"{self.config.system_prompt}\n\n{prompt}"
            
            instruction_data.append({
                'prompt': prompt,
                'completion': completion[:self.config.max_length - 500],
            })
        
        return instruction_data


class PreferenceDatasetBuilder:
    """偏好数据集构建器（用于 DPO/RLHF）"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        random.seed(config.seed)
    
    def build_from_comparisons(self,
                               input_file: str,
                               output_dir: str):
        """
        从比较数据构建偏好数据集
        
        Args:
            input_file: 输入文件，格式：
                {
                    "prompt": "...",
                    "chosen": "...",
                    "rejected": "..."
                }
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data = self._load_jsonl(input_file)
        
        # 划分训练集和验证集
        if self.config.shuffle:
            random.shuffle(data)
        
        split_idx = int(len(data) * (1 - self.config.split_ratio))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # 保存
        self._save_dataset(train_data, output_dir / "train.jsonl")
        self._save_dataset(val_data, output_dir / "val.jsonl")
        
        logger.info(f"偏好数据集构建完成:")
        logger.info(f"  训练集: {len(train_data)}")
        logger.info(f"  验证集: {len(val_data)}")
    
    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """加载 JSONL 文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return data
    
    def _save_dataset(self, data: List[Dict], output_path: Path):
        """保存数据集"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据集构建工具")
    parser.add_argument("--input", "-i", nargs="+", required=True, 
                       help="输入文件列表")
    parser.add_argument("--output", "-o", required=True, 
                       help="输出目录")
    parser.add_argument("--format", default="alpaca",
                       choices=['alpaca', 'sharegpt', 'conversation', 'instruction'],
                       help="数据集格式")
    parser.add_argument("--text-key", default="text", 
                       help="文本字段名")
    parser.add_argument("--system-prompt", 
                       help="系统提示词")
    parser.add_argument("--split-ratio", type=float, default=0.1,
                       help="验证集比例")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    args = parser.parse_args()
    
    config = DatasetConfig(
        format=DatasetFormat(args.format),
        system_prompt=args.system_prompt,
        split_ratio=args.split_ratio,
        seed=args.seed,
    )
    
    builder = DatasetBuilder(config)
    builder.build_from_raw(args.input, args.output, text_key=args.text_key)


if __name__ == "__main__":
    main()
