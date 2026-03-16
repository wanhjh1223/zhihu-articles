"""
数据清洗和预处理模块
"""

import re
import json
import html
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import logging

import jieba
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """清洗配置"""
    remove_html: bool = True
    remove_urls: bool = False
    remove_emails: bool = False
    remove_phone: bool = False
    remove_emoji: bool = False
    normalize_whitespace: bool = True
    min_length: int = 10
    max_length: int = 8192
    remove_duplicates: bool = True
    dedup_window: int = 1000


class TextCleaner:
    """文本清洗器"""
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        
    def clean(self, text: str) -> Optional[str]:
        """
        清洗单条文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本，如果不符合要求返回 None
        """
        if not text or not isinstance(text, str):
            return None
        
        # HTML 解码和清理
        if self.config.remove_html:
            text = self._remove_html(text)
        
        # URL 过滤
        if self.config.remove_urls:
            text = self._remove_urls(text)
        
        # 邮箱过滤
        if self.config.remove_emails:
            text = self._remove_emails(text)
        
        # 手机号过滤
        if self.config.remove_phone:
            text = self._remove_phone(text)
        
        # Emoji 过滤
        if self.config.remove_emoji:
            text = self._remove_emoji(text)
        
        # 空白字符规范化
        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        # 长度检查
        if len(text) < self.config.min_length:
            return None
        if len(text) > self.config.max_length:
            text = text[:self.config.max_length]
        
        return text.strip()
    
    def _remove_html(self, text: str) -> str:
        """移除 HTML 标签"""
        # 先解码 HTML 实体
        text = html.unescape(text)
        # 移除 HTML 标签
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    
    def _remove_urls(self, text: str) -> str:
        """移除 URL"""
        url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
        return re.sub(url_pattern, ' ', text)
    
    def _remove_emails(self, text: str) -> str:
        """移除邮箱地址"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, ' ', text)
    
    def _remove_phone(self, text: str) -> str:
        """移除手机号"""
        phone_pattern = r'1[3-9]\d{9}'
        return re.sub(phone_pattern, '***', text)
    
    def _remove_emoji(self, text: str) -> str:
        """移除 Emoji"""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # 表情符号
            "\U0001F300-\U0001F5FF"  # 符号和象形
            "\U0001F680-\U0001F6FF"  # 交通和地图
            "\U0001F1E0-\U0001F1FF"  # 国旗
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r' ', text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """规范化空白字符"""
        # 替换多种空白为单个空格
        text = re.sub(r'\s+', ' ', text)
        # 移除控制字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text


class DataDeduplicator:
    """数据去重器"""
    
    def __init__(self, method: str = 'hash', threshold: float = 0.9):
        """
        初始化去重器
        
        Args:
            method: 去重方法 'hash' | 'simhash' | 'minhash'
            threshold: 相似度阈值
        """
        self.method = method
        self.threshold = threshold
        self.seen_hashes = set()
        
    def deduplicate(self, texts: List[str]) -> List[str]:
        """
        去重文本列表
        
        Args:
            texts: 文本列表
            
        Returns:
            去重后的文本列表
        """
        if self.method == 'hash':
            return self._hash_deduplicate(texts)
        elif self.method == 'simhash':
            return self._simhash_deduplicate(texts)
        else:
            return self._minhash_deduplicate(texts)
    
    def _hash_deduplicate(self, texts: List[str]) -> List[str]:
        """基于精确哈希的去重"""
        unique_texts = []
        seen = set()
        
        for text in texts:
            text_hash = hash(text)
            if text_hash not in seen:
                seen.add(text_hash)
                unique_texts.append(text)
        
        logger.info(f"去重: {len(texts)} -> {len(unique_texts)}")
        return unique_texts
    
    def _simhash_deduplicate(self, texts: List[str]) -> List[str]:
        """基于 SimHash 的模糊去重"""
        # 简化实现，实际可使用 simhash 库
        return self._hash_deduplicate(texts)
    
    def _minhash_deduplicate(self, texts: List[str]) -> List[str]:
        """基于 MinHash 的模糊去重"""
        # 简化实现，实际可使用 datasketch 库
        return self._hash_deduplicate(texts)


class QualityFilter:
    """质量过滤器"""
    
    def __init__(self):
        # 垃圾信息模式
        self.spam_patterns = [
            r'.{200,}',  # 重复字符
            r'(.)\1{20,}',  # 重复字符
            r'[\u4e00-\u9fa5]{1,2}[a-zA-Z]{1,2}[\u4e00-\u9fa5]{1,2}[a-zA-Z]{1,2}',  # 乱码模式
        ]
        
        # 广告关键词
        self.ad_keywords = [
            '加微信', '加QQ', '扫码', '优惠券', '折扣', '秒杀',
            '限时', '免费领取', '点击链接', '购买', '下单',
        ]
    
    def filter(self, text: str) -> bool:
        """
        判断文本是否通过质量过滤
        
        Args:
            text: 待检测文本
            
        Returns:
            True 表示通过过滤，False 表示过滤掉
        """
        # 检查垃圾模式
        for pattern in self.spam_patterns:
            if re.search(pattern, text):
                return False
        
        # 检查广告关键词密度
        ad_count = sum(1 for kw in self.ad_keywords if kw in text)
        if ad_count >= 3:
            return False
        
        # 检查中文字符比例
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
        if len(text) > 0 and chinese_chars / len(text) < 0.3:
            return False
        
        return True


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, 
                 cleaning_config: Optional[CleaningConfig] = None,
                 dedup_method: str = 'hash'):
        self.cleaner = TextCleaner(cleaning_config)
        self.deduplicator = DataDeduplicator(method=dedup_method)
        self.quality_filter = QualityFilter()
        
    def process_file(self, 
                     input_path: str,
                     output_path: str,
                     text_key: str = 'text',
                     batch_size: int = 1000):
        """
        处理文件
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            text_key: 文本字段名
            batch_size: 批处理大小
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"开始处理文件: {input_path}")
        
        cleaned_data = []
        total = 0
        cleaned = 0
        filtered = 0
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                total += 1
                try:
                    data = json.loads(line)
                    text = data.get(text_key, '')
                    
                    # 清洗
                    cleaned_text = self.cleaner.clean(text)
                    if cleaned_text is None:
                        continue
                    
                    # 质量过滤
                    if not self.quality_filter.filter(cleaned_text):
                        filtered += 1
                        continue
                    
                    # 更新数据
                    data[text_key] = cleaned_text
                    cleaned_data.append(data)
                    cleaned += 1
                    
                except json.JSONDecodeError:
                    continue
        
        # 去重
        if self.cleaner.config.remove_duplicates:
            texts = [d[text_key] for d in cleaned_data]
            unique_texts = self.deduplicator.deduplicate(texts)
            
            # 保留去重后的数据
            seen = set(unique_texts)
            cleaned_data = [d for d in cleaned_data if d[text_key] in seen]
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            for data in cleaned_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        logger.info(f"处理完成:")
        logger.info(f"  原始数据: {total}")
        logger.info(f"  清洗后: {cleaned}")
        logger.info(f"  质量过滤: {filtered}")
        logger.info(f"  最终保留: {len(cleaned_data)}")
        logger.info(f"  输出文件: {output_path}")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据清洗工具")
    parser.add_argument("--input", "-i", required=True, help="输入文件")
    parser.add_argument("--output", "-o", required=True, help="输出文件")
    parser.add_argument("--text-key", default="text", help="文本字段名")
    parser.add_argument("--remove-html", action="store_true", help="移除 HTML")
    parser.add_argument("--remove-urls", action="store_true", help="移除 URL")
    parser.add_argument("--remove-emoji", action="store_true", help="移除 Emoji")
    parser.add_argument("--min-length", type=int, default=10, help="最小长度")
    parser.add_argument("--max-length", type=int, default=8192, help="最大长度")
    parser.add_argument("--no-dedup", action="store_true", help="不去重")
    
    args = parser.parse_args()
    
    config = CleaningConfig(
        remove_html=args.remove_html,
        remove_urls=args.remove_urls,
        remove_emoji=args.remove_emoji,
        min_length=args.min_length,
        max_length=args.max_length,
        remove_duplicates=not args.no_dedup,
    )
    
    processor = DataProcessor(cleaning_config=config)
    processor.process_file(args.input, args.output, text_key=args.text_key)


if __name__ == "__main__":
    main()
