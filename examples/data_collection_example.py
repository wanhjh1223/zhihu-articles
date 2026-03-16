"""
数据收集示例
演示如何使用数据收集模块
"""

import sys
sys.path.insert(0, '../../src')

from data_collection.wechat import WeChatArticleCrawler
from data_collection.preprocessing import DataProcessor, CleaningConfig
from data_collection.dataset_builder import DatasetBuilder, DatasetConfig, DatasetFormat


def example_article_crawling():
    """示例：爬取公众号文章"""
    print("=" * 50)
    print("示例：爬取公众号文章")
    print("=" * 50)
    
    # 创建爬虫
    crawler = WeChatArticleCrawler(delay=2.0)
    
    # 搜索文章
    articles = crawler.search_articles(
        keyword="人工智能",
        num_pages=2
    )
    
    print(f"找到 {len(articles)} 篇文章")
    
    # 保存文章
    crawler.save_articles(articles, "./demo_articles.jsonl")
    print("文章已保存到 ./demo_articles.jsonl")


def example_data_cleaning():
    """示例：数据清洗"""
    print("\n" + "=" * 50)
    print("示例：数据清洗")
    print("=" * 50)
    
    # 创建清洗配置
    config = CleaningConfig(
        remove_html=True,
        remove_urls=True,
        remove_emoji=False,
        min_length=50,
        max_length=2048,
        remove_duplicates=True,
    )
    
    # 创建数据处理器
    processor = DataProcessor(cleaning_config=config)
    
    # 处理文件（假设有输入文件）
    # processor.process_file(
    #     input_path="./raw_data.jsonl",
    #     output_path="./cleaned_data.jsonl",
    #     text_key="text"
    # )
    
    print("清洗配置已创建")
    print(f"  - 移除 HTML: {config.remove_html}")
    print(f"  - 最小长度: {config.min_length}")
    print(f"  - 最大长度: {config.max_length}")


def example_dataset_building():
    """示例：构建数据集"""
    print("\n" + "=" * 50)
    print("示例：构建数据集")
    print("=" * 50)
    
    # 创建数据集配置
    config = DatasetConfig(
        format=DatasetFormat.CONVERSATION,
        system_prompt="你是一个有用的 AI 助手。",
        max_length=2048,
        split_ratio=0.1,
        shuffle=True,
        seed=42,
    )
    
    # 创建数据集构建器
    builder = DatasetBuilder(config)
    
    print("数据集配置已创建")
    print(f"  - 格式: {config.format.value}")
    print(f"  - 系统提示: {config.system_prompt}")
    print(f"  - 验证集比例: {config.split_ratio}")
    
    # 构建数据集（假设有输入文件）
    # builder.build_from_raw(
    #     input_files=["./cleaned_data.jsonl"],
    #     output_dir="./dataset",
    #     text_key="text"
    # )


if __name__ == "__main__":
    # 运行示例
    # example_article_crawling()
    example_data_cleaning()
    example_dataset_building()
    
    print("\n" + "=" * 50)
    print("示例运行完成！")
    print("=" * 50)
