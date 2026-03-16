"""
微信公众号文章爬虫
支持通过搜狗微信搜索抓取公众号文章
"""

import re
import json
import time
import random
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import quote, unquote
import logging

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeChatArticleCrawler:
    """微信公众号文章爬虫"""
    
    def __init__(self, delay: float = 2.0):
        """
        初始化爬虫
        
        Args:
            delay: 请求间隔秒数
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0'
        })
        
    def search_articles(self, 
                        keyword: str, 
                        num_pages: int = 1) -> List[Dict]:
        """
        搜索公众号文章
        
        Args:
            keyword: 搜索关键词
            num_pages: 抓取页数
            
        Returns:
            文章列表
        """
        articles = []
        
        for page in range(1, num_pages + 1):
            logger.info(f"正在搜索第 {page} 页...")
            
            try:
                page_articles = self._fetch_search_page(keyword, page)
                articles.extend(page_articles)
                
                if page < num_pages:
                    time.sleep(self.delay + random.uniform(0, 1))
                    
            except Exception as e:
                logger.error(f"搜索第 {page} 页失败: {e}")
                continue
        
        logger.info(f"共找到 {len(articles)} 篇文章")
        return articles
    
    def _fetch_search_page(self, keyword: str, page: int) -> List[Dict]:
        """获取搜索页面"""
        # 搜狗微信搜索 URL
        url = f"https://weixin.sogou.com/weixin"
        params = {
            'query': keyword,
            'type': 2,  # 搜索文章
            'page': page,
        }
        
        response = self.session.get(url, params=params, timeout=30)
        response.encoding = 'utf-8'
        
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []
        
        # 解析文章列表
        for item in soup.find_all('li', class_='wx-rb'):
            try:
                title_elem = item.find('h3')
                title = title_elem.get_text(strip=True) if title_elem else ''
                
                link_elem = item.find('h3').find('a') if title_elem else None
                link = link_elem.get('href', '') if link_elem else ''
                
                summary_elem = item.find('p', class_='txt-info')
                summary = summary_elem.get_text(strip=True) if summary_elem else ''
                
                account_elem = item.find('a', class_='account')
                account = account_elem.get_text(strip=True) if account_elem else ''
                
                time_elem = item.find('span', class_='s2')
                pub_time = time_elem.get_text(strip=True) if time_elem else ''
                
                articles.append({
                    'title': title,
                    'summary': summary,
                    'account': account,
                    'pub_time': pub_time,
                    'url': link,
                    'crawl_time': datetime.now().isoformat(),
                })
                
            except Exception as e:
                logger.warning(f"解析文章失败: {e}")
                continue
        
        return articles
    
    def fetch_article_content(self, url: str) -> Dict:
        """
        获取文章内容
        
        Args:
            url: 文章 URL
            
        Returns:
            文章内容字典
        """
        try:
            # 搜狗链接需要跳转
            if 'weixin.sogou.com' in url:
                url = self._get_real_url(url)
            
            response = self.session.get(url, timeout=30)
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取标题
            title = soup.find('h1', class_='rich_media_title')
            title = title.get_text(strip=True) if title else ''
            
            # 提取作者
            author = soup.find('a', id='js_name')
            author = author.get_text(strip=True) if author else ''
            
            # 提取发布时间
            pub_time = soup.find('em', id='publish_time')
            pub_time = pub_time.get_text(strip=True) if pub_time else ''
            
            # 提取正文
            content_div = soup.find('div', id='js_content')
            
            if content_div:
                # 提取文本
                text_content = content_div.get_text(separator='\n', strip=True)
                
                # 提取图片
                images = []
                for img in content_div.find_all('img'):
                    img_url = img.get('data-src') or img.get('src', '')
                    if img_url:
                        images.append(img_url)
                
                # 提取链接
                links = []
                for a in content_div.find_all('a'):
                    href = a.get('href', '')
                    if href:
                        links.append({
                            'text': a.get_text(strip=True),
                            'url': href
                        })
            else:
                text_content = ''
                images = []
                links = []
            
            return {
                'title': title,
                'author': author,
                'pub_time': pub_time,
                'content': text_content,
                'images': images,
                'links': links,
                'url': url,
                'crawl_time': datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"获取文章内容失败: {e}")
            return {
                'url': url,
                'error': str(e),
                'crawl_time': datetime.now().isoformat(),
            }
    
    def _get_real_url(self, sogou_url: str) -> str:
        """获取真实文章 URL"""
        try:
            response = self.session.get(sogou_url, allow_redirects=True, timeout=30)
            return response.url
        except Exception as e:
            logger.error(f"获取真实 URL 失败: {e}")
            return sogou_url
    
    def crawl_account_articles(self, 
                               account_name: str,
                               max_articles: int = 100) -> List[Dict]:
        """
        爬取指定公众号文章
        
        Args:
            account_name: 公众号名称
            max_articles: 最大文章数
            
        Returns:
            文章列表
        """
        logger.info(f"开始爬取公众号 '{account_name}' 的文章...")
        
        # 搜索该公众号的文章
        articles = self.search_articles(account_name, num_pages=5)
        
        # 过滤该公众号的文章
        account_articles = [
            a for a in articles 
            if account_name.lower() in a.get('account', '').lower()
        ]
        
        # 获取完整内容
        full_articles = []
        for article in account_articles[:max_articles]:
            if article.get('url'):
                content = self.fetch_article_content(article['url'])
                full_articles.append({**article, **content})
                time.sleep(self.delay)
        
        logger.info(f"成功爬取 {len(full_articles)} 篇文章")
        return full_articles
    
    def save_articles(self, 
                      articles: List[Dict], 
                      output_path: str,
                      format: str = 'jsonl'):
        """
        保存文章
        
        Args:
            articles: 文章列表
            output_path: 输出路径
            format: 格式 jsonl 或 json
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for article in articles:
                    f.write(json.dumps(article, ensure_ascii=False) + '\n')
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"文章已保存到: {output_path}")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="微信公众号文章爬虫")
    parser.add_argument("--keyword", "-k", required=True, help="搜索关键词")
    parser.add_argument("--output", "-o", required=True, help="输出文件")
    parser.add_argument("--pages", "-p", type=int, default=3, help="抓取页数")
    parser.add_argument("--delay", "-d", type=float, default=2.0, help="请求间隔")
    parser.add_argument("--fetch-content", action="store_true", 
                       help="获取完整文章内容")
    
    args = parser.parse_args()
    
    crawler = WeChatArticleCrawler(delay=args.delay)
    
    # 搜索文章
    articles = crawler.search_articles(args.keyword, num_pages=args.pages)
    
    # 获取完整内容
    if args.fetch_content:
        full_articles = []
        for article in articles:
            if article.get('url'):
                content = crawler.fetch_article_content(article['url'])
                full_articles.append({**article, **content})
                time.sleep(args.delay)
        articles = full_articles
    
    # 保存
    crawler.save_articles(articles, args.output)


if __name__ == "__main__":
    main()
