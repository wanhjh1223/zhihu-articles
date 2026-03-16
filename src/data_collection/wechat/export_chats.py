"""
微信聊天记录导出模块
支持导出个人聊天、群聊、公众号文章等
"""

import os
import json
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeChatDataExporter:
    """微信数据导出器"""
    
    def __init__(self, wx_path: Optional[str] = None):
        """
        初始化导出器
        
        Args:
            wx_path: 微信数据目录路径，默认自动查找
        """
        self.wx_path = wx_path or self._find_wechat_path()
        self.output_dir = None
        
    def _find_wechat_path(self) -> str:
        """自动查找微信数据目录"""
        # Windows 默认路径
        possible_paths = [
            os.path.expandvars(r"%USERPROFILE%\Documents\WeChat Files"),
            os.path.expandvars(r"%USERPROFILE%\Documents\微信"),
            "/Users/$(whoami)/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("未找到微信数据目录，请手动指定路径")
    
    def export_chats(self, 
                     output_dir: str,
                     chat_types: List[str] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> str:
        """
        导出聊天记录
        
        Args:
            output_dir: 输出目录
            chat_types: 聊天类型 ['private', 'group', 'public']
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            
        Returns:
            输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        chat_types = chat_types or ['private', 'group']
        
        logger.info(f"开始导出聊天记录到: {output_dir}")
        
        # 导出个人聊天
        if 'private' in chat_types:
            self._export_private_chats(start_date, end_date)
        
        # 导出群聊
        if 'group' in chat_types:
            self._export_group_chats(start_date, end_date)
            
        logger.info(f"聊天记录导出完成")
        return str(self.output_dir)
    
    def _export_private_chats(self, start_date: Optional[str], end_date: Optional[str]):
        """导出个人聊天"""
        logger.info("导出个人聊天...")
        
        private_dir = self.output_dir / "private_chats"
        private_dir.mkdir(exist_ok=True)
        
        # 这里需要连接微信数据库
        # 实际实现需要根据微信数据库结构调整
        # 以下为示例实现框架
        
        chats_data = []
        
        # 模拟读取数据（实际需要从微信数据库读取）
        # 微信数据库通常是加密的 SQLite，需要解密
        
        output_file = private_dir / "private_chats.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for chat in chats_data:
                f.write(json.dumps(chat, ensure_ascii=False) + '\n')
        
        logger.info(f"个人聊天已导出到: {output_file}")
    
    def _export_group_chats(self, start_date: Optional[str], end_date: Optional[str]):
        """导出群聊"""
        logger.info("导出群聊...")
        
        group_dir = self.output_dir / "group_chats"
        group_dir.mkdir(exist_ok=True)
        
        chats_data = []
        
        output_file = group_dir / "group_chats.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for chat in chats_data:
                f.write(json.dumps(chat, ensure_ascii=False) + '\n')
        
        logger.info(f"群聊已导出到: {output_file}")
    
    def export_public_articles(self, 
                               output_dir: str,
                               account_list: Optional[List[str]] = None) -> str:
        """
        导出公众号文章
        
        Args:
            output_dir: 输出目录
            account_list: 指定公众号列表，None表示全部
            
        Returns:
            输出目录路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("导出公众号文章...")
        
        # 这里可以结合爬虫抓取公众号文章
        # 或使用微信内置的收藏/历史记录
        
        articles = []
        
        output_file = output_path / "public_articles.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        logger.info(f"公众号文章已导出到: {output_file}")
        return str(output_path)


class WeChatLoginExporter:
    """通过微信网页版/itchat 导出（需扫码登录）"""
    
    def __init__(self):
        self.bot = None
        
    def login(self):
        """登录微信"""
        try:
            import itchat
            self.bot = itchat
            self.bot.auto_login(hotReload=True)
            logger.info("微信登录成功")
            return True
        except Exception as e:
            logger.error(f"登录失败: {e}")
            return False
    
    def export_contacts(self, output_file: str):
        """导出联系人列表"""
        if not self.bot:
            raise RuntimeError("请先登录")
        
        friends = self.bot.get_friends(update=True)
        contacts = []
        
        for friend in friends:
            contacts.append({
                'nickname': friend.get('NickName', ''),
                'remark_name': friend.get('RemarkName', ''),
                'sex': friend.get('Sex', 0),
                'province': friend.get('Province', ''),
                'city': friend.get('City', ''),
                'signature': friend.get('Signature', ''),
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(contacts, f, ensure_ascii=False, indent=2)
        
        logger.info(f"联系人已导出到: {output_file}")
    
    def export_recent_chats(self, output_dir: str, limit: int = 100):
        """导出最近聊天记录"""
        if not self.bot:
            raise RuntimeError("请先登录")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取聊天列表
        chatrooms = self.bot.get_chatrooms(update=True)
        
        for room in chatrooms[:limit]:
            room_name = room.get('NickName', 'unknown')
            # 获取聊天记录需要额外处理
            
        logger.info(f"最近聊天记录已导出到: {output_dir}")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="微信数据导出工具")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument("--wx-path", help="微信数据目录路径")
    parser.add_argument("--types", nargs="+", default=["private", "group"],
                       help="导出类型: private group public")
    parser.add_argument("--start-date", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--use-login", action="store_true",
                       help="使用扫码登录方式")
    
    args = parser.parse_args()
    
    if args.use_login:
        exporter = WeChatLoginExporter()
        if exporter.login():
            exporter.export_contacts(Path(args.output) / "contacts.json")
    else:
        exporter = WeChatDataExporter(wx_path=args.wx_path)
        exporter.export_chats(
            output_dir=args.output,
            chat_types=args.types,
            start_date=args.start_date,
            end_date=args.end_date
        )


if __name__ == "__main__":
    main()
