#!/usr/bin/env python3
"""
A股模拟盘自动交易脚本
模拟A股每日交易逻辑
"""
import json
import os
import sys
import random
import datetime
from pathlib import Path

# 模拟股票池（A股常见股票）
STOCK_POOL = [
    {"code": "600519.SH", "name": "贵州茅台", "price": 1680.00},
    {"code": "000858.SZ", "name": "五粮液", "price": 145.50},
    {"code": "000333.SZ", "name": "美的集团", "price": 58.20},
    {"code": "002415.SZ", "name": "海康威视", "price": 32.80},
    {"code": "600276.SH", "name": "恒瑞医药", "price": 42.60},
    {"code": "300750.SZ", "name": "宁德时代", "price": 185.30},
    {"code": "601012.SH", "name": "隆基绿能", "price": 22.15},
    {"code": "002594.SZ", "name": "比亚迪", "price": 218.50},
    {"code": "600036.SH", "name": "招商银行", "price": 31.20},
    {"code": "601318.SH", "name": "中国平安", "price": 42.80},
]

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def is_trading_day(date):
    """判断是否为交易日（简化版，仅排除周末）"""
    return date.weekday() < 5

def load_portfolio():
    """加载持仓数据"""
    portfolio_file = DATA_DIR / "portfolio.json"
    if portfolio_file.exists():
        with open(portfolio_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    # 初始资金100万
    return {
        "cash": 1000000.00,
        "positions": {},
        "trade_history": [],
        "created_at": datetime.datetime.now().isoformat()
    }

def save_portfolio(portfolio):
    """保存持仓数据"""
    portfolio_file = DATA_DIR / "portfolio.json"
    portfolio["updated_at"] = datetime.datetime.now().isoformat()
    with open(portfolio_file, 'w', encoding='utf-8') as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)

def simulate_price_change(base_price):
    """模拟股价变动（-3% ~ +3%）"""
    change_pct = random.uniform(-0.03, 0.03)
    return base_price * (1 + change_pct), change_pct

def execute_trade(portfolio, stock, action, quantity, price):
    """执行交易"""
    code = stock["code"]
    total = price * quantity
    
    if action == "buy":
        # 买入：扣除现金，增加持仓
        fee = total * 0.00025  # 佣金万分之2.5
        total_cost = total + fee
        
        if portfolio["cash"] >= total_cost:
            portfolio["cash"] -= total_cost
            if code not in portfolio["positions"]:
                portfolio["positions"][code] = {"name": stock["name"], "quantity": 0, "avg_cost": 0}
            
            pos = portfolio["positions"][code]
            old_value = pos["quantity"] * pos["avg_cost"]
            new_value = old_value + total
            pos["quantity"] += quantity
            pos["avg_cost"] = new_value / pos["quantity"] if pos["quantity"] > 0 else 0
            
            trade_record = {
                "date": datetime.datetime.now().isoformat(),
                "action": "买入",
                "code": code,
                "name": stock["name"],
                "price": round(price, 2),
                "quantity": quantity,
                "fee": round(fee, 2),
                "total": round(total_cost, 2)
            }
            portfolio["trade_history"].append(trade_record)
            return True, trade_record
        return False, "资金不足"
    
    elif action == "sell":
        # 卖出：增加现金，减少持仓
        if code in portfolio["positions"] and portfolio["positions"][code]["quantity"] >= quantity:
            fee = total * 0.00025  # 佣金
            tax = total * 0.001    # 印花税
            net_total = total - fee - tax
            
            portfolio["cash"] += net_total
            portfolio["positions"][code]["quantity"] -= quantity
            if portfolio["positions"][code]["quantity"] == 0:
                del portfolio["positions"][code]
            
            trade_record = {
                "date": datetime.datetime.now().isoformat(),
                "action": "卖出",
                "code": code,
                "name": stock["name"],
                "price": round(price, 2),
                "quantity": quantity,
                "fee": round(fee + tax, 2),
                "total": round(net_total, 2)
            }
            portfolio["trade_history"].append(trade_record)
            return True, trade_record
        return False, "持仓不足"
    
    return False, "未知操作"

def run_trading_strategy(portfolio, market_data):
    """运行交易策略"""
    trades_executed = []
    
    for stock in market_data:
        code = stock["code"]
        current_price = stock["current_price"]
        
        # 简单策略：随机买卖
        decision = random.choice(["buy", "sell", "hold", "hold", "hold"])
        
        if decision == "buy" and portfolio["cash"] > 50000:
            # 买入1-3手
            quantity = random.randint(1, 3) * 100
            if current_price * quantity <= portfolio["cash"] * 0.1:  # 单笔不超过10%
                success, result = execute_trade(portfolio, stock, "buy", quantity, current_price)
                if success:
                    trades_executed.append(result)
        
        elif decision == "sell" and code in portfolio["positions"]:
            pos = portfolio["positions"][code]
            if pos["quantity"] >= 100:
                # 卖出1-2手
                quantity = random.randint(1, 2) * 100
                if quantity <= pos["quantity"]:
                    success, result = execute_trade(portfolio, stock, "sell", quantity, current_price)
                    if success:
                        trades_executed.append(result)
    
    return trades_executed

def main():
    today = datetime.datetime.now()
    
    print(f"=" * 60)
    print(f"A股模拟盘交易 - {today.strftime('%Y年%m月%d日')}")
    print(f"=" * 60)
    
    # 检查是否为交易日
    if not is_trading_day(today):
        print(f"⚠️  今日非交易日（{today.strftime('%A')}），跳过交易")
        return
    
    # 加载持仓
    portfolio = load_portfolio()
    print(f"\n📊 当前账户资金: ¥{portfolio['cash']:,.2f}")
    
    # 生成市场数据（模拟实时价格）
    market_data = []
    print(f"\n📈 市场行情:")
    for stock in STOCK_POOL:
        current_price, change_pct = simulate_price_change(stock["price"])
        market_data.append({
            **stock,
            "current_price": current_price,
            "change_pct": change_pct
        })
        change_symbol = "📈" if change_pct >= 0 else "📉"
        print(f"  {stock['code']} {stock['name']}: ¥{current_price:.2f} ({change_pct:+.2%}) {change_symbol}")
    
    # 执行交易策略
    print(f"\n🤖 执行交易策略...")
    trades = run_trading_strategy(portfolio, market_data)
    
    if trades:
        print(f"\n✅ 执行 {len(trades)} 笔交易:")
        for trade in trades:
            emoji = "🟢" if trade["action"] == "买入" else "🔴"
            print(f"  {emoji} {trade['action']} {trade['name']}({trade['code']}) {trade['quantity']}股 @ ¥{trade['price']}")
    else:
        print(f"\n⏸️  今日无交易")
    
    # 计算总资产
    total_value = portfolio["cash"]
    for code, pos in portfolio["positions"].items():
        # 找到当前价格
        for stock in market_data:
            if stock["code"] == code:
                total_value += pos["quantity"] * stock["current_price"]
                break
    
    portfolio["total_value"] = round(total_value, 2)
    portfolio["daily_pnl"] = round(total_value - 1000000, 2)
    portfolio["daily_return"] = round((total_value - 1000000) / 1000000 * 100, 2)
    
    print(f"\n💰 账户总资产: ¥{total_value:,.2f}")
    print(f"📊 累计盈亏: ¥{portfolio['daily_pnl']:,.2f} ({portfolio['daily_return']:+.2f}%)")
    
    # 保存持仓
    save_portfolio(portfolio)
    print(f"\n✅ 交易完成，数据已保存")

if __name__ == "__main__":
    main()
