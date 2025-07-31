import os
dir = os.path.dirname(os.path.abspath(__file__))

def parse_daily_holdings():
    import pandas as pd
    from tqdm import tqdm
    import json

    # === 读取 CSV 数据 ===
    df = pd.read_csv(f"{dir}/small_cap_100_trades.csv", dtype={"股票代码": str})

    # 将日期转为 datetime 格式
    df["买入日期"] = pd.to_datetime(df["买入日期"])
    df["卖出日期"] = pd.to_datetime(df["卖出日期"])

    # === 目标1：生成每日持仓列表 ===
    # 找出最早买入和最晚卖出时间，构造完整日期序列
    all_days = pd.date_range(df["买入日期"].min(), df["卖出日期"].max(), freq='D')

    # 构造每日持仓字典
    daily_holdings = {}

    for day in tqdm(all_days):
        holding_today = df[(df["买入日期"] <= day) & (df["卖出日期"] > day)]
        daily_holdings[str(day.date())] = list(holding_today["股票代码"])

    # === 目标2：提取股票信息字典 ===
    stock_info = {}
    for _, row in tqdm(df.iterrows()):
        code = row["股票代码"]
        if code not in stock_info:
            stock_info[code] = {
                "name": row["股票名"],
                "industry": row["行业分类"],
                "sub_industry": row["二级行业"]
            }

    # === 输出结果为 JSON 格式 ===
    with open(f"{dir}/daily_holdings.json", "w", encoding="utf-8") as f:
        json.dump(daily_holdings, f, ensure_ascii=False, indent=2)

    with open(f"{dir}/stock_info.json", "w", encoding="utf-8") as f:
        json.dump(stock_info, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    parse_daily_holdings()