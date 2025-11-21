import os
import glob
import pandas as pd


def compute_20240101_hourly_avg():
    """
    计算 2024-01-01 当天每个交易小时(0-23)的平均RRP，并打印当日总体平均价。
    口径：使用5分钟结算数据，按结算时间的小时字段映射：H:00:00~H:55:00 归入交易小时 H。
    """

    aemo_dir = 'aemo'
    start_ts = pd.Timestamp('2024-01-01 00:00:00')
    end_ts = pd.Timestamp('2024-01-01 23:55:00')

    files = sorted(glob.glob(os.path.join(aemo_dir, 'PRICE_AND_DEMAND_*.csv')))
    if not files:
        print('未找到 aemo/PRICE_AND_DEMAND_*.csv 文件')
        return None

    parts = []

    for fp in files:
        try:
            df = pd.read_csv(fp)
            if 'SETTLEMENTDATE' not in df.columns or 'RRP' not in df.columns:
                continue
            dt = pd.to_datetime(df['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S', errors='coerce')
            if dt.isna().any():
                dt = pd.to_datetime(df['SETTLEMENTDATE'], errors='coerce')
            df = df.assign(SETTLEMENTDATE=dt)
            df = df[df['SETTLEMENTDATE'].notna()]
            mask = (df['SETTLEMENTDATE'] >= start_ts) & (df['SETTLEMENTDATE'] <= end_ts)
            df = df.loc[mask]
            if df.empty:
                continue
            df = df[['SETTLEMENTDATE', 'RRP']].copy()
            df['TradingHour'] = df['SETTLEMENTDATE'].dt.hour
            parts.append(df)
        except Exception as e:
            print(f'读取 {os.path.basename(fp)} 出错: {e}')
            continue

    if not parts:
        print('指定日期范围内没有数据点')
        return None

    day_df = pd.concat(parts, ignore_index=True)
    hourly = day_df.groupby('TradingHour')['RRP'].mean().reindex(range(24))

    # 保存逐小时均价
    out_csv = 'AEMO_20240101_逐小时均价.csv'
    hourly.to_frame('AvgRRP').to_csv(out_csv, encoding='utf-8-sig')
    print(f'已保存逐小时均价：{out_csv}')

    # 当日总体平均（全天288个5分钟点的简单平均）
    daily_avg = day_df['RRP'].mean()
    print(f'2024-01-01 当日总体平均电价：{daily_avg:.6f} $/MWh')

    return hourly, daily_avg


if __name__ == '__main__':
    compute_20240101_hourly_avg()



