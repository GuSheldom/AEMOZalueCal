#!/usr/bin/env python3
"""
测试新的周期定义逻辑
"""

import pandas as pd
from datetime import datetime, time as dt_time
from typing import Tuple, List

def get_period_boundaries(period_type: str, selected_period: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """根据周期类型和选择的周期，返回开始和结束时间"""
    if period_type == "天":
        # 单日：从选定日期的23:00到次日08:00
        date = pd.to_datetime(selected_period)
        start_time = pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=23, minute=0)
        end_time = start_time + pd.Timedelta(hours=9)  # 到次日08:00
        return start_time, end_time
    
    elif period_type == "季度":
        # 季度：从上季度最后一天23:00到本季度最后一天08:00
        year_quarter = pd.Period(selected_period)
        year, quarter = year_quarter.year, year_quarter.quarter
        
        # 计算季度的最后一个月
        last_month = quarter * 3
        
        # 上季度最后一天23:00
        if quarter == 1:
            prev_year, prev_month = year - 1, 12
        else:
            prev_year, prev_month = year, (quarter - 2) * 3 + 3
        
        prev_quarter_last_day = pd.Timestamp(year=prev_year, month=prev_month, day=1) + pd.offsets.MonthEnd(0)
        start_time = pd.Timestamp(year=prev_quarter_last_day.year, month=prev_quarter_last_day.month,
                                 day=prev_quarter_last_day.day, hour=23, minute=0)
        
        # 本季度最后一天08:00
        quarter_last_day = pd.Timestamp(year=year, month=last_month, day=1) + pd.offsets.MonthEnd(0)
        end_time = pd.Timestamp(year=quarter_last_day.year, month=quarter_last_day.month,
                               day=quarter_last_day.day, hour=8, minute=0)
        
        return start_time, end_time
    
    return pd.Timestamp.now(), pd.Timestamp.now()

def test_period_boundaries():
    """测试周期边界计算"""
    
    print("🧪 测试周期边界计算")
    print("=" * 50)
    
    # 测试2024Q1
    start, end = get_period_boundaries("季度", "2024Q1")
    print(f"2024Q1:")
    print(f"  开始时间: {start}")
    print(f"  结束时间: {end}")
    print(f"  持续时间: {end - start}")
    print()
    
    # 测试2024Q2
    start, end = get_period_boundaries("季度", "2024Q2")
    print(f"2024Q2:")
    print(f"  开始时间: {start}")
    print(f"  结束时间: {end}")
    print(f"  持续时间: {end - start}")
    print()
    
    # 测试单日
    start, end = get_period_boundaries("天", "2024-01-01")
    print(f"2024-01-01:")
    print(f"  开始时间: {start}")
    print(f"  结束时间: {end}")
    print(f"  持续时间: {end - start}")
    print()
    
    # 验证2024Q1是否包含2023-12-31 23:00
    q1_start, q1_end = get_period_boundaries("季度", "2024Q1")
    test_time = pd.Timestamp("2023-12-31 23:00:00")
    
    print(f"验证: 2023-12-31 23:00 是否在 2024Q1 中?")
    print(f"  测试时间: {test_time}")
    print(f"  Q1开始: {q1_start}")
    print(f"  Q1结束: {q1_end}")
    print(f"  包含?: {q1_start <= test_time <= q1_end}")

if __name__ == "__main__":
    test_period_boundaries() 