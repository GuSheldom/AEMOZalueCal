#!/usr/bin/env python3
"""
最优Z值计算 - 自适应搜索范围版本
根据价格分布智能确定Z值搜索范围，避免无效搜索
"""

import pandas as pd
import numpy as np
import glob
from datetime import datetime, time as dt_time
from typing import List, Tuple, Dict, Optional
import pulp
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time

def assign_cycle_date(ts_str: str) -> str:
    """分配周期日期"""
    ts = pd.to_datetime(ts_str)
    if ts.time() >= dt_time(23, 0):
        return str(ts.normalize().date())
    elif ts.time() < dt_time(8, 0):
        return str((ts - pd.Timedelta(days=1)).normalize().date())
    else:
        return str(ts.normalize().date())

def get_period_boundaries(period_type: str, selected_period: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """根据周期类型和选择的周期，返回开始和结束时间"""
    if period_type == "季度":
        year_quarter = pd.Period(selected_period)
        year, quarter = year_quarter.year, year_quarter.quarter
        
        last_month = quarter * 3
        
        if quarter == 1:
            prev_year, prev_month = year - 1, 12
        else:
            prev_year, prev_month = year, (quarter - 2) * 3 + 3
        
        prev_quarter_last_day = pd.Timestamp(year=prev_year, month=prev_month, day=1) + pd.offsets.MonthEnd(0)
        start_time = pd.Timestamp(year=prev_quarter_last_day.year, month=prev_quarter_last_day.month,
                                 day=prev_quarter_last_day.day, hour=23, minute=0)
        
        quarter_last_day = pd.Timestamp(year=year, month=last_month, day=1) + pd.offsets.MonthEnd(0)
        end_time = pd.Timestamp(year=quarter_last_day.year, month=quarter_last_day.month,
                               day=quarter_last_day.day, hour=8, minute=0)
        
        return start_time, end_time
    
    elif period_type == "半年":
        year = int(selected_period[:4])
        half = int(selected_period[-1])
        
        if half == 1:
            start_time = pd.Timestamp(year=year-1, month=12, day=31, hour=23, minute=0)
            end_time = pd.Timestamp(year=year, month=6, day=30, hour=8, minute=0)
        else:
            start_time = pd.Timestamp(year=year, month=6, day=30, hour=23, minute=0)
            end_time = pd.Timestamp(year=year, month=12, day=31, hour=8, minute=0)
        
        return start_time, end_time
    
    elif period_type == "年":
        year = int(selected_period)
        start_time = pd.Timestamp(year=year-1, month=12, day=31, hour=23, minute=0)
        end_time = pd.Timestamp(year=year, month=12, day=31, hour=8, minute=0)
        return start_time, end_time
    
    return pd.Timestamp.now(), pd.Timestamp.now()

def solve_cycle_with_z_optimal(charge_prices: List[float], discharge_prices: List[float], z: float) -> float:
    """使用线性规划求解给定Z值下的最优收益（简化版，只返回收益）"""
    try:
        prob = pulp.LpProblem("Battery_Optimization", pulp.LpMaximize)
        
        n_charge = len(charge_prices)
        n_discharge = len(discharge_prices)
        
        # 决策变量：充电时段i到放电时段j的能量分配
        x = {}
        for i in range(n_charge):
            for j in range(n_discharge):
                if discharge_prices[j] > charge_prices[i] + z:  # Z值作为筛选条件
                    x[i, j] = pulp.LpVariable(f"x_{i}_{j}", 0, None)
        
        if not x:  # 没有可行的分配
            return 0.0
        
        # 目标函数：最大化总利润（不包含Z值，只是价差）
        profit_terms = []
        for (i, j), var in x.items():
            profit = discharge_prices[j] - charge_prices[i]  # 纯价差，不减去Z
            profit_terms.append(profit * var)
        
        if profit_terms:
            prob += pulp.lpSum(profit_terms)
        else:
            return 0.0
        
        # 约束条件
        for i in range(n_charge):
            charge_vars = [x[i, j] for j in range(n_discharge) if (i, j) in x]
            if charge_vars:
                prob += pulp.lpSum(charge_vars) <= 55.83
        
        for j in range(n_discharge):
            discharge_vars = [x[i, j] for i in range(n_charge) if (i, j) in x]
            if discharge_vars:
                prob += pulp.lpSum(discharge_vars) <= 200.0
        
        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            return pulp.value(prob.objective) if prob.objective else 0.0
        else:
            return 0.0
        
    except Exception as e:
        return 0.0

def evaluate_z_value_parallel(args: Tuple[float, List[Tuple[List[float], List[float]]]]) -> Tuple[float, float]:
    """并行计算单个Z值的总收益"""
    z, cycle_data_list = args
    total_profit = 0.0
    
    for charge_prices, discharge_prices in cycle_data_list:
        cycle_profit = solve_cycle_with_z_optimal(charge_prices, discharge_prices, z)
        total_profit += cycle_profit
    
    return z, total_profit

def determine_smart_search_range(cycle_data_list: List[Tuple[List[float], List[float]]]) -> Tuple[float, float, float]:
    """
    智能确定搜索范围
    基于价格分布和经验规律
    """
    print(f"\n🧠 智能分析搜索范围...")
    
    # 收集所有价格数据
    all_charge_prices = []
    all_discharge_prices = []
    all_spreads = []  # 所有可能的价差
    
    for charge_prices, discharge_prices in cycle_data_list:
        all_charge_prices.extend(charge_prices)
        all_discharge_prices.extend(discharge_prices)
        
        # 计算这个周期内所有可能的价差
        for cp in charge_prices:
            for dp in discharge_prices:
                if dp > cp:  # 只考虑有利可图的组合
                    all_spreads.append(dp - cp)
    
    # 价格统计
    charge_min, charge_max = min(all_charge_prices), max(all_charge_prices)
    discharge_min, discharge_max = min(all_discharge_prices), max(all_discharge_prices)
    
    print(f"💰 价格分析:")
    print(f"   充电价格: {charge_min:.2f} ~ {charge_max:.2f}")
    print(f"   放电价格: {discharge_min:.2f} ~ {discharge_max:.2f}")
    
    if not all_spreads:
        print(f"❌ 没有有利可图的价差，使用默认范围")
        return 0.0, 10.0, 1.0
    
    # 价差分析
    spread_min, spread_max = min(all_spreads), max(all_spreads)
    spread_mean = np.mean(all_spreads)
    spread_std = np.std(all_spreads)
    spread_percentiles = np.percentile(all_spreads, [25, 50, 75, 90, 95, 99])
    
    print(f"📊 价差分析:")
    print(f"   价差范围: {spread_min:.2f} ~ {spread_max:.2f}")
    print(f"   平均价差: {spread_mean:.2f} ± {spread_std:.2f}")
    print(f"   百分位数: P25={spread_percentiles[0]:.2f}, P50={spread_percentiles[1]:.2f}, P75={spread_percentiles[2]:.2f}")
    print(f"   高百分位: P90={spread_percentiles[3]:.2f}, P95={spread_percentiles[4]:.2f}, P99={spread_percentiles[5]:.2f}")
    
    # 智能确定搜索范围
    # 策略：大多数有效的Z值应该在P75到P95之间
    z_min = 0.0  # 总是从0开始
    
    # 上限策略：
    # 1. 如果P95价差 < 50，使用P95 + 10作为上限
    # 2. 如果P95价差很大，使用更保守的策略
    if spread_percentiles[4] <= 50:  # P95 <= 50
        z_max = min(spread_percentiles[4] + 10, 100)
        step_size = 0.5
    elif spread_percentiles[4] <= 200:  # P95 <= 200
        z_max = min(spread_percentiles[4] * 0.5, 100)
        step_size = 1.0
    else:  # P95 > 200，价差很大
        z_max = min(spread_percentiles[3] + 20, 100)  # 使用P90 + 20
        step_size = 2.0
    
    print(f"🎯 搜索策略:")
    print(f"   搜索范围: [{z_min:.1f}, {z_max:.1f}]")
    print(f"   步长: {step_size}")
    print(f"   预计测试点数: {int((z_max - z_min) / step_size) + 1}")
    
    # 合理性检查
    if z_max < 5:
        z_max = 10.0
        print(f"🔧 调整：最小搜索上限为10.0")
    
    return z_min, z_max, step_size

def parallel_grid_search_adaptive(cycle_data_list: List[Tuple[List[float], List[float]]], 
                                 max_workers: Optional[int] = None) -> Tuple[float, float]:
    """
    自适应并行网格搜索最优Z值
    """
    print(f"\n🔍 使用自适应并行网格搜索寻找最优Z值...")
    
    # 智能确定搜索范围
    z_min, z_max, step_size = determine_smart_search_range(cycle_data_list)
    
    z_values = np.arange(z_min, z_max + step_size, step_size)
    print(f"\n📊 搜索配置:")
    print(f"   范围: [{z_min:.1f}, {z_max:.1f}]，步长: {step_size}")
    print(f"   总测试点数: {len(z_values)}")
    
    if max_workers is None:
        max_workers = min(len(z_values), mp.cpu_count())
    
    print(f"⚡ 并行工作进程数: {max_workers}")
    
    start_time = time.time()
    
    # 准备并行任务
    tasks = [(z, cycle_data_list) for z in z_values]
    
    best_z = z_min
    best_profit = 0.0
    completed_tasks = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_z = {executor.submit(evaluate_z_value_parallel, task): task[0] for task in tasks}
        
        # 处理完成的任务
        for future in as_completed(future_to_z):
            z, profit = future.result()
            completed_tasks += 1
            
            if profit > best_profit:
                best_profit = profit
                best_z = z
            
            if completed_tasks % max(1, len(z_values) // 10) == 0:
                progress = completed_tasks / len(z_values) * 100
                print(f"  进度: {completed_tasks}/{len(z_values)} ({progress:.0f}%) - 当前最优: Z={best_z:.1f}, 收益={best_profit:.2f}")
    
    elapsed_time = time.time() - start_time
    
    print(f"✅ 自适应搜索完成:")
    print(f"   用时: {elapsed_time:.2f} 秒")
    print(f"   最优Z值: {best_z:.1f}")
    print(f"   最大收益: {best_profit:.2f}")
    print(f"   平均每次评估: {elapsed_time/len(z_values):.2f} 秒")
    print(f"   搜索效率: 测试了 {len(z_values)} 个点 (vs 固定范围可能需要 100+ 个点)")
    
    return best_z, best_profit

def calculate_optimal_z_for_period_adaptive(df: pd.DataFrame, period_type: str, selected_period: str) -> Tuple[float, float, int]:
    """
    使用自适应搜索计算给定周期数据的最优Z值
    """
    print(f"\n🔍 正在计算{period_type} {selected_period}的最优Z值 (自适应搜索)...")
    
    # 筛选周期数据
    start_time, end_time = get_period_boundaries(period_type, selected_period)
    period_data = df[(df["Timestamp"] >= start_time) & (df["Timestamp"] <= end_time)].copy()
    
    if len(period_data) == 0:
        print(f"❌ {period_type} {selected_period} 没有数据")
        return 0.0, 0.0, 0
    
    unique_cycles = period_data["Cycle_Date"].unique()
    cycle_count = len(unique_cycles)
    print(f"📊 包含 {cycle_count} 个日周期")
    
    # 预先收集所有日周期的价格数据
    cycle_data_list = []
    for cycle_date in unique_cycles:
        cycle_data = period_data[period_data["Cycle_Date"] == cycle_date]
        
        charge_data = cycle_data[cycle_data["Phase"] == "charge"]
        discharge_data = cycle_data[cycle_data["Phase"] == "discharge"]
        
        if len(charge_data) > 0 and len(discharge_data) > 0:
            charge_prices = charge_data["Price_RRP"].tolist()
            discharge_prices = discharge_data["Price_RRP"].tolist()
            cycle_data_list.append((charge_prices, discharge_prices))
    
    print(f"📋 有效日周期: {len(cycle_data_list)} 个")
    
    # 使用自适应搜索
    optimal_z, max_profit = parallel_grid_search_adaptive(cycle_data_list)
    
    print(f"✅ 最优Z值: {optimal_z:.1f}, 最大收益: {max_profit:.2f}")
    return optimal_z, max_profit, cycle_count

def load_all_data():
    """加载所有数据"""
    print("📊 正在加载所有数据文件...")
    
    pattern = "AEMO_23to08_with_opt_*_z0Fast.xlsx"
    excel_files = sorted(glob.glob(pattern))
    
    if not excel_files:
        print("❌ 未找到数据文件")
        return None
    
    all_dataframes = []
    for file in excel_files:
        try:
            df = pd.read_excel(file, sheet_name="23to08_opt")
            df = df.rename(columns={
                "时间": "Timestamp",
                "电价(RRP)": "Price_RRP", 
                "阶段": "Phase"
            })
            all_dataframes.append(df)
            print(f"  ✅ {file}: {len(df)} 行")
        except Exception as e:
            print(f"  ❌ {file}: {e}")
    
    if not all_dataframes:
        return None
    
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    merged_df["Timestamp"] = pd.to_datetime(merged_df["Timestamp"])
    merged_df = merged_df.sort_values("Timestamp").reset_index(drop=True)
    merged_df["Cycle_Date"] = merged_df["Timestamp"].astype(str).apply(assign_cycle_date)
    merged_df["Cycle_Date"] = pd.to_datetime(merged_df["Cycle_Date"])
    
    print(f"✅ 数据加载完成: {len(merged_df)} 行, 时间范围 {merged_df['Timestamp'].min()} 到 {merged_df['Timestamp'].max()}")
    return merged_df

def main():
    """主函数 - 测试一个代表性周期"""
    print("🚀 AEMO电池储能优化系统 - 自适应搜索范围测试")
    print("=" * 70)
    print("🎯 目标：测试自适应搜索范围的效果")
    
    # 检查CPU核心数
    cpu_count = mp.cpu_count()
    print(f"💻 系统CPU核心数: {cpu_count}")
    
    # 加载数据
    df = load_all_data()
    if df is None:
        print("❌ 无法加载数据，程序退出")
        return
    
    # 测试一个代表性周期：2024Q2 (价差最大的季度)
    test_period_type = "季度"
    test_period = "2024Q2"
    
    print(f"\n📋 测试周期: {test_period_type} {test_period}")
    print("💡 选择这个周期是因为它有最大的价格波动范围")
    print("=" * 70)
    
    overall_start_time = time.time()
    
    try:
        optimal_z, max_profit, cycle_count = calculate_optimal_z_for_period_adaptive(df, test_period_type, test_period)
        
        print(f"\n🎯 测试结果:")
        print(f"   周期: {test_period_type} {test_period}")
        print(f"   包含天数: {cycle_count}")
        print(f"   最优Z值: {optimal_z:.1f}")
        print(f"   最大收益: {max_profit:.2f}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    overall_elapsed_time = time.time() - overall_start_time
    
    print(f"\n🎉 测试完成！")
    print(f"⏱️  总用时: {overall_elapsed_time:.2f} 秒")
    
    print(f"\n💡 自适应搜索的优势:")
    print(f"   1. 根据实际价格分布确定搜索范围")
    print(f"   2. 避免在无效范围内浪费计算资源")
    print(f"   3. 自动调整步长以平衡精度和效率")
    print(f"   4. 基于价差百分位数的科学方法")

if __name__ == "__main__":
    main() 