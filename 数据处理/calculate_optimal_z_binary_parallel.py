#!/usr/bin/env python3
"""
最优Z值计算 - 二分法 + 并行计算版本
使用二分搜索和多进程并行计算来加速Z值优化
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
from functools import partial
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

def binary_search_optimal_z(cycle_data_list: List[Tuple[List[float], List[float]]], 
                           z_min: float = 0.0, z_max: float = 50.0, 
                           tolerance: float = 0.1, max_iterations: int = 20,
                           use_parallel: bool = True) -> Tuple[float, float]:
    """
    使用二分搜索找到最优Z值
    
    假设：收益函数关于Z是单峰的（先增后减或单调递减）
    """
    print(f"\n🔍 使用二分搜索寻找最优Z值...")
    print(f"📊 搜索范围: [{z_min:.1f}, {z_max:.1f}]")
    print(f"🎯 精度要求: {tolerance}")
    print(f"🔄 最大迭代次数: {max_iterations}")
    print(f"⚡ 并行计算: {'是' if use_parallel else '否'}")
    
    iteration = 0
    best_z = z_min
    best_profit = 0.0
    
    # 记录所有评估过的点
    evaluated_points = {}
    
    def evaluate_z(z_val: float) -> float:
        """评估单个Z值，使用缓存避免重复计算"""
        if z_val in evaluated_points:
            return evaluated_points[z_val]
        
        if use_parallel:
            # 使用并行计算
            with ProcessPoolExecutor(max_workers=min(4, mp.cpu_count())) as executor:
                args = (z_val, cycle_data_list)
                future = executor.submit(evaluate_z_value_parallel, args)
                _, profit = future.result()
        else:
            # 串行计算
            profit = 0.0
            for charge_prices, discharge_prices in cycle_data_list:
                cycle_profit = solve_cycle_with_z_optimal(charge_prices, discharge_prices, z_val)
                profit += cycle_profit
        
        evaluated_points[z_val] = profit
        return profit
    
    # 初始评估端点
    profit_min = evaluate_z(z_min)
    profit_max = evaluate_z(z_max)
    
    print(f"📋 初始评估:")
    print(f"   Z={z_min:.1f}: 收益={profit_min:.2f}")
    print(f"   Z={z_max:.1f}: 收益={profit_max:.2f}")
    
    # 更新最优值
    if profit_min > best_profit:
        best_z, best_profit = z_min, profit_min
    if profit_max > best_profit:
        best_z, best_profit = z_max, profit_max
    
    # 二分搜索主循环
    left, right = z_min, z_max
    
    while iteration < max_iterations and (right - left) > tolerance:
        iteration += 1
        
        # 计算三分点（使用三分搜索的思想）
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        
        profit1 = evaluate_z(mid1)
        profit2 = evaluate_z(mid2)
        
        print(f"🔄 迭代 {iteration}: Z1={mid1:.2f}(收益={profit1:.2f}), Z2={mid2:.2f}(收益={profit2:.2f})")
        
        # 更新最优值
        if profit1 > best_profit:
            best_z, best_profit = mid1, profit1
        if profit2 > best_profit:
            best_z, best_profit = mid2, profit2
        
        # 三分搜索逻辑：保留包含最优解的区间
        if profit1 > profit2:
            right = mid2  # 最优解在左半部分
        else:
            left = mid1   # 最优解在右半部分
    
    print(f"✅ 二分搜索完成:")
    print(f"   迭代次数: {iteration}")
    print(f"   最优Z值: {best_z:.2f}")
    print(f"   最大收益: {best_profit:.2f}")
    print(f"   总评估点数: {len(evaluated_points)}")
    
    return best_z, best_profit

def parallel_grid_search_optimal_z(cycle_data_list: List[Tuple[List[float], List[float]]], 
                                  z_min: float = 0.0, z_max: float = 50.0, 
                                  step_size: float = 1.0,
                                  max_workers: Optional[int] = None) -> Tuple[float, float]:
    """
    并行网格搜索最优Z值（作为对比）
    """
    print(f"\n🔍 使用并行网格搜索寻找最优Z值...")
    
    z_values = np.arange(z_min, z_max + step_size, step_size)
    print(f"📊 搜索范围: [{z_min:.1f}, {z_max:.1f}]，步长: {step_size}")
    print(f"📋 总测试点数: {len(z_values)}")
    
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
                print(f"  进度: {completed_tasks}/{len(z_values)} ({progress:.1f}%) - 当前最优: Z={best_z:.1f}, 收益={best_profit:.2f}")
    
    elapsed_time = time.time() - start_time
    
    print(f"✅ 并行网格搜索完成:")
    print(f"   用时: {elapsed_time:.2f} 秒")
    print(f"   最优Z值: {best_z:.1f}")
    print(f"   最大收益: {best_profit:.2f}")
    
    return best_z, best_profit

def calculate_optimal_z_for_period_advanced(df: pd.DataFrame, period_type: str, selected_period: str, 
                                          method: str = "binary") -> Tuple[float, float, int]:
    """
    使用高级算法计算给定周期数据的最优Z值
    method: "binary" (二分搜索) 或 "parallel_grid" (并行网格搜索)
    """
    print(f"\n🔍 正在计算{period_type} {selected_period}的最优Z值 (方法: {method})...")
    
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
    
    # 分析价格分布以确定搜索范围
    all_charge_prices = []
    all_discharge_prices = []
    for charge_prices, discharge_prices in cycle_data_list:
        all_charge_prices.extend(charge_prices)
        all_discharge_prices.extend(discharge_prices)
    
    max_possible_spread = max(all_discharge_prices) - min(all_charge_prices) if all_charge_prices and all_discharge_prices else 30
    z_max = min(max_possible_spread, 50.0)
    
    print(f"💰 价格范围: 充电 {min(all_charge_prices):.2f}-{max(all_charge_prices):.2f}, 放电 {min(all_discharge_prices):.2f}-{max(all_discharge_prices):.2f}")
    print(f"📈 最大价差: {max_possible_spread:.2f}, 搜索上限: {z_max:.2f}")
    
    # 根据方法选择算法
    if method == "binary":
        optimal_z, max_profit = binary_search_optimal_z(
            cycle_data_list, z_min=0.0, z_max=z_max, tolerance=0.1, max_iterations=15
        )
    elif method == "parallel_grid":
        optimal_z, max_profit = parallel_grid_search_optimal_z(
            cycle_data_list, z_min=0.0, z_max=z_max, step_size=1.0
        )
    else:
        raise ValueError(f"未知的方法: {method}")
    
    print(f"✅ 最优Z值: {optimal_z:.2f}, 最大收益: {max_profit:.2f}")
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

def get_available_periods(df: pd.DataFrame) -> Dict[str, List[str]]:
    """获取所有可用的周期"""
    max_time = df["Timestamp"].max()
    
    periods = {
        "季度": ["2024Q1", "2024Q2", "2024Q3", "2024Q4"],
        "半年": ["2024H1", "2024H2"],
        "年": ["2024"]
    }
    
    # 2025年的季度和半年
    if max_time.year >= 2025:
        if max_time.month >= 3:
            periods["季度"].append("2025Q1")
        if max_time.month >= 6:
            periods["季度"].append("2025Q2")
            periods["半年"].append("2025H1")
        if max_time.month >= 8:
            periods["季度"].append("2025Q3")
    
    return periods

def main():
    """主函数"""
    print("🚀 AEMO电池储能优化系统 - 二分法 + 并行计算版本")
    print("=" * 70)
    
    # 检查CPU核心数
    cpu_count = mp.cpu_count()
    print(f"💻 系统CPU核心数: {cpu_count}")
    
    # 加载数据
    df = load_all_data()
    if df is None:
        print("❌ 无法加载数据，程序退出")
        return
    
    # 获取所有可用周期
    available_periods = get_available_periods(df)
    
    # 准备结果列表
    results = []
    
    # 选择测试的周期（可以选择几个代表性的周期进行测试）
    test_periods = [
        ("季度", "2024Q1"),
        ("季度", "2024Q2"),
        ("半年", "2024H1"),
        ("年", "2024")
    ]
    
    print(f"\n📋 将测试以下周期:")
    for period_type, period in test_periods:
        print(f"   - {period_type}: {period}")
    
    print(f"\n🎯 开始计算...")
    print("=" * 70)
    
    for i, (period_type, period) in enumerate(test_periods, 1):
        print(f"\n[{i}/{len(test_periods)}] {period_type}: {period}")
        
        try:
            # 使用二分法
            print(f"\n🔍 方法1: 二分搜索")
            start_time = time.time()
            optimal_z_binary, max_profit_binary, cycle_count = calculate_optimal_z_for_period_advanced(
                df, period_type, period, method="binary")
            binary_time = time.time() - start_time
            
            # 使用并行网格搜索（作为对比）
            print(f"\n🔍 方法2: 并行网格搜索")
            start_time = time.time()
            optimal_z_grid, max_profit_grid, _ = calculate_optimal_z_for_period_advanced(
                df, period_type, period, method="parallel_grid")
            grid_time = time.time() - start_time
            
            # 获取时间范围
            start_period_time, end_period_time = get_period_boundaries(period_type, period)
            
            results.append({
                "周期类型": period_type,
                "周期": period,
                "开始时间": start_period_time.strftime("%Y-%m-%d %H:%M"),
                "结束时间": end_period_time.strftime("%Y-%m-%d %H:%M"),
                "包含天数": cycle_count,
                "二分法_Z值": optimal_z_binary,
                "二分法_收益": max_profit_binary,
                "二分法_用时": binary_time,
                "网格法_Z值": optimal_z_grid,
                "网格法_收益": max_profit_grid,
                "网格法_用时": grid_time,
                "收益差异": abs(max_profit_binary - max_profit_grid),
                "计算时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            print(f"\n📊 结果对比:")
            print(f"   二分法: Z={optimal_z_binary:.2f}, 收益={max_profit_binary:.2f}, 用时={binary_time:.2f}秒")
            print(f"   网格法: Z={optimal_z_grid:.2f}, 收益={max_profit_grid:.2f}, 用时={grid_time:.2f}秒")
            print(f"   加速比: {grid_time/binary_time:.2f}x")
            
        except Exception as e:
            print(f"❌ 计算失败: {e}")
            results.append({
                "周期类型": period_type,
                "周期": period,
                "开始时间": "",
                "结束时间": "",
                "包含天数": 0,
                "二分法_Z值": 0.0,
                "二分法_收益": 0.0,
                "二分法_用时": 0.0,
                "网格法_Z值": 0.0,
                "网格法_收益": 0.0,
                "网格法_用时": 0.0,
                "收益差异": 0.0,
                "计算时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # 保存结果到CSV
    output_file = "optimal_z_values_binary_parallel.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 计算完成！")
    print(f"📄 结果已保存到: {output_file}")
    
    # 显示结果摘要
    print(f"\n📋 算法性能对比:")
    print("-" * 100)
    print(f"{'周期':<12} {'二分法用时':<10} {'网格法用时':<10} {'加速比':<8} {'收益差异':<12}")
    print("-" * 100)
    
    for result in results:
        if result['二分法_用时'] > 0 and result['网格法_用时'] > 0:
            speedup = result['网格法_用时'] / result['二分法_用时']
            print(f"{result['周期']:<12} {result['二分法_用时']:<10.2f} {result['网格法_用时']:<10.2f} "
                  f"{speedup:<8.2f} {result['收益差异']:<12.2f}")
    
    print("-" * 100)
    print(f"💾 详细结果请查看: {output_file}")

if __name__ == "__main__":
    main() 