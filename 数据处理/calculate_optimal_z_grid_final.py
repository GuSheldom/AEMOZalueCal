#!/usr/bin/env python3
"""
最优Z值计算 - 网格法最终版本
使用并行网格搜索计算所有季度、半年、年的最优Z值
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
        # 单位换算：RRP是AUD/MWh，变量单位是kWh，需要除以1000转换为MWh
        profit_terms = []
        for (i, j), var in x.items():
            profit_per_mwh = discharge_prices[j] - charge_prices[i]  # AUD/MWh
            profit_terms.append(profit_per_mwh * var / 1000.0)  # AUD/MWh × kWh/1000 = AUD
        
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

def parallel_grid_search_optimal_z(cycle_data_list: List[Tuple[List[float], List[float]]], 
                                  z_min: float = 0.0, z_max: float = 10.0, 
                                  step_size: float = 0.5,
                                  max_workers: Optional[int] = None) -> Tuple[float, float]:
    """
    并行网格搜索最优Z值
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
                print(f"  进度: {completed_tasks}/{len(z_values)} ({progress:.0f}%) - 当前最优: Z={best_z:.1f}, 收益={best_profit:.2f}")
    
    elapsed_time = time.time() - start_time
    
    print(f"✅ 并行网格搜索完成:")
    print(f"   用时: {elapsed_time:.2f} 秒")
    print(f"   最优Z值: {best_z:.1f}")
    print(f"   最大收益: {best_profit:.2f}")
    print(f"   平均每次评估: {elapsed_time/len(z_values):.2f} 秒")
    
    return best_z, best_profit

def calculate_optimal_z_for_period_grid(df: pd.DataFrame, period_type: str, selected_period: str) -> Tuple[float, float, int]:
    """
    使用网格搜索计算给定周期数据的最优Z值
    """
    print(f"\n🔍 正在计算{period_type} {selected_period}的最优Z值...")
    
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
    z_max = min(max_possible_spread, 20.0)  # 限制搜索上限为20
    
    print(f"💰 价格范围: 充电 {min(all_charge_prices):.2f}-{max(all_charge_prices):.2f}, 放电 {min(all_discharge_prices):.2f}-{max(all_discharge_prices):.2f}")
    print(f"📈 最大价差: {max_possible_spread:.2f}, 搜索上限: {z_max:.2f}")
    
    # 使用网格搜索
    optimal_z, max_profit = parallel_grid_search_optimal_z(
        cycle_data_list, z_min=0.0, z_max=z_max, step_size=0.5
    )
    
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
    print("🚀 AEMO电池储能优化系统 - 网格法最终版本")
    print("=" * 70)
    print("🎯 目标：计算所有季度、半年、年的最优Z值")
    
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
    
    # 计算所有周期的最优Z值
    total_calculations = sum(len(periods) for periods in available_periods.values())
    current_calculation = 0
    
    print(f"\n📋 总共需要计算 {total_calculations} 个周期")
    print("💡 使用并行网格搜索，步长0.5，搜索范围动态调整")
    print("=" * 70)
    
    overall_start_time = time.time()
    
    for period_type, periods in available_periods.items():
        print(f"\n📊 开始计算{period_type}周期...")
        
        for period in periods:
            current_calculation += 1
            print(f"\n[{current_calculation}/{total_calculations}] {period_type}: {period}")
            
            try:
                optimal_z, max_profit, cycle_count = calculate_optimal_z_for_period_grid(df, period_type, period)
                
                # 获取时间范围
                start_time, end_time = get_period_boundaries(period_type, period)
                
                results.append({
                    "周期类型": period_type,
                    "周期": period,
                    "开始时间": start_time.strftime("%Y-%m-%d %H:%M"),
                    "结束时间": end_time.strftime("%Y-%m-%d %H:%M"),
                    "包含天数": cycle_count,
                    "最优Z值": optimal_z,
                    "最大收益": max_profit,
                    "计算时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                print(f"🎯 {period_type} {period} 结果: Z={optimal_z:.1f}, 收益={max_profit:.2f}, 天数={cycle_count}")
                
            except Exception as e:
                print(f"❌ 计算失败: {e}")
                results.append({
                    "周期类型": period_type,
                    "周期": period,
                    "开始时间": "",
                    "结束时间": "",
                    "包含天数": 0,
                    "最优Z值": 0.0,
                    "最大收益": 0.0,
                    "计算时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
    
    overall_elapsed_time = time.time() - overall_start_time
    
    # 保存结果到CSV
    output_file = "optimal_z_values.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 计算完成！")
    print(f"⏱️  总用时: {overall_elapsed_time:.2f} 秒 ({overall_elapsed_time/60:.1f} 分钟)")
    print(f"📄 结果已保存到: {output_file}")
    print(f"📊 总共计算了 {len(results)} 个周期")
    
    # 显示结果摘要
    print(f"\n📋 结果摘要:")
    print("-" * 90)
    print(f"{'周期类型':<8} {'周期':<10} {'包含天数':<8} {'最优Z值':<8} {'最大收益':<15} {'时间范围':<25}")
    print("-" * 90)
    
    for result in results:
        time_range = f"{result['开始时间'][:10]} 至 {result['结束时间'][:10]}"
        print(f"{result['周期类型']:<8} {result['周期']:<10} {result['包含天数']:<8} "
              f"{result['最优Z值']:<8.1f} {result['最大收益']:<15.0f} {time_range:<25}")
    
    print("-" * 90)
    print(f"💾 详细结果请查看: {output_file}")
    
    # 分析结果
    z_values = [r['最优Z值'] for r in results if r['最优Z值'] > 0]
    if z_values:
        print(f"\n📊 Z值分析:")
        print(f"  最优Z值范围: {min(z_values):.1f} - {max(z_values):.1f}")
        print(f"  平均最优Z值: {np.mean(z_values):.1f}")
        print(f"  Z=0的周期数: {sum(1 for r in results if r['最优Z值'] == 0)}/{len(results)}")
        
        # 按周期类型分析
        for period_type in ["季度", "半年", "年"]:
            type_results = [r for r in results if r['周期类型'] == period_type]
            if type_results:
                type_z_values = [r['最优Z值'] for r in type_results]
                print(f"  {period_type}最优Z值: {type_z_values}")
    else:
        print(f"\n📊 所有周期的最优Z值都是0.0")
    
    print(f"\n🚀 网格搜索性能:")
    print(f"  平均每个周期用时: {overall_elapsed_time/len(results):.1f} 秒")
    print(f"  并行效率: 使用 {cpu_count} 核CPU")

if __name__ == "__main__":
    main() 