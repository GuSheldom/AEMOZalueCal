#!/usr/bin/env python3
"""
批量计算最优Z值
为所有季度、半年、年周期计算最优Z值并保存到CSV文件
"""

import pandas as pd
import numpy as np
import glob
from datetime import datetime, time as dt_time
from typing import List, Tuple, Dict
import pulp
import csv

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

def filter_data_by_period_boundaries(df: pd.DataFrame, period_type: str, selected_period: str) -> pd.DataFrame:
    """根据时间边界筛选数据"""
    start_time, end_time = get_period_boundaries(period_type, selected_period)
    filtered_df = df[(df["Timestamp"] >= start_time) & (df["Timestamp"] <= end_time)].copy()
    return filtered_df

def solve_cycle_with_z(charge_prices: List[float], discharge_prices: List[float], z: float) -> Tuple[List[float], List[float], float]:
    """使用线性规划求解给定Z值下的最优分配"""
    try:
        prob = pulp.LpProblem("Battery_Optimization", pulp.LpMaximize)
        
        n_charge = len(charge_prices)
        n_discharge = len(discharge_prices)
        
        x = {}
        for i in range(n_charge):
            for j in range(n_discharge):
                if discharge_prices[j] > charge_prices[i] + z:
                    x[i, j] = pulp.LpVariable(f"x_{i}_{j}", 0, None)
        
        if not x:
            return [0.0] * n_charge, [0.0] * n_discharge, 0.0
        
        profit_terms = []
        for (i, j), var in x.items():
            profit = discharge_prices[j] - charge_prices[i]
            profit_terms.append(profit * var)
        
        if profit_terms:
            prob += pulp.lpSum(profit_terms)
        
        for i in range(n_charge):
            charge_vars = [x[i, j] for j in range(n_discharge) if (i, j) in x]
            if charge_vars:
                prob += pulp.lpSum(charge_vars) <= 55.83
        
        for j in range(n_discharge):
            discharge_vars = [x[i, j] for i in range(n_charge) if (i, j) in x]
            if discharge_vars:
                prob += pulp.lpSum(discharge_vars) <= 200.0
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status != pulp.LpStatusOptimal:
            return [0.0] * n_charge, [0.0] * n_discharge, 0.0
        
        charge_energy = [0.0] * n_charge
        discharge_energy = [0.0] * n_discharge
        total_profit = 0.0
        
        for (i, j), var in x.items():
            if var.varValue and var.varValue > 1e-6:
                energy = var.varValue
                charge_energy[i] += energy
                discharge_energy[j] += energy
                profit = discharge_prices[j] - charge_prices[i]
                total_profit += profit * energy
        
        return charge_energy, discharge_energy, total_profit
        
    except Exception as e:
        print(f"求解过程出错 (z={z}): {e}")
        return [0.0] * len(charge_prices), [0.0] * len(discharge_prices), 0.0

def calculate_optimal_z_for_period(df: pd.DataFrame, period_type: str, selected_period: str) -> Tuple[float, float, int]:
    """计算给定周期数据的最优Z值和对应的最大收益"""
    print(f"\n🔍 正在计算{period_type} {selected_period}的最优Z值...")
    
    period_data = filter_data_by_period_boundaries(df, period_type, selected_period)
    
    if len(period_data) == 0:
        print(f"❌ {period_type} {selected_period} 没有数据")
        return 0.0, 0.0, 0
    
    unique_cycles = period_data["Cycle_Date"].unique()
    cycle_count = len(unique_cycles)
    print(f"📊 包含 {cycle_count} 个日周期")
    
    # 测试不同的Z值 (从0到30，步长0.5)
    z_values = np.arange(0.0, 30.1, 0.5)
    best_z = 0.0
    best_total_profit = 0.0
    
    for i, z in enumerate(z_values):
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(z_values)} (Z={z:.1f})")
        
        total_profit = 0.0
        
        # 对每个日周期计算收益
        for cycle_date in unique_cycles:
            cycle_data = period_data[period_data["Cycle_Date"] == cycle_date]
            
            charge_data = cycle_data[cycle_data["Phase"] == "charge"]
            discharge_data = cycle_data[cycle_data["Phase"] == "discharge"]
            
            if len(charge_data) == 0 or len(discharge_data) == 0:
                continue
            
            charge_prices = charge_data["Price_RRP"].tolist()
            discharge_prices = discharge_data["Price_RRP"].tolist()
            
            _, _, cycle_profit = solve_cycle_with_z(charge_prices, discharge_prices, z)
            total_profit += cycle_profit
        
        if total_profit > best_total_profit:
            best_total_profit = total_profit
            best_z = z
    
    print(f"✅ 最优Z值: {best_z:.1f}, 最大收益: {best_total_profit:.2f}")
    return best_z, best_total_profit, cycle_count

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
    print("🚀 AEMO电池储能优化系统 - 批量计算最优Z值")
    print("=" * 60)
    
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
    print("=" * 60)
    
    for period_type, periods in available_periods.items():
        print(f"\n📊 开始计算{period_type}周期...")
        
        for period in periods:
            current_calculation += 1
            print(f"\n[{current_calculation}/{total_calculations}] {period_type}: {period}")
            
            try:
                optimal_z, max_profit, cycle_count = calculate_optimal_z_for_period(df, period_type, period)
                
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
    
    # 保存结果到CSV
    output_file = "optimal_z_values.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 计算完成！")
    print(f"📄 结果已保存到: {output_file}")
    print(f"📊 总共计算了 {len(results)} 个周期")
    
    # 显示结果摘要
    print(f"\n📋 结果摘要:")
    print("-" * 80)
    print(f"{'周期类型':<8} {'周期':<10} {'包含天数':<8} {'最优Z值':<8} {'最大收益':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['周期类型']:<8} {result['周期']:<10} {result['包含天数']:<8} "
              f"{result['最优Z值']:<8.1f} {result['最大收益']:<12.2f}")
    
    print("-" * 80)
    print(f"💾 详细结果请查看: {output_file}")

if __name__ == "__main__":
    main() 