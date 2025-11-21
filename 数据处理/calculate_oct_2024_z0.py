#!/usr/bin/env python3
"""
计算2024年10月的最优Z值
"""

import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time
from typing import List, Tuple
import pulp

def assign_cycle_date(ts_str: str) -> str:
    """分配周期日期"""
    ts = pd.to_datetime(ts_str)
    if ts.time() >= dt_time(23, 0):
        return str(ts.normalize().date())
    elif ts.time() < dt_time(8, 0):
        return str((ts - pd.Timedelta(days=1)).normalize().date())
    else:
        return str(ts.normalize().date())

def solve_cycle_with_z_optimal(charge_prices: List[float], discharge_prices: List[float], z: float) -> float:
    """使用线性规划求解给定Z值下的最优收益"""
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
            profit = discharge_prices[j] - charge_prices[i]  # 纯价差
            profit_terms.append(profit * var)
        
        if profit_terms:
            prob += pulp.lpSum(profit_terms)
        else:
            return 0.0
        
        # 约束条件
        # 充电约束：每个充电时段最多充电55.83kWh
        for i in range(n_charge):
            charge_vars = [x[i, j] for j in range(n_discharge) if (i, j) in x]
            if charge_vars:
                prob += pulp.lpSum(charge_vars) <= 55.83
        
        # 放电约束：每个放电时段最多放电200.0kWh
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
        print(f"求解过程出错 (z={z}): {e}")
        return 0.0

def calculate_optimal_z_for_october() -> Tuple[float, float, int]:
    """计算2024年10月的最优Z值"""
    print("=" * 80)
    print("🔍 开始计算2024年10月的最优Z值")
    print("=" * 80)
    
    # 读取10月份的数据
    file_path = "AEMO_23to08_with_opt_2024-10_z0Fast.xlsx"
    print(f"\n📂 正在读取数据文件: {file_path}")
    
    try:
        df = pd.read_excel(file_path, sheet_name="23to08_opt")
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")
        return 0.0, 0.0, 0
    
    # 重命名列
    df = df.rename(columns={
        "时间": "Timestamp",
        "电价(RRP)": "Price_RRP", 
        "阶段": "Phase"
    })
    
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Cycle_Date"] = df["Timestamp"].astype(str).apply(assign_cycle_date)
    df["Cycle_Date"] = pd.to_datetime(df["Cycle_Date"])
    
    print(f"✅ 数据加载完成: {len(df)} 行")
    print(f"📅 时间范围: {df['Timestamp'].min()} 到 {df['Timestamp'].max()}")
    
    # 获取所有日周期
    unique_cycles = df["Cycle_Date"].unique()
    cycle_count = len(unique_cycles)
    print(f"📊 包含 {cycle_count} 个日周期")
    
    # 预先收集所有日周期的价格数据
    print(f"\n📋 正在收集日周期数据...")
    cycle_data_list = []
    for cycle_date in unique_cycles:
        cycle_data = df[df["Cycle_Date"] == cycle_date]
        
        charge_data = cycle_data[cycle_data["Phase"] == "charge"]
        discharge_data = cycle_data[cycle_data["Phase"] == "discharge"]
        
        if len(charge_data) > 0 and len(discharge_data) > 0:
            charge_prices = charge_data["Price_RRP"].tolist()
            discharge_prices = discharge_data["Price_RRP"].tolist()
            cycle_data_list.append((charge_prices, discharge_prices))
    
    print(f"✅ 有效日周期: {len(cycle_data_list)} 个")
    
    # 分析价格分布
    all_charge_prices = []
    all_discharge_prices = []
    for charge_prices, discharge_prices in cycle_data_list:
        all_charge_prices.extend(charge_prices)
        all_discharge_prices.extend(discharge_prices)
    
    max_charge = max(all_charge_prices) if all_charge_prices else 0
    min_charge = min(all_charge_prices) if all_charge_prices else 0
    max_discharge = max(all_discharge_prices) if all_discharge_prices else 0
    min_discharge = min(all_discharge_prices) if all_discharge_prices else 0
    max_possible_spread = max_discharge - min_charge
    
    print(f"\n💰 价格统计:")
    print(f"   充电电价: {min_charge:.2f} - {max_charge:.2f} $/MWh")
    print(f"   放电电价: {min_discharge:.2f} - {max_discharge:.2f} $/MWh")
    print(f"   最大价差: {max_possible_spread:.2f} $/MWh")
    
    # 智能确定Z值测试范围
    if max_possible_spread > 0:
        z_max = min(max_possible_spread, 50.0)
        z_values = np.arange(0.0, z_max + 0.1, 2.0)  # 步长2.0
    else:
        z_values = np.array([0.0])
    
    print(f"\n🧮 测试Z值范围: 0.0 到 {z_values[-1]:.1f}, 共 {len(z_values)} 个值")
    print(f"=" * 80)
    
    # 搜索最优Z值
    best_z = 0.0
    best_total_profit = 0.0
    z_profit_history = []
    
    print("\n⚡ 开始搜索最优Z值...")
    for i, z in enumerate(z_values):
        print(f"  [{i+1}/{len(z_values)}] 测试 Z = {z:.1f}...", end=" ")
        
        total_profit = 0.0
        
        # 对每个日周期计算收益
        for charge_prices, discharge_prices in cycle_data_list:
            cycle_profit = solve_cycle_with_z_optimal(charge_prices, discharge_prices, z)
            total_profit += cycle_profit
        
        z_profit_history.append((z, total_profit))
        print(f"总收益 = {total_profit:.2f}")
        
        if total_profit > best_total_profit:
            best_total_profit = total_profit
            best_z = z
    
    print(f"=" * 80)
    print(f"\n✅ 搜索完成！")
    print(f"\n🎯 最优结果:")
    print(f"   最优Z值: {best_z:.1f} $/MWh")
    print(f"   最大收益: {best_total_profit:.2f} $")
    print(f"   日均收益: {best_total_profit / cycle_count:.2f} $")
    print(f"=" * 80)
    
    # 保存结果到CSV
    result_df = pd.DataFrame(z_profit_history, columns=['Z值', '总收益'])
    output_csv = "2024_10_z_optimization_results.csv"
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n💾 详细结果已保存到: {output_csv}")
    
    # 显示前10个最佳Z值
    result_df_sorted = result_df.sort_values('总收益', ascending=False)
    print(f"\n📊 前10个最佳Z值:")
    print("-" * 50)
    print(f"{'排名':<6} {'Z值':<10} {'总收益':<15} {'日均收益':<15}")
    print("-" * 50)
    for idx, row in result_df_sorted.head(10).iterrows():
        rank = result_df_sorted.index.get_loc(idx) + 1
        print(f"{rank:<6} {row['Z值']:<10.1f} {row['总收益']:<15.2f} {row['总收益']/cycle_count:<15.2f}")
    print("-" * 50)
    
    return best_z, best_total_profit, cycle_count

def main():
    """主函数"""
    print("\n🚀 AEMO电池储能优化系统 - 2024年10月最优Z值计算")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    optimal_z, max_profit, cycle_count = calculate_optimal_z_for_october()
    
    print(f"\n⏰ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{'='*80}")
    print(f"✨ 计算完成！")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

