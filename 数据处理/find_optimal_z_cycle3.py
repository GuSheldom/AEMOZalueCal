#!/usr/bin/env python3
"""
计算第三个周期的最优Z值
测试不同Z值下的收益，找出最佳策略
"""

import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time
from typing import List, Tuple
import pulp
import glob

def assign_cycle_date(ts: pd.Timestamp) -> pd.Timestamp:
    """分配周期日期"""
    if ts.time() >= dt_time(23, 0):
        return ts.normalize()
    elif ts.time() < dt_time(8, 0):
        return (ts - pd.Timedelta(days=1)).normalize()
    else:
        return ts.normalize()

def solve_cycle_with_z(charge_prices: List[float], discharge_prices: List[float], 
                      z: float) -> Tuple[List[float], List[float], float]:
    """使用线性规划求解给定Z值下的最优分配"""
    try:
        # 创建线性规划问题
        prob = pulp.LpProblem("Battery_Optimization", pulp.LpMaximize)
        
        n_charge = len(charge_prices)
        n_discharge = len(discharge_prices)
        
        # 决策变量：充电时段i到放电时段j的能量分配
        x = {}
        for i in range(n_charge):
            for j in range(n_discharge):
                if discharge_prices[j] > charge_prices[i] + z:  # 只有满足阈值条件才创建变量
                    x[i, j] = pulp.LpVariable(f"x_{i}_{j}", 0, None)
        
        if not x:  # 没有可行的分配
            return [0.0] * n_charge, [0.0] * n_discharge, 0.0
        
        # 目标函数：最大化总利润
        profit_terms = []
        for (i, j), var in x.items():
            profit = discharge_prices[j] - charge_prices[i]
            profit_terms.append(profit * var)
        
        if profit_terms:
            prob += pulp.lpSum(profit_terms)
        
        # 约束条件
        # 1. 充电时段容量约束 (每个时段最多55.83 kWh)
        for i in range(n_charge):
            charge_vars = [x[i, j] for j in range(n_discharge) if (i, j) in x]
            if charge_vars:
                prob += pulp.lpSum(charge_vars) <= 55.83
        
        # 2. 放电时段容量约束 (每个时段最多200 kWh)
        for j in range(n_discharge):
            discharge_vars = [x[i, j] for i in range(n_charge) if (i, j) in x]
            if discharge_vars:
                prob += pulp.lpSum(discharge_vars) <= 200.0
        
        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status != pulp.LpStatusOptimal:
            return [0.0] * n_charge, [0.0] * n_discharge, 0.0
        
        # 提取结果
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

def find_optimal_z_for_third_cycle():
    """找出第三个周期的最优Z值"""
    print("🔍 正在加载所有数据以找到第三个周期...")
    
    # 加载所有数据文件
    pattern = "AEMO_23to08_with_opt_*_z0Fast.xlsx"
    excel_files = sorted(glob.glob(pattern))
    
    if not excel_files:
        print("❌ 未找到数据文件")
        return
    
    # 加载并合并所有数据
    all_dataframes = []
    for file in excel_files:
        try:
            df = pd.read_excel(file, sheet_name="23to08_opt")
            # 重命名列
            df = df.rename(columns={
                "时间": "Timestamp",
                "电价(RRP)": "Price_RRP", 
                "阶段": "Phase"
            })
            all_dataframes.append(df)
            print(f"  加载 {file}: {len(df)} 行")
        except Exception as e:
            print(f"  加载 {file} 失败: {e}")
    
    if not all_dataframes:
        print("❌ 无法加载任何数据文件")
        return
    
    # 合并数据
    df = pd.concat(all_dataframes, ignore_index=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["Cycle_Date"] = df["Timestamp"].apply(assign_cycle_date)
    
    print(f"✅ 数据加载完成，共 {len(df)} 行")
    
    # 获取所有唯一周期并找到第三个
    unique_cycles = sorted(df["Cycle_Date"].unique())
    print(f"📊 总共找到 {len(unique_cycles)} 个周期")
    
    if len(unique_cycles) < 3:
        print("❌ 数据中少于3个周期")
        return
    
    third_cycle_date = unique_cycles[2]  # 索引2是第三个
    third_cycle_data = df[df["Cycle_Date"] == third_cycle_date].copy()
    
    print(f"📅 第三个周期日期: {third_cycle_date.date()}")
    print(f"📊 周期数据点: {len(third_cycle_data)} 行")
    
    # 提取充电和放电数据
    charge_data = third_cycle_data[third_cycle_data["Phase"] == "charge"]
    discharge_data = third_cycle_data[third_cycle_data["Phase"] == "discharge"]
    
    print(f"🔋 充电时段: {len(charge_data)} 个")
    print(f"⚡ 放电时段: {len(discharge_data)} 个")
    
    if len(charge_data) == 0 or len(discharge_data) == 0:
        print("❌ 充电或放电数据为空")
        return
    
    # 获取价格数据
    charge_prices = charge_data["Price_RRP"].tolist()
    discharge_prices = discharge_data["Price_RRP"].tolist()
    
    print(f"💰 充电价格范围: {min(charge_prices):.2f} ~ {max(charge_prices):.2f}")
    print(f"💰 放电价格范围: {min(discharge_prices):.2f} ~ {max(discharge_prices):.2f}")
    
    # 显示一些价格样本
    print(f"📋 充电价格示例: {charge_prices[:5]} ...")
    print(f"📋 放电价格示例: {discharge_prices[:5]} ...")
    
    # 测试不同的Z值
    print("\n🧮 开始测试不同Z值的收益...")
    z_values = np.arange(0.0, 50.1, 0.5)  # 从0到50，步长0.5
    results = []
    
    for i, z in enumerate(z_values):
        charge_energy, discharge_energy, total_profit = solve_cycle_with_z(
            charge_prices, discharge_prices, z)
        
        results.append({
            'z': z,
            'profit': total_profit,
            'total_charge': sum(charge_energy),
            'total_discharge': sum(discharge_energy)
        })
        
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(z_values)} (Z={z:.1f}, 收益={total_profit:.2f})")
    
    # 转换为DataFrame并分析
    results_df = pd.DataFrame(results)
    
    # 找出最优Z值
    best_result = results_df.loc[results_df['profit'].idxmax()]
    
    print(f"\n🎯 第三个周期最优结果:")
    print(f"   周期日期: {third_cycle_date.date()}")
    print(f"   最佳Z值: {best_result['z']:.1f}")
    print(f"   最大收益: {best_result['profit']:.2f}")
    print(f"   总充电量: {best_result['total_charge']:.2f} kWh")
    print(f"   总放电量: {best_result['total_discharge']:.2f} kWh")
    
    # 显示前10个最佳结果
    print(f"\n📊 前10个最佳Z值:")
    top_10 = results_df.nlargest(10, 'profit')
    for idx, row in top_10.iterrows():
        print(f"   Z={row['z']:4.1f}: 收益={row['profit']:7.2f}, 充电={row['total_charge']:6.1f}, 放电={row['total_discharge']:6.1f}")
    
    # 保存详细结果
    filename = f"third_cycle_z_optimization_{third_cycle_date.date()}.csv"
    results_df.to_csv(filename, index=False)
    print(f"\n💾 详细结果已保存到: {filename}")
    
    # 分析收益趋势
    print(f"\n📈 收益趋势分析:")
    key_z_values = [0, 5, 10, 15, 20, 25, 30]
    for z in key_z_values:
        if z <= results_df['z'].max():
            profit = results_df[results_df['z']==z]['profit'].iloc[0]
            print(f"   Z={z:2d}时收益: {profit:7.2f}")
    
    # 找出收益为0的临界Z值
    zero_profit = results_df[results_df['profit'] <= 0]
    if len(zero_profit) > 0:
        critical_z = zero_profit['z'].min()
        print(f"   收益归零的临界Z值: {critical_z:.1f}")
    else:
        print(f"   在测试范围内收益始终为正")
    
    # 比较与第一个周期的差异
    print(f"\n📊 与第一个周期对比:")
    print(f"   第一个周期最优收益: 1037.08")
    print(f"   第三个周期最优收益: {best_result['profit']:.2f}")
    diff = best_result['profit'] - 1037.08
    pct_change = (diff / 1037.08) * 100
    print(f"   收益差异: {diff:+.2f} ({pct_change:+.1f}%)")
    
    return best_result

if __name__ == "__main__":
    find_optimal_z_for_third_cycle() 