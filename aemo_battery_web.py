#!/usr/bin/env python3
"""
AEMO电池储能优化系统 - Web版本 (Streamlit)
显示所有周期数据，支持周期选择和Z值实时调整
"""

import streamlit as st
import pandas as pd
import numpy as np
import glob
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time as dt_time
from typing import List, Dict, Tuple, Optional
import pulp

# 页面配置
st.set_page_config(
    page_title="AEMO电池储能优化系统",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def assign_cycle_date(ts_str: str) -> str:
    """分配周期日期"""
    ts = pd.to_datetime(ts_str)
    if ts.time() >= dt_time(23, 0):
        return str(ts.normalize().date())
    elif ts.time() < dt_time(8, 0):
        return str((ts - pd.Timedelta(days=1)).normalize().date())
    else:
        return str(ts.normalize().date())

@st.cache_data
def load_all_data():
    """加载所有数据"""
    try:
        # 查找所有z0Fast文件
        pattern = "AEMO_23to08_with_opt_*_z0Fast.xlsx"
        excel_files = sorted(glob.glob(pattern))
        
        if not excel_files:
            st.error("未找到数据文件！请确保AEMO_23to08_with_opt_*_z0Fast.xlsx文件存在。")
            return None
        
        st.info(f"正在加载 {len(excel_files)} 个数据文件...")
        
        # 加载所有数据
        all_dataframes = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(excel_files):
            try:
                df = pd.read_excel(file, sheet_name="23to08_opt")
                # 重命名列
                df = df.rename(columns={
                    "时间": "Timestamp",
                    "电价(RRP)": "Price_RRP", 
                    "阶段": "Phase",
                    "z值": "Z_Value",
                    "电量(kWh)": "Energy_kWh",
                    "累计电量(kWh)": "Cumulative_Energy_kWh",
                    "成本/收益": "Cost_Revenue",
                    "周期总收益": "Cycle_Total_Revenue"
                })
                all_dataframes.append(df)
                progress_bar.progress((i + 1) / len(excel_files))
            except Exception as e:
                st.warning(f"加载 {file} 失败: {e}")
        
        if not all_dataframes:
            st.error("无法加载任何数据文件！")
            return None
        
        # 合并数据
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        merged_df["Timestamp"] = pd.to_datetime(merged_df["Timestamp"])
        merged_df = merged_df.sort_values("Timestamp").reset_index(drop=True)
        
        # 添加周期信息
        merged_df["Cycle_Date"] = merged_df["Timestamp"].astype(str).apply(assign_cycle_date)
        
        st.success(f"数据加载完成！共 {len(merged_df)} 行数据，{merged_df['Cycle_Date'].nunique()} 个周期")
        return merged_df
        
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return None

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
        st.error(f"求解过程出错: {e}")
        return [0.0] * len(charge_prices), [0.0] * len(discharge_prices), 0.0

def update_cycle_data_with_z(cycle_data: pd.DataFrame, z_value: float) -> pd.DataFrame:
    """根据新的Z值更新周期数据"""
    updated_data = cycle_data.copy()
    
    # 提取充电和放电数据
    charge_data = updated_data[updated_data["Phase"] == "charge"]
    discharge_data = updated_data[updated_data["Phase"] == "discharge"]
    
    if len(charge_data) == 0 or len(discharge_data) == 0:
        return updated_data
    
    # 获取价格数据
    charge_prices = charge_data["Price_RRP"].tolist()
    discharge_prices = discharge_data["Price_RRP"].tolist()
    
    # 求解优化问题
    charge_energy, discharge_energy, total_profit = solve_cycle_with_z(
        charge_prices, discharge_prices, z_value)
    
    # 重置所有能量值
    updated_data["Z_Value"] = z_value
    updated_data["Energy_kWh"] = 0.0
    updated_data["Cost_Revenue"] = 0.0
    
    # 更新充电数据
    charge_indices = charge_data.index
    for i, idx in enumerate(charge_indices):
        if i < len(charge_energy):
            energy = charge_energy[i]
            price = updated_data.at[idx, "Price_RRP"]
            updated_data.at[idx, "Energy_kWh"] = energy
            updated_data.at[idx, "Cost_Revenue"] = -price * energy / 1000
    
    # 更新放电数据
    discharge_indices = discharge_data.index
    for i, idx in enumerate(discharge_indices):
        if i < len(discharge_energy):
            energy = -discharge_energy[i]  # 放电为负值
            price = updated_data.at[idx, "Price_RRP"]
            updated_data.at[idx, "Energy_kWh"] = energy
            updated_data.at[idx, "Cost_Revenue"] = -price * energy / 1000
    
    # 计算累计电量
    cumulative_energy = 0
    for idx in updated_data.index:
        energy = updated_data.at[idx, "Energy_kWh"]
        cumulative_energy = max(0, cumulative_energy + energy)
        updated_data.at[idx, "Cumulative_Energy_kWh"] = cumulative_energy
    
    # 设置周期总收益
    updated_data["Cycle_Total_Revenue"] = total_profit
    
    return updated_data

def main():
    """主函数"""
    st.title("⚡ AEMO电池储能优化系统")
    st.markdown("---")
    
    # 加载数据
    if 'all_data' not in st.session_state:
        with st.spinner("正在加载数据..."):
            st.session_state.all_data = load_all_data()
    
    all_data = st.session_state.all_data
    if all_data is None:
        st.stop()
    
    # 侧边栏控制
    st.sidebar.header("🎛️ 控制面板")
    
    # 获取所有周期
    unique_cycles = sorted(all_data["Cycle_Date"].unique())
    
    # 周期选择
    selected_cycle = st.sidebar.selectbox(
        "📅 选择周期",
        unique_cycles,
        index=0
    )
    
    # Z值输入
    z_value = st.sidebar.number_input(
        "⚡ Z值",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.5,
        format="%.1f"
    )
    
    # 获取选定周期的数据
    cycle_data = all_data[all_data["Cycle_Date"] == selected_cycle].copy()
    
    if len(cycle_data) == 0:
        st.error("选定周期没有数据")
        st.stop()
    
    # 根据Z值更新数据
    if st.sidebar.button("🔄 重新计算", type="primary"):
        with st.spinner("正在计算最优策略..."):
            cycle_data = update_cycle_data_with_z(cycle_data, z_value)
            st.session_state.current_cycle_data = cycle_data
    
    # 使用缓存的数据或原始数据
    if 'current_cycle_data' in st.session_state:
        display_data = st.session_state.current_cycle_data
    else:
        display_data = update_cycle_data_with_z(cycle_data, z_value)
        st.session_state.current_cycle_data = display_data
    
    # 显示统计信息
    col1, col2, col3, col4 = st.columns(4)
    
    total_profit = display_data["Cycle_Total_Revenue"].iloc[0] if len(display_data) > 0 else 0
    total_charge = display_data[display_data["Phase"] == "charge"]["Energy_kWh"].sum()
    total_discharge = -display_data[display_data["Phase"] == "discharge"]["Energy_kWh"].sum()
    max_cumulative = display_data["Cumulative_Energy_kWh"].max()
    
    with col1:
        st.metric("📊 周期总收益", f"{total_profit:.2f}", delta=None)
    
    with col2:
        st.metric("🔋 总充电量", f"{total_charge:.1f} kWh", delta=None)
    
    with col3:
        st.metric("⚡ 总放电量", f"{total_discharge:.1f} kWh", delta=None)
    
    with col4:
        st.metric("📈 最大储能", f"{max_cumulative:.1f} kWh", delta=None)
    
    # 创建两列布局
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📋 周期详细数据")
        
        # 为展示计算新增列：充电成本(按周期内电量净累计) 与 周期内累加收益
        display_data = display_data.sort_values(["Cycle_Date", "Timestamp"]).copy()
        display_data["Charge_Cost"] = display_data.groupby("Cycle_Date")["Energy_kWh"].cumsum()
        display_data["Cycle_Cum_Revenue"] = display_data.groupby("Cycle_Date")["Cost_Revenue"].cumsum()

        # 准备显示用的数据
        display_df = display_data.copy()
        display_df["时间"] = display_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        display_df["电价(RRP)"] = display_df["Price_RRP"].round(2)
        display_df["阶段"] = display_df["Phase"].map({"charge": "充电", "discharge": "放电"})
        # 将原“Z值”列替换为“充电成本”（按周期内电量净累计）
        display_df["充电成本"] = display_df["Charge_Cost"].round(2)
        display_df["电量(kWh)"] = display_df["Energy_kWh"].round(2)
        display_df["累计电量(kWh)"] = display_df["Cumulative_Energy_kWh"].round(2)
        display_df["成本/收益"] = display_df["Cost_Revenue"].round(2)
        # 将原“周期总收益”列替换为“周期内累加收益”（按周期内累计到当前行），并追加展示“周期总收益”
        display_df["周期内累加收益"] = display_df["Cycle_Cum_Revenue"].round(2)
        display_df["周期总收益"] = display_df["Cycle_Total_Revenue"].round(2)
        
        # 显示表格
        st.dataframe(
            display_df[["时间", "电价(RRP)", "阶段", "充电成本", "电量(kWh)", 
                       "累计电量(kWh)", "成本/收益", "周期内累加收益", "周期总收益"]],
            use_container_width=True,
            height=400
        )
    
    with col_right:
        st.subheader("📊 可视化分析")
        
        # 电价趋势图
        fig_price = px.line(
            display_data, 
            x="Timestamp", 
            y="Price_RRP",
            color="Phase",
            title="电价趋势",
            color_discrete_map={"charge": "blue", "discharge": "red"}
        )
        fig_price.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_price, use_container_width=True)
        
        # 能量分布图
        fig_energy = px.bar(
            display_data[display_data["Energy_kWh"] != 0], 
            x="Timestamp", 
            y="Energy_kWh",
            color="Phase",
            title="充放电能量分布",
            color_discrete_map={"charge": "green", "discharge": "orange"}
        )
        fig_energy.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_energy, use_container_width=True)
        
        # 累计电量图
        fig_cumulative = px.line(
            display_data, 
            x="Timestamp", 
            y="Cumulative_Energy_kWh",
            title="累计储能量",
            color_discrete_sequence=["purple"]
        )
        fig_cumulative.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_cumulative, use_container_width=True)
    
    # 显示周期概览
    st.subheader("🔍 周期概览")
    
    # 按阶段分组统计
    phase_stats = display_data.groupby("Phase").agg({
        "Energy_kWh": ["sum", "count"],
        "Cost_Revenue": "sum",
        "Price_RRP": ["mean", "min", "max"]
    }).round(2)
    
    phase_stats.columns = ["总能量", "时段数", "总成本收益", "平均电价", "最低电价", "最高电价"]
    phase_stats.index = phase_stats.index.map({"charge": "充电阶段", "discharge": "放电阶段"})
    
    st.dataframe(phase_stats, use_container_width=True)
    
    # 侧边栏显示系统信息
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ 系统信息")
    st.sidebar.info(f"""
    **当前周期**: {selected_cycle}  
    **Z值**: {z_value}  
    **数据点**: {len(display_data)} 行  
    **充电时段**: {len(display_data[display_data['Phase'] == 'charge'])} 个  
    **放电时段**: {len(display_data[display_data['Phase'] == 'discharge'])} 个
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("💡 **使用说明**:")
    st.sidebar.markdown("""
    1. 选择要查看的周期
    2. 调整Z值（最低利润阈值）
    3. 点击"重新计算"更新结果
    4. 查看表格和图表了解详情
    """)

if __name__ == "__main__":
    main() 