#!/usr/bin/env python3
"""
合并所有月份的Excel文件为一个完整的Excel文件
"""

import pandas as pd
import glob
import os
from typing import List

def merge_monthly_excel_files():
    """合并所有月份的Excel文件"""
    
    # 查找所有z0Fast的Excel文件
    pattern = "AEMO_23to08_with_opt_*_z0Fast.xlsx"
    excel_files = sorted(glob.glob(pattern))
    
    if not excel_files:
        print("未找到任何z0Fast的Excel文件")
        return
    
    print(f"找到 {len(excel_files)} 个Excel文件:")
    for file in excel_files:
        print(f"  - {file}")
    
    # 读取并合并所有Excel文件
    all_dataframes: List[pd.DataFrame] = []
    total_rows = 0
    
    for file in excel_files:
        try:
            df = pd.read_excel(file, sheet_name="23to08_opt")
            
            # 确保列名为中文（防止有些文件列名不一致）
            expected_columns = {
                "time": "时间",
                "rrp": "电价(RRP)",
                "label": "阶段", 
                "z": "z值",
                "qty_kwh": "电量(kWh)",
                "cum_kwh": "累计电量(kWh)",
                "pnl": "成本/收益",
                "cycle_total_pnl": "周期总收益",
                # 处理可能的英文列名
                "时间": "时间",
                "电价(RRP)": "电价(RRP)",
                "阶段": "阶段",
                "z值": "z值", 
                "电量(kWh)": "电量(kWh)",
                "累计电量(kWh)": "累计电量(kWh)",
                "成本/收益": "成本/收益",
                "周期总收益": "周期总收益"
            }
            
            # 重命名列名为中文
            df = df.rename(columns=expected_columns)
            
            # 确保有所有必需的列
            required_columns = ["时间", "电价(RRP)", "阶段", "z值", "电量(kWh)", "累计电量(kWh)", "成本/收益", "周期总收益"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"警告: 文件 {file} 缺少列: {missing_columns}")
                continue
            
            # 选择需要的列
            df = df[required_columns]
            
            all_dataframes.append(df)
            total_rows += len(df)
            print(f"  读取 {file}: {len(df)} 行")
            
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
            continue
    
    if not all_dataframes:
        print("没有成功读取任何Excel文件")
        return
    
    # 合并所有数据
    print(f"\n合并 {len(all_dataframes)} 个文件，总共 {total_rows} 行...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    # 按时间排序
    merged_df["时间"] = pd.to_datetime(merged_df["时间"])
    merged_df = merged_df.sort_values("时间").reset_index(drop=True)
    
    # 输出统计信息
    print(f"合并后总行数: {len(merged_df)}")
    print(f"时间范围: {merged_df['时间'].min()} 到 {merged_df['时间'].max()}")
    print(f"总周期数: {merged_df['时间'].dt.normalize().nunique()}")
    
    # 输出到Excel文件
    output_file = "AEMO_23to08_合并所有月份_z0Fast.xlsx"
    
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        merged_df.to_excel(writer, index=False, sheet_name="所有数据")
        
        # 获取工作表对象用于格式化
        workbook = writer.book
        worksheet = writer.sheets["所有数据"]
        
        # 设置列宽
        worksheet.set_column('A:A', 20)  # 时间列
        worksheet.set_column('B:B', 12)  # 电价列
        worksheet.set_column('C:C', 8)   # 阶段列
        worksheet.set_column('D:D', 8)   # z值列
        worksheet.set_column('E:E', 12)  # 电量列
        worksheet.set_column('F:F', 15)  # 累计电量列
        worksheet.set_column('G:G', 12)  # 成本/收益列
        worksheet.set_column('H:H', 15)  # 周期总收益列
        
        # 设置数字格式
        number_format = workbook.add_format({'num_format': '0.00'})
        worksheet.set_column('B:B', 12, number_format)  # 电价
        worksheet.set_column('D:D', 8, number_format)   # z值
        worksheet.set_column('E:E', 12, number_format)  # 电量
        worksheet.set_column('F:F', 15, number_format)  # 累计电量
        worksheet.set_column('G:G', 12, number_format)  # 成本/收益
        worksheet.set_column('H:H', 15, number_format)  # 周期总收益
        
        # 冻结首行
        worksheet.freeze_panes(1, 0)
    
    print(f"\n✅ 合并完成! 输出文件: {output_file}")
    print(f"文件大小: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    merge_monthly_excel_files() 