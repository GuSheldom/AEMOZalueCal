import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob

# 设置数据文件夹路径
data_folder = "1MW PV Monash -Electricity generation data-May 2024 to April 2025"

# 获取所有CSV文件路径
csv_files = glob.glob(os.path.join(data_folder, "*.csv"))

# 初始化数据框列表和月度发电量列表
dfs = []
monthly_generation = []
months = []

# 处理每个月的数据文件
for file_path in sorted(csv_files):
    # 从文件名中提取月份和年份
    file_name = os.path.basename(file_path)
    date_parts = file_name.split("_")[2:4]
    month_year = date_parts[0]
    month_date = datetime.strptime(month_year, "%Y-%m-%d")
    month_name = month_date.strftime("%Y-%m")
    months.append(month_name)
    
    # 读取CSV文件
    df = pd.read_csv(file_path, skiprows=15)  # 跳过文件头信息
    
    # 将日期和时间列合并为datetime格式
    if 'Date' in df.columns:
        df['DateTime'] = pd.to_datetime(df['Date'], dayfirst=True)
        
        # 计算总发电量 (kWh)
        # 注意: 基于CSV文件格式可能需要调整列名
        if 'Consumption(kwh)' in df.columns:
            total_generation = df['Consumption(kwh)'].sum()
            monthly_generation.append(total_generation)
            print(f"{month_name}: 总发电量 = {total_generation:.2f} kWh")
        
        # 存储处理后的数据框
        dfs.append(df)
    else:
        print(f"警告: 文件 {file_path} 格式不正确或无法读取")

# 合并所有月份的数据
if dfs:
    all_data = pd.concat(dfs, ignore_index=True)
    
    # 创建月度发电量图表
    plt.figure(figsize=(12, 6))
    plt.bar(months, monthly_generation)
    plt.title('每月发电量 (kWh)')
    plt.xlabel('月份')
    plt.ylabel('发电量 (kWh)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('monthly_generation.png')
    
    # 保存处理后的数据
    all_data.to_csv('processed_pv_data.csv', index=False)
    
    # 输出总发电量
    total_annual_generation = sum(monthly_generation)
    print(f"\n总年发电量: {total_annual_generation:.2f} kWh")
    print(f"平均月发电量: {total_annual_generation/len(monthly_generation):.2f} kWh")
    
    # 如果有AEMO电价数据，可以在这里添加收益分析
    # 例如: revenue = total_generation * price_per_kwh
else:
    print("未找到有效数据文件") 