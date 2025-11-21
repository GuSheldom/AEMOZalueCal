import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# 设置数据文件夹路径
data_folder = "1MW PV Monash -Electricity generation data-May 2024 to April 2025"

# 获取所有CSV文件路径
csv_files = glob.glob(os.path.join(data_folder, "*.csv"))

# 初始化数据框
dfs = []
seasons = {
    '夏季': ['12', '01', '02'],
    '秋季': ['03', '04', '05'],
    '冬季': ['06', '07', '08'],
    '春季': ['09', '10', '11']
}

# 处理每个月的数据
for file_path in sorted(csv_files):
    # 从文件名中提取月份
    file_name = os.path.basename(file_path)
    date_parts = file_name.split("_")[2:4]
    month_year = date_parts[0]
    month_date = datetime.strptime(month_year, "%Y-%m-%d")
    month = month_date.strftime("%m")
    
    # 确定季节
    season = next((s for s, months in seasons.items() if month in months), "未知")
    
    try:
        # 读取CSV文件，跳过头部信息
        df = pd.read_csv(file_path, skiprows=15)
        
        if 'Date' in df.columns and 'Consumption(kwh)' in df.columns:
            # 转换日期和时间
            df['DateTime'] = pd.to_datetime(df['Date'], dayfirst=True)
            df['Date'] = df['DateTime'].dt.date
            df['Time'] = df['DateTime'].dt.time
            df['Hour'] = df['DateTime'].dt.hour
            df['Month'] = df['DateTime'].dt.month
            df['Season'] = season
            
            # 确保消费列是数值型
            df['Consumption(kwh)'] = pd.to_numeric(df['Consumption(kwh)'], errors='coerce').fillna(0)
            
            # 存储处理后的数据
            dfs.append(df)
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")

# 合并所有数据
if dfs:
    all_data = pd.concat(dfs, ignore_index=True)
    print(f"成功加载 {len(all_data)} 条记录")
    
    # 创建输出文件夹
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # 图1: 每日发电曲线 (取平均值)
    plt.figure(figsize=(12, 6))
    hourly_avg = all_data.groupby('Hour')['Consumption(kwh)'].mean()
    sns.lineplot(x=hourly_avg.index, y=hourly_avg.values)
    plt.title('平均每日发电曲线')
    plt.xlabel('小时')
    plt.ylabel('平均发电量 (kWh)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/daily_generation_curve.png')
    
    # 图2: 季节性发电模式
    plt.figure(figsize=(12, 6))
    season_hour_avg = all_data.groupby(['Season', 'Hour'])['Consumption(kwh)'].mean().reset_index()
    sns.lineplot(data=season_hour_avg, x='Hour', y='Consumption(kwh)', hue='Season')
    plt.title('季节性发电曲线')
    plt.xlabel('小时')
    plt.ylabel('平均发电量 (kWh)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='季节')
    plt.tight_layout()
    plt.savefig('visualizations/seasonal_generation.png')
    
    # 图3: 月度总发电量热图
    plt.figure(figsize=(14, 8))
    monthly_total = all_data.groupby(['Month', 'Hour'])['Consumption(kwh)'].mean().reset_index().pivot('Month', 'Hour', 'Consumption(kwh)')
    sns.heatmap(monthly_total, cmap='YlOrRd', annot=False, fmt='.1f', cbar_kws={'label': '平均发电量 (kWh)'})
    plt.title('月度-小时发电量热图')
    plt.xlabel('小时')
    plt.ylabel('月份')
    plt.tight_layout()
    plt.savefig('visualizations/monthly_hourly_heatmap.png')
    
    # 图4: 每月总发电量比较
    plt.figure(figsize=(12, 6))
    monthly_generation = all_data.groupby('Month')['Consumption(kwh)'].sum()
    sns.barplot(x=monthly_generation.index, y=monthly_generation.values)
    plt.title('月度总发电量')
    plt.xlabel('月份')
    plt.ylabel('总发电量 (kWh)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/monthly_total_generation.png')
    
    # 图5: 峰值发电量分析
    plt.figure(figsize=(12, 6))
    daily_peak = all_data.groupby('Date')['Consumption(kwh)'].max()
    plt.plot(daily_peak.index, daily_peak.values)
    plt.title('每日峰值发电量')
    plt.xlabel('日期')
    plt.ylabel('峰值发电量 (kWh)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/daily_peak_generation.png')
    
    # 图6: 年度发电量曲线
    plt.figure(figsize=(14, 8))
    
    # 根据日期排序
    all_data['Date_Only'] = pd.to_datetime(all_data['Date'])
    date_gen = all_data.groupby('Date_Only')['Consumption(kwh)'].sum().reset_index()
    date_gen = date_gen.sort_values('Date_Only')
    
    plt.plot(date_gen['Date_Only'], date_gen['Consumption(kwh)'])
    plt.title('年度发电量曲线')
    plt.xlabel('日期')
    plt.ylabel('日发电量 (kWh)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置x轴日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()  # 自动格式化x轴日期标签
    
    plt.tight_layout()
    plt.savefig('visualizations/yearly_generation_curve.png')
    
    # 输出统计信息
    print("\n===== 统计分析 =====")
    print(f"年总发电量: {all_data['Consumption(kwh)'].sum():.2f} kWh")
    print(f"平均日发电量: {all_data.groupby('Date')['Consumption(kwh)'].sum().mean():.2f} kWh")
    print(f"最高单日发电量: {all_data.groupby('Date')['Consumption(kwh)'].sum().max():.2f} kWh")
    print(f"最高峰值发电量: {all_data['Consumption(kwh)'].max():.2f} kWh")
    
    # 计算容量因子 (实际年发电量 / 理论最大发电量)
    # 1MW电站理论最大年发电量 = 1MW * 24小时 * 365天 = 8760 MWh = 8,760,000 kWh
    capacity_factor = (all_data['Consumption(kwh)'].sum() / 8760000) * 100
    print(f"容量因子: {capacity_factor:.2f}%")
    
    # 每个季节的平均发电量
    season_avg = all_data.groupby('Season')['Consumption(kwh)'].sum()
    print("\n季节发电量:")
    for season, gen in season_avg.items():
        print(f"{season}: {gen:.2f} kWh")
else:
    print("没有找到有效数据进行分析") 