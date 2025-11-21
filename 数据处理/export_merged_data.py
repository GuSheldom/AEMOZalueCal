import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime, timedelta

# 设置文件夹路径
pv_data_folder = "1MW PV Monash -Electricity generation data-May 2024 to April 2025"
aemo_data_folder = "aemo"
output_folder = "verification_data"

# 创建输出目录
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 处理AEMO电价数据
def process_aemo_price_data():
    print("正在处理AEMO电价数据...")
    aemo_files = glob.glob(os.path.join(aemo_data_folder, "*.csv"))
    
    # 初始化存储所有AEMO数据的列表
    all_aemo_data = []
    
    for file_path in sorted(aemo_files):
        # 从文件名中提取月份和年份
        file_name = os.path.basename(file_path)
        parts = file_name.split("_")
        
        if len(parts) >= 3:
            # 从文件名提取日期部分，例如: PRICE_AND_DEMAND_202405_SA1.csv
            date_str = parts[2].split("_")[0]  # 例如: 202405
            
            # 确保日期格式正确
            if len(date_str) >= 6 and date_str.isdigit():
                year_month = date_str[:4] + "-" + date_str[4:6]
            else:
                print(f"  警告: 文件名日期格式不规范: {date_str}, 使用文件修改日期")
                file_date = datetime.fromtimestamp(os.path.getmtime(file_path))
                year_month = file_date.strftime("%Y-%m")
        else:
            print(f"  警告: 文件名格式不规范: {file_name}, 使用文件修改日期")
            file_date = datetime.fromtimestamp(os.path.getmtime(file_path))
            year_month = file_date.strftime("%Y-%m")
        
        try:
            # 读取AEMO数据
            aemo_df = pd.read_csv(file_path)
            
            # 转换日期时间格式
            aemo_df['SETTLEMENTDATE'] = pd.to_datetime(aemo_df['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
            
            # 添加年月标识符用于后续分析
            aemo_df['YearMonth'] = year_month
            
            # 存储处理后的数据
            all_aemo_data.append(aemo_df)
            
            print(f"  成功加载 {year_month} 的AEMO数据: {len(aemo_df)} 条记录")
        except Exception as e:
            print(f"  处理AEMO文件 {file_path} 时出错: {str(e)}")
    
    # 合并所有AEMO数据
    if all_aemo_data:
        aemo_combined = pd.concat(all_aemo_data, ignore_index=True)
        print(f"  总计加载 {len(aemo_combined)} 条AEMO价格记录")
        return aemo_combined
    else:
        print("  未找到有效的AEMO数据")
        return None

# 处理光伏发电量数据 - 分开处理solar和solarimp
def process_pv_generation_data():
    print("正在处理光伏发电量数据...")
    pv_files = glob.glob(os.path.join(pv_data_folder, "*.csv"))
    
    # 初始化数据框列表，分别存储solar和solarimp数据
    solar_data = []
    solarimp_data = []
    
    for file_path in sorted(pv_files):
        # 从文件名中提取月份和年份
        file_name = os.path.basename(file_path)
        date_parts = file_name.split("_")[2:4]
        month_year = date_parts[0]
        
        # 处理不规范的日期格式，如 "2024-05--30" 中的双破折号
        if '--' in month_year:
            month_year = month_year.replace('--', '-')
        
        try:
            month_date = datetime.strptime(month_year, "%Y-%m-%d")
            year_month = month_date.strftime("%Y-%m")
        except ValueError:
            # 如果日期格式仍有问题，则提取年月
            print(f"  警告: 日期格式不规范: {month_year}, 尝试提取年月...")
            # 尝试直接从字符串提取年月部分
            if len(month_year) >= 7:  # 至少应该有 "YYYY-MM"
                year_month = month_year[:7]
            else:
                # 如果无法提取，使用文件创建日期
                print(f"  无法从 {month_year} 提取年月，使用当前日期")
                year_month = datetime.now().strftime("%Y-%m")
        
        try:
            # 读取CSV文件，包括文件头以识别数据类型
            raw_df = pd.read_csv(file_path, nrows=20)
            
            # 查找"Site Name"行以确定solar和solarimp的位置
            site_name_row = None
            for i, row in raw_df.iterrows():
                if isinstance(row[0], str) and "Site Name" in row[0]:
                    site_name_row = i
                    break
            
            if site_name_row is not None:
                # 如果找到了Site Name行，读取第二行确定列名称
                column_names = raw_df.iloc[site_name_row].tolist()
                
                # 确定solar和solarimp所在的列索引
                solar_col = None
                solarimp_col = None
                
                for i, name in enumerate(column_names):
                    if isinstance(name, str) and "SOLAR" in name.upper() and "IMP" not in name.upper():
                        solar_col = i
                    elif isinstance(name, str) and "SOLAR" in name.upper() and "IMP" in name.upper():
                        solarimp_col = i
                
                if solar_col is None or solarimp_col is None:
                    print(f"  警告: 无法在 {file_path} 中找到solar和solarimp列")
                    continue
                
                # 读取实际数据，跳过前15行
                df = pd.read_csv(file_path, skiprows=15)
                
                # 将日期列转换为datetime
                if 'Date' in df.columns:
                    df['DateTime'] = pd.to_datetime(df['Date'], dayfirst=True)
                    df['YearMonth'] = year_month
                    
                    # 分别处理solar和solarimp数据
                    # 找到正确的数据列
                    consumption_cols = [col for col in df.columns if 'Consumption(kwh)' in col]
                    
                    if len(consumption_cols) >= 2:  # 至少需要两个消耗列
                        # 假设第一个消耗列是solar，第二个是solarimp
                        solar_consumption = consumption_cols[0]
                        solarimp_consumption = consumption_cols[1]
                        
                        # 复制dataframe并保留相关列
                        solar_df = df[['DateTime', 'Date', 'YearMonth', solar_consumption]].copy()
                        solar_df.rename(columns={solar_consumption: 'Consumption(kwh)'}, inplace=True)
                        solar_df['DataType'] = 'solar'
                        
                        solarimp_df = df[['DateTime', 'Date', 'YearMonth', solarimp_consumption]].copy()
                        solarimp_df.rename(columns={solarimp_consumption: 'Consumption(kwh)'}, inplace=True)
                        solarimp_df['DataType'] = 'solarimp'
                        
                        # 确保发电量列是数值型
                        solar_df['Consumption(kwh)'] = pd.to_numeric(solar_df['Consumption(kwh)'], errors='coerce').fillna(0)
                        solarimp_df['Consumption(kwh)'] = pd.to_numeric(solarimp_df['Consumption(kwh)'], errors='coerce').fillna(0)
                        
                        # 添加到各自的列表中
                        solar_data.append(solar_df)
                        solarimp_data.append(solarimp_df)
                        
                        print(f"  成功加载 {year_month} 的发电量数据: solar={len(solar_df)}条记录, solarimp={len(solarimp_df)}条记录")
                    else:
                        print(f"  警告: 未找到足够的Consumption(kwh)列 in {file_path}")
                else:
                    print(f"  警告: 文件 {file_path} 中未找到'Date'列")
            else:
                print(f"  警告: 在 {file_path} 中找不到Site Name行")
        except Exception as e:
            print(f"  处理文件 {file_path} 时出错: {str(e)}")
    
    # 合并所有数据
    combined_solar = pd.concat(solar_data, ignore_index=True) if solar_data else None
    combined_solarimp = pd.concat(solarimp_data, ignore_index=True) if solarimp_data else None
    
    if combined_solar is not None:
        print(f"  总计加载 {len(combined_solar)} 条solar发电量记录")
    if combined_solarimp is not None:
        print(f"  总计加载 {len(combined_solarimp)} 条solarimp发电量记录")
    
    return combined_solar, combined_solarimp

# 整合电价和发电量数据并导出详细对照表
def export_merged_data_with_prices(pv_data, aemo_data, data_type):
    print(f"正在为{data_type}数据创建详细的RRP对照表...")
    
    if pv_data is None or aemo_data is None:
        print("  数据不完整，无法继续分析")
        return None
    
    # 创建每30分钟的电价数据
    # 使用ceil将每个5分钟间隔的结束时间向上取整到30分钟
    # 例如：7:05、7:10、7:15、7:20、7:25 -> 7:30；7:30、7:35、7:40、7:45、7:50、7:55 -> 8:00
    aemo_data['TimeSlot'] = aemo_data['SETTLEMENTDATE'].dt.ceil('30min')
    
    # 每30分钟内有5-6个5分钟的价格点，计算统计值
    price_30min = aemo_data.groupby('TimeSlot').agg({
        'RRP': ['mean', 'min', 'max', 'count', 'std']  # 计算平均值、最小值、最大值、价格点数量和标准差
    }).reset_index()
    
    # 展平多级列名
    price_30min.columns = ['TimeSlot', 'AvgPrice', 'MinPrice', 'MaxPrice', 'PricePointCount', 'PriceStdDev']
    
    # 输出一些统计信息，了解价格点的完整性
    avg_points = price_30min['PricePointCount'].mean()
    print(f"  每30分钟时段内平均有 {avg_points:.2f} 个价格点（理想值为6个）")
    
    # 为PV数据添加时间段标识
    pv_data['TimeSlot'] = pv_data['DateTime'].dt.floor('30min')
    
    # 将价格数据合并到PV数据中
    merged_data = pd.merge(pv_data, price_30min, on='TimeSlot', how='left')
    
    # 计算每半小时的收益
    # RRP单位是AUD/MWh，需要除以1000转换为AUD/kWh
    merged_data['Revenue'] = merged_data['Consumption(kwh)'] * merged_data['AvgPrice'] / 1000
    
    # 添加实际时间信息便于核对
    merged_data['Date_Formatted'] = merged_data['DateTime'].dt.strftime('%Y-%m-%d')
    merged_data['Time_Formatted'] = merged_data['DateTime'].dt.strftime('%H:%M')
    
    # 整理列顺序，使核对更容易
    cols_order = ['Date_Formatted', 'Time_Formatted', 'Consumption(kwh)', 
                 'AvgPrice', 'MinPrice', 'MaxPrice', 'PriceStdDev', 'PricePointCount',
                 'Revenue', 'YearMonth', 'DataType']
    
    export_data = merged_data[cols_order].copy()
    
    # 保存为CSV文件
    output_file = os.path.join(output_folder, f"{data_type}_with_rrp_details.csv")
    export_data.to_csv(output_file, index=False)
    
    print(f"  成功导出{data_type}数据与RRP对照表: {output_file}")
    return merged_data

def main():
    # 处理AEMO电价数据
    aemo_data = process_aemo_price_data()
    
    # 处理光伏发电量数据
    solar_data, solarimp_data = process_pv_generation_data()
    
    # 导出带有详细RRP信息的数据表
    if solar_data is not None and aemo_data is not None:
        export_merged_data_with_prices(solar_data, aemo_data, "solar")
    
    if solarimp_data is not None and aemo_data is not None:
        export_merged_data_with_prices(solarimp_data, aemo_data, "solarimp")
    
    print("数据导出完成，您可以在verification_data目录下找到详细的对照表进行核对。")

if __name__ == "__main__":
    main() 