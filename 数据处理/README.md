# ⚡ AEMO电池储能优化系统

一个基于线性规划的电池储能优化分析工具，用于分析AEMO电力市场的充放电策略和收益。

## 🌟 功能特点

- 📊 **实时数据分析**：加载2023年12月-2025年8月的AEMO价格数据
- ⚡ **智能优化**：使用线性规划(PuLP)计算最优充放电策略
- 🎛️ **交互界面**：Web界面支持周期选择和Z值调整
- 📈 **可视化分析**：电价趋势、能量分布、累计储能图表
- 🔄 **实时计算**：Z值变化时自动重新优化策略

## 🚀 在线访问

**立即体验**: [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)

## 📋 本地运行

### 环境要求
- Python 3.8+
- 依赖包见 `requirements.txt`

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/your-username/aemo-battery-optimization.git
   cd aemo-battery-optimization
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **运行应用**
   ```bash
   streamlit run aemo_battery_web.py
   ```

4. **访问应用**
   - 本地访问: http://localhost:8501

## 📊 数据说明

### 数据来源
- **AEMO电力市场**价格数据（2023年12月-2025年8月）
- 每5分钟一个数据点
- 包含23:00-08:00的充放电时段

### 数据文件
- `AEMO_23to08_with_opt_*_z0Fast.xlsx`: 各月优化结果
- 每个周期108个数据点（72个充电+36个放电）

## 🎮 使用方法

### 界面操作
1. **选择周期**：从下拉菜单选择要分析的日期
2. **调整Z值**：设置最低利润阈值（0-50）
3. **重新计算**：点击按钮应用新的Z值
4. **查看结果**：观察表格和图表变化

### 参数说明
- **Z值**：最低利润阈值，只有当`放电价格 > 充电价格 + Z值`时才进行交易
- **充电功率**：670kW（每5分钟最多55.83kWh）
- **放电功率**：2400kW（每5分钟最多200kWh）

## 📈 核心算法

### 线性规划优化
- **目标**：最大化总收益
- **约束**：充电/放电功率限制
- **条件**：满足Z值阈值要求

### 计算公式
```
收益 = Σ(放电价格 - 充电价格) × 能量传输量
约束：放电价格 > 充电价格 + Z值
```

## 🛠️ 技术栈

- **前端界面**：Streamlit
- **数据处理**：Pandas, NumPy
- **优化算法**：PuLP (线性规划)
- **可视化**：Plotly
- **数据存储**：Excel (openpyxl)

## 📁 项目结构

```
├── aemo_battery_web.py      # 主应用程序
├── requirements.txt         # 依赖包列表
├── README.md               # 项目说明
├── .gitignore             # Git忽略文件
├── AEMO_23to08_with_opt_*_z0Fast.xlsx  # 数据文件
└── README_deployment.md    # 部署指南
```

## 🔒 数据隐私

- 使用公开的AEMO市场数据
- 不包含个人敏感信息
- 数据仅用于学术和分析目的

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 📧 Email: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/aemo-battery-optimization/issues)

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

---

⚡ **让电池储能优化变得简单高效！** 