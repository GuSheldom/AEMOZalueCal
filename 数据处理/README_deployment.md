# AEMO电池储能优化系统 - 部署指南

## 🌐 让其他人访问您的应用程序

### 方案1：Streamlit Cloud部署（免费，推荐）

#### 步骤：
1. **创建GitHub仓库**
   - 将代码上传到GitHub
   - 包含：`aemo_battery_web.py`, `requirements.txt`, 数据文件

2. **登录Streamlit Cloud**
   - 访问：https://share.streamlit.io/
   - 用GitHub账号登录

3. **部署应用**
   - 点击"New app"
   - 选择您的GitHub仓库
   - 主文件：`aemo_battery_web.py`
   - 点击"Deploy"

4. **获得公开链接**
   - 系统会生成类似：https://your-app-name.streamlit.app
   - 任何人都可以访问这个链接

#### 优点：
- ✅ 完全免费
- ✅ 自动HTTPS
- ✅ 全球访问
- ✅ 自动更新

#### 缺点：
- ❌ 需要GitHub账号
- ❌ 代码必须公开（除非付费）

---

### 方案2：使用ngrok隧道（临时访问）

#### 安装ngrok：
```bash
# macOS
brew install ngrok

# 或下载：https://ngrok.com/download
```

#### 使用方法：
```bash
# 在新终端运行
ngrok http 8501
```

#### 获得公开链接：
- ngrok会显示类似：https://abc123.ngrok.io
- 任何人都可以访问这个临时链接

#### 优点：
- ✅ 立即可用
- ✅ 不需要修改代码
- ✅ 数据保持私有

#### 缺点：
- ❌ 临时链接（8小时后失效）
- ❌ 免费版有连接限制
- ❌ 需要保持本地运行

---

### 方案3：VPS服务器部署

#### 使用云服务器：
- 阿里云、腾讯云、AWS等
- 安装Docker或直接部署

#### 部署命令：
```bash
# 在服务器上
git clone your-repo
cd your-repo
pip install -r requirements.txt
streamlit run aemo_battery_web.py --server.port 8501 --server.address 0.0.0.0
```

#### 优点：
- ✅ 完全控制
- ✅ 永久访问
- ✅ 可自定义域名

#### 缺点：
- ❌ 需要付费
- ❌ 需要技术维护

---

## 🔒 数据安全考虑

### 当前数据文件：
- `AEMO_23to08_with_opt_*_z0Fast.xlsx` 文件包含真实数据
- 如果数据敏感，建议：
  1. 使用示例数据替换
  2. 或选择私有部署方案

### 建议：
- 公开部署：使用脱敏或示例数据
- 私有访问：使用ngrok或VPS方案

---

## 🚀 快速开始（推荐方案）

### 最简单的方法：使用ngrok

1. **安装ngrok**：
   ```bash
   brew install ngrok
   ```

2. **启动应用**（如果还没运行）：
   ```bash
   streamlit run aemo_battery_web.py
   ```

3. **创建公开隧道**：
   ```bash
   ngrok http 8501
   ```

4. **分享链接**：
   - ngrok显示的https链接可以直接分享给任何人
   - 他们可以在任何地方访问您的应用

### 示例输出：
```
Session Status                online
Account                       your-email@example.com
Version                       3.0.0
Region                        United States (us)
Forwarding                    https://abc123.ngrok.io -> http://localhost:8501
Forwarding                    http://abc123.ngrok.io -> http://localhost:8501
```

**分享这个链接**：https://abc123.ngrok.io

---

## 📞 需要帮助？

如果您需要：
- GitHub仓库设置帮助
- Streamlit Cloud部署指导  
- 数据隐私处理建议
- 自定义域名配置

请告诉我您的具体需求！ 