# 🚀 BullBear Arena 配置指南

## 📋 前置要求

- Python >= 3.8
- pip 包管理器
- DeepSeek API Key

---

## 🔐 API Key配置 (三种方式)

### 方式1: 使用 .env 文件 (⭐推荐 - 本地开发)

#### 步骤:

**1. 复制示例文件**
```bash
cp .env.example .env
```

**2. 编辑 `.env` 文件**
```bash
nano .env
# 或使用任何文本编辑器
```

**3. 填入你的API Key**
```bash
DEEPSEEK_API_KEY=sk-你的真实API-Key
DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
```

**4. 保存并运行**
```bash
streamlit run streamlit_app.py
```

✅ 系统会自动从 `.env` 加载API Key

---

### 方式2: Streamlit Secrets (⭐推荐 - 云端部署)

#### 本地测试:

**1. 创建配置目录**
```bash
mkdir -p .streamlit
```

**2. 创建 secrets.toml**
```bash
nano .streamlit/secrets.toml
```

**3. 添加配置**
```toml
[api]
deepseek_key = "sk-你的真实API-Key"
deepseek_url = "https://api.deepseek.com/v1/chat/completions"
```

#### 云端部署 (Streamlit Cloud):

**1. 推送代码到GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

**2. 访问 [Streamlit Cloud](https://streamlit.io/cloud)**

**3. 连接GitHub仓库并部署**

**4. 在App Settings → Secrets 中添加:**
```toml
[api]
deepseek_key = "sk-你的真实API-Key"
```

✅ 云端自动使用配置的Key,无需手动输入

---

### 方式3: 手动输入 (临时使用)

**1. 启动应用**
```bash
streamlit run streamlit_app.py
```

**2. 在侧边栏输入API Key**

⚠️ 仅当前会话有效,关闭后需重新输入

---

## 🎯 获取DeepSeek API Key

**1. 访问:** https://platform.deepseek.com

**2. 注册/登录账号**

**3. 进入API Keys页面**

**4. 创建新的API Key**

**5. 复制Key** (格式: `sk-xxxxxxxxxx`)

⚠️ **安全提示:**
- API Key相当于密码,请妥善保管
- 不要将Key提交到公开仓库
- 定期轮换Key
- 限制Key的使用权限

---

## 📦 完整安装流程
```bash
# 1. 克隆项目
git clone https://github.com/your-repo/BullBear-Arena.git
cd BullBear-Arena

# 2. 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置API Key (选择一种方式)
cp .env.example .env
nano .env  # 填入API Key

# 5. 运行应用
streamlit run streamlit_app.py
```

---

## ✅ 验证配置

启动应用后,检查:

- ✅ 侧边栏显示 "API Key已自动加载"
- ✅ 显示Key来源 (环境变量/.env)
- ✅ 显示部分Key用于确认

---

## ❓ 常见问题

### Q: 为什么我的.env文件不生效?

**A:** 确保:
1. 文件名正确: `.env` (不是 `env.txt` 或其他)
2. 文件在项目根目录
3. 已安装 `python-dotenv`
4. 重启了应用

### Q: 如何检查.env是否存在?
```bash
ls -la .env
```

### Q: 可以看到.env的内容吗?
```bash
cat .env
```

⚠️ 不要分享输出内容!

### Q: .env会被上传到GitHub吗?

**A:** 不会! `.env` 已在 `.gitignore` 中,Git会自动忽略

### Q: 如何验证API Key是否有效?

**A:** 运行应用,尝试进行一次分析,如果成功则Key有效

---

## 🔒 安全最佳实践

### ✅ 做:
- 使用 `.env` 文件本地开发
- 使用 Streamlit Secrets 云端部署
- 定期轮换API Key
- 限制Key的使用范围

### ❌ 不要:
- 将 `.env` 提交到Git
- 在代码中硬编码API Key
- 公开分享包含Key的截图
- 在公共场合展示完整Key

---

## 📞 获取帮助

遇到问题?

- 📖 查看 [README.md](README.md)
- 🐛 提交 [Issue](https://github.com/your-repo/BullBear-Arena/issues)
- 💬 加入讨论 [Discussions](https://github.com/your-repo/BullBear-Arena/discussions)
```

---

## ✅ 完整文件清单

现在你有了:
```
BullBear-Arena/
├── streamlit_app.py           ✅ (完整的,支持多种API Key配置)
├── .env.example               ✅ (API Key配置示例)
├── requirements.txt           ✅ (更新了python-dotenv)
├── SETUP.md                   ✅ (详细配置指南)
├── README.md                  ✅ (专业项目文档)
└── .gitignore                 ✅ (已包含.env)
