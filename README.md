# 🐂🐻 BullBear Arena

> Multi-Agent AI System for US Stock Analysis with Adversarial Voting Mechanism

一个基于对抗投票机制的美股分析多智能体系统,由资深数据科学家打造,专为金融机构量化分析需求设计。

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DeepSeek](https://img.shields.io/badge/Powered%20by-DeepSeek-orange.svg)](https://deepseek.com)

## 🎯 核心特性

- **🤖 多Agent协同**: 4个专业分析Agent从不同维度评估股票
  - 📊 基本面Agent - 10-K/10-Q财报深度挖掘
  - 📈 技术面Agent - 量化指标与趋势分析
  - 💬 情绪面Agent - 新闻与社交媒体情感
  - ⚠️  风险面Agent - 波动率与风险量化

- **⚔️ 对抗投票机制**: Arena Judge裁判Agent通过加权投票产生最终决策
- **🎓 企业级架构**: 基于千万级调用量生产经验设计
- **🔌 灵活扩展**: 模块化设计,易于添加新Agent或数据源

## 🚀 快速开始

### 安装
```bash
git clone https://github.com/your-username/BullBear-Arena.git
cd BullBear-Arena
pip install -r requirements.txt
```

### 配置API密钥
```bash
cp config/api_keys.yaml.example config/api_keys.yaml
# 编辑 api_keys.yaml 填入你的 DeepSeek API Key
```

### 单股票分析
```python
from bullbear_arena import Arena

# 初始化竞技场
arena = Arena(api_key="your-deepseek-api-key")

# 分析单只股票
result = arena.analyze("AAPL")

print(f"投资建议: {result.recommendation}")
print(f"综合评分: {result.final_score}/100")
print(f"置信度: {result.confidence:.1%}")
```

## 🏗️ 系统架构
```
                    ┌─────────────────┐
                    │  Arena Judge    │
                    │  (裁判Agent)     │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Fundamental  │    │  Technical   │    │  Sentiment   │
│    Agent     │    │    Agent     │    │    Agent     │
└──────────────┘    └──────────────┘    └──────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ▼
                    ┌──────────────┐
                    │  Risk Agent  │
                    └──────────────┘
```

## 📊 Agent详细说明

### 基本面Agent (Fundamental Agent)
- **数据源**: yfinance (10-K, 10-Q财报)
- **分析维度**:
  - 财务健康度 (ROE, ROA, 负债率)
  - 现金流质量 (FCF, 转化率)
  - 运营效率 (周转率, 利润率)
- **输出**: 0-100评分 + BUY/HOLD/SELL建议

### 技术面Agent (Technical Agent)
- **指标体系**:
  - 趋势: MA(5,10,20,50,200), MACD
  - 动量: RSI, KDJ
  - 波动: Bollinger Bands, ATR
- **输出**: 技术评分 + 支撑/阻力位

### 情绪面Agent (Sentiment Agent)
- **数据源**: 新闻API, 社交媒体
- **技术**: NLP情感分析
- **输出**: -1到1情感评分

### 风险Agent (Risk Agent)
- **指标**:
  - 波动率 (历史/隐含)
  - VaR (Value at Risk)
  - Beta系数
- **输出**: 风险等级 + 风险调整后评分

## 📈 使用案例

### 1. 批量股票筛选
```python
from bullbear_arena import Arena

arena = Arena(api_key="your-key")

# 分析多只股票
watchlist = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]
results = arena.batch_analyze(watchlist)

# 按评分排序
top_picks = sorted(results, key=lambda x: x.final_score, reverse=True)

for stock in top_picks[:3]:
    print(f"{stock.ticker}: {stock.recommendation} ({stock.final_score:.1f})")
```

### 2. 定制化Agent权重
```python
# 更看重基本面
arena = Arena(
    api_key="your-key",
    weights={
        "fundamental": 0.4,
        "technical": 0.2,
        "sentiment": 0.2,
        "risk": 0.2
    }
)
```

## 🛠️ 开发计划

- [x] 基本面Agent
- [ ] 技术分析Agent
- [ ] 情绪分析Agent
- [ ] 风险评估Agent
- [ ] Arena Judge投票机制
- [ ] 回测系统
- [ ] Web可视化界面

## 📝 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 👨‍💻 作者

**湘影Flora** - 数据科学家 | CQF
- 5年Python量化经验
- 领导团队开发千万级调用智能体平台

## 🤝 贡献

欢迎提交Issue和Pull Request!

## ⭐ Star History

如果这个项目对你有帮助,请给个Star支持一下!
