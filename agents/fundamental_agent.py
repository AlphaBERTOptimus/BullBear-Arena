# ============================================================================
# 基本面分析Agent - 生产版本 (供裁判Agent调用)
# fundamental_agent.py
# ============================================================================

import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from pydantic import BaseModel, Field

# ============================================================================
# 数据模型定义
# ============================================================================

class FinancialMetrics(BaseModel):
    """财务指标"""
    revenue: float = Field(description="总收入")
    revenue_growth: float = Field(description="收入增长率 (%)")
    gross_profit_margin: float = Field(description="毛利率 (%)")
    operating_margin: float = Field(description="营业利润率 (%)")
    net_profit_margin: float = Field(description="净利润率 (%)")
    roe: float = Field(description="净资产收益率 (%)")
    roa: float = Field(description="总资产收益率 (%)")
    current_ratio: float = Field(description="流动比率")
    quick_ratio: float = Field(description="速动比率")
    debt_to_equity: float = Field(description="负债权益比")
    interest_coverage: float = Field(description="利息保障倍数")
    
class CashFlowAnalysis(BaseModel):
    """现金流分析"""
    operating_cash_flow: float = Field(description="经营活动现金流")
    investing_cash_flow: float = Field(description="投资活动现金流")
    financing_cash_flow: float = Field(description="融资活动现金流")
    free_cash_flow: float = Field(description="自由现金流")
    fcf_growth: float = Field(description="自由现金流增长率 (%)")
    cash_conversion_rate: float = Field(description="现金转化率 (%)")
    cash_health_score: str = Field(description="现金健康度评级")
    
class OperationalMetrics(BaseModel):
    """运营指标"""
    asset_turnover: float = Field(description="资产周转率")
    inventory_turnover: float = Field(description="存货周转率")
    receivables_turnover: float = Field(description="应收账款周转率")
    working_capital: float = Field(description="营运资本")
    operational_efficiency_score: float = Field(description="运营效率评分 0-100")

class FundamentalAnalysisResult(BaseModel):
    """基本面分析结果 - 标准输出格式"""
    agent_name: str = "Fundamental Analysis Agent"
    ticker: str
    analysis_date: str
    score: float = Field(description="综合评分 0-100", ge=0, le=100)
    recommendation: str = Field(description="投资建议: BUY/HOLD/SELL")
    confidence: float = Field(description="置信度 0-1", ge=0, le=1)
    financial_metrics: FinancialMetrics
    cash_flow_analysis: CashFlowAnalysis
    operational_metrics: OperationalMetrics
    key_strengths: List[str]
    key_risks: List[str]
    analysis_summary: str

# ============================================================================
# 基本面分析Agent类
# ============================================================================

class FundamentalAgent:
    """
    基本面分析智能体
    
    专注于:
    - 10-K/10-Q财务报表分析
    - 现金流健康度评估
    - 盈利能力与运营效率分析
    """
    
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com/v1/chat/completions"):
        self.api_key = api_key
        self.api_url = api_url
        self.model = "deepseek-chat"
        self.agent_name = "Fundamental Analysis Agent"
    
    def call_deepseek_api(self, prompt: str) -> str:
        """调用DeepSeek API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def fetch_financial_data(self, ticker: str) -> Dict[str, Any]:
        """获取财务数据"""
        stock = yf.Ticker(ticker)
        
        return {
            "income_statement": stock.income_stmt,
            "balance_sheet": stock.balance_sheet,
            "cash_flow": stock.cashflow,
            "info": stock.info,
            "quarterly_income": stock.quarterly_income_stmt,
            "quarterly_balance": stock.quarterly_balance_sheet,
            "quarterly_cashflow": stock.quarterly_cashflow
        }
    
    def calculate_financial_metrics(self, data: Dict) -> FinancialMetrics:
        """计算财务指标"""
        try:
            income = data["income_statement"]
            balance = data["balance_sheet"]
            info = data["info"]
            
            latest_income = income.iloc[:, 0] if not income.empty else pd.Series()
            prev_income = income.iloc[:, 1] if income.shape[1] > 1 else pd.Series()
            latest_balance = balance.iloc[:, 0] if not balance.empty else pd.Series()
            
            revenue = latest_income.get('Total Revenue', 0)
            prev_revenue = prev_income.get('Total Revenue', revenue)
            revenue_growth = ((revenue - prev_revenue) / prev_revenue * 100) if prev_revenue else 0
            
            gross_profit = latest_income.get('Gross Profit', 0)
            operating_income = latest_income.get('Operating Income', 0)
            net_income = latest_income.get('Net Income', 0)
            
            total_assets = latest_balance.get('Total Assets', 1)
            total_equity = latest_balance.get('Stockholders Equity', 1)
            current_assets = latest_balance.get('Current Assets', 0)
            current_liabilities = latest_balance.get('Current Liabilities', 1)
            inventory = latest_balance.get('Inventory', 0)
            total_debt = latest_balance.get('Total Debt', 0)
            
            return FinancialMetrics(
                revenue=float(revenue),
                revenue_growth=float(revenue_growth),
                gross_profit_margin=float(gross_profit / revenue * 100) if revenue else 0,
                operating_margin=float(operating_income / revenue * 100) if revenue else 0,
                net_profit_margin=float(net_income / revenue * 100) if revenue else 0,
                roe=float(net_income / total_equity * 100) if total_equity else 0,
                roa=float(net_income / total_assets * 100) if total_assets else 0,
                current_ratio=float(current_assets / current_liabilities) if current_liabilities else 0,
                quick_ratio=float((current_assets - inventory) / current_liabilities) if current_liabilities else 0,
                debt_to_equity=float(total_debt / total_equity) if total_equity else 0,
                interest_coverage=float(info.get('interestCoverage', 0))
            )
        except Exception as e:
            return FinancialMetrics(
                revenue=0, revenue_growth=0, gross_profit_margin=0,
                operating_margin=0, net_profit_margin=0, roe=0,
                roa=0, current_ratio=0, quick_ratio=0,
                debt_to_equity=0, interest_coverage=0
            )
    
    def analyze_cash_flow(self, data: Dict) -> CashFlowAnalysis:
        """分析现金流"""
        try:
            cash_flow = data["cash_flow"]
            income = data["income_statement"]
            
            latest_cf = cash_flow.iloc[:, 0] if not cash_flow.empty else pd.Series()
            prev_cf = cash_flow.iloc[:, 1] if cash_flow.shape[1] > 1 else pd.Series()
            latest_income = income.iloc[:, 0] if not income.empty else pd.Series()
            
            operating_cf = latest_cf.get('Operating Cash Flow', 0)
            investing_cf = latest_cf.get('Investing Cash Flow', 0)
            financing_cf = latest_cf.get('Financing Cash Flow', 0)
            capex = abs(latest_cf.get('Capital Expenditure', 0))
            
            fcf = operating_cf - capex
            prev_fcf = prev_cf.get('Free Cash Flow', fcf) if not prev_cf.empty else fcf
            fcf_growth = ((fcf - prev_fcf) / abs(prev_fcf) * 100) if prev_fcf else 0
            
            net_income = latest_income.get('Net Income', 1)
            cash_conversion = (operating_cf / net_income * 100) if net_income else 0
            
            if fcf > 0 and operating_cf > 0 and cash_conversion > 80:
                health = "Excellent"
            elif fcf > 0 and operating_cf > 0:
                health = "Good"
            elif operating_cf > 0:
                health = "Fair"
            else:
                health = "Poor"
            
            return CashFlowAnalysis(
                operating_cash_flow=float(operating_cf),
                investing_cash_flow=float(investing_cf),
                financing_cash_flow=float(financing_cf),
                free_cash_flow=float(fcf),
                fcf_growth=float(fcf_growth),
                cash_conversion_rate=float(cash_conversion),
                cash_health_score=health
            )
        except Exception as e:
            return CashFlowAnalysis(
                operating_cash_flow=0, investing_cash_flow=0,
                financing_cash_flow=0, free_cash_flow=0,
                fcf_growth=0, cash_conversion_rate=0,
                cash_health_score="Unknown"
            )
    
    def calculate_operational_metrics(self, data: Dict) -> OperationalMetrics:
        """计算运营指标"""
        try:
            income = data["income_statement"]
            balance = data["balance_sheet"]
            
            latest_income = income.iloc[:, 0] if not income.empty else pd.Series()
            latest_balance = balance.iloc[:, 0] if not balance.empty else pd.Series()
            
            revenue = latest_income.get('Total Revenue', 1)
            cogs = latest_income.get('Cost Of Revenue', 0)
            total_assets = latest_balance.get('Total Assets', 1)
            inventory = latest_balance.get('Inventory', 1)
            receivables = latest_balance.get('Accounts Receivable', 1)
            current_assets = latest_balance.get('Current Assets', 0)
            current_liabilities = latest_balance.get('Current Liabilities', 0)
            
            asset_turnover = revenue / total_assets if total_assets else 0
            inventory_turnover = cogs / inventory if inventory else 0
            receivables_turnover = revenue / receivables if receivables else 0
            working_capital = current_assets - current_liabilities
            
            efficiency_score = min(100, (
                (asset_turnover * 20) +
                (min(inventory_turnover / 10, 1) * 30) +
                (min(receivables_turnover / 10, 1) * 30) +
                (20 if working_capital > 0 else 0)
            ))
            
            return OperationalMetrics(
                asset_turnover=float(asset_turnover),
                inventory_turnover=float(inventory_turnover),
                receivables_turnover=float(receivables_turnover),
                working_capital=float(working_capital),
                operational_efficiency_score=float(efficiency_score)
            )
        except Exception as e:
            return OperationalMetrics(
                asset_turnover=0, inventory_turnover=0,
                receivables_turnover=0, working_capital=0,
                operational_efficiency_score=0
            )
    
    def generate_ai_analysis(self, ticker: str, metrics: Dict) -> Dict:
        """使用AI生成深度分析"""
        prompt = f"""你是一位资深的股票基本面分析师。请基于以下财务数据对 {ticker} 进行深度分析:

财务指标:
{json.dumps(metrics['financial_metrics'], indent=2, ensure_ascii=False)}

现金流分析:
{json.dumps(metrics['cash_flow'], indent=2, ensure_ascii=False)}

运营指标:
{json.dumps(metrics['operational'], indent=2, ensure_ascii=False)}

请提供:
1. 综合评分 (0-100)
2. 投资建议 (BUY/HOLD/SELL)
3. 置信度 (0-1)
4. 3-5个关键优势
5. 3-5个关键风险
6. 200字左右的分析总结

以JSON格式返回,结构如下:
{{
  "score": 75.5,
  "recommendation": "BUY",
  "confidence": 0.85,
  "strengths": ["优势1", "优势2", ...],
  "risks": ["风险1", "风险2", ...],
  "summary": "分析总结..."
}}
"""
        
        try:
            response_text = self.call_deepseek_api(prompt)
            if response_text:
                result = json.loads(response_text)
                return result
            else:
                raise Exception("API返回为空")
        except Exception as e:
            return {
                "score": 50,
                "recommendation": "HOLD",
                "confidence": 0.5,
                "strengths": ["数据分析受限"],
                "risks": ["分析不完整"],
                "summary": "AI分析暂时不可用,建议人工复核。"
            }
    
    def analyze(self, ticker: str, verbose: bool = False) -> FundamentalAnalysisResult:
        """
        执行完整的基本面分析
        
        Args:
            ticker: 股票代码
            verbose: 是否打印详细过程
            
        Returns:
            FundamentalAnalysisResult: 标准化的分析结果
        """
        if verbose:
            print(f"[{self.agent_name}] 开始分析 {ticker}...")
        
        # 1. 获取数据
        financial_data = self.fetch_financial_data(ticker)
        if not financial_data:
            raise ValueError(f"无法获取 {ticker} 的财务数据")
        
        # 2. 计算各项指标
        financial_metrics = self.calculate_financial_metrics(financial_data)
        cash_flow_analysis = self.analyze_cash_flow(financial_data)
        operational_metrics = self.calculate_operational_metrics(financial_data)
        
        # 3. 准备AI分析数据
        metrics_for_ai = {
            "financial_metrics": financial_metrics.model_dump(),
            "cash_flow": cash_flow_analysis.model_dump(),
            "operational": operational_metrics.model_dump()
        }
        
        # 4. AI深度分析
        ai_analysis = self.generate_ai_analysis(ticker, metrics_for_ai)
        
        # 5. 组装最终结果
        result = FundamentalAnalysisResult(
            agent_name=self.agent_name,
            ticker=ticker,
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            score=ai_analysis["score"],
            recommendation=ai_analysis["recommendation"],
            confidence=ai_analysis["confidence"],
            financial_metrics=financial_metrics,
            cash_flow_analysis=cash_flow_analysis,
            operational_metrics=operational_metrics,
            key_strengths=ai_analysis["strengths"],
            key_risks=ai_analysis["risks"],
            analysis_summary=ai_analysis["summary"]
        )
        
        if verbose:
            print(f"[{self.agent_name}] 分析完成: {result.recommendation} (评分: {result.score:.1f})")
        
        return result
    
    def get_output_for_ensemble(self, ticker: str) -> Dict:
        """
        为集成投票Agent提供标准化输出
        
        Returns:
            Dict: 包含所有必要信息的字典
        """
        result = self.analyze(ticker, verbose=False)
        return {
            "agent_name": self.agent_name,
            "score": result.score,
            "recommendation": result.recommendation,
            "confidence": result.confidence,
            "summary": result.analysis_summary,
            "detailed_data": result.model_dump()
        }
```

---

## 二、项目命名建议 🎯

基于你的背景(银行首席技术官、200人团队、千万级调用量),我推荐以下命名:

### **推荐方案:**

#### 🏆 **1. QuantumEdge** (量子边缘)
```
寓意: 利用AI在金融市场获得量子级别的信息优势
特点: 科技感强、国际化、易记
```

#### 🚀 **2. VortexAI** (漩涡AI)
```
寓意: 多Agent系统像漩涡一样汇聚智慧
特点: 动感、富有力量、专业
```

#### 💎 **3. PrismAlpha** (棱镜阿尔法)
```
寓意: 像棱镜分解光线一样,多角度分析股票获取超额收益
特点: 优雅、专业、富有深度
```

#### ⚡ **4. NexusQuant** (关联量化)
```
寓意: 多Agent连接(Nexus)形成量化决策网络
特点: 专业、现代、技术感
```

#### 🎯 **5. ApexSignal** (顶点信号)
```
寓意: 捕捉市场最顶级的交易信号
特点: 简洁、目标明确、专业
```

#### 🌟 **6. SynergyAI** (协同AI)
```
寓意: 多Agent协同产生1+1>2的效果
特点: 强调团队合作、符合你的管理背景
```

---

### **个性化方案 (结合你的经历):**

#### 🏦 **7. TitanQuant** (泰坦量化)
```
寓意: 银行级(Titan)的量化分析能力
特点: 大气、稳重、权威
```

#### 🧠 **8. CogniTrade** (认知交易)
```
寓意: 基于AI认知能力的智能交易
特点: 学术感、创新
```

#### 🔮 **9. OracleEdge** (神谕边缘)
```
寓意: AI像神谕一样预测市场
特点: 神秘、智慧、专业
```

---

### **我的Top 3推荐:**

1. **🥇 PrismAlpha** - 优雅专业,体现多维分析
2. **🥈 QuantumEdge** - 科技前沿,符合AI特性  
3. **🥉 NexusQuant** - 强调连接,突出多Agent架构

---

## GitHub项目结构建议:
```
PrismAlpha/  (或你选择的名字)
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── agents/
│   ├── __init__.py
│   ├── fundamental_agent.py      # ✅ 已完成
│   ├── technical_agent.py         # 待开发
│   ├── sentiment_agent.py         # 待开发
│   └── risk_agent.py              # 待开发
├── ensemble/
│   └── voting_agent.py            # 最终裁判
├── utils/
│   ├── data_fetcher.py
│   └── visualization.py
├── tests/
│   └── test_agents.py
└── examples/
    └── demo.ipynb
