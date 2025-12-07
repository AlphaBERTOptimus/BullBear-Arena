# ============================================================================
# BullBear Arena - 风险分析Agent
# bullbear_arena/agents/risk_agent.py
# ============================================================================
"""
风险分析Agent - ⚠️ Risk Guardian

专注于:
- 波动率分析 (历史波动率、波动率趋势)
- 风险价值VaR (95%/99% VaR, 最大回撤)
- 市场风险 (Beta系数、系统性风险)
- 风险调整收益 (夏普比率、索提诺比率)
- 风险预警系统

核心决策逻辑:
1. EXTREME风险等级 → 强烈SELL
2. 最大回撤>30% → SELL
3. 夏普比率<0 → SELL
4. 夏普比率>2.0 + 低风险 → BUY
5. 多维度风险评估综合决策

输出标准格式供Arena Judge裁判使用
"""

import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from scipy import stats

# ============================================================================
# 数据模型定义
# ============================================================================

class VolatilityAnalysis(BaseModel):
    """波动率分析"""
    historical_volatility: float = Field(description="历史波动率 (年化%)")
    volatility_percentile: float = Field(description="波动率百分位 0-100")
    volatility_trend: str = Field(description="波动率趋势: INCREASING/STABLE/DECREASING")
    volatility_level: str = Field(description="波动率水平: EXTREME/HIGH/MODERATE/LOW")
    recent_volatility: float = Field(description="近期波动率 (30天)")
    
class ValueAtRisk(BaseModel):
    """风险价值VaR"""
    var_95: float = Field(description="95%置信度VaR (%)")
    var_99: float = Field(description="99%置信度VaR (%)")
    expected_shortfall: float = Field(description="预期损失ES/CVaR (%)")
    max_drawdown: float = Field(description="最大回撤 (%)")
    max_drawdown_duration: int = Field(description="最大回撤持续天数")
    
class MarketRisk(BaseModel):
    """市场风险"""
    beta: float = Field(description="Beta系数 (相对市场)")
    correlation_with_market: float = Field(description="与市场相关性")
    systematic_risk: float = Field(description="系统性风险占比 (%)")
    unsystematic_risk: float = Field(description="非系统性风险占比 (%)")
    market_sensitivity: str = Field(description="市场敏感度: VERY_HIGH/HIGH/MODERATE/LOW")
    
class RiskAdjustedReturns(BaseModel):
    """风险调整收益"""
    sharpe_ratio: float = Field(description="夏普比率")
    sortino_ratio: float = Field(description="索提诺比率")
    calmar_ratio: float = Field(description="卡玛比率")
    risk_return_grade: str = Field(description="风险收益评级: A+/A/B/C/D")
    return_volatility_ratio: float = Field(description="收益波动率比")
    
class RiskWarnings(BaseModel):
    """风险预警"""
    extreme_volatility_alert: bool = Field(description="极端波动预警")
    high_drawdown_alert: bool = Field(description="高回撤预警")
    market_crash_risk: bool = Field(description="市场崩盘风险")
    liquidity_risk: bool = Field(description="流动性风险")
    concentration_risk: bool = Field(description="集中度风险")
    risk_level: str = Field(description="综合风险等级: EXTREME/HIGH/MODERATE/LOW")

class RiskAnalysisResult(BaseModel):
    """风险分析结果 - 标准输出格式"""
    agent_name: str = "⚠️ Risk Guardian"
    ticker: str
    analysis_date: str
    score: float = Field(description="综合评分 0-100 (越高越安全)", ge=0, le=100)
    recommendation: str = Field(description="投资建议: BUY/HOLD/SELL")
    confidence: float = Field(description="置信度 0-1", ge=0, le=1)
    volatility_analysis: VolatilityAnalysis
    value_at_risk: ValueAtRisk
    market_risk: MarketRisk
    risk_adjusted_returns: RiskAdjustedReturns
    risk_warnings: RiskWarnings
    key_risks: List[str]
    mitigation_strategies: List[str]
    analysis_summary: str

# ============================================================================
# 风险分析Agent类
# ============================================================================

class RiskAgent:
    """
    风险分析智能体 - BullBear Arena
    
    角色: ⚠️ Risk Guardian (风险守护者)
    职责: 量化评估投资风险,提供风险预警和对冲建议
    
    核心分析:
    1. 波动率分析 (历史波动率、波动率趋势)
    2. VaR风险价值 (95%/99% VaR, 最大回撤)
    3. 市场风险 (Beta系数、系统性风险)
    4. 风险调整收益 (夏普比率、索提诺比率)
    5. 风险预警系统
    
    决策逻辑:
    1. EXTREME风险等级 → 强烈SELL
    2. 最大回撤>30% → SELL
    3. 市场崩盘风险 → SELL
    4. 夏普比率<0 → SELL
    5. 夏普比率>2.0 + 低风险 → BUY
    """
    
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com/v1/chat/completions"):
        """
        初始化风险分析Agent
        
        Args:
            api_key: DeepSeek API密钥
            api_url: API端点
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = "deepseek-chat"
        self.agent_name = "⚠️ Risk Guardian"
        self.agent_type = "risk"
        self.risk_free_rate = 0.045  # 无风险利率 4.5%
    
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
    
    def fetch_price_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """获取价格数据"""
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"无法获取 {ticker} 的价格数据")
        
        return df
    
    def fetch_market_data(self, period: str = "1y") -> pd.DataFrame:
        """获取市场基准数据 (S&P 500)"""
        spy = yf.Ticker("SPY")
        return spy.history(period=period)
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """计算日收益率"""
        return df['Close'].pct_change().dropna()
    
    def analyze_volatility(self, df: pd.DataFrame) -> VolatilityAnalysis:
        """分析波动率"""
        returns = self.calculate_returns(df)
        
        # 历史波动率 (年化)
        historical_vol = returns.std() * np.sqrt(252) * 100
        
        # 近期波动率 (30天)
        recent_returns = returns.tail(30)
        recent_vol = recent_returns.std() * np.sqrt(252) * 100 if len(recent_returns) > 0 else historical_vol
        
        # 波动率趋势
        if len(returns) >= 60:
            vol_first_half = returns.iloc[:len(returns)//2].std() * np.sqrt(252) * 100
            vol_second_half = returns.iloc[len(returns)//2:].std() * np.sqrt(252) * 100
            
            if vol_second_half > vol_first_half * 1.2:
                volatility_trend = "INCREASING"
            elif vol_second_half < vol_first_half * 0.8:
                volatility_trend = "DECREASING"
            else:
                volatility_trend = "STABLE"
        else:
            volatility_trend = "STABLE"
        
        # 波动率水平判断
        if historical_vol > 60:
            volatility_level = "EXTREME"
        elif historical_vol > 40:
            volatility_level = "HIGH"
        elif historical_vol > 20:
            volatility_level = "MODERATE"
        else:
            volatility_level = "LOW"
        
        # 波动率百分位
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100
        volatility_percentile = stats.percentileofscore(rolling_vol.dropna(), historical_vol)
        
        return VolatilityAnalysis(
            historical_volatility=float(historical_vol),
            volatility_percentile=float(volatility_percentile),
            volatility_trend=volatility_trend,
            volatility_level=volatility_level,
            recent_volatility=float(recent_vol)
        )
    
    def calculate_var(self, df: pd.DataFrame) -> ValueAtRisk:
        """计算风险价值VaR和最大回撤"""
        returns = self.calculate_returns(df)
        
        # VaR计算 (历史模拟法)
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        
        # 预期损失 ES/CVaR
        returns_beyond_var = returns[returns <= np.percentile(returns, 5)]
        expected_shortfall = returns_beyond_var.mean() * 100 if len(returns_beyond_var) > 0 else var_95
        
        # 最大回撤计算
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        
        max_drawdown = drawdown.min()
        
        # 最大回撤持续时间
        if max_drawdown < 0:
            max_dd_idx = drawdown.idxmin()
            recovery_idx = drawdown[drawdown.index > max_dd_idx].ge(0).idxmax() if any(drawdown[drawdown.index > max_dd_idx] >= 0) else drawdown.index[-1]
            max_drawdown_duration = (recovery_idx - max_dd_idx).days if hasattr((recovery_idx - max_dd_idx), 'days') else 0
        else:
            max_drawdown_duration = 0
        
        return ValueAtRisk(
            var_95=float(var_95),
            var_99=float(var_99),
            expected_shortfall=float(expected_shortfall),
            max_drawdown=float(max_drawdown),
            max_drawdown_duration=int(max_drawdown_duration)
        )
    
    def analyze_market_risk(self, df: pd.DataFrame, market_df: pd.DataFrame) -> MarketRisk:
        """分析市场风险 (Beta系数)"""
        returns = self.calculate_returns(df)
        market_returns = self.calculate_returns(market_df)
        
        # 对齐日期
        aligned_data = pd.DataFrame({
            'stock': returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned_data) < 30:
            return MarketRisk(
                beta=1.0,
                correlation_with_market=0.5,
                systematic_risk=50.0,
                unsystematic_risk=50.0,
                market_sensitivity="MODERATE"
            )
        
        # 计算Beta
        covariance = aligned_data['stock'].cov(aligned_data['market'])
        market_variance = aligned_data['market'].var()
        beta = covariance / market_variance if market_variance > 0 else 1.0
        
        # 相关系数
        correlation = aligned_data['stock'].corr(aligned_data['market'])
        
        # R平方 (系统性风险占比)
        r_squared = correlation ** 2 * 100
        systematic_risk = r_squared
        unsystematic_risk = 100 - systematic_risk
        
        # 市场敏感度
        if abs(beta) > 1.5:
            market_sensitivity = "VERY_HIGH"
        elif abs(beta) > 1.2:
            market_sensitivity = "HIGH"
        elif abs(beta) > 0.8:
            market_sensitivity = "MODERATE"
        else:
            market_sensitivity = "LOW"
        
        return MarketRisk(
            beta=float(beta),
            correlation_with_market=float(correlation),
            systematic_risk=float(systematic_risk),
            unsystematic_risk=float(unsystematic_risk),
            market_sensitivity=market_sensitivity
        )
    
    def calculate_risk_adjusted_returns(
        self, 
        df: pd.DataFrame, 
        volatility: float,
        max_drawdown: float
    ) -> RiskAdjustedReturns:
        """计算风险调整收益"""
        returns = self.calculate_returns(df)
        
        # 年化收益率
        total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1)
        days = len(df)
        annual_return = (1 + total_return) ** (252 / days) - 1
        annual_return_pct = annual_return * 100
        
        # 夏普比率
        excess_return = annual_return - self.risk_free_rate
        sharpe_ratio = excess_return / (volatility / 100) if volatility > 0 else 0
        
        # 索提诺比率 (只考虑下行波动)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else volatility
        sortino_ratio = excess_return / (downside_std / 100) if downside_std > 0 else 0
        
        # 卡玛比率
        calmar_ratio = annual_return / abs(max_drawdown / 100) if max_drawdown < 0 else 0
        
        # 收益波动率比
        return_volatility_ratio = annual_return_pct / volatility if volatility > 0 else 0
        
        # 风险收益评级
        if sharpe_ratio > 2.0:
            risk_return_grade = "A+"
        elif sharpe_ratio > 1.5:
            risk_return_grade = "A"
        elif sharpe_ratio > 1.0:
            risk_return_grade = "B"
        elif sharpe_ratio > 0.5:
            risk_return_grade = "C"
        else:
            risk_return_grade = "D"
        
        return RiskAdjustedReturns(
            sharpe_ratio=float(sharpe_ratio),
            sortino_ratio=float(sortino_ratio),
            calmar_ratio=float(calmar_ratio),
            risk_return_grade=risk_return_grade,
            return_volatility_ratio=float(return_volatility_ratio)
        )
    
    def generate_risk_warnings(
        self,
        volatility: VolatilityAnalysis,
        var: ValueAtRisk,
        market_risk: MarketRisk,
        risk_adjusted: RiskAdjustedReturns
    ) -> RiskWarnings:
        """生成风险预警"""
        warnings_count = 0
        
        # 极端波动预警
        extreme_volatility_alert = volatility.volatility_level in ["EXTREME", "HIGH"]
        if extreme_volatility_alert:
            warnings_count += 2
        
        # 高回撤预警
        high_drawdown_alert = var.max_drawdown < -20
        if high_drawdown_alert:
            warnings_count += 2
        
        # 市场崩盘风险
        market_crash_risk = var.var_99 < -10
        if market_crash_risk:
            warnings_count += 3
        
        # 流动性风险
        liquidity_risk = False
        
        # 集中度风险
        concentration_risk = abs(market_risk.beta) > 1.5
        if concentration_risk:
            warnings_count += 1
        
        # 综合风险等级
        if warnings_count >= 5:
            risk_level = "EXTREME"
        elif warnings_count >= 3:
            risk_level = "HIGH"
        elif warnings_count >= 1:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        return RiskWarnings(
            extreme_volatility_alert=extreme_volatility_alert,
            high_drawdown_alert=high_drawdown_alert,
            market_crash_risk=market_crash_risk,
            liquidity_risk=liquidity_risk,
            concentration_risk=concentration_risk,
            risk_level=risk_level
        )
    
    def make_risk_based_decision(
        self,
        volatility: VolatilityAnalysis,
        var: ValueAtRisk,
        market_risk: MarketRisk,
        risk_adjusted: RiskAdjustedReturns,
        risk_warnings: RiskWarnings
    ) -> Tuple[str, float, float]:
        """
        基于风险的投资建议
        
        Returns:
            (recommendation, confidence, score)
        """
        sell_signals = 0
        hold_signals = 0
        buy_signals = 0
        
        # 规则1: 极端风险 → SELL
        if risk_warnings.risk_level == "EXTREME":
            sell_signals += 5
        elif risk_warnings.risk_level == "HIGH":
            sell_signals += 3
        
        # 规则2: 高回撤 → SELL
        if var.max_drawdown < -30:
            sell_signals += 3
        elif var.max_drawdown < -20:
            sell_signals += 2
        
        # 规则3: 市场崩盘风险 → SELL
        if risk_warnings.market_crash_risk:
            sell_signals += 3
        
        # 规则4: 夏普比率低 → SELL/HOLD
        if risk_adjusted.sharpe_ratio < 0:
            sell_signals += 2
        elif risk_adjusted.sharpe_ratio < 0.5:
            hold_signals += 1
        
        # 规则5: 夏普比率高 → BUY
        if risk_adjusted.sharpe_ratio > 2.0:
            buy_signals += 3
        elif risk_adjusted.sharpe_ratio > 1.5:
            buy_signals += 2
        elif risk_adjusted.sharpe_ratio > 1.0:
            buy_signals += 1
        
        # 规则6: 低风险 + 好收益 → BUY
        if risk_warnings.risk_level == "LOW" and risk_adjusted.sharpe_ratio > 1.0:
            buy_signals += 2
        
        # 最终决策
        if sell_signals >= 4:
            recommendation = "SELL"
            confidence = min(0.9, 0.6 + sell_signals * 0.05)
        elif buy_signals >= 3 and sell_signals == 0:
            recommendation = "BUY"
            confidence = min(0.9, 0.6 + buy_signals * 0.05)
        elif sell_signals > buy_signals:
            recommendation = "SELL"
            confidence = 0.7
        elif buy_signals > sell_signals:
            recommendation = "BUY"
            confidence = 0.7
        else:
            recommendation = "HOLD"
            confidence = 0.6
        
        # 评分 (0-100, 越高越安全)
        base_score = 50
        base_score -= (volatility.historical_volatility / 2)
        base_score -= (abs(var.max_drawdown) / 2)
        base_score += risk_adjusted.sharpe_ratio * 10
        
        if risk_warnings.risk_level == "EXTREME":
            base_score -= 30
        elif risk_warnings.risk_level == "HIGH":
            base_score -= 20
        elif risk_warnings.risk_level == "MODERATE":
            base_score -= 10
        
        score = max(0, min(100, base_score))
        
        return recommendation, confidence, score
    
    def generate_ai_analysis(self, ticker: str, metrics: Dict, decision: Dict) -> Dict:
        """使用AI生成深度分析"""
        prompt = f"""你是一位资深的风险管理专家。基于以下风险数据对 {ticker} 进行深度分析:

【风险引擎决策】
建议: {decision['recommendation']}
置信度: {decision['confidence']:.1%}
评分: {decision['score']:.1f}/100

【详细数据】
波动率: {json.dumps(metrics['volatility'], indent=2, ensure_ascii=False)}
VaR: {json.dumps(metrics['var'], indent=2, ensure_ascii=False)}
市场风险: {json.dumps(metrics['market_risk'], indent=2, ensure_ascii=False)}
风险调整收益: {json.dumps(metrics['risk_adjusted'], indent=2, ensure_ascii=False)}
风险预警: {json.dumps(metrics['risk_warnings'], indent=2, ensure_ascii=False)}

请提供:
1. 确认评分 (0-100)
2. 投资建议 (BUY/HOLD/SELL)
3. 置信度 (0-1)
4. 3-5个关键风险点
5. 3-5个风险缓解策略
6. 200字风险分析总结

JSON格式:
{{
  "score": 65.5,
  "recommendation": "HOLD",
  "confidence": 0.75,
  "risks": ["风险1", ...],
  "strategies": ["策略1", ...],
  "summary": "分析总结..."
}}
"""
        
        try:
            response_text = self.call_deepseek_api(prompt)
            if response_text:
                return json.loads(response_text)
        except:
            pass
        
        return {
            "score": decision['score'],
            "recommendation": decision['recommendation'],
            "confidence": decision['confidence'],
            "risks": ["风险引擎主导决策"],
            "strategies": ["建议人工复核"],
            "summary": f"风险等级: {metrics['risk_warnings']['risk_level']}"
        }
    
    def analyze(self, ticker: str, period: str = "1y", verbose: bool = False) -> RiskAnalysisResult:
        """
        执行完整的风险分析
        
        Args:
            ticker: 股票代码
            period: 分析周期
            verbose: 是否打印详细过程
            
        Returns:
            RiskAnalysisResult: 标准化的分析结果
        """
        if verbose:
            print(f"[{self.agent_name}] 开始分析 {ticker}...")
        
        # 1. 获取数据
        df = self.fetch_price_data(ticker, period)
        market_df = self.fetch_market_data(period)
        
        # 2. 各项风险分析
        volatility_analysis = self.analyze_volatility(df)
        value_at_risk = self.calculate_var(df)
        market_risk = self.analyze_market_risk(df, market_df)
        risk_adjusted_returns = self.calculate_risk_adjusted_returns(
            df, 
            volatility_analysis.historical_volatility,
            value_at_risk.max_drawdown
        )
        risk_warnings = self.generate_risk_warnings(
            volatility_analysis, value_at_risk, market_risk, risk_adjusted_returns
        )
        
        # 3. 风险决策
        recommendation, confidence, score = self.make_risk_based_decision(
            volatility_analysis, value_at_risk, market_risk, risk_adjusted_returns, risk_warnings
        )
        
        # 4. 准备AI分析数据
        metrics_for_ai = {
            "volatility": volatility_analysis.model_dump(),
            "var": value_at_risk.model_dump(),
            "market_risk": market_risk.model_dump(),
            "risk_adjusted": risk_adjusted_returns.model_dump(),
            "risk_warnings": risk_warnings.model_dump()
        }
        
        decision = {
            "recommendation": recommendation,
            "confidence": confidence,
            "score": score
        }
        
        # 5. AI深度分析
        ai_analysis = self.generate_ai_analysis(ticker, metrics_for_ai, decision)
        
        # 6. 组装最终结果
        result = RiskAnalysisResult(
            agent_name=self.agent_name,
            ticker=ticker,
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            score=ai_analysis["score"],
            recommendation=ai_analysis["recommendation"],
            confidence=ai_analysis["confidence"],
            volatility_analysis=volatility_analysis,
            value_at_risk=value_at_risk,
            market_risk=market_risk,
            risk_adjusted_returns=risk_adjusted_returns,
            risk_warnings=risk_warnings,
            key_risks=ai_analysis["risks"],
            mitigation_strategies=ai_analysis["strategies"],
            analysis_summary=ai_analysis["summary"]
        )
        
        if verbose:
            print(f"[{self.agent_name}] 分析完成: {result.recommendation} (评分: {result.score:.1f})")
        
        return result
    
    def get_arena_output(self, ticker: str, period: str = "1y") -> Dict:
        """
        为Arena Judge提供标准化输出
        
        这是提供给最终裁判Agent的接口
        
        Returns:
            Dict: 竞技场标准格式,包含所有必要的投票信息
        """
        result = self.analyze(ticker, period, verbose=False)
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "ticker": result.ticker,
            "score": result.score,
            "recommendation": result.recommendation,
            "confidence": result.confidence,
            "vote_weight": 1.0,
            "summary": result.analysis_summary,
            "key_points": {
                "risks": result.key_risks,
                "strategies": result.mitigation_strategies
            },
            "detailed_metrics": {
                "volatility": result.volatility_analysis.model_dump(),
                "var": result.value_at_risk.model_dump(),
                "market_risk": result.market_risk.model_dump(),
                "risk_adjusted": result.risk_adjusted_returns.model_dump(),
                "risk_warnings": result.risk_warnings.model_dump()
            }
        }
