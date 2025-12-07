# ============================================================================
# BullBear Arena - 技术分析Agent
# bullbear_arena/agents/technical_agent.py
# ============================================================================
"""
技术分析Agent - 📈 Technical Analyst

专注于:
- 价格趋势分析 (移动平均线、ADX)
- 技术指标 (RSI, MACD, KDJ)
- 支撑阻力位计算
- 动量分析 (成交量、MFI、ROC)

输出标准格式供Arena Judge裁判使用
"""

import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# ============================================================================
# 数据模型定义
# ============================================================================

class TrendAnalysis(BaseModel):
    """趋势分析"""
    primary_trend: str = Field(description="主要趋势: BULLISH/BEARISH/SIDEWAYS")
    trend_strength: float = Field(description="趋势强度 0-100")
    ma_alignment: str = Field(description="均线排列: BULLISH/BEARISH/MIXED")
    ma_signals: Dict[str, str] = Field(description="各周期均线信号")
    price_vs_ma200: float = Field(description="价格相对MA200的位置 (%)")
    adx: float = Field(description="ADX趋势强度指标")

class TechnicalIndicators(BaseModel):
    """技术指标"""
    rsi: float = Field(description="RSI相对强弱指标 0-100")
    rsi_signal: str = Field(description="RSI信号: OVERBOUGHT/OVERSOLD/NEUTRAL")
    macd: float = Field(description="MACD值")
    macd_signal: float = Field(description="MACD信号线")
    macd_histogram: float = Field(description="MACD柱状图")
    macd_trend: str = Field(description="MACD趋势: BULLISH/BEARISH")
    stochastic_k: float = Field(description="随机指标K值")
    stochastic_d: float = Field(description="随机指标D值")
    stochastic_signal: str = Field(description="KDJ信号")

class SupportResistance(BaseModel):
    """支撑阻力"""
    current_price: float = Field(description="当前价格")
    resistance_1: float = Field(description="第一阻力位")
    resistance_2: float = Field(description="第二阻力位")
    support_1: float = Field(description="第一支撑位")
    support_2: float = Field(description="第二支撑位")
    distance_to_resistance: float = Field(description="距离阻力位 (%)")
    distance_to_support: float = Field(description="距离支撑位 (%)")
    pivot_point: float = Field(description="枢轴点")

class MomentumAnalysis(BaseModel):
    """动量分析"""
    momentum_score: float = Field(description="动量评分 0-100")
    volume_trend: str = Field(description="成交量趋势: INCREASING/DECREASING/STABLE")
    price_momentum: float = Field(description="价格动量 (%)")
    volume_ratio: float = Field(description="成交量比率")
    money_flow_index: float = Field(description="资金流量指标 MFI")
    rate_of_change: float = Field(description="变化率 ROC")

class TechnicalAnalysisResult(BaseModel):
    """技术分析结果 - 标准输出格式"""
    agent_name: str = "📈 Technical Analyst"
    ticker: str
    analysis_date: str
    score: float = Field(description="综合评分 0-100", ge=0, le=100)
    recommendation: str = Field(description="投资建议: BUY/HOLD/SELL")
    confidence: float = Field(description="置信度 0-1", ge=0, le=1)
    trend_analysis: TrendAnalysis
    technical_indicators: TechnicalIndicators
    support_resistance: SupportResistance
    momentum_analysis: MomentumAnalysis
    key_signals: List[str]
    key_warnings: List[str]
    analysis_summary: str

# ============================================================================
# 技术分析Agent类
# ============================================================================

class TechnicalAgent:
    """
    技术分析智能体 - BullBear Arena
    
    角色: 📈 Technical Analyst (技术分析师)
    职责: 从价格走势和技术指标角度评估交易时机
    """
    
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com/v1/chat/completions"):
        """
        初始化技术分析Agent
        
        Args:
            api_key: DeepSeek API密钥
            api_url: API端点
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = "deepseek-chat"
        self.agent_name = "📈 Technical Analyst"
        self.agent_type = "technical"
    
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
    
    def fetch_price_data(self, ticker: str, period: str = "6mo") -> pd.DataFrame:
        """获取价格数据"""
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"无法获取 {ticker} 的价格数据")
        
        return df
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> Dict:
        """计算移动平均线"""
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        current_price = df['Close'].iloc[-1]
        ma5 = df['MA5'].iloc[-1]
        ma10 = df['MA10'].iloc[-1]
        ma20 = df['MA20'].iloc[-1]
        ma50 = df['MA50'].iloc[-1]
        ma200 = df['MA200'].iloc[-1] if len(df) >= 200 else current_price
        
        # 判断均线排列
        bullish_alignment = (ma5 > ma10 > ma20 > ma50)
        bearish_alignment = (ma5 < ma10 < ma20 < ma50)
        
        if bullish_alignment:
            alignment = "BULLISH"
        elif bearish_alignment:
            alignment = "BEARISH"
        else:
            alignment = "MIXED"
        
        # 各周期信号
        signals = {
            "MA5": "BULLISH" if current_price > ma5 else "BEARISH",
            "MA10": "BULLISH" if current_price > ma10 else "BEARISH",
            "MA20": "BULLISH" if current_price > ma20 else "BEARISH",
            "MA50": "BULLISH" if current_price > ma50 else "BEARISH",
            "MA200": "BULLISH" if current_price > ma200 else "BEARISH"
        }
        
        return {
            "alignment": alignment,
            "signals": signals,
            "price_vs_ma200": ((current_price - ma200) / ma200 * 100) if ma200 else 0,
            "df": df
        }
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算ADX趋势强度指标"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # 计算+DI和-DI
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0
    
    def analyze_trend(self, df: pd.DataFrame) -> TrendAnalysis:
        """分析趋势"""
        ma_data = self.calculate_moving_averages(df)
        df = ma_data["df"]
        
        # 计算ADX
        adx = self.calculate_adx(df)
        
        # 判断主要趋势
        alignment = ma_data["alignment"]
        current_price = df['Close'].iloc[-1]
        
        # 20日内价格变化
        price_20d_ago = df['Close'].iloc[-20] if len(df) >= 20 else df['Close'].iloc[0]
        price_change = (current_price - price_20d_ago) / price_20d_ago * 100
        
        if alignment == "BULLISH" and price_change > 5:
            primary_trend = "BULLISH"
            trend_strength = min(100, adx + 20)
        elif alignment == "BEARISH" and price_change < -5:
            primary_trend = "BEARISH"
            trend_strength = min(100, adx + 20)
        else:
            primary_trend = "SIDEWAYS"
            trend_strength = adx
        
        return TrendAnalysis(
            primary_trend=primary_trend,
            trend_strength=float(trend_strength),
            ma_alignment=alignment,
            ma_signals=ma_data["signals"],
            price_vs_ma200=float(ma_data["price_vs_ma200"]),
            adx=float(adx)
        )
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算RSI指标"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
    
    def calculate_macd(self, df: pd.DataFrame) -> Dict:
        """计算MACD指标"""
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return {
            "macd": float(macd.iloc[-1]),
            "signal": float(signal.iloc[-1]),
            "histogram": float(histogram.iloc[-1])
        }
    
    def calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> Dict:
        """计算随机指标KDJ"""
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        
        k = 100 * (df['Close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=3).mean()
        
        return {
            "k": float(k.iloc[-1]) if not pd.isna(k.iloc[-1]) else 50,
            "d": float(d.iloc[-1]) if not pd.isna(d.iloc[-1]) else 50
        }
    
    def analyze_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """分析技术指标"""
        # RSI
        rsi = self.calculate_rsi(df)
        if rsi > 70:
            rsi_signal = "OVERBOUGHT"
        elif rsi < 30:
            rsi_signal = "OVERSOLD"
        else:
            rsi_signal = "NEUTRAL"
        
        # MACD
        macd_data = self.calculate_macd(df)
        macd_trend = "BULLISH" if macd_data["histogram"] > 0 else "BEARISH"
        
        # KDJ
        stoch = self.calculate_stochastic(df)
        if stoch["k"] > 80 and stoch["d"] > 80:
            stoch_signal = "OVERBOUGHT"
        elif stoch["k"] < 20 and stoch["d"] < 20:
            stoch_signal = "OVERSOLD"
        else:
            stoch_signal = "NEUTRAL"
        
        return TechnicalIndicators(
            rsi=float(rsi),
            rsi_signal=rsi_signal,
            macd=macd_data["macd"],
            macd_signal=macd_data["signal"],
            macd_histogram=macd_data["histogram"],
            macd_trend=macd_trend,
            stochastic_k=stoch["k"],
            stochastic_d=stoch["d"],
            stochastic_signal=stoch_signal
        )
    
    def calculate_support_resistance(self, df: pd.DataFrame) -> SupportResistance:
        """计算支撑阻力位"""
        current_price = df['Close'].iloc[-1]
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # 使用枢轴点方法
        pivot = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
        
        # 计算阻力位和支撑位
        resistance_1 = 2 * pivot - low.iloc[-1]
        resistance_2 = pivot + (high.iloc[-1] - low.iloc[-1])
        support_1 = 2 * pivot - high.iloc[-1]
        support_2 = pivot - (high.iloc[-1] - low.iloc[-1])
        
        # 距离百分比
        distance_to_resistance = (resistance_1 - current_price) / current_price * 100
        distance_to_support = (current_price - support_1) / current_price * 100
        
        return SupportResistance(
            current_price=float(current_price),
            resistance_1=float(resistance_1),
            resistance_2=float(resistance_2),
            support_1=float(support_1),
            support_2=float(support_2),
            distance_to_resistance=float(distance_to_resistance),
            distance_to_support=float(distance_to_support),
            pivot_point=float(pivot)
        )
    
    def calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算资金流量指标MFI"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        
        return float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50
    
    def analyze_momentum(self, df: pd.DataFrame) -> MomentumAnalysis:
        """分析动量"""
        # 价格动量
        price_10d_ago = df['Close'].iloc[-10] if len(df) >= 10 else df['Close'].iloc[0]
        price_momentum = (df['Close'].iloc[-1] - price_10d_ago) / price_10d_ago * 100
        
        # 成交量趋势
        avg_volume_20 = df['Volume'].iloc[-20:].mean()
        recent_volume = df['Volume'].iloc[-5:].mean()
        volume_ratio = recent_volume / avg_volume_20 if avg_volume_20 > 0 else 1
        
        if volume_ratio > 1.2:
            volume_trend = "INCREASING"
        elif volume_ratio < 0.8:
            volume_trend = "DECREASING"
        else:
            volume_trend = "STABLE"
        
        # MFI资金流量指标
        mfi = self.calculate_mfi(df)
        
        # ROC变化率
        roc = ((df['Close'].iloc[-1] - df['Close'].iloc[-10]) / df['Close'].iloc[-10] * 100) if len(df) >= 10 else 0
        
        # 动量评分
        momentum_score = min(100, max(0, 50 + price_momentum + (volume_ratio - 1) * 20))
        
        return MomentumAnalysis(
            momentum_score=float(momentum_score),
            volume_trend=volume_trend,
            price_momentum=float(price_momentum),
            volume_ratio=float(volume_ratio),
            money_flow_index=float(mfi),
            rate_of_change=float(roc)
        )
    
    def generate_ai_analysis(self, ticker: str, metrics: Dict) -> Dict:
        """使用AI生成深度分析"""
        prompt = f"""你是一位资深的技术分析师。请基于以下技术数据对 {ticker} 进行深度分析:

趋势分析:
{json.dumps(metrics['trend'], indent=2, ensure_ascii=False)}

技术指标:
{json.dumps(metrics['indicators'], indent=2, ensure_ascii=False)}

支撑阻力:
{json.dumps(metrics['support_resistance'], indent=2, ensure_ascii=False)}

动量分析:
{json.dumps(metrics['momentum'], indent=2, ensure_ascii=False)}

请提供:
1. 综合评分 (0-100)
2. 投资建议 (BUY/HOLD/SELL)
3. 置信度 (0-1)
4. 3-5个关键交易信号
5. 3-5个关键警告
6. 200字左右的技术分析总结

以JSON格式返回,结构如下:
{{
  "score": 75.5,
  "recommendation": "BUY",
  "confidence": 0.85,
  "signals": ["信号1", "信号2", ...],
  "warnings": ["警告1", "警告2", ...],
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
                "signals": ["技术分析受限"],
                "warnings": ["分析不完整"],
                "summary": "AI分析暂时不可用,建议人工复核技术指标。"
            }
    
    def analyze(self, ticker: str, period: str = "6mo", verbose: bool = False) -> TechnicalAnalysisResult:
        """
        执行完整的技术分析
        
        Args:
            ticker: 股票代码
            period: 分析周期 (1mo, 3mo, 6mo, 1y, 2y)
            verbose: 是否打印详细过程
            
        Returns:
            TechnicalAnalysisResult: 标准化的分析结果
        """
        if verbose:
            print(f"[{self.agent_name}] 开始分析 {ticker}...")
        
        # 1. 获取价格数据
        df = self.fetch_price_data(ticker, period)
        
        # 2. 各项技术分析
        trend_analysis = self.analyze_trend(df)
        technical_indicators = self.analyze_indicators(df)
        support_resistance = self.calculate_support_resistance(df)
        momentum_analysis = self.analyze_momentum(df)
        
        # 3. 准备AI分析数据
        metrics_for_ai = {
            "trend": trend_analysis.model_dump(),
            "indicators": technical_indicators.model_dump(),
            "support_resistance": support_resistance.model_dump(),
            "momentum": momentum_analysis.model_dump()
        }
        
        # 4. AI深度分析
        ai_analysis = self.generate_ai_analysis(ticker, metrics_for_ai)
        
        # 5. 组装最终结果
        result = TechnicalAnalysisResult(
            agent_name=self.agent_name,
            ticker=ticker,
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            score=ai_analysis["score"],
            recommendation=ai_analysis["recommendation"],
            confidence=ai_analysis["confidence"],
            trend_analysis=trend_analysis,
            technical_indicators=technical_indicators,
            support_resistance=support_resistance,
            momentum_analysis=momentum_analysis,
            key_signals=ai_analysis["signals"],
            key_warnings=ai_analysis["warnings"],
            analysis_summary=ai_analysis["summary"]
        )
        
        if verbose:
            print(f"[{self.agent_name}] 分析完成: {result.recommendation} (评分: {result.score:.1f})")
        
        return result
    
    def get_arena_output(self, ticker: str, period: str = "6mo") -> Dict:
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
            "vote_weight": 1.0,  # 基础权重,可由Arena Judge动态调整
            "summary": result.analysis_summary,
            "key_points": {
                "signals": result.key_signals,
                "warnings": result.key_warnings
            },
            "detailed_metrics": {
                "trend": result.trend_analysis.model_dump(),
                "indicators": result.technical_indicators.model_dump(),
                "support_resistance": result.support_resistance.model_dump(),
                "momentum": result.momentum_analysis.model_dump()
            }
        }
