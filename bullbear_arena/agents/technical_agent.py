# ============================================================================
# BullBear Arena - Technical Analysis Agent
# bullbear_arena/agents/technical_agent.py
# ============================================================================
"""
Technical Analysis Agent - Technical Analyst

Focus on:
- Price trend analysis (Moving averages, ADX)
- Technical indicators (RSI, MACD, KDJ)
- Support and resistance calculation
- Momentum analysis (Volume, MFI, ROC)

Standard output format for Arena Judge
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
# Data Models
# ============================================================================

class TrendAnalysis(BaseModel):
    """Trend analysis"""
    primary_trend: str = Field(description="Primary trend: BULLISH/BEARISH/SIDEWAYS")
    trend_strength: float = Field(description="Trend strength 0-100")
    ma_alignment: str = Field(description="MA alignment: BULLISH/BEARISH/MIXED")
    ma_signals: Dict[str, str] = Field(description="MA signals for each period")
    price_vs_ma200: float = Field(description="Price vs MA200 position (%)")
    adx: float = Field(description="ADX trend strength indicator")

class TechnicalIndicators(BaseModel):
    """Technical indicators"""
    rsi: float = Field(description="RSI 0-100")
    rsi_signal: str = Field(description="RSI signal: OVERBOUGHT/OVERSOLD/NEUTRAL")
    macd: float = Field(description="MACD value")
    macd_signal: float = Field(description="MACD signal line")
    macd_histogram: float = Field(description="MACD histogram")
    macd_trend: str = Field(description="MACD trend: BULLISH/BEARISH")
    stochastic_k: float = Field(description="Stochastic K")
    stochastic_d: float = Field(description="Stochastic D")
    stochastic_signal: str = Field(description="KDJ signal")

class SupportResistance(BaseModel):
    """Support and resistance"""
    current_price: float = Field(description="Current price")
    resistance_1: float = Field(description="First resistance level")
    resistance_2: float = Field(description="Second resistance level")
    support_1: float = Field(description="First support level")
    support_2: float = Field(description="Second support level")
    distance_to_resistance: float = Field(description="Distance to resistance (%)")
    distance_to_support: float = Field(description="Distance to support (%)")
    pivot_point: float = Field(description="Pivot point")

class MomentumAnalysis(BaseModel):
    """Momentum analysis"""
    momentum_score: float = Field(description="Momentum score 0-100")
    volume_trend: str = Field(description="Volume trend: INCREASING/DECREASING/STABLE")
    price_momentum: float = Field(description="Price momentum (%)")
    volume_ratio: float = Field(description="Volume ratio")
    money_flow_index: float = Field(description="Money Flow Index MFI")
    rate_of_change: float = Field(description="Rate of Change ROC")

class TechnicalAnalysisResult(BaseModel):
    """Technical analysis result - Standard output format"""
    agent_name: str = "Technical Analyst"
    ticker: str
    analysis_date: str
    score: float = Field(description="Overall score 0-100", ge=0, le=100)
    recommendation: str = Field(description="Investment recommendation: BUY/HOLD/SELL")
    confidence: float = Field(description="Confidence 0-1", ge=0, le=1)
    trend_analysis: TrendAnalysis
    technical_indicators: TechnicalIndicators
    support_resistance: SupportResistance
    momentum_analysis: MomentumAnalysis
    key_signals: List[str]
    key_warnings: List[str]
    analysis_summary: str

# ============================================================================
# Technical Analysis Agent Class
# ============================================================================

class TechnicalAgent:
    """
    Technical Analysis Agent - BullBear Arena
    
    Role: Technical Analyst
    Responsibility: Evaluate trading timing from price trends and technical indicators
    """
    
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com/v1/chat/completions"):
        """
        Initialize Technical Analysis Agent
        
        Args:
            api_key: DeepSeek API key
            api_url: API endpoint
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = "deepseek-chat"
        self.agent_name = "Technical Analyst"
        self.agent_type = "technical"
    
    def call_deepseek_api(self, prompt: str) -> str:
        """Call DeepSeek API"""
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
        """Fetch price data"""
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"Cannot fetch price data for {ticker}")
        
        return df
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> Dict:
        """Calculate moving averages"""
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
        
        bullish_alignment = (ma5 > ma10 > ma20 > ma50)
        bearish_alignment = (ma5 < ma10 < ma20 < ma50)
        
        if bullish_alignment:
            alignment = "BULLISH"
        elif bearish_alignment:
            alignment = "BEARISH"
        else:
            alignment = "MIXED"
        
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
        """Calculate ADX trend strength indicator"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
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
        """Analyze trend"""
        ma_data = self.calculate_moving_averages(df)
        df = ma_data["df"]
        
        adx = self.calculate_adx(df)
        
        alignment = ma_data["alignment"]
        current_price = df['Close'].iloc[-1]
        
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
        """Calculate RSI"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
    
    def calculate_macd(self, df: pd.DataFrame) -> Dict:
        """Calculate MACD"""
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
        """Calculate Stochastic KDJ"""
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        
        k = 100 * (df['Close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=3).mean()
        
        return {
            "k": float(k.iloc[-1]) if not pd.isna(k.iloc[-1]) else 50,
            "d": float(d.iloc[-1]) if not pd.isna(d.iloc[-1]) else 50
        }
    
    def analyze_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Analyze technical indicators"""
        rsi = self.calculate_rsi(df)
        if rsi > 70:
            rsi_signal = "OVERBOUGHT"
        elif rsi < 30:
            rsi_signal = "OVERSOLD"
        else:
            rsi_signal = "NEUTRAL"
        
        macd_data = self.calculate_macd(df)
        macd_trend = "BULLISH" if macd_data["histogram"] > 0 else "BEARISH"
        
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
        """Calculate support and resistance levels"""
        current_price = df['Close'].iloc[-1]
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        pivot = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
        
        resistance_1 = 2 * pivot - low.iloc[-1]
        resistance_2 = pivot + (high.iloc[-1] - low.iloc[-1])
        support_1 = 2 * pivot - high.iloc[-1]
        support_2 = pivot - (high.iloc[-1] - low.iloc[-1])
        
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
        """Calculate Money Flow Index MFI"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        
        return float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50
    
    def analyze_momentum(self, df: pd.DataFrame) -> MomentumAnalysis:
        """Analyze momentum"""
        price_10d_ago = df['Close'].iloc[-10] if len(df) >= 10 else df['Close'].iloc[0]
        price_momentum = (df['Close'].iloc[-1] - price_10d_ago) / price_10d_ago * 100
        
        avg_volume_20 = df['Volume'].iloc[-20:].mean()
        recent_volume = df['Volume'].iloc[-5:].mean()
        volume_ratio = recent_volume / avg_volume_20 if avg_volume_20 > 0 else 1
        
        if volume_ratio > 1.2:
            volume_trend = "INCREASING"
        elif volume_ratio < 0.8:
            volume_trend = "DECREASING"
        else:
            volume_trend = "STABLE"
        
        mfi = self.calculate_mfi(df)
        
        roc = ((df['Close'].iloc[-1] - df['Close'].iloc[-10]) / df['Close'].iloc[-10] * 100) if len(df) >= 10 else 0
        
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
        """Generate AI analysis"""
        prompt = f"""You are a senior technical analyst. Provide in-depth technical analysis of {ticker} based on:

Trend Analysis:
{json.dumps(metrics['trend'], indent=2)}

Technical Indicators:
{json.dumps(metrics['indicators'], indent=2)}

Support/Resistance:
{json.dumps(metrics['support_resistance'], indent=2)}

Momentum Analysis:
{json.dumps(metrics['momentum'], indent=2)}

Provide:
1. Overall score (0-100)
2. Investment recommendation (BUY/HOLD/SELL)
3. Confidence (0-1)
4. 3-5 key trading signals
5. 3-5 key warnings
6. 200-word technical analysis summary

JSON format:
{{
  "score": 75.5,
  "recommendation": "BUY",
  "confidence": 0.85,
  "signals": ["Signal 1", "Signal 2", ...],
  "warnings": ["Warning 1", "Warning 2", ...],
  "summary": "Analysis summary..."
}}
"""
        
        try:
            response_text = self.call_deepseek_api(prompt)
            if response_text:
                result = json.loads(response_text)
                return result
            else:
                raise Exception("API returned empty response")
        except Exception as e:
            return {
                "score": 50,
                "recommendation": "HOLD",
                "confidence": 0.5,
                "signals": ["Limited technical analysis"],
                "warnings": ["Incomplete analysis"],
                "summary": "AI analysis temporarily unavailable, please manually review technical indicators."
            }
    
    def analyze(self, ticker: str, period: str = "6mo", verbose: bool = False) -> TechnicalAnalysisResult:
        """
        Execute complete technical analysis
        
        Args:
            ticker: Stock ticker
            period: Analysis period (1mo, 3mo, 6mo, 1y, 2y)
            verbose: Print detailed process
            
        Returns:
            TechnicalAnalysisResult: Standardized analysis result
        """
        if verbose:
            print(f"[{self.agent_name}] Starting analysis for {ticker}...")
        
        df = self.fetch_price_data(ticker, period)
        
        trend_analysis = self.analyze_trend(df)
        technical_indicators = self.analyze_indicators(df)
        support_resistance = self.calculate_support_resistance(df)
        momentum_analysis = self.analyze_momentum(df)
        
        metrics_for_ai = {
            "trend": trend_analysis.model_dump(),
            "indicators": technical_indicators.model_dump(),
            "support_resistance": support_resistance.model_dump(),
            "momentum": momentum_analysis.model_dump()
        }
        
        ai_analysis = self.generate_ai_analysis(ticker, metrics_for_ai)
        
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
            print(f"[{self.agent_name}] Analysis complete: {result.recommendation} (Score: {result.score:.1f})")
        
        return result
    
    def get_arena_output(self, ticker: str, period: str = "6mo") -> Dict:
        """
        Provide standardized output for Arena Judge
        
        Returns:
            Dict: Arena standard format with all necessary voting info
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
