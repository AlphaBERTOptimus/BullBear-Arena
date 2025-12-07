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
