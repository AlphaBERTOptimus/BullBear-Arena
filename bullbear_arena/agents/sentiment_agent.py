# ============================================================================
# BullBear Arena - 情绪分析Agent
# bullbear_arena/agents/sentiment_agent.py
# ============================================================================
"""
情绪分析Agent - Sentiment Monitor

专注于:
- 新闻情感分析 (金融关键词NLP)
- 社交媒体情绪追踪
- 市场情绪指标 (恐惧贪婪指数)
- 分析师估值智能判断

核心决策逻辑:
1. EXTREME_FEAR市场情绪 -> 自动SELL
2. 价格超过目标价20% -> 自动SELL
3. EXTREME_GREED -> SELL (防止追高)
4. 低估值 + 正面情绪 -> BUY
5. 多维度规则引擎综合决策

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

# ============================================================================
# 金融情感词典
# ============================================================================

POSITIVE_KEYWORDS = {
    # Super strong positive (0.8-1.0)
    'surge': 0.9, 'soar': 0.9, 'skyrocket': 1.0, 'boom': 0.85, 'breakout': 0.8,
    # Strong positive (0.6-0.8)
    'rally': 0.75, 'beat': 0.7, 'exceed': 0.65, 'outperform': 0.7, 'upgrade': 0.75,
    'bullish': 0.8, 'strong': 0.6, 'record': 0.7, 'breakthrough': 0.75,
    # Moderate positive (0.4-0.6)
    'gains': 0.5, 'growth': 0.55, 'profit': 0.5, 'success': 0.6, 'innovation': 0.55,
    'momentum': 0.5, 'optimistic': 0.6, 'opportunity': 0.45, 'expansion': 0.5,
    # Weak positive (0.2-0.4)
    'up': 0.3, 'rise': 0.4, 'increase': 0.3, 'gain': 0.4, 'positive': 0.4,
    'better': 0.35, 'improve': 0.4, 'recovery': 0.5, 'partnership': 0.35
}

NEGATIVE_KEYWORDS = {
    # Super strong negative (-0.8 to -1.0)
    'crash': -1.0, 'collapse': -0.95, 'plunge': -0.9, 'plummet': -0.9, 
    'bankruptcy': -1.0, 'scandal': -0.85,
    # Strong negative (-0.6 to -0.8)
    'tumble': -0.75, 'decline': -0.6, 'bearish': -0.8, 'downgrade': -0.75,
    'crisis': -0.8, 'lawsuit': -0.7, 'investigation': -0.65, 'layoff': -0.7,
    # Moderate negative (-0.4 to -0.6)
    'fall': -0.5, 'drop': -0.5, 'weak': -0.5, 'loss': -0.6, 'miss': -0.65,
    'disappoint': -0.6, 'concern': -0.45, 'risk': -0.4, 'warning': -0.55,
    'threat': -0.6, 'debt': -0.45, 'struggle': -0.5,
    # Weak negative (-0.2 to -0.4)
    'down': -0.3, 'decrease': -0.35, 'lower': -0.3, 'negative': -0.4,
    'worse': -0.4, 'pressure': -0.35, 'challenge': -0.3, 'volatility': -0.35
}

# ============================================================================
# 数据模型定义
# ============================================================================

class NewsSentiment(BaseModel):
    """新闻情感分析"""
    sentiment_score: float = Field(description="News sentiment score -1 to 1")
    sentiment_label: str = Field(description="Sentiment label: VERY_POSITIVE/POSITIVE/NEUTRAL/NEGATIVE/VERY_NEGATIVE")
    news_count: int = Field(description="Number of news analyzed")
    positive_ratio: float = Field(description="Positive news ratio 0-1")
    negative_ratio: float = Field(description="Negative news ratio 0-1")
    recent_headlines: List[str] = Field(description="Recent important headlines")
    sentiment_intensity: str = Field(description="Sentiment intensity: EXTREME/STRONG/MODERATE/WEAK")
    
class SocialSentiment(BaseModel):
    """社交媒体情绪"""
    social_score: float = Field(description="Social media sentiment score -1 to 1")
    discussion_volume: str = Field(description="Discussion volume: VIRAL/HIGH/MEDIUM/LOW")
    trending_topics: List[str] = Field(description="Trending topics")
    sentiment_trend: str = Field(description="Sentiment trend: SURGING/IMPROVING/STABLE/DECLINING/COLLAPSING")
    buzz_level: float = Field(description="Buzz level 0-100")
    
class MarketSentiment(BaseModel):
    """市场情绪指标"""
    fear_greed_index: float = Field(description="Fear & Greed Index 0-100")
    put_call_ratio: float = Field(description="Put/Call ratio")
    volatility_index: float = Field(description="Volatility index")
    market_mood: str = Field(description="Market mood: EXTREME_FEAR/FEAR/NEUTRAL/GREED/EXTREME_GREED")
    market_signal: str = Field(description="Market signal: STRONG_SELL/SELL/HOLD/BUY/STRONG_BUY")
    
class AnalystValuation(BaseModel):
    """分析师估值"""
    current_price: float = Field(description="Current price")
    target_price: float = Field(description="Analyst target price")
    upside_potential: float = Field(description="Upside potential (%)")
    valuation_signal: str = Field(description="Valuation signal: OVERVALUED/FAIRLY_VALUED/UNDERVALUED")
    price_vs_target: str = Field(description="Price vs target position")
    analyst_rating: str = Field(description="Analyst rating")

class EventImpact(BaseModel):
    """事件影响评估"""
    recent_events: List[str] = Field(description="Recent major events")
    earnings_sentiment: str = Field(description="Earnings sentiment: POSITIVE/NEUTRAL/NEGATIVE/PENDING")
    analyst_valuation: AnalystValuation = Field(description="Analyst valuation analysis")
    institutional_activity: str = Field(description="Institutional activity: BUYING/SELLING/NEUTRAL")

class SentimentAnalysisResult(BaseModel):
    """情绪分析结果 - 标准输出格式"""
    agent_name: str = "Sentiment Monitor"
    ticker: str
    analysis_date: str
    score: float = Field(description="Overall score 0-100", ge=0, le=100)
    recommendation: str = Field(description="Investment recommendation: BUY/HOLD/SELL")
    confidence: float = Field(description="Confidence 0-1", ge=0, le=1)
    news_sentiment: NewsSentiment
    social_sentiment: SocialSentiment
    market_sentiment: MarketSentiment
    event_impact: EventImpact
    key_catalysts: List[str]
    key_concerns: List[str]
    analysis_summary: str
    decision_rationale: str = Field(description="Decision rationale")

# ============================================================================
# 情绪分析Agent类
# ============================================================================

class SentimentAgent:
    """
    情绪分析智能体 - BullBear Arena
    
    Role: Sentiment Monitor
    Responsibility: Evaluate investor sentiment from news, social media, and market mood
    
    Core decision logic:
    1. EXTREME_FEAR market mood -> auto SELL
    2. Price exceeds target by 20%+ -> auto SELL
    3. EXTREME_GREED + overbought -> SELL
    4. Multiple negative news + downtrend -> SELL
    5. Strong positive sentiment + undervalued -> BUY
    """
    
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com/v1/chat/completions"):
        """
        Initialize Sentiment Analysis Agent
        
        Args:
            api_key: DeepSeek API key
            api_url: API endpoint
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = "deepseek-chat"
        self.agent_name = "Sentiment Monitor"
        self.agent_type = "sentiment"
    
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
    
    def fetch_stock_info(self, ticker: str) -> Dict:
        """Fetch stock basic info"""
        stock = yf.Ticker(ticker)
        return stock.info
    
    def fetch_news(self, ticker: str) -> List[Dict]:
        """Fetch news data"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            if not news:
                return [{"title": f"{ticker} maintains steady performance", "publisher": "Market Watch"}]
            return news[:20]
        except Exception as e:
            return []
    
    def analyze_text_sentiment_enhanced(self, text: str) -> float:
        """Enhanced text sentiment analysis (with financial keywords)"""
        text_lower = text.lower()
        
        base_sentiment = 0.0
        
        keyword_score = 0.0
        keyword_count = 0
        
        for keyword, weight in POSITIVE_KEYWORDS.items():
            if keyword in text_lower:
                keyword_score += weight
                keyword_count += 1
        
        for keyword, weight in NEGATIVE_KEYWORDS.items():
            if keyword in text_lower:
                keyword_score += weight
                keyword_count += 1
        
        if keyword_count > 0:
            final_score = keyword_score * 0.8 + base_sentiment * 0.2
        else:
            final_score = base_sentiment * 2.0
        
        return max(-1, min(1, final_score))
    
    def analyze_news_sentiment(self, ticker: str) -> NewsSentiment:
        """Analyze news sentiment"""
        news_items = self.fetch_news(ticker)
        
        if not news_items:
            return NewsSentiment(
                sentiment_score=0.0,
                sentiment_label="NEUTRAL",
                news_count=0,
                positive_ratio=0.0,
                negative_ratio=0.0,
                recent_headlines=["No news data available"],
                sentiment_intensity="WEAK"
            )
        
        sentiments = []
        headlines = []
        
        for item in news_items:
            title = item.get('title', '')
            if title:
                headlines.append(title)
                sentiment = self.analyze_text_sentiment_enhanced(title)
                sentiments.append(sentiment)
        
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        
        positive_count = sum(1 for s in sentiments if s > 0.02)
        negative_count = sum(1 for s in sentiments if s < -0.02)
        total = len(sentiments) if sentiments else 1
        
        if avg_sentiment > 0.25:
            label = "VERY_POSITIVE"
        elif avg_sentiment > 0.05:
            label = "POSITIVE"
        elif avg_sentiment < -0.25:
            label = "VERY_NEGATIVE"
        elif avg_sentiment < -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        
        abs_avg = abs(avg_sentiment)
        if abs_avg > 0.4:
            intensity = "EXTREME"
        elif abs_avg > 0.2:
            intensity = "STRONG"
        elif abs_avg > 0.08:
            intensity = "MODERATE"
        else:
            intensity = "WEAK"
        
        return NewsSentiment(
            sentiment_score=float(avg_sentiment),
            sentiment_label=label,
            news_count=len(news_items),
            positive_ratio=float(positive_count / total),
            negative_ratio=float(negative_count / total),
            recent_headlines=headlines[:5],
            sentiment_intensity=intensity
        )
    
    def analyze_social_sentiment(self, ticker: str, info: Dict) -> SocialSentiment:
        """Analyze social media sentiment"""
        company_name = info.get('longName', ticker)
        volume = info.get('volume', 0)
        avg_volume = info.get('averageVolume', 1)
        price_change = info.get('regularMarketChangePercent', 0)
        
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio > 5:
            discussion_volume = "VIRAL"
            social_score = 0.7
            buzz_level = 100
        elif volume_ratio > 2.5:
            discussion_volume = "HIGH"
            social_score = 0.4
            buzz_level = 80
        elif volume_ratio > 1.2:
            discussion_volume = "MEDIUM"
            social_score = 0.1
            buzz_level = 50
        else:
            discussion_volume = "LOW"
            social_score = -0.1
            buzz_level = 20
        
        if price_change > 10:
            social_score += 0.5
            sentiment_trend = "SURGING"
        elif price_change > 3:
            social_score += 0.25
            sentiment_trend = "IMPROVING"
        elif price_change < -10:
            social_score -= 0.5
            sentiment_trend = "COLLAPSING"
        elif price_change < -3:
            social_score -= 0.25
            sentiment_trend = "DECLINING"
        else:
            sentiment_trend = "STABLE"
        
        social_score = max(-1, min(1, social_score))
        
        trending_topics = [f"{company_name} {sentiment_trend.lower()}", f"{ticker} analysis"]
        
        return SocialSentiment(
            social_score=float(social_score),
            discussion_volume=discussion_volume,
            trending_topics=trending_topics,
            sentiment_trend=sentiment_trend,
            buzz_level=float(buzz_level)
        )
    
    def analyze_market_sentiment(self, ticker: str, info: Dict) -> MarketSentiment:
        """Analyze market sentiment indicators"""
        stock = yf.Ticker(ticker)
        
        try:
            options_dates = stock.options
            if options_dates:
                opt_chain = stock.option_chain(options_dates[0])
                puts_volume = opt_chain.puts['volume'].sum()
                calls_volume = opt_chain.calls['volume'].sum()
                put_call_ratio = puts_volume / calls_volume if calls_volume > 0 else 1.0
            else:
                put_call_ratio = 1.0
        except:
            put_call_ratio = 1.0
        
        try:
            hist = stock.history(period="1mo")
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
        except:
            volatility = 20.0
        
        price_momentum = info.get('regularMarketChangePercent', 0)
        
        fear_greed = 50
        fear_greed += price_momentum * 4
        fear_greed -= (put_call_ratio - 1) * 40
        fear_greed -= (volatility - 20) * 1.5
        
        fear_greed = max(0, min(100, fear_greed))
        
        if fear_greed < 10:
            market_mood = "EXTREME_FEAR"
            market_signal = "STRONG_SELL"
        elif fear_greed < 30:
            market_mood = "FEAR"
            market_signal = "SELL"
        elif fear_greed < 70:
            market_mood = "NEUTRAL"
            market_signal = "HOLD"
        elif fear_greed < 90:
            market_mood = "GREED"
            market_signal = "BUY"
        else:
            market_mood = "EXTREME_GREED"
            market_signal = "SELL"
        
        return MarketSentiment(
            fear_greed_index=float(fear_greed),
            put_call_ratio=float(put_call_ratio),
            volatility_index=float(volatility),
            market_mood=market_mood,
            market_signal=market_signal
        )
    
    def analyze_analyst_valuation(self, info: Dict) -> AnalystValuation:
        """Analyze analyst valuation"""
        current_price = info.get('regularMarketPrice', 0) or info.get('currentPrice', 0)
        target_price = info.get('targetMeanPrice', 0)
        analyst_rating = info.get('recommendationKey', 'hold')
        
        if target_price and current_price:
            upside_potential = (target_price - current_price) / current_price * 100
            
            if current_price > target_price * 1.20:
                valuation_signal = "OVERVALUED"
                price_vs_target = f"Exceeds target by {abs(upside_potential):.1f}% -> Strong sell signal!"
            elif current_price > target_price * 1.10:
                valuation_signal = "OVERVALUED"
                price_vs_target = f"Exceeds target by {abs(upside_potential):.1f}% -> Consider taking profit"
            elif current_price < target_price * 0.80:
                valuation_signal = "UNDERVALUED"
                price_vs_target = f"Below target by {upside_potential:.1f}% -> Buy opportunity"
            else:
                valuation_signal = "FAIRLY_VALUED"
                price_vs_target = f"Near target (upside {upside_potential:.1f}%)"
        else:
            upside_potential = 0.0
            valuation_signal = "FAIRLY_VALUED"
            price_vs_target = "No target price data"
        
        return AnalystValuation(
            current_price=float(current_price),
            target_price=float(target_price),
            upside_potential=float(upside_potential),
            valuation_signal=valuation_signal,
            price_vs_target=price_vs_target,
            analyst_rating=analyst_rating
        )
    
    def analyze_event_impact(self, ticker: str, info: Dict) -> EventImpact:
        """Analyze event impact"""
        recent_events = []
        
        earnings_date = info.get('earningsDate', None)
        if earnings_date:
            recent_events.append(f"Earnings date: {earnings_date}")
            earnings_sentiment = "PENDING"
        else:
            earnings_sentiment = "NEUTRAL"
        
        analyst_valuation = self.analyze_analyst_valuation(info)
        
        institutional_holders = info.get('heldPercentInstitutions', 0)
        if institutional_holders > 0.7:
            institutional_activity = "BUYING"
        elif institutional_holders < 0.3:
            institutional_activity = "SELLING"
        else:
            institutional_activity = "NEUTRAL"
        
        if not recent_events:
            recent_events.append("No major events")
        
        return EventImpact(
            recent_events=recent_events,
            earnings_sentiment=earnings_sentiment,
            analyst_valuation=analyst_valuation,
            institutional_activity=institutional_activity
        )
    
    def make_final_decision(
        self, 
        news_sentiment: NewsSentiment,
        social_sentiment: SocialSentiment,
        market_sentiment: MarketSentiment,
        event_impact: EventImpact
    ) -> Tuple[str, float, str]:
        """
        Final decision logic (rule engine)
        
        Returns:
            (recommendation, confidence, rationale)
        """
        reasons = []
        sell_signals = 0
        buy_signals = 0
        
        # Rule 1: Extreme market fear -> SELL
        if market_sentiment.market_mood == "EXTREME_FEAR":
            sell_signals += 3
            reasons.append("Extreme market fear (EXTREME_FEAR)")
        
        # Rule 2: Exceeds target by 20%+ -> SELL
        if event_impact.analyst_valuation.valuation_signal == "OVERVALUED":
            if event_impact.analyst_valuation.upside_potential < -15:
                sell_signals += 3
                reasons.append(f"Price exceeds target by {abs(event_impact.analyst_valuation.upside_potential):.1f}%")
            elif event_impact.analyst_valuation.upside_potential < -5:
                sell_signals += 2
                reasons.append("Price exceeds target")
        
        # Rule 3: Extreme greed -> SELL
        if market_sentiment.market_mood == "EXTREME_GREED":
            sell_signals += 2
            reasons.append("Extreme market greed")
        
        # Rule 4: Very negative news -> SELL
        if news_sentiment.sentiment_label == "VERY_NEGATIVE":
            sell_signals += 2
            reasons.append(f"Very negative news (score: {news_sentiment.sentiment_score:.2f})")
        
        # Rule 5: Collapsing social sentiment -> SELL
        if social_sentiment.sentiment_trend == "COLLAPSING":
            sell_signals += 2
            reasons.append("Collapsing social sentiment")
        
        # Rule 6: Undervalued + positive sentiment -> BUY
        if event_impact.analyst_valuation.valuation_signal == "UNDERVALUED":
            if event_impact.analyst_valuation.upside_potential > 20:
                buy_signals += 3
                reasons.append(f"Significantly below target {event_impact.analyst_valuation.upside_potential:.1f}%")
            elif event_impact.analyst_valuation.upside_potential > 10:
                buy_signals += 2
                reasons.append("Below target price")
        
        # Rule 7: Very positive news -> BUY
        if news_sentiment.sentiment_label == "VERY_POSITIVE":
            buy_signals += 2
            reasons.append(f"Very positive news (score: {news_sentiment.sentiment_score:.2f})")
        
        # Rule 8: Surging social sentiment -> BUY
        if social_sentiment.sentiment_trend == "SURGING":
            buy_signals += 1
            reasons.append("Surging social sentiment")
        
        # Rule 9: Market fear + undervalued -> BUY
        if market_sentiment.market_mood == "FEAR" and event_impact.analyst_valuation.upside_potential > 15:
            buy_signals += 2
            reasons.append("Market fear but fair valuation")
        
        # Final decision
        if sell_signals >= 3:
            recommendation = "SELL"
            confidence = min(0.95, 0.6 + sell_signals * 0.1)
        elif buy_signals >= 3:
            recommendation = "BUY"
            confidence = min(0.95, 0.6 + buy_signals * 0.1)
        elif sell_signals > buy_signals:
            recommendation = "SELL"
            confidence = 0.6
        elif buy_signals > sell_signals:
            recommendation = "BUY"
            confidence = 0.6
        else:
            recommendation = "HOLD"
            confidence = 0.5
            reasons.append("Neutral signals")
        
        rationale = " | ".join(reasons) if reasons else "Comprehensive neutral analysis"
        
        return recommendation, confidence, rationale
    
    def generate_ai_analysis(self, ticker: str, metrics: Dict, rule_decision: Dict) -> Dict:
        """Generate AI analysis"""
        prompt = f"""You are a senior market sentiment analyst. Provide in-depth analysis of {ticker} based on:

Rule Engine Decision:
Recommendation: {rule_decision['recommendation']}
Confidence: {rule_decision['confidence']:.1%}
Rationale: {rule_decision['rationale']}

Detailed Data:
News Sentiment: {json.dumps(metrics['news_sentiment'], indent=2)}
Social Sentiment: {json.dumps(metrics['social_sentiment'], indent=2)}
Market Sentiment: {json.dumps(metrics['market_sentiment'], indent=2)}
Event Impact: {json.dumps(metrics['event_impact'], indent=2)}

Provide:
1. Overall score (0-100)
2. Investment recommendation (BUY/HOLD/SELL)
3. Confidence (0-1)
4. 3-5 key catalysts
5. 3-5 key concerns
6. 200-word analysis summary

JSON format:
{{
  "score": 75.5,
  "recommendation": "BUY",
  "confidence": 0.85,
  "catalysts": ["Catalyst 1", ...],
  "concerns": ["Concern 1", ...],
  "summary": "Analysis summary..."
}}
"""
        
        try:
            response_text = self.call_deepseek_api(prompt)
            if response_text:
                return json.loads(response_text)
        except:
            pass
        
        score_map = {"SELL": 25, "HOLD": 50, "BUY": 75}
        return {
            "score": score_map.get(rule_decision['recommendation'], 50),
            "recommendation": rule_decision['recommendation'],
            "confidence": rule_decision['confidence'],
            "catalysts": ["Rule engine driven decision"],
            "concerns": ["AI analysis unavailable"],
            "summary": rule_decision['rationale']
        }
    
    def analyze(self, ticker: str, verbose: bool = False) -> SentimentAnalysisResult:
        """
        Execute complete sentiment analysis
        
        Args:
            ticker: Stock ticker
            verbose: Print detailed process
            
        Returns:
            SentimentAnalysisResult: Standardized analysis result
        """
        if verbose:
            print(f"[{self.agent_name}] Starting analysis for {ticker}...")
        
        info = self.fetch_stock_info(ticker)
        
        news_sentiment = self.analyze_news_sentiment(ticker)
        social_sentiment = self.analyze_social_sentiment(ticker, info)
        market_sentiment = self.analyze_market_sentiment(ticker, info)
        event_impact = self.analyze_event_impact(ticker, info)
        
        recommendation, confidence, rationale = self.make_final_decision(
            news_sentiment, social_sentiment, market_sentiment, event_impact
        )
        
        metrics_for_ai = {
            "news_sentiment": news_sentiment.model_dump(),
            "social_sentiment": social_sentiment.model_dump(),
            "market_sentiment": market_sentiment.model_dump(),
            "event_impact": event_impact.model_dump()
        }
        
        rule_decision = {
            "recommendation": recommendation,
            "confidence": confidence,
            "rationale": rationale
        }
        
        ai_analysis = self.generate_ai_analysis(ticker, metrics_for_ai, rule_decision)
        
        result = SentimentAnalysisResult(
            agent_name=self.agent_name,
            ticker=ticker,
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            score=ai_analysis["score"],
            recommendation=ai_analysis["recommendation"],
            confidence=ai_analysis["confidence"],
            news_sentiment=news_sentiment,
            social_sentiment=social_sentiment,
            market_sentiment=market_sentiment,
            event_impact=event_impact,
            key_catalysts=ai_analysis["catalysts"],
            key_concerns=ai_analysis["concerns"],
            analysis_summary=ai_analysis["summary"],
            decision_rationale=rationale
        )
        
        if verbose:
            print(f"[{self.agent_name}] Analysis complete: {result.recommendation} (Score: {result.score:.1f})")
        
        return result
    
    def get_arena_output(self, ticker: str) -> Dict:
        """
        Provide standardized output for Arena Judge
        
        Returns:
            Dict: Arena standard format with all necessary voting info
        """
        result = self.analyze(ticker, verbose=False)
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "ticker": result.ticker,
            "score": result.score,
            "recommendation": result.recommendation,
            "confidence": result.confidence,
            "vote_weight": 1.0,
            "summary": result.analysis_summary,
            "decision_rationale": result.decision_rationale,
            "key_points": {
                "catalysts": result.key_catalysts,
                "concerns": result.key_concerns
            },
            "detailed_metrics": {
                "news_sentiment": result.news_sentiment.model_dump(),
                "social_sentiment": result.social_sentiment.model_dump(),
                "market_sentiment": result.market_sentiment.model_dump(),
                "event_impact": result.event_impact.model_dump()
            }
        }
