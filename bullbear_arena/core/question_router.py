"""
Question Router - Analyzes user questions and routes to appropriate agents

Determines which agents to call based on question content.
"""

import json
import requests
from typing import List, Dict
from pydantic import BaseModel, Field


class RoutingDecision(BaseModel):
    """Question routing decision"""
    question: str = Field(default="", description="Original user question")
    question_type: str = Field(description="Type: single_stock / comparison / market_overview")
    tickers: List[str] = Field(description="Stock tickers mentioned")
    agents_needed: List[str] = Field(description="Agents to call: fundamental/technical/sentiment/risk")
    execution_mode: str = Field(description="Execution mode: single / comparison / market")
    reasoning: str = Field(description="Why this routing decision")


class QuestionRouter:
    """
    Routes user questions to appropriate agents
    
    Examples:
    - "What's AAPL's PE ratio?" -> fundamental agent, single stock
    - "Compare NVDA and AMD" -> all agents, comparison mode
    - "How's the market?" -> sentiment agent, market overview
    """
    
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com/v1/chat/completions"):
        self.api_key = api_key
        self.api_url = api_url
        self.model = "deepseek-chat"
    
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
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    
    def analyze_question(self, question: str) -> RoutingDecision:
        """
        Analyze question and determine routing
        
        Args:
            question: User question
            
        Returns:
            RoutingDecision: Routing decision with tickers and agents
        """
        prompt = f"""Analyze this investment question and determine routing:

Question: "{question}"

Determine:
1. question_type: "single_stock" (1 ticker), "comparison" (2-5 tickers), or "market_overview" (no specific ticker)
2. tickers: List of stock ticker symbols mentioned (e.g., ["AAPL", "NVDA"])
3. agents_needed: Which agents to call based on question keywords:
   - "fundamental": PE, revenue, earnings, ROE, debt, cash flow, valuation
   - "technical": chart, price, trend, RSI, MACD, support, resistance, moving average
   - "sentiment": news, sentiment, social media, analyst, market mood
   - "risk": risk, volatility, drawdown, Sharpe, VaR
   - If asking for "complete analysis" or comparison, use all: ["fundamental", "technical", "sentiment", "risk"]
4. execution_mode: "single" (1 stock), "comparison" (multiple stocks), "market" (general market)
5. reasoning: Brief explanation of routing decision

Return JSON:
{{
  "question_type": "single_stock",
  "tickers": ["AAPL"],
  "agents_needed": ["fundamental"],
  "execution_mode": "single",
  "reasoning": "Question asks about PE ratio which is fundamental analysis"
}}"""
        
        try:
            response_text = self.call_deepseek_api(prompt)
            result = json.loads(response_text)
            
            # Ensure tickers are uppercase
            if result.get('tickers'):
                result['tickers'] = [t.upper() for t in result['tickers']]
            
            # Add original question
            result['question'] = question
            
            return RoutingDecision(**result)
        
        except Exception as e:
            # Fallback: simple keyword-based routing
            question_upper = question.upper()
            
            # Extract potential tickers (simple heuristic)
            words = question_upper.split()
            tickers = [w for w in words if len(w) <= 5 and w.isalpha() and w.isupper()]
            
            # Determine agents based on keywords
            agents = []
            if any(kw in question.lower() for kw in ['pe', 'revenue', 'earnings', 'roe', 'debt', 'cash', 'valuation', 'fundamental']):
                agents.append('fundamental')
            if any(kw in question.lower() for kw in ['chart', 'price', 'trend', 'rsi', 'macd', 'technical', 'support', 'resistance']):
                agents.append('technical')
            if any(kw in question.lower() for kw in ['news', 'sentiment', 'analyst', 'mood']):
                agents.append('sentiment')
            if any(kw in question.lower() for kw in ['risk', 'volatility', 'drawdown', 'sharpe', 'var']):
                agents.append('risk')
            
            # If no specific agent detected or asking for "analysis"/"compare", use all
            if not agents or 'analysis' in question.lower() or 'compare' in question.lower():
                agents = ['fundamental', 'technical', 'sentiment', 'risk']
            
            # Determine question type
            if len(tickers) == 0:
                question_type = "market_overview"
                execution_mode = "market"
            elif len(tickers) == 1:
                question_type = "single_stock"
                execution_mode = "single"
            else:
                question_type = "comparison"
                execution_mode = "comparison"
            
            return RoutingDecision(
                question=question,
                question_type=question_type,
                tickers=tickers,
                agents_needed=agents,
                execution_mode=execution_mode,
                reasoning="Fallback routing based on keyword detection"
            )
