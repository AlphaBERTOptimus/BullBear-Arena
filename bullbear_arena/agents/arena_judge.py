"""
Arena Judge - Final Investment Decision Maker

Integrates analysis from 4 agents and makes final investment recommendation.
"""

import json
import requests
from datetime import datetime
from typing import Dict, List
from pydantic import BaseModel, Field


class InvestmentHorizon(BaseModel):
    """Investment time horizon configuration"""
    horizon_type: str
    duration_description: str
    data_timeframe_used: str
    recommended_weights: Dict[str, float]


class AgentVote(BaseModel):
    """Individual agent vote details"""
    agent_name: str
    agent_type: str
    recommendation: str
    confidence: float
    score: float
    vote_weight: float
    weighted_score: float
    detailed_analysis: Dict


class VotingBreakdown(BaseModel):
    """Voting statistics across all agents"""
    buy_votes: int
    hold_votes: int
    sell_votes: int
    buy_weight: float
    hold_weight: float
    sell_weight: float
    consensus_level: str


class WeightDistribution(BaseModel):
    """Dynamic weight allocation across agents"""
    fundamental_weight: float
    technical_weight: float
    sentiment_weight: float
    risk_weight: float
    weighting_rationale: str
    adjustment_factors: List[str]


class ArenaJudgeResult(BaseModel):
    """Final judgment result from Arena Judge"""
    ticker: str
    analysis_date: str
    investment_horizon: InvestmentHorizon
    final_recommendation: str
    confidence: float
    consensus_score: float
    agent_votes: List[AgentVote]
    voting_breakdown: VotingBreakdown
    weight_distribution: WeightDistribution
    detailed_reasoning: str
    action_plan: str
    risk_disclosure: str
    key_insights: List[str]
    divergent_views: List[str]


class ArenaJudge:
    """
    Arena Judge - Makes final investment decisions
    
    Investment Philosophy:
    - Long-term: Fundamental 50%, Risk 30%, Technical 10%, Sentiment 10%
    - Medium-term: Technical 35%, Fundamental 30%, Sentiment 20%, Risk 15%
    - Short-term: Technical 45%, Sentiment 30%, Risk 15%, Fundamental 10%
    """
    
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com/v1/chat/completions"):
        self.api_key = api_key
        self.api_url = api_url
        self.model = "deepseek-chat"
        self.agent_name = "Arena Judge"
    
    def call_deepseek_api(self, prompt: str) -> str:
        """Call DeepSeek API for AI analysis"""
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
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    
    def determine_investment_horizon(self, ticker: str, investment_period: str = "LONG_TERM") -> InvestmentHorizon:
        """Determine investment horizon and weight configuration"""
        configs = {
            "LONG_TERM": {
                "duration": "Long-term investment (>1 year)",
                "data_range": "Past 3-5 years data",
                "weights": {"fundamental": 0.50, "risk": 0.30, "technical": 0.10, "sentiment": 0.10}
            },
            "MEDIUM_TERM": {
                "duration": "Medium-term investment (3-12 months)",
                "data_range": "Past 1-2 years data",
                "weights": {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.20, "risk": 0.15}
            },
            "SHORT_TERM": {
                "duration": "Short-term investment (<3 months)",
                "data_range": "Past 3-6 months data",
                "weights": {"technical": 0.45, "sentiment": 0.30, "risk": 0.15, "fundamental": 0.10}
            }
        }
        config = configs.get(investment_period, configs["LONG_TERM"])
        return InvestmentHorizon(
            horizon_type=investment_period,
            duration_description=config["duration"],
            data_timeframe_used=config["data_range"],
            recommended_weights=config["weights"]
        )
    
    def calculate_smart_weights(self, agent_outputs: List[Dict], horizon: InvestmentHorizon) -> WeightDistribution:
        """Calculate dynamic weights based on agent confidence"""
        base_weights = horizon.recommended_weights.copy()
        adjustment_factors = []
        
        # Map outputs to agent types based on order
        agent_types = ['fundamental', 'technical', 'sentiment', 'risk']
        
        for i, output in enumerate(agent_outputs):
            # Get agent_type from output or use index
            agent_type = output.get('agent_type', agent_types[i] if i < len(agent_types) else 'unknown')
            
            if agent_type not in base_weights:
                continue
            
            confidence = output.get('confidence', 0.7)
            
            if confidence > 0.85:
                base_weights[agent_type] *= 1.1
                adjustment_factors.append(f"{output.get('agent_name', agent_type)} high confidence (+10%)")
            elif confidence < 0.60:
                base_weights[agent_type] *= 0.9
                adjustment_factors.append(f"{output.get('agent_name', agent_type)} low confidence (-10%)")
        
        total = sum(base_weights.values())
        final_weights = {k: v/total for k, v in base_weights.items()}
        
        return WeightDistribution(
            fundamental_weight=final_weights['fundamental'],
            technical_weight=final_weights['technical'],
            sentiment_weight=final_weights['sentiment'],
            risk_weight=final_weights['risk'],
            weighting_rationale=f"Based on {horizon.duration_description}",
            adjustment_factors=adjustment_factors
        )
    
    def collect_detailed_votes(self, agent_outputs: List[Dict], weights: WeightDistribution) -> List[AgentVote]:
        """Collect votes from all agents with weights"""
        votes = []
        weight_map = {
            'fundamental': weights.fundamental_weight,
            'technical': weights.technical_weight,
            'sentiment': weights.sentiment_weight,
            'risk': weights.risk_weight
        }
        
        # Map outputs to agent types based on order
        agent_types = ['fundamental', 'technical', 'sentiment', 'risk']
        
        for i, output in enumerate(agent_outputs):
            # Get agent_type from output or use index
            agent_type = output.get('agent_type', agent_types[i] if i < len(agent_types) else 'unknown')
            weight = weight_map.get(agent_type, 0.25)
            
            detailed_analysis = {
                "summary": output.get('summary', ''),
                "key_points": output.get('key_points', {}),
                "detailed_metrics": output.get('detailed_metrics', {})
            }
            
            vote = AgentVote(
                agent_name=output.get('agent_name', agent_type.title()),
                agent_type=agent_type,
                recommendation=output.get('recommendation', 'HOLD'),
                confidence=output.get('confidence', 0.7),
                score=output.get('score', 50.0),
                vote_weight=weight,
                weighted_score=output.get('score', 50.0) * weight,
                detailed_analysis=detailed_analysis
            )
            votes.append(vote)
        
        return votes
    
    def analyze_voting(self, votes: List[AgentVote]) -> VotingBreakdown:
        """Analyze voting patterns and consensus level"""
        buy_votes = sum(1 for v in votes if v.recommendation == "BUY")
        hold_votes = sum(1 for v in votes if v.recommendation == "HOLD")
        sell_votes = sum(1 for v in votes if v.recommendation == "SELL")
        
        buy_weight = sum(v.vote_weight for v in votes if v.recommendation == "BUY")
        hold_weight = sum(v.vote_weight for v in votes if v.recommendation == "HOLD")
        sell_weight = sum(v.vote_weight for v in votes if v.recommendation == "SELL")
        
        max_votes = max(buy_votes, hold_votes, sell_votes)
        
        if max_votes >= 3:
            consensus_level = "STRONG"
        elif buy_votes == sell_votes == 2:
            consensus_level = "DIVIDED"
        elif max_votes == 2:
            consensus_level = "MODERATE"
        else:
            consensus_level = "WEAK"
        
        return VotingBreakdown(
            buy_votes=buy_votes,
            hold_votes=hold_votes,
            sell_votes=sell_votes,
            buy_weight=buy_weight,
            hold_weight=hold_weight,
            sell_weight=sell_weight,
            consensus_level=consensus_level
        )
    
    def ai_professional_judgment(self, ticker: str, votes: List[AgentVote], breakdown: VotingBreakdown, 
                                 horizon: InvestmentHorizon, weights: WeightDistribution) -> Dict:
        """Get AI professional judgment on final recommendation"""
        prompt = f"""You are a senior financial analyst. Analyze {ticker} for investment.

Investment Horizon: {horizon.duration_description}
Data Range: {horizon.data_timeframe_used}
Date: {datetime.now().strftime("%Y-%m-%d")}

Weight Configuration:
- Fundamental: {weights.fundamental_weight:.1%}
- Technical: {weights.technical_weight:.1%}
- Sentiment: {weights.sentiment_weight:.1%}
- Risk: {weights.risk_weight:.1%}

Agent Votes:
- BUY: {breakdown.buy_votes} votes (weight {breakdown.buy_weight:.1%})
- HOLD: {breakdown.hold_votes} votes (weight {breakdown.hold_weight:.1%})
- SELL: {breakdown.sell_votes} votes (weight {breakdown.sell_weight:.1%})
- Consensus: {breakdown.consensus_level}

Provide professional analysis in JSON format:
{{
  "final_recommendation": "BUY/HOLD/SELL",
  "confidence": 0.75,
  "consensus_score": 70.0,
  "detailed_reasoning": "Comprehensive analysis (300+ words)",
  "action_plan": "Specific investment actions",
  "key_insights": ["insight 1", "insight 2", "insight 3"],
  "divergent_views": ["divergence 1", "divergence 2"],
  "risk_disclosure": "Specific risks for {ticker} (150+ words)"
}}"""
        
        try:
            response_text = self.call_deepseek_api(prompt)
            return json.loads(response_text)
        except Exception as e:
            return {
                "final_recommendation": "HOLD",
                "confidence": 0.5,
                "consensus_score": 50.0,
                "detailed_reasoning": "AI analysis temporarily unavailable. Manual review recommended.",
                "action_plan": "Wait for system recovery or consult human advisor.",
                "key_insights": ["System limitations"],
                "divergent_views": ["Incomplete data"],
                "risk_disclosure": "Investment involves risks. This analysis is for reference only and does not constitute investment advice."
            }
    
    def judge(self, ticker: str, fundamental_output: Dict, technical_output: Dict, 
              sentiment_output: Dict, risk_output: Dict, investment_period: str = "LONG_TERM") -> ArenaJudgeResult:
        """Execute final judgment on investment decision"""
        agent_outputs = [fundamental_output, technical_output, sentiment_output, risk_output]
        
        horizon = self.determine_investment_horizon(ticker, investment_period)
        weights = self.calculate_smart_weights(agent_outputs, horizon)
        votes = self.collect_detailed_votes(agent_outputs, weights)
        breakdown = self.analyze_voting(votes)
        ai_result = self.ai_professional_judgment(ticker, votes, breakdown, horizon, weights)
        
        return ArenaJudgeResult(
            ticker=ticker,
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            investment_horizon=horizon,
            final_recommendation=ai_result['final_recommendation'],
            confidence=ai_result['confidence'],
            consensus_score=ai_result['consensus_score'],
            agent_votes=votes,
            voting_breakdown=breakdown,
            weight_distribution=weights,
            detailed_reasoning=ai_result['detailed_reasoning'],
            action_plan=ai_result['action_plan'],
            risk_disclosure=ai_result['risk_disclosure'],
            key_insights=ai_result['key_insights'],
            divergent_views=ai_result['divergent_views']
        )
