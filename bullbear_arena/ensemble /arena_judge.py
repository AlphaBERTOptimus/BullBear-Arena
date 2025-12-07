import json
import requests
from datetime import datetime
from typing import Dict, List
from pydantic import BaseModel, Field

class InvestmentHorizon(BaseModel):
    horizon_type: str
    duration_description: str
    data_timeframe_used: str
    recommended_weights: Dict[str, float]

class AgentVote(BaseModel):
    agent_name: str
    agent_type: str
    recommendation: str
    confidence: float
    score: float
    vote_weight: float
    weighted_score: float
    detailed_analysis: Dict

class VotingBreakdown(BaseModel):
    buy_votes: int
    hold_votes: int
    sell_votes: int
    buy_weight: float
    hold_weight: float
    sell_weight: float
    consensus_level: str

class WeightDistribution(BaseModel):
    fundamental_weight: float
    technical_weight: float
    sentiment_weight: float
    risk_weight: float
    weighting_rationale: str
    adjustment_factors: List[str]

class ArenaJudgeResult(BaseModel):
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
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com/v1/chat/completions"):
        self.api_key = api_key
        self.api_url = api_url
        self.model = "deepseek-chat"
        self.agent_name = "Arena Judge"
    
    def call_deepseek_api(self, prompt: str) -> str:
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
        configs = {
            "LONG_TERM": {
                "duration": "Long-term (>1 year)",
                "data_range": "Past 3-5 years",
                "weights": {"fundamental": 0.50, "risk": 0.30, "technical": 0.10, "sentiment": 0.10}
            },
            "MEDIUM_TERM": {
                "duration": "Medium-term (3-12 months)",
                "data_range": "Past 1-2 years",
                "weights": {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.20, "risk": 0.15}
            },
            "SHORT_TERM": {
                "duration": "Short-term (<3 months)",
                "data_range": "Past 3-6 months",
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
        base_weights = horizon.recommended_weights.copy()
        adjustment_factors = []
        for output in agent_outputs:
            agent_type = output['agent_type']
            confidence = output['confidence']
            if confidence > 0.85:
                base_weights[agent_type] *= 1.1
                adjustment_factors.append(f"{output['agent_name']} high confidence")
            elif confidence < 0.60:
                base_weights[agent_type] *= 0.9
                adjustment_factors.append(f"{output['agent_name']} low confidence")
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
        votes = []
        weight_map = {
            'fundamental': weights.fundamental_weight,
            'technical': weights.technical_weight,
            'sentiment': weights.sentiment_weight,
            'risk': weights.risk_weight
        }
        for output in agent_outputs:
            agent_type = output['agent_type']
            weight = weight_map.get(agent_type, 0.25)
            detailed_analysis = {
                "summary": output.get('summary', ''),
                "key_points": output.get('key_points', {}),
                "detailed_metrics": output.get('detailed_metrics', {})
            }
            vote = AgentVote(
                agent_name=output['agent_name'],
                agent_type=agent_type,
                recommendation=output['recommendation'],
                confidence=output['confidence'],
                score=output['score'],
                vote_weight=weight,
                weighted_score=output['score'] * weight,
                detailed_analysis=detailed_analysis
            )
            votes.append(vote)
        return votes
    
    def analyze_voting(self, votes: List[AgentVote]) -> VotingBreakdown:
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
            buy_votes=buy_votes, hold_votes=hold_votes, sell_votes=sell_votes,
            buy_weight=buy_weight, hold_weight=hold_weight, sell_weight=sell_weight,
            consensus_level=consensus_level
        )
    
    def ai_professional_judgment(self, ticker: str, votes: List[AgentVote], breakdown: VotingBreakdown, horizon: InvestmentHorizon, weights: WeightDistribution) -> Dict:
        prompt = f"Analyze {ticker} based on 4 agent votes. Investment horizon: {horizon.duration_description}. Voting: BUY={breakdown.buy_votes}, HOLD={breakdown.hold_votes}, SELL={breakdown.sell_votes}. Return JSON with: final_recommendation (BUY/HOLD/SELL), confidence (0-1), consensus_score (0-100), detailed_reasoning (300+ words), action_plan, key_insights (list), divergent_views (list), risk_disclosure (150+ words)."
        try:
            response_text = self.call_deepseek_api(prompt)
            return json.loads(response_text)
        except:
            return {
                "final_recommendation": "HOLD",
                "confidence": 0.5,
                "consensus_score": 50.0,
                "detailed_reasoning": "AI analysis unavailable",
                "action_plan": "Manual review required",
                "key_insights": ["System limited"],
                "divergent_views": ["Incomplete data"],
                "risk_disclosure": "Investment involves risks. For reference only."
            }
    
    def judge(self, ticker: str, fundamental_output: Dict, technical_output: Dict, sentiment_output: Dict, risk_output: Dict, investment_period: str = "LONG_TERM") -> ArenaJudgeResult:
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
