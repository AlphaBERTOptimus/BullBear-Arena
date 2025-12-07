# ============================================================================
# BullBear Arena - Arena Judge
# bullbear_arena/ensemble/arena_judge.py
# ============================================================================
"""
Arena Judge - Final Arbiter

Role: Senior financial analyst
Responsibility: Integrate analysis from 4 agents for professional investment decisions

Investment Philosophy:
- Long-term (>1 year): Fundamental 50% + Risk 30% + Technical 10% + Sentiment 10%
- Medium-term (3-12 months): Technical 35% + Fundamental 30% + Sentiment 20% + Risk 15%
- Short-term (<3 months): Technical 45% + Sentiment 30% + Risk 15% + Fundamental 10%

Core Mechanisms:
1. Smart investment horizon detection
2. Dynamic weight allocation
3. Complete data presentation
4. AI deep analysis
5. Professional risk disclosure
"""

import json
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# ============================================================================
# Data Models
# ============================================================================

class InvestmentHorizon(BaseModel):
    """Investment time horizon"""
    horizon_type: str = Field(description="Investment horizon: LONG_TERM/MEDIUM_TERM/SHORT_TERM")
    duration_description: str = Field(description="Duration description")
    data_timeframe_used: str = Field(description="Data timeframe used")
    recommended_weights: Dict[str, float] = Field(description="Recommended weight distribution")

class AgentVote(BaseModel):
    """Agent vote"""
    agent_name: str
    agent_type: str
    recommendation: str
    confidence: float
    score: float
    vote_weight: float
    weighted_score: float
    detailed_analysis: Dict = Field(description="Detailed analysis data")

class VotingBreakdown(BaseModel):
    """Voting breakdown"""
    buy_votes: int
    hold_votes: int
    sell_votes: int
    buy_weight: float
    hold_weight: float
    sell_weight: float
    consensus_level: str

class WeightDistribution(BaseModel):
    """Weight distribution"""
    fundamental_weight: float
    technical_weight: float
    sentiment_weight: float
    risk_weight: float
    weighting_rationale: str = Field(description="Weight allocation rationale")
    adjustment_factors: List[str] = Field(description="Adjustment factors")

class ArenaJudgeResult(BaseModel):
    """Arena Judge result"""
    ticker: str
    analysis_date: str
    investment_horizon: InvestmentHorizon
    final_recommendation: str
    confidence: float
    consensus_score: float
    agent_votes: List[AgentVote]
    voting_breakdown: VotingBreakdown
    weight_distribution: WeightDistribution
    detailed_reasoning: str = Field(description="Detailed reasoning")
    action_plan: str = Field(description="Action plan")
    risk_disclosure: str = Field(description="Risk disclosure")
    key_insights: List[str]
    divergent_views: List[str]

# ============================================================================
# Arena Judge Class
# ============================================================================

class ArenaJudge:
    """
    Arena Judge - Professional financial analyst
    
    Investment Philosophy:
    - Long-term (>1 year): Fundamental 50% + Risk 30% + Technical 10% + Sentiment 10%
    - Medium-term (3-12 months): Technical 35% + Fundamental 30% + Sentiment 20% + Risk 15%
    - Short-term (<3 months): Technical 45% + Sentiment 30% + Risk 15% + Fundamental 10%
    """
    
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com/v1/chat/completions"):
        """
        Initialize Arena Judge
        
        Args:
            api_key: DeepSeek API key
            api_url: API endpoint
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = "deepseek-chat"
        self.agent_name = "Arena Judge"
    
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
        
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    
    def determine_investment_horizon(
        self,
        ticker: str,
        investment_period: str = "LONG_TERM"
    ) -> InvestmentHorizon:
        """
        Determine investment horizon and data range
        
        Args:
            ticker: Stock ticker
            investment_period: Investment period (LONG_TERM/MEDIUM_TERM/SHORT_TERM)
            
        Returns:
            InvestmentHorizon: Investment horizon configuration
        """
        horizon_configs = {
            "LONG_TERM": {
                "duration": "Long-term investment (>1 year)",
                "data_range": "Past 3-5 years data",
                "weights": {
                    "fundamental": 0.50,
                    "risk": 0.30,
                    "technical": 0.10,
                    "sentiment": 0.10
                }
            },
            "MEDIUM_TERM": {
                "duration": "Medium-term investment (3-12 months)",
                "data_range": "Past 1-2 years data",
                "weights": {
                    "technical": 0.35,
                    "fundamental": 0.30,
                    "sentiment": 0.20,
                    "risk": 0.15
                }
            },
            "SHORT_TERM": {
                "duration": "Short-term investment (<3 months)",
                "data_range": "Past 3-6 months data",
                "weights": {
                    "technical": 0.45,
                    "sentiment": 0.30,
                    "risk": 0.15,
                    "fundamental": 0.10
                }
            }
        }
        
        config = horizon_configs.get(investment_period, horizon_configs["LONG_TERM"])
        
        return InvestmentHorizon(
            horizon_type=investment_period,
            duration_description=config["duration"],
            data_timeframe_used=config["data_range"],
            recommended_weights=config["weights"]
        )
    
    def calculate_smart_weights(
        self,
        agent_outputs: List[Dict],
        horizon: InvestmentHorizon
    ) -> WeightDistribution:
        """
        Smart weight allocation (based on investment horizon)
        
        Args:
            agent_outputs: Outputs from 4 agents
            horizon: Investment horizon configuration
            
        Returns:
            WeightDistribution: Weight distribution result
        """
        base_weights = horizon.recommended_weights.copy()
        adjustment_factors = []
        
        for output in agent_outputs:
            agent_type = output['agent_type']
            confidence = output['confidence']
            
            if confidence > 0.85:
                base_weights[agent_type] *= 1.1
                adjustment_factors.append(f"{output['agent_name']} high confidence (+10%)")
            elif confidence < 0.60:
                base_weights[agent_type] *= 0.9
                adjustment_factors.append(f"{output['agent_name']} low confidence (-10%)")
        
        total = sum(base_weights.values())
        final_weights = {k: v/total for k, v in base_weights.items()}
        
        rationale = f"Based on {horizon.duration_description}, using professional weight configuration"
        
        return WeightDistribution(
            fundamental_weight=final_weights['fundamental'],
            technical_weight=final_weights['technical'],
            sentiment_weight=final_weights['sentiment'],
            risk_weight=final_weights['risk'],
            weighting_rationale=rationale,
            adjustment_factors=adjustment_factors
        )
    
    def collect_detailed_votes(
        self,
        agent_outputs: List[Dict],
        weights: WeightDistribution
    ) -> List[AgentVote]:
        """
        Collect detailed votes (including complete analysis data)
        
        Args:
            agent_outputs: Outputs from 4 agents
            weights: Weight distribution
            
        Returns:
            List[AgentVote]: Vote list
        """
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
        """
        Analyze voting distribution
        
        Args:
            votes: Vote list
            
        Returns:
            VotingBreakdown: Voting breakdown result
        """
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
    
    def ai_professional_judgment(
        self,
        ticker: str,
        votes: List[AgentVote],
        breakdown: VotingBreakdown,
        horizon: InvestmentHorizon,
        weights: WeightDistribution
    ) -> Dict:
        """
        AI professional judgment (DeepSeek as senior analyst)
        
        Args:
            ticker: Stock ticker
            votes: Vote list
            breakdown: Voting breakdown
            horizon: Investment horizon
            weights: Weight distribution
            
        Returns:
            Dict: AI analysis result
        """
        prompt = f"""You are a senior financial analyst with 20 years of experience. Provide professional investment analysis for {ticker}.

Investment Background:
Investment Horizon: {horizon.duration_description}
Data Timeframe: {horizon.data_timeframe_used}
Analysis Date: {datetime.now().strftime("%Y-%m-%d")}

Weight Configuration:
{weights.weighting_rationale}
- Fundamental Weight: {weights.fundamental_weight:.1%}
- Technical Weight: {weights.technical_weight:.1%}
- Sentiment Weight: {weights.sentiment_weight:.1%}
- Risk Weight: {weights.risk_weight:.1%}

Adjustment Factors: {', '.join(weights.adjustment_factors) if weights.adjustment_factors else 'None'}

Complete Analysis from 4 Professional Agents:
"""
        
        for vote in votes:
            prompt += f"\n{'='*60}\n"
            prompt += f"{vote.agent_name} ({vote.agent_type})\n"
            prompt += f"{'='*60}\n"
            prompt += f"Recommendation: {vote.recommendation}\n"
            prompt += f"Score: {vote.score:.1f}/100\n"
            prompt += f"Confidence: {vote.confidence:.1%}\n"
            prompt += f"Weight: {vote.vote_weight:.1%}\n"
            prompt += f"\nAnalysis Summary:\n{vote.detailed_analysis.get('summary', '')}\n"
            
            if 'key_points' in vote.detailed_analysis:
                prompt += f"\nKey Points:\n{json.dumps(vote.detailed_analysis['key_points'], indent=2)}\n"
            
            if 'detailed_metrics' in vote.detailed_analysis:
                prompt += f"\nDetailed Metrics:\n{json.dumps(vote.detailed_analysis['detailed_metrics'], indent=2)}\n"
        
        prompt += f"""
{'='*60}
Voting Statistics:
- BUY: {breakdown.buy_votes} votes (weight {breakdown.buy_weight:.1%})
- HOLD: {breakdown.hold_votes} votes (weight {breakdown.hold_weight:.1%})
- SELL: {breakdown.sell_votes} votes (weight {breakdown.sell_weight:.1%})
- Consensus Level: {breakdown.consensus_level}

Your Task:
Provide in-depth professional analysis based on all above information:

1. Carefully review detailed data from each agent
2. Judge importance of each dimension based on investment horizon
3. Identify points of divergence and consensus among agents
4. Provide professional final investment recommendation

Return in JSON format:
{{
  "final_recommendation": "BUY/HOLD/SELL",
  "confidence": 0.85,
  "consensus_score": 75.0,
  "detailed_reasoning": "Detailed reasoning (at least 300 words)",
  "action_plan": "Specific action recommendations",
  "key_insights": ["Key insight 1", "Key insight 2", "Key insight 3"],
  "divergent_views": ["Divergent view 1", "Divergent view 2"],
  "risk_disclosure": "Risk disclosure specific to {ticker} (at least 150 words)"
}}

Requirements:
- detailed_reasoning must be comprehensive
- Must clearly explain why certain agent recommendations were adopted or not
- If DIVIDED consensus, must explain decision logic in detail
- risk_disclosure must be specific, not generic
"""
        
        try:
            response_text = self.call_deepseek_api(prompt)
            ai_result = json.loads(response_text)
            return ai_result
        except Exception as e:
            return {
                "final_recommendation": "HOLD",
                "confidence": 0.5,
                "consensus_score": 50.0,
                "detailed_reasoning": "AI analysis temporarily unavailable, recommend manual review",
                "action_plan": "Wait for AI system recovery",
                "key_insights": ["System limited"],
                "divergent_views": ["Incomplete data"],
                "risk_disclosure": "Investment involves risks. This analysis is for reference only."
            }
    
    def judge(
        self,
        ticker: str,
        fundamental_output: Dict,
        technical_output: Dict,
        sentiment_output: Dict,
        risk_output: Dict,
        investment_period: str = "LONG_TERM"
    ) -> ArenaJudgeResult:
        """
        Execute professional judgment
        
        Args:
            ticker: Stock ticker
            fundamental_output: Fundamental agent output
            technical_output: Technical agent output
            sentiment_output: Sentiment agent output
            risk_output: Risk agent output
            investment_period: Investment period (LONG_TERM/MEDIUM_TERM/SHORT_TERM)
            
        Returns:
            ArenaJudgeResult: Final judgment result
        """
        agent_outputs = [
            fundamental_output,
            technical_output,
            sentiment_output,
            risk_output
        ]
        
        horizon = self.determine_investment_horizon(ticker, investment_period)
        weights = self.calculate_smart_weights(agent_outputs, horizon)
        votes = self.collect_detailed_votes(agent_outputs, weights)
        breakdown = self.analyze_voting(votes)
        ai_result = self.ai_professional_judgment(ticker, votes, breakdown, horizon, weights)
        
        result = ArenaJudgeResult(
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
        
        return result
