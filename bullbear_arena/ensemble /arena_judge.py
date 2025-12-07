# ============================================================================
# BullBear Arena - 竞技场裁判
# bullbear_arena/ensemble/arena_judge.py
# ============================================================================
"""
Arena Judge - 🏆 最终裁判

角色: 资深财经分析师
职责: 整合4个Agent的分析,进行专业投资决策

投资哲学:
- 长期投资(>1年): 基本面50% + 风险30% + 技术10% + 情绪10%
- 中期投资(3-12月): 技术35% + 基本面30% + 情绪20% + 风险15%
- 短期投资(<3月): 技术45% + 情绪30% + 风险15% + 基本面10%

核心机制:
1. 投资周期智能识别
2. 动态权重分配
3. 完整数据展示
4. AI深度分析
5. 专业风险提示
"""

import json
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# ============================================================================
# 数据模型定义
# ============================================================================

class InvestmentHorizon(BaseModel):
    """投资时间周期"""
    horizon_type: str = Field(description="投资周期: LONG_TERM/MEDIUM_TERM/SHORT_TERM")
    duration_description: str = Field(description="周期描述")
    data_timeframe_used: str = Field(description="使用的数据时间范围")
    recommended_weights: Dict[str, float] = Field(description="推荐权重分配")

class AgentVote(BaseModel):
    """Agent投票"""
    agent_name: str
    agent_type: str
    recommendation: str
    confidence: float
    score: float
    vote_weight: float
    weighted_score: float
    detailed_analysis: Dict = Field(description="详细分析数据")

class VotingBreakdown(BaseModel):
    """投票分解"""
    buy_votes: int
    hold_votes: int
    sell_votes: int
    buy_weight: float
    hold_weight: float
    sell_weight: float
    consensus_level: str

class WeightDistribution(BaseModel):
    """权重分配"""
    fundamental_weight: float
    technical_weight: float
    sentiment_weight: float
    risk_weight: float
    weighting_rationale: str = Field(description="权重分配理由")
    adjustment_factors: List[str] = Field(description="调整因素")

class ArenaJudgeResult(BaseModel):
    """竞技场裁判结果"""
    ticker: str
    analysis_date: str
    investment_horizon: InvestmentHorizon
    final_recommendation: str
    confidence: float
    consensus_score: float
    agent_votes: List[AgentVote]
    voting_breakdown: VotingBreakdown
    weight_distribution: WeightDistribution
    detailed_reasoning: str = Field(description="详细推理过程")
    action_plan: str = Field(description="行动计划")
    risk_disclosure: str = Field(description="风险提示")
    key_insights: List[str]
    divergent_views: List[str]

# ============================================================================
# Arena Judge类
# ============================================================================

class ArenaJudge:
    """
    竞技场裁判 - 专业财经分析师
    
    投资哲学:
    - 长期投资(>1年): 基本面50% + 风险30% + 技术10% + 情绪10%
    - 中期投资(3-12月): 技术35% + 基本面30% + 情绪20% + 风险15%
    - 短期投资(<3月): 技术45% + 情绪30% + 风险15% + 基本面10%
    """
    
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com/v1/chat/completions"):
        """
        初始化Arena Judge
        
        Args:
            api_key: DeepSeek API密钥
            api_url: API端点
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = "deepseek-chat"
        self.agent_name = "🏆 Arena Judge"
    
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
        
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    
    def determine_investment_horizon(
        self,
        ticker: str,
        investment_period: str = "LONG_TERM"
    ) -> InvestmentHorizon:
        """
        确定投资时间周期和数据范围
        
        Args:
            ticker: 股票代码
            investment_period: 投资周期 (LONG_TERM/MEDIUM_TERM/SHORT_TERM)
            
        Returns:
            InvestmentHorizon: 投资周期配置
        """
        horizon_configs = {
            "LONG_TERM": {
                "duration": "1年以上长期投资",
                "data_range": "过去3-5年数据",
                "weights": {
                    "fundamental": 0.50,
                    "risk": 0.30,
                    "technical": 0.10,
                    "sentiment": 0.10
                }
            },
            "MEDIUM_TERM": {
                "duration": "3-12个月中期投资",
                "data_range": "过去1-2年数据",
                "weights": {
                    "technical": 0.35,
                    "fundamental": 0.30,
                    "sentiment": 0.20,
                    "risk": 0.15
                }
            },
            "SHORT_TERM": {
                "duration": "3个月内短期投资",
                "data_range": "过去3-6个月数据",
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
        智能权重分配 (基于投资周期)
        
        Args:
            agent_outputs: 4个Agent的输出
            horizon: 投资周期配置
            
        Returns:
            WeightDistribution: 权重分配结果
        """
        base_weights = horizon.recommended_weights.copy()
        adjustment_factors = []
        
        # 置信度调整
        for output in agent_outputs:
            agent_type = output['agent_type']
            confidence = output['confidence']
            
            if confidence > 0.85:
                base_weights[agent_type] *= 1.1
                adjustment_factors.append(f"{output['agent_name']}置信度高(+10%)")
            elif confidence < 0.60:
                base_weights[agent_type] *= 0.9
                adjustment_factors.append(f"{output['agent_name']}置信度低(-10%)")
        
        # 标准化
        total = sum(base_weights.values())
        final_weights = {k: v/total for k, v in base_weights.items()}
        
        rationale = f"基于{horizon.duration_description},采用专业权重配置"
        
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
        收集详细投票 (包含完整分析数据)
        
        Args:
            agent_outputs: 4个Agent的输出
            weights: 权重分配
            
        Returns:
            List[AgentVote]: 投票列表
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
        分析投票分布
        
        Args:
            votes: 投票列表
            
        Returns:
            VotingBreakdown: 投票分解结果
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
        AI专业裁决 (DeepSeek作为资深分析师)
        
        Args:
            ticker: 股票代码
            votes: 投票列表
            breakdown: 投票分解
            horizon: 投资周期
            weights: 权重分配
            
        Returns:
            Dict: AI分析结果
        """
        prompt = f"""你是一位拥有20年经验的资深财经分析师,现在需要对 {ticker} 进行专业投资分析。

【投资背景】
投资周期: {horizon.duration_description}
数据时间范围: {horizon.data_timeframe_used}
分析日期: {datetime.now().strftime("%Y-%m-%d")}

【权重配置逻辑】
{weights.weighting_rationale}
- 基本面权重: {weights.fundamental_weight:.1%}
- 技术面权重: {weights.technical_weight:.1%}
- 情绪面权重: {weights.sentiment_weight:.1%}
- 风险面权重: {weights.risk_weight:.1%}

调整因素: {', '.join(weights.adjustment_factors) if weights.adjustment_factors else '无'}

【4个专业Agent的完整分析】
"""
        
        for vote in votes:
            prompt += f"\n{'='*60}\n"
            prompt += f"{vote.agent_name} ({vote.agent_type})\n"
            prompt += f"{'='*60}\n"
            prompt += f"建议: {vote.recommendation}\n"
            prompt += f"评分: {vote.score:.1f}/100\n"
            prompt += f"置信度: {vote.confidence:.1%}\n"
            prompt += f"权重: {vote.vote_weight:.1%}\n"
            prompt += f"\n分析摘要:\n{vote.detailed_analysis.get('summary', '')}\n"
            
            if 'key_points' in vote.detailed_analysis:
                prompt += f"\n关键点:\n{json.dumps(vote.detailed_analysis['key_points'], indent=2, ensure_ascii=False)}\n"
            
            if 'detailed_metrics' in vote.detailed_analysis:
                prompt += f"\n详细指标:\n{json.dumps(vote.detailed_analysis['detailed_metrics'], indent=2, ensure_ascii=False)}\n"
        
        prompt += f"""
{'='*60}
【投票统计】
- BUY: {breakdown.buy_votes}票 (权重{breakdown.buy_weight:.1%})
- HOLD: {breakdown.hold_votes}票 (权重{breakdown.hold_weight:.1%})
- SELL: {breakdown.sell_votes}票 (权重{breakdown.sell_weight:.1%})
- 共识程度: {breakdown.consensus_level}

【你的任务】
作为资深分析师,请综合以上所有信息,进行深度专业分析:

1. 仔细审查每个Agent提供的详细数据
2. 根据投资周期({horizon.duration_description})判断各维度的重要性
3. 识别Agent之间的分歧点和共识点
4. 给出专业的最终投资建议

分析要点:
- 长线投资注重基本面和风险Agent的意见
- 中线投资着重考虑技术面的支撑位和阻力位
- 短线投资考虑新闻的及时性,加大sentiment agent权重

请以JSON格式返回:
{{
  "final_recommendation": "BUY/HOLD/SELL",
  "confidence": 0.85,
  "consensus_score": 75.0,
  "detailed_reasoning": "详细推理过程,至少300字,包括:
    - 为什么选择这个建议
    - 各Agent分析的权衡
    - 关键决策因素
    - 风险收益分析",
  "action_plan": "具体行动建议,包括仓位、入场时机、止损位等",
  "key_insights": ["关键洞察1", "关键洞察2", "关键洞察3"],
  "divergent_views": ["分歧观点1", "分歧观点2"],
  "risk_disclosure": "针对{ticker}的风险提示,至少150字"
}}

要求:
- detailed_reasoning必须详细,体现专业分析师的思考过程
- 必须明确说明为何采用或不采用某个Agent的建议
- 如果是DIVIDED共识,必须详细解释决策逻辑
- risk_disclosure必须具体,不要套话
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
                "detailed_reasoning": "AI分析暂时不可用,建议人工复核",
                "action_plan": "等待AI系统恢复",
                "key_insights": ["系统受限"],
                "divergent_views": ["数据不完整"],
                "risk_disclosure": "投资有风险,决策需谨慎。本分析仅供参考,不构成投资建议。"
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
        执行专业裁决
        
        Args:
            ticker: 股票代码
            fundamental_output: 基本面Agent输出
            technical_output: 技术面Agent输出
            sentiment_output: 情绪面Agent输出
            risk_output: 风险面Agent输出
            investment_period: 投资周期 (LONG_TERM/MEDIUM_TERM/SHORT_TERM)
            
        Returns:
            ArenaJudgeResult: 最终裁决结果
        """
        agent_outputs = [
            fundamental_output,
            technical_output,
            sentiment_output,
            risk_output
        ]
        
        # 1. 确定投资周期
        horizon = self.determine_investment_horizon(ticker, investment_period)
        
        # 2. 智能权重分配
        weights = self.calculate_smart_weights(agent_outputs, horizon)
        
        # 3. 收集详细投票
        votes = self.collect_detailed_votes(agent_outputs, weights)
        
        # 4. 分析投票
        breakdown = self.analyze_voting(votes)
        
        # 5. AI专业裁决
        ai_result = self.ai_professional_judgment(
            ticker, votes, breakdown, horizon, weights
        )
        
        # 6. 组装结果
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
