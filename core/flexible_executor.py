# ============================================================================
# BullBear Arena - 灵活执行器
# bullbear_arena/core/flexible_executor.py
# ============================================================================
"""
灵活执行器

职责: 根据QuestionRouter的分析结果,实际调用对应的Agent

支持:
1. 单Agent查询
2. 多Agent查询
3. 完整分析流程
4. 多股票对比 (2-5只)
5. 宽泛市场问题
"""

from typing import Dict, List

# ============================================================================
# 灵活执行器
# ============================================================================

class FlexibleExecutor:
    """
    灵活执行器
    
    根据问题路由结果,动态调用对应的Agent
    """
    
    def __init__(
        self,
        fundamental_agent,
        technical_agent,
        sentiment_agent,
        risk_agent,
        arena_judge=None
    ):
        """
        初始化执行器
        
        Args:
            fundamental_agent: 基本面Agent实例
            technical_agent: 技术面Agent实例
            sentiment_agent: 情绪面Agent实例
            risk_agent: 风险面Agent实例
            arena_judge: Arena Judge实例 (可选)
        """
        self.fundamental_agent = fundamental_agent
        self.technical_agent = technical_agent
        self.sentiment_agent = sentiment_agent
        self.risk_agent = risk_agent
        self.arena_judge = arena_judge
        
        # Agent映射
        self.agent_map = {
            "fundamental": self.fundamental_agent,
            "technical": self.technical_agent,
            "sentiment": self.sentiment_agent,
            "risk": self.risk_agent
        }
        
        # 配置
        self.max_comparison_stocks = 5
    
    def execute(self, routing) -> Dict:
        """
        执行分析
        
        Args:
            routing: QuestionAnalysis对象
            
        Returns:
            Dict: 执行结果
        """
        # 情况1: 宽泛市场问题
        if not routing.tickers and routing.fallback_strategy == "sentiment_agent":
            return self._handle_market_general(routing)
        
        # 情况2: 完整分析
        if routing.question_type == "full_analysis":
            return self._handle_full_analysis(routing)
        
        # 情况3: 单Agent查询
        if len(routing.agents_needed) == 1 and len(routing.tickers) == 1:
            return self._handle_single_agent(routing)
        
        # 情况4: 多Agent查询 (单股票)
        if len(routing.agents_needed) > 1 and len(routing.tickers) == 1:
            return self._handle_multiple_agents(routing)
        
        # 情况5: 对比分析
        if routing.question_type == "comparison" or len(routing.tickers) > 1:
            return self._handle_comparison(routing)
        
        # 情况6: 无法处理
        return self._handle_unsupported(routing)
    
    def _handle_market_general(self, routing) -> Dict:
        """处理宽泛市场问题"""
        return {
            "execution_type": "market_general",
            "question": routing.specific_request,
            "response": f"关于'{routing.specific_request}'的市场分析",
            "agents_used": ["sentiment"],
            "data": {
                "market_sentiment": "NEUTRAL",
                "key_news": ["市场整体稳定", "投资者情绪中性"],
                "summary": "市场整体分析摘要..."
            }
        }
    
    def _handle_full_analysis(self, routing) -> Dict:
        """处理完整分析"""
        if not routing.tickers:
            return {"error": "完整分析需要指定股票代码"}
        
        ticker = routing.tickers[0]
        
        # 调用所有4个Agent
        agent_results = {}
        for agent_type in ["fundamental", "technical", "sentiment", "risk"]:
            agent = self.agent_map[agent_type]
            agent_results[agent_type] = agent.get_arena_output(ticker)
        
        # 如果有Arena Judge,进行最终裁决
        if self.arena_judge:
            judge_result = self.arena_judge.judge(
                ticker=ticker,
                fundamental_output=agent_results["fundamental"],
                technical_output=agent_results["technical"],
                sentiment_output=agent_results["sentiment"],
                risk_output=agent_results["risk"],
                investment_period=routing.time_horizon
            )
            
            return {
                "execution_type": "full_analysis",
                "ticker": ticker,
                "agents_used": ["fundamental", "technical", "sentiment", "risk"],
                "agent_results": agent_results,
                "judge_result": judge_result.model_dump() if hasattr(judge_result, 'model_dump') else judge_result,
                "final_recommendation": judge_result.final_recommendation,
                "confidence": judge_result.confidence
            }
        
        return {
            "execution_type": "full_analysis",
            "ticker": ticker,
            "agents_used": ["fundamental", "technical", "sentiment", "risk"],
            "agent_results": agent_results,
            "note": "需要Arena Judge进行最终裁决"
        }
    
    def _handle_single_agent(self, routing) -> Dict:
        """处理单Agent查询"""
        agent_type = routing.agents_needed[0]
        ticker = routing.tickers[0]
        
        agent = self.agent_map[agent_type]
        result = agent.get_arena_output(ticker)
        
        return {
            "execution_type": "single_agent",
            "ticker": ticker,
            "agent_type": agent_type,
            "agents_used": [agent_type],
            "result": result,
            "summary": result.get("summary", "")
        }
    
    def _handle_multiple_agents(self, routing) -> Dict:
        """处理多Agent查询"""
        ticker = routing.tickers[0]
        
        agent_results = {}
        for agent_type in routing.agents_needed:
            agent = self.agent_map[agent_type]
            agent_results[agent_type] = agent.get_arena_output(ticker)
        
        return {
            "execution_type": "multiple_agents",
            "ticker": ticker,
            "agents_used": routing.agents_needed,
            "agent_results": agent_results
        }
    
    def _handle_comparison(self, routing) -> Dict:
        """处理对比分析 (支持2-5只股票)"""
        if len(routing.tickers) < 2:
            return {"error": "对比分析需要至少2只股票"}
        
        if len(routing.tickers) > self.max_comparison_stocks:
            return {
                "error": f"对比分析最多支持{self.max_comparison_stocks}只股票"
            }
        
        # 对每只股票调用所需的Agent
        comparison_results = {}
        for ticker in routing.tickers:
            ticker_results = {}
            for agent_type in routing.agents_needed:
                agent = self.agent_map[agent_type]
                ticker_results[agent_type] = agent.get_arena_output(ticker)
            comparison_results[ticker] = ticker_results
        
        # 生成对比摘要
        comparison_summary = self._generate_comparison_summary(
            comparison_results, 
            routing.agents_needed
        )
        
        return {
            "execution_type": "comparison",
            "tickers": routing.tickers,
            "stock_count": len(routing.tickers),
            "agents_used": routing.agents_needed,
            "comparison_results": comparison_results,
            "comparison_summary": comparison_summary
        }
    
    def _generate_comparison_summary(
        self, 
        comparison_results: Dict, 
        agents_used: List[str]
    ) -> Dict:
        """生成对比摘要"""
        summary = {
            "rankings": {},
            "best_performers": {},
            "worst_performers": {}
        }
        
        for agent_type in agents_used:
            scores = {
                ticker: results[agent_type].get("score", 0)
                for ticker, results in comparison_results.items()
            }
            
            sorted_tickers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            summary["rankings"][agent_type] = [
                {"ticker": ticker, "score": score}
                for ticker, score in sorted_tickers
            ]
            
            if sorted_tickers:
                summary["best_performers"][agent_type] = sorted_tickers[0][0]
                summary["worst_performers"][agent_type] = sorted_tickers[-1][0]
        
        return summary
    
    def _handle_unsupported(self, routing) -> Dict:
        """处理不支持的请求"""
        return {
            "execution_type": "unsupported",
            "error": "无法处理此类问题",
            "suggestion": "请重新表述您的问题"
        }
