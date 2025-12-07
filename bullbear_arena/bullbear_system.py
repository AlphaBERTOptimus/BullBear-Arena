# ============================================================================
# BullBear Arena - 统一系统入口
# bullbear_arena/bullbear_system.py
# ============================================================================
"""
BullBear Arena 统一系统入口

这是用户唯一需要交互的接口

支持两种模式:
1. ask() - 自由提问模式
2. analyze() - 完整分析模式
"""

from typing import Dict, Optional

# ============================================================================
# BullBear System
# ============================================================================

class BullBearSystem:
    """
    BullBear Arena 统一系统入口
    
    用户接口:
    - system.ask(question) - 自由提问
    - system.analyze(ticker, period) - 完整分析
    
    内部流程:
    1. QuestionRouter 分析问题
    2. FlexibleExecutor 调用Agent
    3. 返回结果
    """
    
    def __init__(self, api_key: str):
        """
        初始化BullBear系统
        
        Args:
            api_key: DeepSeek API密钥
        """
        from bullbear_arena.core.question_router import QuestionRouter
        from bullbear_arena.core.flexible_executor import FlexibleExecutor
        from bullbear_arena.agents.fundamental_agent import FundamentalAgent
        from bullbear_arena.agents.technical_agent import TechnicalAgent
        from bullbear_arena.agents.sentiment_agent import SentimentAgent
        from bullbear_arena.agents.risk_agent import RiskAgent
        from bullbear_arena.ensemble.arena_judge import ArenaJudge
        
        self.api_key = api_key
        
        # 初始化4个Agent
        self.fundamental_agent = FundamentalAgent(api_key)
        self.technical_agent = TechnicalAgent(api_key)
        self.sentiment_agent = SentimentAgent(api_key)
        self.risk_agent = RiskAgent(api_key)
        self.arena_judge = ArenaJudge(api_key)
        
        # 初始化核心模块
        self.question_router = QuestionRouter(api_key)
        self.executor = FlexibleExecutor(
            fundamental_agent=self.fundamental_agent,
            technical_agent=self.technical_agent,
            sentiment_agent=self.sentiment_agent,
            risk_agent=self.risk_agent,
            arena_judge=self.arena_judge
        )
    
    def ask(self, question: str, verbose: bool = False) -> Dict:
        """
        自由提问模式
        
        Args:
            question: 用户问题
            verbose: 是否打印详细过程
            
        Returns:
            Dict: 分析结果
        
        Examples:
            system.ask("MU的PE怎么样?")
            system.ask("NVDA技术指标如何?")
            system.ask("比较MU和AMD")
            system.ask("最近市场怎么样?")
        """
        # Step 1: 问题路由
        routing = self.question_router.analyze_question(question)
        
        # Step 2: 执行分析
        result = self.executor.execute(routing)
        
        return {
            "question": question,
            "routing": routing.model_dump() if hasattr(routing, 'model_dump') else routing,
            "result": result
        }
    
    def analyze(
        self,
        ticker: str,
        investment_period: str = "LONG_TERM",
        verbose: bool = False
    ) -> Dict:
        """
        完整分析模式
        
        Args:
            ticker: 股票代码
            investment_period: 投资周期 (LONG_TERM/MEDIUM_TERM/SHORT_TERM)
            verbose: 是否打印详细过程
            
        Returns:
            Dict: 完整分析结果
        
        Examples:
            system.analyze("AAPL", "LONG_TERM")
            system.analyze("TSLA", "MEDIUM_TERM")
        """
        # 获取4个Agent的输出
        fundamental_output = self.fundamental_agent.get_arena_output(ticker)
        technical_output = self.technical_agent.get_arena_output(ticker)
        sentiment_output = self.sentiment_agent.get_arena_output(ticker)
        risk_output = self.risk_agent.get_arena_output(ticker)
        
        # Arena Judge最终裁决
        judge_result = self.arena_judge.judge(
            ticker=ticker,
            fundamental_output=fundamental_output,
            technical_output=technical_output,
            sentiment_output=sentiment_output,
            risk_output=risk_output,
            investment_period=investment_period
        )
        
        return {
            "ticker": ticker,
            "investment_period": investment_period,
            "agent_results": {
                "fundamental": fundamental_output,
                "technical": technical_output,
                "sentiment": sentiment_output,
                "risk": risk_output
            },
            "judge_result": judge_result.model_dump() if hasattr(judge_result, 'model_dump') else judge_result
        }
