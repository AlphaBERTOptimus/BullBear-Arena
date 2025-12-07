"""
BullBear Arena - Unified System Entry

This is the main interface for users to interact with the system.

Supports two modes:
1. ask() - Free question mode (requires QuestionRouter)
2. analyze() - Complete analysis mode (works standalone)
"""

from typing import Dict


class BullBearSystem:
    """
    BullBear Arena Unified System
    
    Main interface:
    - system.ask(question) - Free questions
    - system.analyze(ticker, period) - Complete analysis
    """
    
    def __init__(self, api_key: str):
        """
        Initialize BullBear System
        
        Args:
            api_key: DeepSeek API key
        """
        # Import agents using relative imports
        from .agents.fundamental_agent import FundamentalAgent
        from .agents.technical_agent import TechnicalAgent
        from .agents.sentiment_agent import SentimentAgent
        from .agents.risk_agent import RiskAgent
        from .agents.arena_judge import ArenaJudge
        
        self.api_key = api_key
        
        # Initialize 4 agents
        self.fundamental_agent = FundamentalAgent(api_key)
        self.technical_agent = TechnicalAgent(api_key)
        self.sentiment_agent = SentimentAgent(api_key)
        self.risk_agent = RiskAgent(api_key)
        self.arena_judge = ArenaJudge(api_key)
        
        # Try to import core modules (optional for now)
        try:
            from .core.question_router import QuestionRouter
            from .core.flexible_executor import FlexibleExecutor
            
            self.question_router = QuestionRouter(api_key)
            self.executor = FlexibleExecutor(
                fundamental_agent=self.fundamental_agent,
                technical_agent=self.technical_agent,
                sentiment_agent=self.sentiment_agent,
                risk_agent=self.risk_agent,
                arena_judge=self.arena_judge
            )
            self.has_core_modules = True
        except ImportError:
            self.question_router = None
            self.executor = None
            self.has_core_modules = False
    
    def ask(self, question: str, verbose: bool = False) -> Dict:
        """
        Free question mode (requires core modules)
        
        Args:
            question: User question
            verbose: Print detailed process
            
        Returns:
            Dict: Analysis result
        """
        if not self.has_core_modules:
            return {
                "error": "Free question mode requires QuestionRouter and FlexibleExecutor modules",
                "suggestion": "Use analyze() method for direct stock analysis"
            }
        
        # Step 1: Question routing
        routing = self.question_router.analyze_question(question)
        
        # Step 2: Execute analysis
        result = self.executor.execute(routing)
        
        return {
            "question": question,
            "routing": routing.model_dump() if hasattr(routing, 'model_dump') else routing,
            "result": result
        }
    
    def analyze(self, ticker: str, investment_period: str = "LONG_TERM", verbose: bool = False) -> Dict:
        """
        Complete analysis mode (works standalone)
        
        Args:
            ticker: Stock ticker symbol
            investment_period: LONG_TERM / MEDIUM_TERM / SHORT_TERM
            verbose: Print detailed process
            
        Returns:
            Dict: Complete analysis result
        """
        # Get outputs from 4 agents
        fundamental_output = self.fundamental_agent.get_arena_output(ticker)
        technical_output = self.technical_agent.get_arena_output(ticker)
        sentiment_output = self.sentiment_agent.get_arena_output(ticker)
        risk_output = self.risk_agent.get_arena_output(ticker)
        
        # Arena Judge final judgment
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
