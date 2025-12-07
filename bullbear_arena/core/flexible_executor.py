"""
Flexible Executor - Executes analysis based on routing decisions

Handles different execution modes:
- Quick answers for simple questions
- Single stock analysis
- Multi-stock comparison
- Market overview
"""

from typing import Dict, List, Any


class FlexibleExecutor:
    """
    Executes agent calls based on routing decisions
    
    Supports:
    - Quick answers: Direct answers for simple factual questions
    - Single stock: Call specific agents for one ticker
    - Comparison: Compare multiple tickers across agents
    - Market overview: General market sentiment
    """
    
    def __init__(self, fundamental_agent, technical_agent, sentiment_agent, risk_agent, arena_judge):
        """
        Initialize executor with agents
        
        Args:
            fundamental_agent: Fundamental analysis agent
            technical_agent: Technical analysis agent
            sentiment_agent: Sentiment analysis agent
            risk_agent: Risk analysis agent
            arena_judge: Arena judge for final decisions
        """
        self.agents = {
            'fundamental': fundamental_agent,
            'technical': technical_agent,
            'sentiment': sentiment_agent,
            'risk': risk_agent
        }
        self.arena_judge = arena_judge
    
    def execute(self, routing) -> Dict[str, Any]:
        """
        Execute analysis based on routing decision
        
        Args:
            routing: RoutingDecision from QuestionRouter
            
        Returns:
            Dict: Analysis results
        """
        execution_mode = routing.execution_mode
        
        if execution_mode == "single":
            return self._execute_single_stock(routing)
        elif execution_mode == "comparison":
            return self._execute_comparison(routing)
        elif execution_mode == "market":
            return self._execute_market_overview(routing)
        else:
            return {"error": f"Unknown execution mode: {execution_mode}"}
    
    def _get_quick_answer(self, question: str, ticker: str, agent_type: str) -> str:
        """
        Get quick answer from specific agent
        
        Args:
            question: User question
            ticker: Stock ticker
            agent_type: Which agent to query (fundamental/technical/sentiment/risk)
            
        Returns:
            str: Quick answer or None if not applicable
        """
        agent = self.agents.get(agent_type)
        if not agent:
            return None
        
        # Check if agent has quick_query method
        if hasattr(agent, 'quick_query'):
            try:
                return agent.quick_query(question, ticker)
            except:
                pass
        
        # Fallback: try to extract answer from full analysis
        try:
            result = agent.get_arena_output(ticker)
            
            question_lower = question.lower()
            
            # Fundamental questions
            if agent_type == 'fundamental':
                metrics = result.get('detailed_metrics', {})
                
                if 'pe' in question_lower:
                    pe = metrics.get('pe_ratio')
                    if pe:
                        return f"{ticker}'s PE ratio is {pe:.2f}x"
                
                if 'roe' in question_lower:
                    roe = metrics.get('roe')
                    if roe:
                        return f"{ticker}'s ROE is {roe:.2f}%"
                
                if 'valuation' in question_lower or 'estimate' in question_lower or '估值' in question_lower:
                    rec = result.get('recommendation', 'N/A')
                    score = result.get('score', 0)
                    summary = result.get('summary', '')
                    # Extract valuation opinion from summary
                    if 'undervalued' in summary.lower():
                        return f"{ticker} appears undervalued based on fundamental analysis (Score: {score:.1f}/100, Recommendation: {rec})"
                    elif 'overvalued' in summary.lower():
                        return f"{ticker} appears overvalued based on fundamental analysis (Score: {score:.1f}/100, Recommendation: {rec})"
                    else:
                        return f"{ticker}'s valuation: {rec} (Score: {score:.1f}/100). {summary[:200]}"
            
            # Technical questions
            elif agent_type == 'technical':
                metrics = result.get('detailed_metrics', {})
                
                if 'rsi' in question_lower:
                    rsi = metrics.get('rsi')
                    if rsi:
                        status = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
                        return f"{ticker}'s RSI is {rsi:.2f} ({status})"
                
                if 'macd' in question_lower:
                    macd_signal = metrics.get('macd_signal')
                    if macd_signal:
                        return f"{ticker}'s MACD signal: {macd_signal}"
                    else:
                        rec = result.get('recommendation', 'N/A')
                        return f"{ticker}'s technical indicators suggest {rec}"
                
                if 'technical' in question_lower or 'indicator' in question_lower or '技术' in question_lower:
                    rec = result.get('recommendation', 'N/A')
                    score = result.get('score', 0)
                    summary = result.get('summary', '')
                    return f"{ticker}'s technical analysis: {rec} (Score: {score:.1f}/100). {summary[:200]}"
            
            # Sentiment questions
            elif agent_type == 'sentiment':
                if 'news' in question_lower or 'sentiment' in question_lower or '新闻' in question_lower or '情绪' in question_lower:
                    rec = result.get('recommendation', 'N/A')
                    score = result.get('score', 0)
                    summary = result.get('summary', '')
                    return f"{ticker}'s market sentiment: {rec} (Score: {score:.1f}/100). {summary[:300]}"
            
            # Risk questions
            elif agent_type == 'risk':
                metrics = result.get('detailed_metrics', {})
                
                if 'volatility' in question_lower or '波动' in question_lower:
                    vol = metrics.get('volatility')
                    if vol:
                        return f"{ticker}'s volatility is {vol:.2f}%"
                
                if 'risk' in question_lower or '风险' in question_lower:
                    rec = result.get('recommendation', 'N/A')
                    score = result.get('score', 0)
                    summary = result.get('summary', '')
                    return f"{ticker}'s risk assessment: {rec} (Score: {score:.1f}/100). {summary[:200]}"
            
            return None
            
        except Exception as e:
            return None
    
    def _execute_single_stock(self, routing) -> Dict[str, Any]:
        """Execute single stock analysis"""
        ticker = routing.tickers[0] if routing.tickers else None
        
        if not ticker:
            return {"error": "No ticker specified"}
        
        question = getattr(routing, 'question', '')
        
        # Try quick answer for single-agent questions
        if question and len(routing.agents_needed) == 1:
            agent_type = routing.agents_needed[0]
            quick_answer = self._get_quick_answer(question, ticker, agent_type)
            
            if quick_answer:
                return {
                    "execution_type": "quick_answer",
                    "ticker": ticker,
                    "agent_type": agent_type,
                    "answer": quick_answer,
                    "summary": quick_answer
                }
        
        # Full analysis for complex or multi-agent questions
        results = {}
        
        # Call requested agents
        for agent_type in routing.agents_needed:
            agent = self.agents.get(agent_type)
            if agent:
                try:
                    results[agent_type] = agent.get_arena_output(ticker)
                except Exception as e:
                    results[agent_type] = {"error": str(e)}
        
        # Generate summary
        summary = self._generate_single_summary(ticker, results, routing.agents_needed)
        
        return {
            "execution_type": "single_stock",
            "ticker": ticker,
            "agents_called": routing.agents_needed,
            "results": results,
            "summary": summary
        }
    
    def _execute_comparison(self, routing) -> Dict[str, Any]:
        """Execute multi-stock comparison"""
        tickers = routing.tickers
        
        if len(tickers) < 2:
            return {"error": "Comparison requires at least 2 tickers"}
        
        question = getattr(routing, 'question', '')
        
        # For focused comparison questions (e.g., "compare MU and AMD's PE")
        if len(routing.agents_needed) == 1:
            agent_type = routing.agents_needed[0]
            comparison_result = self._quick_comparison(question, tickers, agent_type)
            
            if comparison_result:
                return {
                    "execution_type": "quick_comparison",
                    "tickers": tickers,
                    "agent_type": agent_type,
                    "comparison": comparison_result,
                    "summary": comparison_result
                }
        
        # Full comparison for comprehensive analysis
        all_results = {}
        for ticker in tickers:
            ticker_results = {}
            for agent_type in routing.agents_needed:
                agent = self.agents.get(agent_type)
                if agent:
                    try:
                        ticker_results[agent_type] = agent.get_arena_output(ticker)
                    except Exception as e:
                        ticker_results[agent_type] = {"error": str(e)}
            all_results[ticker] = ticker_results
        
        # Generate comparison summary
        comparison_summary = self._generate_comparison_summary(tickers, all_results, routing.agents_needed)
        
        return {
            "execution_type": "comparison",
            "tickers": tickers,
            "agents_called": routing.agents_needed,
            "results": all_results,
            "comparison_summary": comparison_summary
        }
    
    def _quick_comparison(self, question: str, tickers: List[str], agent_type: str) -> str:
        """
        Quick comparison for focused questions
        
        Args:
            question: User question
            tickers: List of tickers to compare
            agent_type: Which agent to use
            
        Returns:
            str: Comparison result
        """
        try:
            agent = self.agents.get(agent_type)
            if not agent:
                return None
            
            # Get results for all tickers
            ticker_results = {}
            for ticker in tickers:
                try:
                    result = agent.get_arena_output(ticker)
                    ticker_results[ticker] = result
                except:
                    ticker_results[ticker] = None
            
            # Extract comparison metric
            question_lower = question.lower()
            
            if agent_type == 'fundamental':
                if 'pe' in question_lower:
                    comparison = []
                    for ticker, result in ticker_results.items():
                        if result:
                            metrics = result.get('detailed_metrics', {})
                            pe = metrics.get('pe_ratio')
                            if pe:
                                comparison.append(f"{ticker}: PE={pe:.2f}x")
                    
                    if comparison:
                        ranked = sorted(comparison)
                        return f"PE ratio comparison: {', '.join(ranked)}"
                
                # Generic fundamental comparison
                comparison = []
                for ticker, result in ticker_results.items():
                    if result:
                        rec = result.get('recommendation', 'N/A')
                        score = result.get('score', 0)
                        comparison.append({
                            'ticker': ticker,
                            'recommendation': rec,
                            'score': score
                        })
                
                comparison.sort(key=lambda x: x['score'], reverse=True)
                result_text = "Fundamental comparison:\n"
                for i, item in enumerate(comparison, 1):
                    result_text += f"{i}. {item['ticker']}: {item['recommendation']} (Score: {item['score']:.1f}/100)\n"
                
                return result_text
            
            # Similar logic for other agent types
            # ...
            
            return None
            
        except Exception as e:
            return None
    
    def _execute_market_overview(self, routing) -> Dict[str, Any]:
        """Execute market overview (mainly sentiment)"""
        results = {}
        question = getattr(routing, 'question', '')
        
        # For market sentiment questions
        if 'sentiment' in routing.agents_needed:
            agent = self.agents.get('sentiment')
            if agent:
                try:
                    # Use SPY as market proxy
                    results['sentiment'] = agent.get_arena_output('SPY')
                    
                    # Extract sentiment summary
                    summary = results['sentiment'].get('summary', '')
                    return {
                        "execution_type": "market_overview",
                        "results": results,
                        "summary": f"Market sentiment: {summary[:300]}"
                    }
                except Exception as e:
                    results['sentiment'] = {"error": str(e)}
        
        summary = "Market overview analysis based on available data"
        
        return {
            "execution_type": "market_overview",
            "results": results,
            "summary": summary
        }
    
    def _generate_single_summary(self, ticker: str, results: Dict, agents_called: List[str]) -> str:
        """Generate summary for single stock analysis"""
        summary_parts = [f"Analysis of {ticker}:"]
        
        for agent_type in agents_called:
            if agent_type in results and 'error' not in results[agent_type]:
                data = results[agent_type]
                rec = data.get('recommendation', 'N/A')
                score = data.get('score', 0)
                summary_parts.append(f"- {agent_type.title()}: {rec} (Score: {score:.1f})")
        
        return " ".join(summary_parts)
    
    def _generate_comparison_summary(self, tickers: List[str], all_results: Dict, agents_called: List[str]) -> Dict:
        """Generate comparison summary across tickers"""
        rankings = {}
        
        for agent_type in agents_called:
            agent_rankings = []
            
            for ticker in tickers:
                if ticker in all_results and agent_type in all_results[ticker]:
                    data = all_results[ticker][agent_type]
                    if 'error' not in data:
                        score = data.get('score', 0)
                        rec = data.get('recommendation', 'N/A')
                        agent_rankings.append({
                            'ticker': ticker,
                            'score': score,
                            'recommendation': rec
                        })
            
            # Sort by score
            agent_rankings.sort(key=lambda x: x['score'], reverse=True)
            rankings[agent_type] = agent_rankings
        
        return {
            "rankings": rankings,
            "tickers_compared": tickers,
            "agents_used": agents_called
        }
