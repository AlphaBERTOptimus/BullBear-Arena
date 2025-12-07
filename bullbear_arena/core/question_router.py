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
        question_lower = question.lower()
        
        # For fundamental questions, try to fetch data directly first
        if agent_type == 'fundamental':
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # PE ratio
                if 'pe' in question_lower or 'p/e' in question_lower:
                    pe = info.get('trailingPE') or info.get('forwardPE')
                    if pe:
                        return f"{ticker}'s PE ratio is {pe:.2f}x"
                    else:
                        return f"{ticker}'s PE ratio data is not available"
                
                # PB ratio
                if 'pb' in question_lower or 'p/b' in question_lower or 'book' in question_lower:
                    pb = info.get('priceToBook')
                    if pb:
                        return f"{ticker}'s PB ratio is {pb:.2f}x"
                    else:
                        return f"{ticker}'s PB ratio data is not available"
                
                # PS ratio
                if 'ps' in question_lower or 'p/s' in question_lower:
                    ps = info.get('priceToSalesTrailing12Months')
                    if ps:
                        return f"{ticker}'s PS ratio is {ps:.2f}x"
                    else:
                        return f"{ticker}'s PS ratio data is not available"
                
                # ROE
                if 'roe' in question_lower or 'return on equity' in question_lower:
                    roe = info.get('returnOnEquity')
                    if roe:
                        return f"{ticker}'s ROE (Return on Equity) is {roe*100:.2f}%"
                    else:
                        return f"{ticker}'s ROE data is not available"
                
                # ROA
                if 'roa' in question_lower or 'return on assets' in question_lower:
                    roa = info.get('returnOnAssets')
                    if roa:
                        return f"{ticker}'s ROA (Return on Assets) is {roa*100:.2f}%"
                    else:
                        return f"{ticker}'s ROA data is not available"
                
                # Revenue
                if 'revenue' in question_lower or 'sales' in question_lower:
                    revenue = info.get('totalRevenue')
                    if revenue:
                        revenue_b = revenue / 1e9
                        return f"{ticker}'s total revenue is ${revenue_b:.2f}B"
                    else:
                        return f"{ticker}'s revenue data is not available"
                
                # Market Cap
                if 'market cap' in question_lower or 'marketcap' in question_lower:
                    mcap = info.get('marketCap')
                    if mcap:
                        mcap_b = mcap / 1e9
                        return f"{ticker}'s market cap is ${mcap_b:.2f}B"
                    else:
                        return f"{ticker}'s market cap data is not available"
                
                # Debt to Equity
                if 'debt' in question_lower and 'equity' in question_lower:
                    de = info.get('debtToEquity')
                    if de:
                        return f"{ticker}'s Debt-to-Equity ratio is {de:.2f}"
                    else:
                        return f"{ticker}'s Debt-to-Equity ratio data is not available"
                
                # EPS
                if 'eps' in question_lower:
                    eps = info.get('trailingEps')
                    if eps:
                        return f"{ticker}'s EPS (Earnings Per Share) is ${eps:.2f}"
                    else:
                        return f"{ticker}'s EPS data is not available"
                
                # Dividend Yield
                if 'dividend' in question_lower:
                    div_yield = info.get('dividendYield')
                    if div_yield:
                        return f"{ticker}'s dividend yield is {div_yield*100:.2f}%"
                    else:
                        return f"{ticker} does not pay dividends or data is not available"
                
                # For valuation questions, use agent analysis
                if 'valuation' in question_lower or 'estimate' in question_lower:
                    pass  # Fall through to agent analysis below
                
            except Exception as e:
                pass  # Fall through to agent analysis
        
        # For technical questions
        elif agent_type == 'technical':
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                
                # Current price
                if 'price' in question_lower and any(w in question_lower for w in ['current', 'now', 'today']):
                    info = stock.info
                    price = info.get('currentPrice') or info.get('regularMarketPrice')
                    if price:
                        return f"{ticker}'s current price is ${price:.2f}"
                
                # RSI
                if 'rsi' in question_lower:
                    hist = stock.history(period="1mo")
                    if not hist.empty and len(hist) >= 14:
                        delta = hist['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs))
                        current_rsi = rsi.iloc[-1]
                        
                        if current_rsi < 30:
                            status = "oversold"
                        elif current_rsi > 70:
                            status = "overbought"
                        else:
                            status = "neutral"
                        
                        return f"{ticker}'s RSI is {current_rsi:.2f} ({status})"
                
                # Moving averages
                if 'ma' in question_lower or 'moving average' in question_lower:
                    hist = stock.history(period="6mo")
                    if not hist.empty:
                        ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
                        ma200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else None
                        current_price = hist['Close'].iloc[-1]
                        
                        result = f"{ticker}'s current price is ${current_price:.2f}, 50-day MA is ${ma50:.2f}"
                        if ma200:
                            result += f", 200-day MA is ${ma200:.2f}"
                        return result
                
            except Exception as e:
                pass
        
        # For sentiment questions, always use agent (need real-time news)
        elif agent_type == 'sentiment':
            try:
                agent = self.agents.get('sentiment')
                if agent:
                    result = agent.get_arena_output(ticker)
                    summary = result.get('summary', '')
                    
                    if 'news' in question_lower:
                        return f"{ticker} recent news sentiment: {summary[:400]}"
                    else:
                        return f"{ticker}'s market sentiment: {summary[:300]}"
            except:
                pass
        
        # For risk questions
        elif agent_type == 'risk':
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
                
                if not hist.empty:
                    # Volatility
                    if 'volatility' in question_lower:
                        returns = hist['Close'].pct_change()
                        volatility = returns.std() * (252 ** 0.5) * 100
                        return f"{ticker}'s annualized volatility is {volatility:.2f}%"
                    
                    # Beta
                    if 'beta' in question_lower:
                        info = stock.info
                        beta = info.get('beta')
                        if beta:
                            return f"{ticker}'s beta is {beta:.2f}"
            except:
                pass
        
        # Fallback: use agent analysis
        agent = self.agents.get(agent_type)
        if not agent:
            return None
        
        try:
            result = agent.get_arena_output(ticker)
            
            # For general questions, return a focused summary
            rec = result.get('recommendation', 'N/A')
            score = result.get('score', 0)
            summary = result.get('summary', '')
            
            return f"{ticker} {agent_type} analysis: {rec} (Score: {score:.1f}/100). {summary[:200]}"
            
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
        
        # For focused comparison questions
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
        
        # Full comparison
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
        """Quick comparison for focused questions"""
        try:
            import yfinance as yf
            question_lower = question.lower()
            
            if agent_type == 'fundamental':
                if 'pe' in question_lower:
                    comparisons = []
                    for ticker in tickers:
                        try:
                            stock = yf.Ticker(ticker)
                            info = stock.info
                            pe = info.get('trailingPE') or info.get('forwardPE')
                            if pe:
                                comparisons.append((ticker, pe))
                        except:
                            pass
                    
                    if comparisons:
                        comparisons.sort(key=lambda x: x[1])
                        result = "PE ratio comparison:\n"
                        for i, (ticker, pe) in enumerate(comparisons, 1):
                            result += f"{i}. {ticker}: {pe:.2f}x\n"
                        return result
            
            return None
            
        except Exception as e:
            return None
    
    def _execute_market_overview(self, routing) -> Dict[str, Any]:
        """Execute market overview"""
        results = {}
        question = getattr(routing, 'question', '')
        
        if 'sentiment' in routing.agents_needed:
            agent = self.agents.get('sentiment')
            if agent:
                try:
                    results['sentiment'] = agent.get_arena_output('SPY')
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
            
            agent_rankings.sort(key=lambda x: x['score'], reverse=True)
            rankings[agent_type] = agent_rankings
        
        return {
            "rankings": rankings,
            "tickers_compared": tickers,
            "agents_used": agents_called
        }
