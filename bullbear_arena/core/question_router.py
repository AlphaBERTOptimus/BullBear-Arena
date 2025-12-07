# ============================================================================
# BullBear Arena - 问题路由器
# bullbear_arena/core/question_router.py
# ============================================================================
"""
问题智能路由器

功能:
1. 识别用户问题类型
2. 提取股票代码
3. 确定需要调用哪些Agent
4. 判断投资周期

特别处理:
- 宽泛市场问题 → sentiment agent
- 名人/公司新闻 → sentiment agent  
- 商品投资(黄金/石油) → sentiment agent
"""

import json
import requests
from typing import List
from pydantic import BaseModel, Field

# ============================================================================
# 数据模型
# ============================================================================

class QuestionAnalysis(BaseModel):
    """问题分析结果"""
    question_type: str = Field(
        description="问题类型: full_analysis/fundamental/technical/sentiment/risk/comparison/market_general"
    )
    tickers: List[str] = Field(description="股票代码列表")
    agents_needed: List[str] = Field(description="需要调用的Agent: fundamental/technical/sentiment/risk")
    specific_request: str = Field(description="具体请求内容")
    time_horizon: str = Field(description="投资周期: LONG_TERM/MEDIUM_TERM/SHORT_TERM")
    response_style: str = Field(description="响应风格: detailed/concise")
    fallback_strategy: str = Field(description="后备策略: sentiment_agent/web_search/polite_decline")

# ============================================================================
# 问题路由器
# ============================================================================

class QuestionRouter:
    """
    问题智能路由器
    
    根据用户问题,智能判断需要调用哪些Agent
    
    支持的问题类型:
    - full_analysis: 完整投资分析
    - fundamental: 基本面问题
    - technical: 技术面问题
    - sentiment: 情绪/新闻问题
    - risk: 风险问题
    - comparison: 对比分析
    - market_general: 宽泛市场问题
    """
    
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com/v1/chat/completions"):
        """
        初始化问题路由器
        
        Args:
            api_key: DeepSeek API密钥
            api_url: API端点
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = "deepseek-chat"
    
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
    
    def analyze_question(self, question: str) -> QuestionAnalysis:
        """
        分析用户问题
        
        Args:
            question: 用户问题
            
        Returns:
            QuestionAnalysis: 分析结果
        """
        prompt = f"""你是一个投资问题分类专家。请分析用户的问题,判断需要调用哪些分析模块。

【用户问题】
{question}

【可用的分析Agent】
1. fundamental - 基本面分析 (财务指标、估值、盈利能力)
2. technical - 技术分析 (K线、均线、MACD、RSI、支撑阻力位)
3. sentiment - 情绪分析 (新闻、社交媒体、市场情绪、宽泛市场话题)
4. risk - 风险分析 (波动率、VaR、夏普比率、最大回撤)

【问题类型定义】
- full_analysis: 需要完整投资分析 (调用所有4个Agent + Judge)
- fundamental: 只关于基本面 (PE、PB、ROE、营收、利润等)
- technical: 只关于技术面 (均线、指标、K线形态等)
- sentiment: 只关于情绪/新闻 (最近新闻、市场情绪、舆论)
- risk: 只关于风险 (波动率、风险指标、回撤)
- comparison: 对比多只股票
- market_general: 宽泛市场问题/名人话题/商品投资 (由sentiment agent处理)

【CRITICAL: 宽泛问题处理规则】
以下情况必须归类为 market_general,并由sentiment agent处理:
1. 没有明确股票代码的市场整体问题
2. 关于名人、CEO、企业家的问题 (马斯克、巴菲特等)
3. 商品投资问题 (黄金、石油、比特币等)
4. 宏观经济问题 (通胀、利率、经济走势)
5. 行业趋势问题 (AI行业、半导体行业)

对于这些问题:
- question_type设为 "market_general"
- tickers设为 []
- agents_needed设为 ["sentiment"]
- fallback_strategy设为 "sentiment_agent"

【投资周期判断】
- LONG_TERM: 明确提到"长期"、"长线"、"价值投资"或没有明确时间
- MEDIUM_TERM: 提到"中期"、"波段"、"几个月"
- SHORT_TERM: 提到"短期"、"短线"、"日内"、"最近"

【响应风格】
- detailed: 用户要求详细分析、完整报告
- concise: 用户只想快速了解某个指标

【后备策略】
- sentiment_agent: 由sentiment agent处理 (宽泛市场问题)
- web_search: 需要网络搜索 (超出系统能力)
- polite_decline: 礼貌拒绝 (完全无关问题)

请以JSON格式返回:
{{"question_type": "market_general", "tickers": [], "agents_needed": ["sentiment"], "specific_request": "查询最近市场整体情绪和新闻", "time_horizon": "SHORT_TERM", "response_style": "concise", "fallback_strategy": "sentiment_agent"}}

注意:
1. 从问题中提取股票代码 (如果有)
2. tickers必须是大写
3. 如果没有明确股票代码但是宽泛市场问题,使用sentiment agent
4. 马斯克、黄金、比特币等都归类为market_general
5. 对于完全无关的问题(如"天气怎么样"),fallback_strategy设为"polite_decline"
"""
        
        try:
            response_text = self.call_deepseek_api(prompt)
            result_dict = json.loads(response_text)
            return QuestionAnalysis(**result_dict)
            
        except Exception as e:
            # 默认使用sentiment agent处理
            return QuestionAnalysis(
                question_type="market_general",
                tickers=[],
                agents_needed=["sentiment"],
                specific_request=question,
                time_horizon="SHORT_TERM",
                response_style="concise",
                fallback_strategy="sentiment_agent"
            )
