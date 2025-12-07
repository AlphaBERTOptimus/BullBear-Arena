# ============================================================================
# BullBear Arena - Streamlit界面 (生产版)
# streamlit_app.py
# ============================================================================

import streamlit as st
import json
from datetime import datetime
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="BullBear Arena - AI投资分析",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 导入BullBear系统
# ============================================================================

try:
    from bullbear_arena.bullbear_system import BullBearSystem
    SYSTEM_AVAILABLE = True
except ImportError:
    SYSTEM_AVAILABLE = False
    st.error("⚠️ 无法导入BullBear系统,请确保已安装所有依赖")

# ============================================================================
# 主界面
# ============================================================================

def main():
    # 标题
    st.title("🏆 BullBear Arena")
    st.markdown("### AI驱动的多维度投资分析系统")
    st.markdown("---")
    
    # 侧边栏 - API配置
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        api_key = st.text_input(
            "DeepSeek API Key",
            value="",
            type="password",
            help="在 https://platform.deepseek.com 获取"
        )
        
        if api_key:
            st.success("✅ API已配置")
        else:
            st.warning("⚠️ 请输入API Key")
        
        st.markdown("---")
        
        st.header("📊 系统架构")
        st.markdown("""
        **4个专业Agent:**
        - 🐂 基本面分析 (Fundamental)
        - 📈 技术分析 (Technical)
        - 💬 情绪分析 (Sentiment)
        - ⚠️ 风险分析 (Risk)
        
        **最终裁判:**
        - 🏆 Arena Judge
        """)
        
        st.markdown("---")
        
        st.header("💡 使用提示")
        st.info("""
        **自由提问模式:**
        - 快速查询单一维度
        - 支持对比分析
        
        **完整分析模式:**
        - 4个Agent完整分析
        - Arena Judge最终裁决
        - 详细投资建议
        """)
        
        st.markdown("---")
        st.caption("Powered by DeepSeek API")
    
    # 检查系统可用性
    if not SYSTEM_AVAILABLE:
        st.error("系统未正确安装,请检查依赖")
        return
    
    if not api_key:
        st.warning("⚠️ 请在侧边栏输入DeepSeek API Key")
        st.info("👈 在左侧侧边栏配置API Key以开始使用")
        return
    
    # 初始化系统
    try:
        with st.spinner("初始化BullBear系统..."):
            system = BullBearSystem(api_key=api_key)
        st.success("✅ 系统初始化成功!")
    except Exception as e:
        st.error(f"❌ 系统初始化失败: {e}")
        return
    
    # 模式选择
    st.markdown("---")
    mode = st.radio(
        "选择分析模式",
        ["💬 自由提问", "📊 完整分析"],
        horizontal=True
    )
    
    # ========================================================================
    # 模式1: 自由提问
    # ========================================================================
    
    if mode == "💬 自由提问":
        st.markdown("---")
        st.header("💬 自由提问模式")
        
        # 示例问题
        with st.expander("💡 示例问题", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **基本面问题:**
                - MU的PE怎么样?
                - AAPL的ROE是多少?
                - NVDA估值合理吗?
                
                **技术面问题:**
                - TSLA的技术指标如何?
                - MU的RSI是多少?
                - AMD的MACD金叉了吗?
                """)
            
            with col2:
                st.markdown("""
                **情绪面问题:**
                - NVDA最近有什么新闻?
                - 市场情绪怎么样?
                - AI行业现在如何?
                
                **综合问题:**
                - 给我AAPL的完整分析
                - 比较MU和AMD
                - 对比NVDA、AMD、INTC
                """)
        
        # 问题输入
        question = st.text_input(
            "💭 输入你的问题:",
            placeholder="例如: MU的PE怎么样?",
            key="free_question"
        )
        
        col1, col2 = st.columns([1, 5])
        
        with col1:
            analyze_button = st.button("🔍 分析", type="primary", use_container_width=True)
        
        if analyze_button and question:
            with st.spinner("🤖 AI分析中,请稍候..."):
                try:
                    result = system.ask(question, verbose=False)
                    
                    # 显示结果
                    st.markdown("---")
                    st.success("✅ 分析完成!")
                    
                    # 基本信息
                    st.markdown(f"**问题:** {result['question']}")
                    
                    # 路由信息
                    routing = result.get('routing', {})
                    st.markdown(f"**分析类型:** {routing.get('question_type', 'N/A')}")
                    if routing.get('tickers'):
                        st.markdown(f"**股票代码:** {', '.join(routing.get('tickers', []))}")
                    
                    # 结果展示
                    analysis_result = result.get('result', {})
                    
                    if 'summary' in analysis_result:
                        st.info(analysis_result['summary'])
                    
                    # 详细结果
                    with st.expander("📄 查看详细分析结果", expanded=False):
                        st.json(result)
                    
                except Exception as e:
                    st.error(f"❌ 分析失败: {e}")
                    st.exception(e)
    
    # ========================================================================
    # 模式2: 完整分析
    # ========================================================================
    
    else:
        st.markdown("---")
        st.header("📊 完整分析模式")
        
        # 输入区域
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ticker = st.text_input(
                "📌 股票代码",
                value="AAPL",
                placeholder="输入股票代码 (如: AAPL, TSLA, MU)",
                key="full_ticker"
            ).upper()
        
        with col2:
            period = st.selectbox(
                "⏰ 投资周期",
                ["LONG_TERM", "MEDIUM_TERM", "SHORT_TERM"],
                format_func=lambda x: {
                    "LONG_TERM": "长期 (>1年)",
                    "MEDIUM_TERM": "中期 (3-12月)",
                    "SHORT_TERM": "短期 (<3月)"
                }[x],
                key="full_period"
            )
        
        analyze_button = st.button("🚀 开始完整分析", type="primary", use_container_width=True)
        
        if analyze_button and ticker:
            with st.spinner(f"🤖 正在对 {ticker} 进行4维度分析,请稍候 (约30-60秒)..."):
                try:
                    result = system.analyze(ticker, period, verbose=False)
                    
                    # 显示结果
                    st.markdown("---")
                    st.success("✅ 完整分析完成!")
                    
                    judge_result = result.get('judge_result', {})
                    
                    # 核心结果卡片
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        rec = judge_result.get('final_recommendation', 'N/A')
                        color = "🟢" if rec == "BUY" else "🟡" if rec == "HOLD" else "🔴"
                        st.metric(
                            "最终建议",
                            f"{color} {rec}"
                        )
                    
                    with col2:
                        confidence = judge_result.get('confidence', 0)
                        st.metric(
                            "置信度",
                            f"{confidence:.1%}"
                        )
                    
                    with col3:
                        consensus = judge_result.get('consensus_score', 0)
                        st.metric(
                            "共识评分",
                            f"{consensus:.0f}/100"
                        )
                    
                    # 详细分析
                    st.markdown("---")
                    
                    # Tab展示
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📝 分析师推理", 
                        "🎯 行动建议", 
                        "⚠️ 风险提示",
                        "📊 4个Agent详情"
                    ])
                    
                    with tab1:
                        st.markdown("### 详细推理过程")
                        st.write(judge_result.get('detailed_reasoning', ''))
                    
                    with tab2:
                        st.markdown("### 行动建议")
                        st.info(judge_result.get('action_plan', ''))
                    
                    with tab3:
                        st.markdown("### 风险提示")
                        st.warning(judge_result.get('risk_disclosure', ''))
                    
                    with tab4:
                        st.markdown("### 4个Agent分析详情")
                        agent_results = result.get('agent_results', {})
                        
                        for agent_type, agent_data in agent_results.items():
                            with st.expander(f"{agent_type.upper()} Agent", expanded=False):
                                st.json(agent_data)
                    
                    # 完整结果
                    with st.expander("📄 查看完整分析结果 (JSON)", expanded=False):
                        st.json(result)
                    
                except Exception as e:
                    st.error(f"❌ 分析失败: {e}")
                    st.exception(e)

# ============================================================================
# 运行
# ============================================================================

if __name__ == "__main__":
    main()
