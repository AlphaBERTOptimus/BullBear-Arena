# ============================================================================
# BullBear Arena - Streamlit界面 (生产版)
# streamlit_app.py
# ============================================================================

import streamlit as st
import json
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

# 页面配置
st.set_page_config(
    page_title="BullBear Arena - AI投资分析",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# API Key管理
# ============================================================================

def get_api_key():
    """
    多来源API Key获取
    
    优先级:
    1. Streamlit Secrets (云端部署)
    2. 环境变量 .env (本地开发)
    3. 用户手动输入 (临时使用)
    
    Returns:
        tuple: (api_key, source)
    """
    api_key = None
    source = None
    
    # 优先级1: Streamlit Secrets
    try:
        api_key = st.secrets["api"]["deepseek_key"]
        source = "Streamlit Secrets (云端配置)"
        return api_key, source
    except:
        pass
    
    # 优先级2: 环境变量
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        source = "环境变量 (.env文件)"
        return api_key, source
    
    return None, None

# ============================================================================
# 导入BullBear系统
# ============================================================================

try:
    from bullbear_arena.bullbear_system import BullBearSystem
    SYSTEM_AVAILABLE = True
except ImportError as e:
    SYSTEM_AVAILABLE = False
    IMPORT_ERROR = str(e)

# ============================================================================
# 主界面
# ============================================================================

def main():
    # 标题
    st.title("🏆 BullBear Arena")
    st.markdown("### AI驱动的多维度投资分析系统")
    st.markdown("---")
    
    # ========================================================================
    # 侧边栏 - API配置
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        # 获取预配置的API Key
        default_api_key, source = get_api_key()
        
        if default_api_key:
            # 已有配置 - 显示状态
            st.success("✅ API Key已自动加载")
            st.caption(f"📍 来源: {source}")
            
            # 显示部分key用于确认 (安全显示)
            masked_key = default_api_key[:8] + "•••" + default_api_key[-4:]
            st.caption(f"🔑 Key: `{masked_key}`")
            
            # 高级选项: 允许手动覆盖
            with st.expander("🔧 手动覆盖API Key", expanded=False):
                st.caption("仅在需要临时使用其他Key时使用")
                manual_key = st.text_input(
                    "输入新的API Key",
                    type="password",
                    key="manual_api_key",
                    help="将覆盖当前配置的Key"
                )
                if manual_key:
                    api_key = manual_key
                    st.info("✓ 使用手动输入的Key")
                else:
                    api_key = default_api_key
        else:
            # 没有预配置 - 需要手动输入
            st.warning("⚠️ 未检测到API Key配置")
            
            st.info("""
            **配置方法 (推荐):**
            
            1️⃣ 创建 `.env` 文件:
```
            DEEPSEEK_API_KEY=your-key-here
```
            
            2️⃣ 重启应用
            
            **或在下方直接输入** (临时使用):
            """)
            
            api_key = st.text_input(
                "DeepSeek API Key",
                type="password",
                help="从 https://platform.deepseek.com 获取",
                placeholder="sk-..."
            )
            
            if not api_key:
                st.error("❌ 请输入API Key")
        
        st.markdown("---")
        
        # 系统架构说明
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
        
        # 使用提示
        st.header("💡 使用提示")
        st.info("""
        **自由提问模式:**
        - 快速查询单一维度
        - 支持对比分析 (2-5只股票)
        - 响应时间: 10-20秒
        
        **完整分析模式:**
        - 4个Agent完整分析
        - Arena Judge最终裁决
        - 详细投资建议
        - 响应时间: 30-60秒
        """)
        
        st.markdown("---")
        
        # 版本信息
        st.caption("**Version:** 1.0.0")
        st.caption("**Powered by:** DeepSeek API")
        st.caption("**GitHub:** [BullBear Arena](https://github.com/your-repo)")
    
    # ========================================================================
    # 检查系统可用性
    # ========================================================================
    
    if not SYSTEM_AVAILABLE:
        st.error("❌ 系统未正确安装")
        st.code(f"错误信息: {IMPORT_ERROR}", language="python")
        st.info("""
        **解决方法:**
        1. 确保所有依赖已安装: `pip install -r requirements.txt`
        2. 检查项目结构是否完整
        3. 确认所有Agent文件存在
        """)
        return
    
    # 检查API Key
    if not api_key:
        st.warning("⚠️ 请先配置DeepSeek API Key")
        st.info("👈 在左侧侧边栏输入或配置API Key")
        
        # 显示配置指南
        with st.expander("📖 查看详细配置指南", expanded=True):
            st.markdown("""
            ### 方式1: 使用 .env 文件 (推荐)
            
            1. 在项目根目录创建 `.env` 文件
            2. 添加以下内容:
```
               DEEPSEEK_API_KEY=your-actual-api-key
```
            3. 重启Streamlit应用
            
            ### 方式2: 直接输入
            
            在左侧侧边栏的输入框中输入你的API Key
            
            ### 获取API Key
            
            访问 [DeepSeek Platform](https://platform.deepseek.com) 注册并获取
            """)
        return
    
    # 初始化系统
    try:
        with st.spinner("🔄 初始化BullBear系统..."):
            system = BullBearSystem(api_key=api_key)
        st.success("✅ 系统初始化成功!")
    except Exception as e:
        st.error(f"❌ 系统初始化失败: {e}")
        st.exception(e)
        return
    
    # ========================================================================
    # 模式选择
    # ========================================================================
    
    st.markdown("---")
    mode = st.radio(
        "**选择分析模式**",
        ["💬 自由提问", "📊 完整分析"],
        horizontal=True,
        help="自由提问适合快速查询,完整分析提供深度投资建议"
    )
    
    # ========================================================================
    # 模式1: 自由提问
    # ========================================================================
    
    if mode == "💬 自由提问":
        st.markdown("---")
        st.header("💬 自由提问模式")
        
        # 示例问题
        with st.expander("💡 示例问题参考", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📊 基本面问题:**
                - MU的PE怎么样?
                - AAPL的ROE是多少?
                - NVDA估值合理吗?
                
                **📈 技术面问题:**
                - TSLA的技术指标如何?
                - MU的RSI是多少?
                - AMD的MACD金叉了吗?
                """)
            
            with col2:
                st.markdown("""
                **💬 情绪面问题:**
                - NVDA最近有什么新闻?
                - 市场情绪怎么样?
                - AI行业现在如何?
                
                **🔍 综合问题:**
                - 给我AAPL的完整分析
                - 比较MU和AMD的基本面
                - 对比NVDA、AMD、INTC
                """)
        
        # 问题输入
        question = st.text_input(
            "💭 **输入你的问题:**",
            placeholder="例如: MU的PE怎么样? 或 比较NVDA和AMD的基本面",
            key="free_question",
            help="支持单股票查询、对比分析(2-5只股票)、市场整体问题"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            analyze_button = st.button(
                "🔍 开始分析",
                type="primary",
                use_container_width=True
            )
        
        with col2:
            clear_button = st.button(
                "🗑️ 清空",
                use_container_width=True
            )
        
        if clear_button:
            st.rerun()
        
        if analyze_button and question:
            with st.spinner("🤖 AI正在分析中,请稍候..."):
                try:
                    # 调用系统
                    result = system.ask(question, verbose=False)
                    
                    # 显示结果
                    st.markdown("---")
                    st.success("✅ 分析完成!")
                    
                    # 基本信息卡片
                    st.markdown(f"**🎯 问题:** {result['question']}")
                    
                    # 路由信息
                    routing = result.get('routing', {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("分析类型", routing.get('question_type', 'N/A').upper())
                    with col2:
                        tickers = routing.get('tickers', [])
                        st.metric("股票数量", len(tickers) if tickers else 0)
                    with col3:
                        agents = routing.get('agents_needed', [])
                        st.metric("调用Agent", len(agents))
                    
                    if tickers:
                        st.info(f"**📌 分析股票:** {', '.join(tickers)}")
                    
                    # 结果展示
                    st.markdown("---")
                    analysis_result = result.get('result', {})
                    
                    # 摘要
                    if 'summary' in analysis_result:
                        st.markdown("### 📋 分析摘要")
                        st.info(analysis_result['summary'])
                    
                    # 对比结果
                    if analysis_result.get('execution_type') == 'comparison':
                        st.markdown("### 📊 对比结果")
                        summary = analysis_result.get('comparison_summary', {})
                        
                        if 'rankings' in summary:
                            for agent_type, rankings in summary['rankings'].items():
                                st.markdown(f"**{agent_type.upper()} 维度排名:**")
                                
                                # 创建排名表格
                                rank_data = []
                                for i, rank in enumerate(rankings, 1):
                                    rank_data.append({
                                        "排名": f"#{i}",
                                        "股票": rank['ticker'],
                                        "评分": f"{rank['score']:.1f}"
                                    })
                                
                                st.table(rank_data)
                                st.markdown("")
                    
                    # 详细结果
                    with st.expander("📄 查看完整分析结果 (JSON)", expanded=False):
                        st.json(result)
                    
                except Exception as e:
                    st.error(f"❌ 分析失败: {e}")
                    st.exception(e)
                    st.info("💡 提示: 请检查问题格式或稍后重试")
    
    # ========================================================================
    # 模式2: 完整分析
    # ========================================================================
    
    else:
        st.markdown("---")
        st.header("📊 完整分析模式")
        
        st.info("""
        **完整分析流程:**
        1. 🐂 基本面Agent分析财务指标
        2. 📈 技术面Agent分析价格趋势
        3. 💬 情绪面Agent分析市场情绪
        4. ⚠️ 风险面Agent评估投资风险
        5. 🏆 Arena Judge综合裁决
        """)
        
        # 输入区域
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ticker = st.text_input(
                "📌 **股票代码**",
                value="AAPL",
                placeholder="输入股票代码 (如: AAPL, TSLA, MU)",
                key="full_ticker",
                help="输入美股股票代码"
            ).upper().strip()
        
        with col2:
            period = st.selectbox(
                "⏰ **投资周期**",
                ["LONG_TERM", "MEDIUM_TERM", "SHORT_TERM"],
                format_func=lambda x: {
                    "LONG_TERM": "🎯 长期 (>1年)",
                    "MEDIUM_TERM": "📅 中期 (3-12月)",
                    "SHORT_TERM": "⚡ 短期 (<3月)"
                }[x],
                key="full_period",
                help="投资周期影响权重分配"
            )
        
        # 权重说明
        with st.expander("ℹ️ 投资周期权重说明", expanded=False):
            if period == "LONG_TERM":
                st.markdown("""
                **长期投资权重分配:**
                - 🐂 基本面: **50%** (最重要 - 关注企业价值)
                - ⚠️ 风险面: **30%** (次要 - 长期风险管理)
                - 📈 技术面: **10%** (参考 - 趋势确认)
                - 💬 情绪面: **10%** (辅助 - 情绪参考)
                """)
            elif period == "MEDIUM_TERM":
                st.markdown("""
                **中期投资权重分配:**
                - 📈 技术面: **35%** (最重要 - 把握波段)
                - 🐂 基本面: **30%** (次要 - 价值支撑)
                - 💬 情绪面: **20%** (参考 - 催化剂)
                - ⚠️ 风险面: **15%** (辅助 - 风险控制)
                """)
            else:
                st.markdown("""
                **短期投资权重分配:**
                - 📈 技术面: **45%** (最重要 - 趋势为王)
                - 💬 情绪面: **30%** (次要 - 新闻驱动)
                - ⚠️ 风险面: **15%** (参考 - 快速止损)
                - 🐂 基本面: **10%** (辅助 - 底层支撑)
                """)
        
        analyze_button = st.button(
            "🚀 开始完整分析",
            type="primary",
            use_container_width=True,
            help="将调用4个Agent进行全面分析,约需30-60秒"
        )
        
        if analyze_button and ticker:
            # 进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1
                status_text.text("🔄 正在初始化分析...")
                progress_bar.progress(10)
                
                # Step 2
                status_text.text(f"🐂 基本面Agent正在分析 {ticker}...")
                progress_bar.progress(30)
                
                # Step 3
                status_text.text(f"📈 技术面Agent正在分析 {ticker}...")
                progress_bar.progress(50)
                
                # Step 4
                status_text.text(f"💬 情绪面Agent正在分析 {ticker}...")
                progress_bar.progress(70)
                
                # Step 5
                status_text.text(f"⚠️ 风险面Agent正在分析 {ticker}...")
                progress_bar.progress(85)
                
                # 调用系统
                result = system.analyze(ticker, period, verbose=False)
                
                # Step 6
                status_text.text("🏆 Arena Judge正在综合裁决...")
                progress_bar.progress(95)
                
                # 完成
                progress_bar.progress(100)
                status_text.text("✅ 分析完成!")
                
                # 清理进度条
                import time
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # 显示结果
                st.markdown("---")
                st.success(f"✅ {ticker} 完整分析完成!")
                
                judge_result = result.get('judge_result', {})
                
                # 核心结果卡片
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    rec = judge_result.get('final_recommendation', 'N/A')
                    color_map = {
                        "BUY": "🟢",
                        "HOLD": "🟡",
                        "SELL": "🔴"
                    }
                    color = color_map.get(rec, "⚪")
                    st.metric(
                        "🎯 最终建议",
                        f"{color} {rec}",
                        help="基于4个Agent综合分析"
                    )
                
                with col2:
                    confidence = judge_result.get('confidence', 0)
                    st.metric(
                        "📊 置信度",
                        f"{confidence:.1%}",
                        help="AI对此建议的信心程度"
                    )
                
                with col3:
                    consensus = judge_result.get('consensus_score', 0)
                    st.metric(
                        "🤝 共识评分",
                        f"{consensus:.0f}/100",
                        help="4个Agent的一致性程度"
                    )
                
                # 详细分析Tab
                st.markdown("---")
                
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📝 AI推理分析",
                    "🎯 行动建议",
                    "⚠️ 风险提示",
                    "📊 4个Agent详情",
                    "📄 完整数据"
                ])
                
                with tab1:
                    st.markdown("### 🤖 AI分析师详细推理")
                    reasoning = judge_result.get('detailed_reasoning', '')
                    if reasoning:
                        st.write(reasoning)
                    else:
                        st.warning("暂无详细推理")
                
                with tab2:
                    st.markdown("### 💼 具体行动建议")
                    action = judge_result.get('action_plan', '')
                    if action:
                        st.info(action)
                    else:
                        st.warning("暂无行动建议")
                
                with tab3:
                    st.markdown("### ⚠️ 风险提示")
                    risk = judge_result.get('risk_disclosure', '')
                    if risk:
                        st.warning(risk)
                    else:
                        st.info("暂无特别风险提示")
                
                with tab4:
                    st.markdown("### 📊 4个Agent分析详情")
                    
                    agent_results = result.get('agent_results', {})
                    
                    if agent_results:
                        for agent_type, agent_data in agent_results.items():
                            agent_names = {
                                "fundamental": "🐂 基本面分析",
                                "technical": "📈 技术分析",
                                "sentiment": "💬 情绪分析",
                                "risk": "⚠️ 风险分析"
                            }
                            
                            with st.expander(f"{agent_names.get(agent_type, agent_type)}", expanded=False):
                                # 基本信息
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("建议", agent_data.get('recommendation', 'N/A'))
                                with col2:
                                    st.metric("评分", f"{agent_data.get('score', 0):.1f}")
                                with col3:
                                    st.metric("置信度", f"{agent_data.get('confidence', 0):.1%}")
                                
                                # 摘要
                                if 'summary' in agent_data:
                                    st.markdown("**分析摘要:**")
                                    st.info(agent_data['summary'])
                                
                                # 完整数据
                                st.markdown("**完整数据:**")
                                st.json(agent_data)
                    else:
                        st.warning("未获取到Agent数据")
                
                with tab5:
                    st.markdown("### 📄 完整分析结果 (JSON)")
                    st.json(result)
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                
                st.error(f"❌ 分析失败: {e}")
                st.exception(e)
                
                st.info("""
                **可能的原因:**
                1. 股票代码不存在或输入错误
                2. 网络连接问题
                3. API调用失败
                4. 数据获取失败
                
                **建议:**
                - 检查股票代码是否正确
                - 稍后重试
                - 查看详细错误信息
                """)
    
    # ========================================================================
    # 页脚
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>
    🏆 BullBear Arena v1.0.0 | 
    Powered by DeepSeek API | 
    <a href='https://github.com/your-repo/BullBear-Arena' target='_blank'>GitHub</a>
    </p>
    <p style='font-size: 0.8em;'>
    ⚠️ 免责声明: 本系统仅供学习研究使用,不构成任何投资建议。投资有风险,决策需谨慎。
    </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 运行
# ============================================================================

if __name__ == "__main__":
    main()
