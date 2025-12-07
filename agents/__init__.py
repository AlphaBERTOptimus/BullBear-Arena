# ============================================================================
# BullBear Arena - Main Package
# bullbear_arena/__init__.py
# ============================================================================
"""
BullBear Arena - AI驱动的多维度投资分析系统

主要组件:
- BullBearSystem: 统一系统入口
- 4个专业Agent: Fundamental, Technical, Sentiment, Risk
- ArenaJudge: 最终裁判
- QuestionRouter: 问题路由
- FlexibleExecutor: 灵活执行

使用示例:
    from bullbear_arena import BullBearSystem
    
    system = BullBearSystem(api_key="your-api-key")
    
    # 自由提问
    result = system.ask("MU的PE怎么样?")
    
    # 完整分析
    result = system.analyze("AAPL", "LONG_TERM")
"""

from bullbear_arena.bullbear_system import BullBearSystem
from bullbear_arena.agents import (
    FundamentalAgent,
    TechnicalAgent,
    SentimentAgent,
    RiskAgent
)
from bullbear_arena.ensemble import ArenaJudge
from bullbear_arena.core import (
    QuestionRouter,
    QuestionAnalysis,
    FlexibleExecutor
)

__version__ = "1.0.0"
__author__ = "BullBear Team"
__email__ = "support@bullbeararena.com"

__all__ = [
    # 主要系统
    "BullBearSystem",
    
    # 4个Agent
    "FundamentalAgent",
    "TechnicalAgent",
    "SentimentAgent",
    "RiskAgent",
    
    # 集成模块
    "ArenaJudge",
    
    # 核心模块
    "QuestionRouter",
    "QuestionAnalysis",
    "FlexibleExecutor"
]
```

---

## ✅ 完整的 `__init__.py` 文件清单

现在你有了所有的 `__init__.py` 文件:
```
BullBear-Arena/
├── bullbear_arena/
│   ├── __init__.py                    ✅ (根包)
│   ├── agents/
│   │   └── __init__.py                ✅ (agents包)
│   ├── ensemble/
│   │   └── __init__.py                ✅ (ensemble包)
│   └── core/
│       └── __init__.py                ✅ (core包)
