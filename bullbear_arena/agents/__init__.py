# ============================================================================
# BullBear Arena - Agents Package
# bullbear_arena/agents/__init__.py
# ============================================================================
"""
BullBear Arena 专业分析Agent包
"""

from bullbear_arena.agents.fundamental_agent import FundamentalAgent
from bullbear_arena.agents.technical_agent import TechnicalAgent
from bullbear_arena.agents.sentiment_agent import SentimentAgent
from bullbear_arena.agents.risk_agent import RiskAgent

__all__ = [
    "FundamentalAgent",
    "TechnicalAgent", 
    "SentimentAgent",
    "RiskAgent"
]

__version__ = "1.0.0"
```

4. **Commit:** "Create agents package structure"

---

#### **2.3 上传4个Agent文件**

现在在 `bullbear_arena/agents/` 文件夹中:

1. **点击 "Add file" → "Upload files"**

2. **上传这4个文件:**
   - `fundamental_agent.py`
   - `technical_agent.py`
   - `sentiment_agent.py`
   - `risk_agent.py`

3. **Commit:** "Add 4 agent files"

---

#### **2.4 创建 `ensemble` 文件夹**

1. **返回 `bullbear_arena` 文件夹**

2. **点击 "Add file" → "Create new file"**

3. **文件名:**
```
   ensemble/__init__.py
