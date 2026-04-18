"""
core - 危险轨迹生成系统

建议导入方式：

第一阶段（轨迹数据处理）：
    from core.stage1.main import main
    from core.stage1.processor import WaymoDataLoader, ScenarioProcessor

第二阶段（意图生成 + 穷举变异）：
    from core.stage2.intention_generator import LLMIntentionGenerator
    from core.stage2.mutator import TrajectoryMutator

LLM通用模块：
    from core.stage2.llm import UnifiedLLMClient, TrajectoryPromptBuilder
"""

# 子模块
from core import stage1
from core import stage2
