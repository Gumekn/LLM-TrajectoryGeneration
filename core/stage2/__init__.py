"""
core.stage2 - 第二阶段：意图生成 + 穷举变异

包含：
- intention_generator: LLM意图生成入口
- mutator: 轨迹变异穷举算法
- llm: LLM通用模块（客户端、提示词构造等）

使用示例：
    from core.stage2.intention_generator import LLMIntentionGenerator
    from core.stage2.mutator import TrajectoryMutator
    from core.stage2.llm import UnifiedLLMClient
"""
