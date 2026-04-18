"""
core/stage2/llm - LLM 模块

支持：
- qwen (阿里千问)
- openai (OpenAI GPT)
- gemini (Google Gemini)

包含：
- config - LLM API 配置（UnifiedLLMClient、提供商配置）
- prompt_builder - 轨迹信息提示词构造器
- mutator - 轨迹变异穷举算法

使用示例：
    from core.stage2.llm import UnifiedLLMClient, TrajectoryPromptBuilder, SYSTEM_PROMPT

    # LLM 客户端
    client = UnifiedLLMClient(provider="qwen", model="qwen3.6-plus")
    client.chat("你好")

    # 轨迹提示词构造
    builder = TrajectoryPromptBuilder(sample_interval=2)
    prompt = builder.build_prompt(fragment)
"""

# LLM 客户端配置
from core.stage2.llm.config import (
    UnifiedLLMClient,
    list_providers,
    get_provider_config,
)

# 提示词构造
from core.stage2.llm.prompt_builder import (
    TrajectoryPromptBuilder,
    SYSTEM_PROMPT,
    build_intention_query_prompt,
)

# 轨迹变异
from core.stage2.mutator import (
    TrajectoryMutator,
    merge_intentions,
    dfs_mutate_with_pruning,
    get_mutation_combinations,
    compute_block_risk_score,
    build_variant_output,
    IntentBlock,
    TrajectoryState,
    IntentionMutation,
    DrivingIntention,
    extract_initial_state,
    mutate_vector_acceleration,
    kinematically_integrate,
)

__all__ = [
    # 统一客户端（config.py）
    "UnifiedLLMClient",
    "list_providers",
    "get_provider_config",

    # 提示词构造（prompt_builder.py）
    "TrajectoryPromptBuilder",
    "SYSTEM_PROMPT",
    "build_intention_query_prompt",

    # 轨迹变异（mutator.py）
    "TrajectoryMutator",
    "merge_intentions",
    "dfs_mutate_with_pruning",
    "get_mutation_combinations",
    "compute_block_risk_score",
    "build_variant_output",
    "IntentBlock",
    "TrajectoryState",
    "IntentionMutation",
    "DrivingIntention",
    "extract_initial_state",
    "mutate_vector_acceleration",
    "kinematically_integrate",
]
