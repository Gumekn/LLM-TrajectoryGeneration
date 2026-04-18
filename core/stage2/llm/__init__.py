"""
core/stage2/llm - LLM 模块

支持：
- qwen (阿里千问)
- openai (OpenAI GPT)
- gemini (Google Gemini)

包含：
- UnifiedLLMClient - 统一 LLM 客户端
- TrajectoryPromptBuilder - 轨迹信息提示词构造器
- SYSTEM_PROMPT - LLM 系统提示词
- IntentionFrame - 意图帧数据类型
- identify_key_frames, generate_intention, parse_intention_response - 意图生成函数

使用示例：
    from core.stage2.llm import UnifiedLLMClient, TrajectoryPromptBuilder, SYSTEM_PROMPT

    # LLM 客户端
    client = UnifiedLLMClient(provider="qwen", model="qwen3.6-plus")
    client.chat("你好")

    # 轨迹提示词构造
    builder = TrajectoryPromptBuilder(sample_interval=2)
    prompt = builder.build_prompt(fragment)

    # 关键帧识别
    from core.stage2.llm.intention_models import identify_key_frames
    key_frames = identify_key_frames(fragment)
"""

from core.stage2.llm.intention_models import (
    UnifiedLLMClient,
    list_providers,
    get_provider_config,
    identify_key_frames,
    generate_intention,
    parse_intention_response,
    extract_json_from_response,
    IntentionFrame,
)

from core.stage2.llm.prompt_builder import (
    TrajectoryPromptBuilder,
    SYSTEM_PROMPT,
    build_intention_query_prompt,
)

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
    # 统一客户端
    "UnifiedLLMClient",
    "list_providers",
    "get_provider_config",

    # 意图生成函数
    "identify_key_frames",
    "build_intention_query_prompt",
    "generate_intention",
    "parse_intention_response",
    "extract_json_from_response",
    "IntentionFrame",

    # 提示词构造
    "TrajectoryPromptBuilder",
    "SYSTEM_PROMPT",

    # 轨迹变异
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
