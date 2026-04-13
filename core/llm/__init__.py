"""
core/llm - LLM 统一客户端和提示词构造

支持：
- qwen (阿里千问)
- openai (OpenAI GPT)
- gemini (Google Gemini)

包含：
- UnifiedLLMClient - 统一 LLM 客户端
- TrajectoryPromptBuilder - 轨迹信息提示词构造器
- SYSTEM_PROMPT - LLM 系统提示词
- DrivingIntention, IntentionPhase, IntentionSequence - 数据类型

使用示例：
    from core.llm import UnifiedLLMClient, TrajectoryPromptBuilder, SYSTEM_PROMPT

    # LLM 客户端
    client = UnifiedLLMClient(provider="qwen", model="qwen3.6-plus")
    client.chat("你好")

    # 轨迹提示词构造
    builder = TrajectoryPromptBuilder(sample_interval=2)
    prompt = builder.build_prompt(fragment)
"""

from core.llm.intention_models import (
    UnifiedLLMClient,
    list_providers,
    get_provider_config,
)

from core.llm.prompt_builder import (
    TrajectoryPromptBuilder,
    SYSTEM_PROMPT,
    DrivingIntention,
    MutationParams,
    IntentionPhase,
    IntentionSequence,
)

__all__ = [
    # 统一客户端
    "UnifiedLLMClient",
    "list_providers",
    "get_provider_config",

    # 数据模型
    "DrivingIntention",
    "MutationParams",
    "IntentionPhase",
    "IntentionSequence",

    # 提示词构造
    "TrajectoryPromptBuilder",
    "SYSTEM_PROMPT",
]
