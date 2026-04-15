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
- DrivingIntention, IntentionPhase, IntentionSequence, IntentionFrame - 数据类型
- identify_key_frames, generate_intention, parse_intention_response - 意图生成函数

使用示例：
    from core.llm import UnifiedLLMClient, TrajectoryPromptBuilder, SYSTEM_PROMPT

    # LLM 客户端
    client = UnifiedLLMClient(provider="qwen", model="qwen3.6-plus")
    client.chat("你好")

    # 轨迹提示词构造
    builder = TrajectoryPromptBuilder(sample_interval=2)
    prompt = builder.build_prompt(fragment)

    # 关键帧识别
    from core.llm.intention_models import identify_key_frames
    key_frames = identify_key_frames(fragment)
"""

from core.llm.intention_models import (
    UnifiedLLMClient,
    list_providers,
    get_provider_config,
    identify_key_frames,
    generate_intention,
    parse_intention_response,
    extract_json_from_response,
    IntentionFrame,
)

from core.llm.prompt_builder import (
    TrajectoryPromptBuilder,
    SYSTEM_PROMPT,
    build_intention_query_prompt,
    DrivingIntention,
    IntentionPhase,
    IntentionSequence,
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

    # 数据模型
    "DrivingIntention",
    "IntentionPhase",
    "IntentionSequence",

    # 提示词构造
    "TrajectoryPromptBuilder",
    "SYSTEM_PROMPT",
]
