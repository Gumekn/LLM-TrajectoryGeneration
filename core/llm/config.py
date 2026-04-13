"""
core/llm/config.py - LLM 配置

支持的提供商及模型：

1. qwen (阿里千问)
   - base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
   - 模型示例: qwen3.6-plus, qwen-plus, qwen-max, qwen-turbo
   - 环境变量: DASHSCOPE_API_KEY

2. openai (OpenAI)
   - base_url: https://api.openai.com/v1
   - 模型示例: gpt-4o, gpt-4o-mini, gpt-4-turbo
   - 环境变量: OPENAI_API_KEY

3. gemini (Google)
   - base_url: 无（使用 google.genai 默认端点）
   - 模型示例: gemini-2.0-flash, gemini-2.0-pro, gemini-1.5-flash
   - 环境变量: GEMINI_API_KEY

使用示例：
    from core.llm import UnifiedLLMClient

    client = UnifiedLLMClient(provider="qwen", model="qwen3.6-plus")
    client = UnifiedLLMClient(provider="openai", model="gpt-4o")
    client = UnifiedLLMClient(provider="gemini", model="gemini-2.0-flash")
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ProviderConfig:
    """提供商配置"""
    name: str                    # 提供商名称
    api_key_env: str             # 环境变量名
    base_url: Optional[str]      # API 基础地址
    description: str             # 描述
    models: str                  # 支持的模型示例


# 提供商配置
PROVIDER_CONFIGS: Dict[str, ProviderConfig] = {
    "qwen": ProviderConfig(
        name="qwen",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="阿里千问",
        models="qwen3.6-plus, qwen-plus, qwen-max, qwen-turbo",
    ),
    "openai": ProviderConfig(
        name="openai",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        description="OpenAI GPT",
        models="gpt-4o, gpt-4o-mini, gpt-4-turbo",
    ),
    "gemini": ProviderConfig(
        name="gemini",
        api_key_env="GEMINI_API_KEY",
        base_url=None,  # 使用 google.genai 默认端点
        description="Google Gemini",
        models="gemini-2.0-flash, gemini-2.0-pro, gemini-1.5-flash, gemini-1.5-pro",
    ),
}


def list_providers() -> Dict[str, ProviderConfig]:
    """列出所有支持的提供商"""
    return PROVIDER_CONFIGS


def get_provider_config(provider: str) -> ProviderConfig:
    """获取提供商配置"""
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(
            f"不支持的提供商: {provider}\n"
            f"支持的提供商: {list(PROVIDER_CONFIGS.keys())}"
        )
    return PROVIDER_CONFIGS[provider]
