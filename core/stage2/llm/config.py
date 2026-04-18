"""
core/stage2/llm/config.py - LLM API 配置与客户端

包含：
- 提供商配置（api_key 环境变量、base_url）
- UnifiedLLMClient 统一客户端

使用示例：
    from core.stage2.llm.config import UnifiedLLMClient, get_provider_config, list_providers

    client = UnifiedLLMClient(provider="qwen", model="qwen3.6-plus")
"""

import os
from typing import Dict, Any, Optional


# =============================================================================
# 提供商配置
# =============================================================================

SUPPORTED_PROVIDERS: Dict[str, str] = {
    "qwen": "阿里千问 (qwen3.6-plus, qwen-plus, qwen-max, qwen-turbo)",
    "openai": "OpenAI GPT (gpt-4o, gpt-4o-mini, gpt-4-turbo)",
    "gemini": "Google Gemini (gemini-2.0-flash, gemini-2.0-pro)",
}

PROVIDER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "qwen": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": None,
    },
}


def list_providers() -> Dict[str, str]:
    """列出所有支持的 LLM 提供商"""
    return SUPPORTED_PROVIDERS


def get_provider_config(provider: str) -> Dict[str, Any]:
    """
    获取 LLM 提供商的配置

    Args:
        provider: 提供商名称 (qwen/openai/gemini)

    Returns:
        包含 api_key_env 和 base_url 的配置字典

    Raises:
        ValueError: 不支持的提供商
    """
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(f"不支持的提供商: {provider}，支持的: {list(PROVIDER_CONFIGS.keys())}")
    return PROVIDER_CONFIGS[provider]


# =============================================================================
# 统一 LLM 客户端
# =============================================================================

class UnifiedLLMClient:
    """
    统一 LLM 客户端 - 封装不同提供商的 API 调用

    使用示例：
        from core.stage2.llm.config import UnifiedLLMClient
        client = UnifiedLLMClient(provider="qwen", model="qwen3.6-plus")
        response = client.chat("你的问题")
    """

    def __init__(self, provider: str, model: str, api_key: Optional[str] = None):
        """
        初始化 LLM 客户端

        Args:
            provider: 提供商名称 (qwen/openai/gemini)
            model: 模型名称
            api_key: 可选，手动指定 API key（默认从环境变量读取）
        """
        config = get_provider_config(provider)
        self.provider = provider
        self.model = model

        self.api_key = api_key or os.environ.get(config["api_key_env"])
        if not self.api_key:
            raise ValueError(f"请设置环境变量 {config['api_key_env']}")

        self.base_url = config["base_url"]
        self._init_client()

    def _init_client(self):
        """根据提供商初始化底层客户端"""
        if self.provider in ("qwen", "openai"):
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("请安装 openai 包: pip install openai")
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        elif self.provider == "gemini":
            try:
                from google import genai
            except ImportError:
                raise ImportError("请安装 google-genai 包: pip install google-genai")
            self._client = genai.Client(api_key=self.api_key)

    @property
    def name(self) -> str:
        """返回 'provider-model' 格式的名称"""
        return f"{self.provider}-{self.model}"

    def chat(self, prompt: str, **kwargs) -> str:
        """
        发送对话请求

        Args:
            prompt: 用户提示词
            temperature: 温度参数（默认0.7）
            max_tokens: 最大token数（默认4096）

        Returns:
            LLM 响应的文本内容
        """
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 4096)

        if self.provider in ("qwen", "openai"):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        elif self.provider == "gemini":
            response = self._client.models.generate_content(
                model=self.model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config={"temperature": temperature}
            )
            return response.text

        raise NotImplementedError(f"未实现的提供商: {self.provider}")