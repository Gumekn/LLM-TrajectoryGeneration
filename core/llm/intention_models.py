"""
core/llm/intention_models.py - 意图相关函数和类

包含：
1. 统一 LLM 客户端 (UnifiedLLMClient)

使用示例：
    from core.llm.intention_models import UnifiedLLMClient

    client = UnifiedLLMClient(provider="qwen", model="qwen3.6-plus")
    client.chat("你好")
"""

import os
from typing import Dict, Any, Optional


# =============================================================================
# 统一 LLM 客户端
# =============================================================================

def list_providers() -> Dict[str, str]:
    """列出所有支持的提供商"""
    return {
        "qwen": "阿里千问 (qwen3.6-plus, qwen-plus, qwen-max, qwen-turbo)",
        "openai": "OpenAI GPT (gpt-4o, gpt-4o-mini, gpt-4-turbo)",
        "gemini": "Google Gemini (gemini-2.0-flash, gemini-2.0-pro)",
    }


def get_provider_config(provider: str) -> Dict[str, Any]:
    """获取提供商配置"""
    configs = {
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
    if provider not in configs:
        raise ValueError(f"不支持的提供商: {provider}，支持的: {list(configs.keys())}")
    return configs[provider]


class UnifiedLLMClient:
    """
    统一 LLM 客户端

    使用示例：
        client = UnifiedLLMClient(provider="qwen", model="qwen3.6-plus")
        response = client.chat("你好")
    """

    def __init__(self, provider: str, model: str, api_key: Optional[str] = None):
        config = get_provider_config(provider)
        self.provider = provider
        self.model = model

        self.api_key = api_key or os.environ.get(config["api_key_env"])
        if not self.api_key:
            raise ValueError(f"请设置环境变量 {config['api_key_env']}")

        self.base_url = config["base_url"]
        self._init_client()

    def _init_client(self):
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
        return f"{self.provider}-{self.model}"

    def chat(self, prompt: str, **kwargs) -> str:
        """发送对话请求，返回文本响应"""
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
