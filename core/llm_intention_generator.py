"""
core/llm_intention_generator.py - LLM 意图生成器

调用流程：
1. build_trajectory_prompt() - 从 prompt_builder.py 调用轨迹提示词构造
2. generate_intention() - 调用 LLM API 生成意图（已注释）

提示词构建统一在 prompt_builder.py 中，方便后续优化。

使用示例：
    from core.llm_intention_generator import build_trajectory_prompt

    prompt = build_trajectory_prompt(fragment)
"""

from core.llm.prompt_builder import (
    TrajectoryPromptBuilder,
    SYSTEM_PROMPT,
    DrivingIntention,
    IntentionPhase,
    IntentionSequence,
)
from core.llm.intention_models import UnifiedLLMClient

import os
import json



# =============================================================================
# 便捷函数 - 提示词构建
# =============================================================================

def build_trajectory_prompt(fragment: dict, sample_interval: int = 2) -> str:
    """
    构造轨迹信息提示词（Step 1）

    Args:
        fragment: 轨迹片段数据
        sample_interval: 采样间隔，默认2

    Returns:
        格式化后的轨迹信息文本
    """
    builder = TrajectoryPromptBuilder(sample_interval=sample_interval)
    return builder.build_prompt(fragment)


def get_system_prompt() -> str:
    """获取系统提示词"""
    return SYSTEM_PROMPT


# =============================================================================
# 意图生成器（Step 2 已注释）
# =============================================================================

class LLMIntentionGenerator:
    """
    基于 LLM 的意图生成器

    使用示例：
        generator = LLMIntentionGenerator(provider="qwen", model="qwen3.6-plus")
        intention_seq = generator.generate(fragment)
    """

    def __init__(self, provider: str = "qwen", model: str = "qwen3.6-plus", api_key: str = None):
        self.client = UnifiedLLMClient(provider=provider, model=model, api_key=api_key)
        self.provider = provider
        self.model = model
        self.prompt_builder = TrajectoryPromptBuilder()

    def build_trajectory_prompt(self, fragment: dict) -> str:
        """Step 1: 构造轨迹信息提示词"""
        return self.prompt_builder.build_prompt(fragment)

    def generate(self, fragment: dict) -> IntentionSequence:
        """完整流程：Step 1 + Step 2"""
        trajectory_prompt = self.build_trajectory_prompt(fragment)

        print("【提示】Step 2 已注释，请测试 Step 1 确认无误后取消注释")
        return self._default_intention_sequence(fragment, trajectory_prompt)

    def _default_intention_sequence(self, fragment: dict, trajectory_prompt: str = "") -> IntentionSequence:
        """返回默认意图序列"""
        frag = fragment.get("fragment", fragment) if "fragment" in fragment else fragment
        frame_count = frag.get("frame_count", 50)

        return IntentionSequence(
            phases=[
                IntentionPhase(
                    start_frame=0,
                    end_frame=frame_count,
                    intention=DrivingIntention.CRUISE_MAINTAIN,
                    confidence=0.5,
                    reasoning="默认匀速意图"
                )
            ],
            overall_strategy="默认策略",
            raw_response="",
            trajectory_prompt=trajectory_prompt
        )


# =============================================================================
# 数据存储
# =============================================================================

DEFAULT_INTENTION_DIR = "data/intention"


def save_fragment_with_intention(
    fragment: dict,
    intention_seq: IntentionSequence,
    output_dir: str = DEFAULT_INTENTION_DIR,
    provider: str = "qwen",
    model: str = "qwen3.6-plus"
) -> str:
    """
    保存带意图的片段数据

    Args:
        fragment: 原始片段数据
        intention_seq: 意图序列
        output_dir: 输出目录
        provider: LLM 提供商
        model: LLM 模型

    Returns:
        保存的文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    meta = fragment.get("metadata", {})
    frag = fragment.get("fragment", fragment) if "fragment" in fragment else fragment
    scenario_id = meta.get("scenario_id", "unknown")
    ego_id = meta.get("ego_vehicle_id", "unknown")
    target_id = frag.get("target_trajectory", {}).get("vehicle_id", "unknown")
    fragment_id = frag.get("fragment_id", f"{scenario_id}_{ego_id}_{target_id}")

    file_name = f"{fragment_id}_with_intention.json"
    file_path = os.path.join(output_dir, file_name)

    output_data = {
        "original_fragment": fragment,
        "intention_sequence": intention_seq.to_dict(),
        "trajectory_prompt": intention_seq.trajectory_prompt,
        "provider": provider,
        "model": model,
    }

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"已保存到: {file_path}")
    return file_path
