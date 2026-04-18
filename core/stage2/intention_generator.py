"""
core/stage2/intention_generator.py - LLM 意图生成器（第二阶段：意图分析）

调用流程：
1. build_trajectory_prompt() - 构造轨迹信息提示词
2. generate_intention() - 调用 LLM API 生成意图
3. generate_trajectory_variants() - 基于意图生成轨迹变异（第三阶段）

提示词构建统一在 core/stage2/llm/prompt_builder.py 中，方便后续优化。

使用示例：
    from core.stage2.intention_generator import build_trajectory_prompt, generate_intention

    # Step 1: 构造轨迹提示词
    prompt = build_trajectory_prompt(fragment)

    # Step 2: 生成意图
    generator = LLMIntentionGenerator(provider="qwen", model="qwen3.6-plus")
    intention_result = generator.generate(fragment)

    # Step 3: 生成轨迹变异
    from core.stage2.intention_generator import generate_trajectory_variants, save_variants_to_json
    variants = generate_trajectory_variants(intention_result)
    save_variants_to_json(variants, intention_result)
"""

from core.stage2.llm.prompt_builder import (
    TrajectoryPromptBuilder,
    SYSTEM_PROMPT,
    build_intention_query_prompt,
)
from core.stage2.llm.intention_models import (
    UnifiedLLMClient,
    identify_key_frames,
    generate_intention as call_llm_generate_intention,
    parse_intention_response,
    IntentionFrame,
)

import os
import json
from typing import Dict, Any


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
# 意图生成器
# =============================================================================

class LLMIntentionGenerator:
    """
    基于 LLM 的意图生成器

    使用示例：
        generator = LLMIntentionGenerator(provider="qwen", model="qwen3.6-plus")
        intention_result = generator.generate(fragment)
    """

    def __init__(self, provider: str = "qwen", model: str = "qwen3.6-plus", api_key: str = None):
        self.client = UnifiedLLMClient(provider=provider, model=model, api_key=api_key)
        self.provider = provider
        self.model = model
        self.prompt_builder = TrajectoryPromptBuilder()

    def build_trajectory_prompt(self, fragment: dict) -> str:
        """Step 1: 构造轨迹信息提示词"""
        return self.prompt_builder.build_prompt(fragment)

    def generate(self, fragment: dict) -> Dict[str, Any]:
        """
        完整流程：Step 1 + Step 2

        Args:
            fragment: 轨迹片段数据

        Returns:
            意图分析结果，包含 intention_frames 列表和 trajectory_prompt
        """
        # Step 1: 轨迹提示词
        trajectory_prompt = self.build_trajectory_prompt(fragment)

        # Step 2: 计算关键帧
        key_frames = identify_key_frames(fragment)

        # Step 3: 构造完整询问提示词
        full_prompt = build_intention_query_prompt(
            trajectory_prompt, key_frames, SYSTEM_PROMPT
        )

        # Step 4: 调用 LLM
        response = call_llm_generate_intention(
            self.client, full_prompt, self.model
        )

        # Step 5: 解析响应
        result = parse_intention_response(response, key_frames, trajectory_prompt)

        return result

    def generate_with_fallback(self, fragment: dict) -> Dict[str, Any]:
        """
        带默认值的意图生成（LLM 失败时返回空列表）

        Args:
            fragment: 轨迹片段数据

        Returns:
            意图分析结果
        """
        try:
            return self.generate(fragment)
        except Exception as e:
            print(f"意图生成失败: {e}")
            trajectory_prompt = self.build_trajectory_prompt(fragment)
            return {
                "intention_frames": [],
                "trajectory_prompt": trajectory_prompt,
                "error": str(e)
            }


# =============================================================================
# 数据存储
# =============================================================================

DEFAULT_INTENTION_DIR = "data/intention"


def save_fragment_with_intention(
    fragment: dict,
    intention_result: Dict[str, Any],
    output_dir: str = DEFAULT_INTENTION_DIR,
    provider: str = "qwen",
    model: str = "qwen3.6-plus"
) -> str:
    """
    保存带意图的片段数据

    Args:
        fragment: 原始片段数据
        intention_result: generate() 返回的意图分析结果
        output_dir: 输出目录
        provider: LLM 提供商
        model: LLM 模型

    Returns:
        保存的文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取片段信息
    frag = fragment.get("fragment", fragment) if "fragment" in fragment else fragment
    meta = fragment.get("metadata", frag.get("metadata", {}))
    scenario_id = meta.get("scenario_id", "unknown")
    ego_id = meta.get("ego_vehicle_id", "unknown")
    target_id = frag.get("target_trajectory", {}).get("vehicle_id", "unknown")
    fragment_id = frag.get("fragment_id", f"{scenario_id}_{ego_id}_{target_id}")

    file_name = f"{fragment_id}_with_intention.json"
    file_path = os.path.join(output_dir, file_name)

    # 构造输出数据
    intention_frames = intention_result.get("intention_frames", [])

    # 转换为可序列化格式
    intention_frames_data = []
    if isinstance(intention_frames, list):
        for f in intention_frames:
            if isinstance(f, IntentionFrame):
                intention_frames_data.append(f.to_dict())
            elif isinstance(f, dict):
                intention_frames_data.append(f)
    elif isinstance(intention_frames, list) and len(intention_frames) == 0:
        intention_frames_data = []

    output_data = {
        "fragment_id": fragment_id,
        "metadata": meta,
        "ego_trajectory": frag.get("ego_trajectory", {}),
        "target_trajectory": frag.get("target_trajectory", {}),
        "interaction_features": frag.get("interaction_features", {}),
        "interaction_stats": frag.get("interaction_stats", {}),
        "intention_analysis": {
            "provider": provider,
            "model": model,
            "intention_frames": intention_frames_data,
            "trajectory_prompt": intention_result.get("trajectory_prompt", ""),
        }
    }

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"已保存到: {file_path}")
    return file_path


# =============================================================================
# 轨迹变异便捷函数（第三阶段）
# =============================================================================

DEFAULT_VARIANT_DIR = "data/variants"


def generate_trajectory_variants(
    fragment_with_intention: dict,
    top_k: float = 10.0,
    random_seed: int = None
) -> list:
    """
    基于意图分析结果生成轨迹变异（第三阶段）

    支持多种输入格式：
    - 直接传入 generate() 的返回值
    - 带 original_fragment 嵌套的格式
    - 带 intention_sequence 的格式

    Args:
        fragment_with_intention: 包含意图分析结果的片段数据
        top_k: 保留危险分数前K%的分支
        random_seed: 随机种子

    Returns:
        变异轨迹列表
    """
    from core.stage2.mutator import TrajectoryMutator
    mutator = TrajectoryMutator(random_seed=random_seed)
    return mutator.mutate(fragment_with_intention, top_k=top_k)


def save_variants_to_json(
    variants: list,
    fragment: dict,
    output_dir: str = DEFAULT_VARIANT_DIR,
    provider: str = "qwen",
    model: str = "qwen3.6-plus"
) -> str:
    """将轨迹变异结果持久化到 JSON 文件"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    frag = fragment.get("original_fragment", fragment)
    meta = frag.get("metadata", {})
    scenario_id = meta.get("scenario_id", "unknown")
    ego_id = meta.get("ego_vehicle_id", "unknown")
    target_id = frag.get("target_trajectory", {}).get("vehicle_id", "unknown")
    fragment_id = frag.get("fragment_id", f"{scenario_id}_{ego_id}_{target_id}")

    file_name = f"{fragment_id}_variants.json"
    file_path = os.path.join(output_dir, file_name)

    output_data = {
        "fragment_id": fragment_id,
        "metadata": meta,
        "variant_count": len(variants),
        "variants": variants,
        "generation_info": {"provider": provider, "model": model},
    }

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"已保存 {len(variants)} 条变异轨迹到: {file_path}")
    return file_path
