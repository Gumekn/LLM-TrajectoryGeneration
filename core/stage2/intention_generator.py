"""
core/stage2/intention_generator.py - 意图生成主入口

职责：
- 数据流编排：读取 → LLM调用 → 存储
- 参数配置
- 对外提供 run_* 便捷函数

存储功能已分离到 core/stage2/storage.py
LLM调用在 core/stage2/llm/intention_models.py
提示词构建在 core/stage2/llm/prompt_builder.py
轨迹变异在 core/stage2/mutator.py
"""

from core.stage2.llm.intention_models import LLMIntentionGenerator
from core.stage2.mutator import TrajectoryMutator
from core.stage2.storage import (
    save_fragment_with_intention,
    save_variants_to_json,
)


# =============================================================================
# 默认参数配置
# =============================================================================

DEFAULT_PROVIDER = "qwen"
DEFAULT_MODEL = "qwen3.6-plus"
DEFAULT_INTENTION_DIR = "data/intention"
DEFAULT_VARIANT_DIR = "data/variants"


# =============================================================================
# 意图分析主流程
# =============================================================================

def run_intention_analysis(
    fragment: dict,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    output_dir: str = DEFAULT_INTENTION_DIR,
    save: bool = True,
) -> dict:
    """
    意图分析完整流程：轨迹提示词构建 → LLM调用 → 解析结果

    Args:
        fragment: 轨迹片段数据
        provider: LLM提供商
        model: 模型名称
        output_dir: 输出目录
        save: 是否保存结果

    Returns:
        意图分析结果 dict，包含 intention_frames 和 trajectory_prompt
    """
    generator = LLMIntentionGenerator(provider=provider, model=model)
    result = generator.generate(fragment)

    if save:
        save_fragment_with_intention(
            fragment, result, output_dir, provider, model
        )

    return result


def run_intention_analysis_with_fallback(
    fragment: dict,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    output_dir: str = DEFAULT_INTENTION_DIR,
) -> dict:
    """
    带降级的意图分析（LLM失败时返回空意图）

    Args:
        fragment: 轨迹片段数据
        provider: LLM提供商
        model: 模型名称
        output_dir: 输出目录

    Returns:
        意图分析结果，失败时 intention_frames 为空列表
    """
    generator = LLMIntentionGenerator(provider=provider, model=model)
    result = generator.generate_with_fallback(fragment)

    save_fragment_with_intention(
        fragment, result, output_dir, provider, model
    )

    return result


# =============================================================================
# 轨迹变异主流程
# =============================================================================

def run_mutation(
    fragment_with_intention: dict,
    top_k: float = 10.0,
    output_dir: str = DEFAULT_VARIANT_DIR,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    save: bool = True,
    random_seed: int = None,
) -> list:
    """
    基于意图分析结果生成轨迹变异

    Args:
        fragment_with_intention: 带意图的片段数据
        top_k: 保留危险分数前K%的分支
        output_dir: 输出目录
        provider: LLM提供商（仅用于保存元信息）
        model: LLM模型（仅用于保存元信息）
        save: 是否保存结果
        random_seed: 随机种子

    Returns:
        变异轨迹列表
    """
    mutator = TrajectoryMutator(random_seed=random_seed)
    variants = mutator.mutate(fragment_with_intention, top_k=top_k)

    if save:
        save_variants_to_json(
            variants, fragment_with_intention, output_dir, provider, model
        )

    return variants


# =============================================================================
# 便捷函数（兼容旧API）
# =============================================================================

def generate_trajectory_variants(
    fragment_with_intention: dict,
    top_k: float = 10.0,
    random_seed: int = None,
) -> list:
    """generate_trajectory_variants 的别名，保持向后兼容"""
    return run_mutation(
        fragment_with_intention,
        top_k=top_k,
        save=False,
        random_seed=random_seed,
    )