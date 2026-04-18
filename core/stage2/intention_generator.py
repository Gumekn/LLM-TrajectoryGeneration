"""
core/stage2/intention_generator.py - 意图生成主入口

职责：
- 意图帧数据类 (IntentionFrame)
- 关键帧识别
- LLM 调用与响应解析
- 数据流编排：读取 → LLM调用 → 存储
- 对外提供 run_* 便捷函数

存储功能在 core/stage2/storage.py
LLM 客户端在 core/stage2/llm/config.py
提示词构建在 core/stage2/llm/prompt_builder.py
轨迹变异在 core/stage2/mutator.py
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, Any, List

from core.stage2.llm.config import UnifiedLLMClient
from core.stage2.llm.prompt_builder import (
    TrajectoryPromptBuilder,
    SYSTEM_PROMPT,
    build_intention_query_prompt,
)
from core.stage2.mutator import TrajectoryMutator
from core.stage2.storage import (
    save_fragment_with_intention,
    save_variants_to_json,
)


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class IntentionFrame:
    """
    意图帧 - 单个关键帧的驾驶意图分析结果

    属性：
        frame: 帧索引
        frame_type: 帧类型 (anchor/min_ttc/min_dist/max_closing/start/end/pre_mid/post_mid)
        intention: 驾驶意图字符串 (accelerate_through/decelerate_to_yield 等)
        reasoning: 推理原因（不超过10字）
    """
    frame: int
    frame_type: str
    intention: str
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame,
            "frame_type": self.frame_type,
            "intention": self.intention,
            "reasoning": self.reasoning,
        }


# =============================================================================
# 关键帧识别
# =============================================================================

def identify_key_frames(fragment: dict) -> List[Dict[str, Any]]:
    """
    从轨迹片段中识别关键帧

    关键帧类型：
        - anchor: 锚点帧（最危险时刻）
        - min_ttc: 最小TTC帧（碰撞时间最短）
        - min_dist: 最近距离帧（两车距离最近）
        - max_closing: 最大接近速度帧（接近最快）
        - start: 起始帧（frame 0）
        - end: 结束帧（最后一帧）
        - pre_mid: 锚点前中点帧
        - post_mid: 锚点后中点帧
    """
    frag_meta = fragment.get("metadata", fragment)
    anchor_frame = frag_meta.get("anchor_frame", 0)
    n_before = frag_meta.get("n_before", 0)
    n_after = frag_meta.get("n_after", 0)
    frame_count = frag_meta.get("frame_count", 50)

    ifeatures = fragment.get("interaction_features", {})
    ttc_long = ifeatures.get("ttc_long", [])
    rel_dist = ifeatures.get("rel_dist", [])
    rel_vel_x = ifeatures.get("rel_vel_x", [])

    key_frames = []

    key_frames.append({"frame": anchor_frame, "frame_type": "anchor"})

    if ttc_long:
        valid_ttc = [(i, t) for i, t in enumerate(ttc_long) if t < float('inf')]
        if valid_ttc:
            min_ttc_idx = min(valid_ttc, key=lambda x: x[1])[0]
            if min_ttc_idx != anchor_frame:
                key_frames.append({"frame": min_ttc_idx, "frame_type": "min_ttc"})

    if rel_dist:
        min_dist_idx = min(range(len(rel_dist)), key=lambda i: rel_dist[i])
        if min_dist_idx != anchor_frame:
            key_frames.append({"frame": min_dist_idx, "frame_type": "min_dist"})

    if rel_vel_x:
        min_idx = min(range(len(rel_vel_x)), key=lambda i: rel_vel_x[i])
        if min_idx != anchor_frame:
            key_frames.append({"frame": min_idx, "frame_type": "max_closing"})

    if anchor_frame != 0:
        key_frames.append({"frame": 0, "frame_type": "start"})

    if anchor_frame != frame_count - 1:
        key_frames.append({"frame": frame_count - 1, "frame_type": "end"})

    if n_before > 0:
        pre_mid_frame = anchor_frame - n_before // 2
        if pre_mid_frame > 0 and pre_mid_frame != anchor_frame:
            key_frames.append({"frame": pre_mid_frame, "frame_type": "pre_mid"})

    if n_after > 0:
        post_mid_frame = anchor_frame + n_after // 2
        if post_mid_frame < frame_count - 1 and post_mid_frame != anchor_frame:
            key_frames.append({"frame": post_mid_frame, "frame_type": "post_mid"})

    key_frames.sort(key=lambda x: x["frame"])
    return key_frames


# =============================================================================
# LLM 调用与解析
# =============================================================================

def generate_intention(
    client: UnifiedLLMClient,
    full_prompt: str,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 4096
) -> str:
    """调用 LLM 生成意图"""
    return client.chat(full_prompt, temperature=temperature, max_tokens=max_tokens)


def extract_json_from_response(text: str) -> str:
    """从 LLM 响应文本中提取 JSON 字符串"""
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if match:
        return match.group(1)
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    return text


def parse_intention_response(
    response: str,
    key_frames: List[Dict[str, Any]],
    trajectory_prompt: str
) -> Dict[str, Any]:
    """解析 LLM 响应为结构化意图数据"""
    json_str = extract_json_from_response(response)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析失败: {e}\n原始响应: {response}")

    intention_frames = data.get("intention_frames", [])
    result_frames = []
    for item in intention_frames:
        frame = item.get("frame")
        frame_type = item.get("frame_type", "")
        intention = item.get("intention", "")
        reasoning = item.get("reasoning", "")
        if frame is None or intention is None:
            continue
        result_frames.append(IntentionFrame(
            frame=frame,
            frame_type=frame_type,
            intention=intention,
            reasoning=reasoning
        ))

    result_frames.sort(key=lambda x: x.frame)
    return {"intention_frames": result_frames, "trajectory_prompt": trajectory_prompt}


# =============================================================================
# LLM 意图生成器
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
        return self.prompt_builder.build_prompt(fragment)

    def generate(self, fragment: dict) -> Dict[str, Any]:
        """完整流程：轨迹提示词 → 关键帧 → LLM调用 → 解析响应"""
        trajectory_prompt = self.build_trajectory_prompt(fragment)
        key_frames = identify_key_frames(fragment)
        full_prompt = build_intention_query_prompt(trajectory_prompt, key_frames, SYSTEM_PROMPT)
        response = generate_intention(self.client, full_prompt, self.model)
        return parse_intention_response(response, key_frames, trajectory_prompt)

    def generate_with_fallback(self, fragment: dict) -> Dict[str, Any]:
        """带默认值的意图生成（LLM 失败时返回空列表）"""
        try:
            return self.generate(fragment)
        except Exception as e:
            print(f"意图生成失败: {e}")
            trajectory_prompt = self.build_trajectory_prompt(fragment)
            return {"intention_frames": [], "trajectory_prompt": trajectory_prompt, "error": str(e)}


# =============================================================================
# 意图分析主流程
# =============================================================================

def run_intention_analysis(
    fragment: dict,
    provider: str,
    model: str,
    output_dir: str,
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
        save_fragment_with_intention(fragment, result, output_dir, provider, model)

    return result


def run_intention_analysis_with_fallback(
    fragment: dict,
    provider: str,
    model: str,
    output_dir: str,
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
    save_fragment_with_intention(fragment, result, output_dir, provider, model)
    return result


# =============================================================================
# 轨迹变异主流程
# =============================================================================

def run_mutation(
    fragment_with_intention: dict,
    top_k: float,
    output_dir: str,
    save: bool = True,
    random_seed: int = None,
) -> list:
    """
    基于意图分析结果生成轨迹变异

    Args:
        fragment_with_intention: 带意图的片段数据
        top_k: 保留危险分数前K%的分支
        output_dir: 输出目录
        save: 是否保存结果
        random_seed: 随机种子

    Returns:
        变异轨迹列表
    """
    mutator = TrajectoryMutator(random_seed=random_seed)
    variants = mutator.mutate(fragment_with_intention, top_k=top_k)

    if save:
        save_variants_to_json(variants, fragment_with_intention, output_dir)

    return variants


def generate_trajectory_variants(
    fragment_with_intention: dict,
    top_k: float,
    output_dir: str,
    random_seed: int = None,
) -> list:
    """generate_trajectory_variants 的别名，保持向后兼容"""
    return run_mutation(
        fragment_with_intention,
        top_k=top_k,
        output_dir=output_dir,
        save=False,
        random_seed=random_seed,
    )