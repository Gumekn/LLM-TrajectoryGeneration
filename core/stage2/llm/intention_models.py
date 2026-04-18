"""
core/stage2/llm/intention_models.py - LLM 意图生成核心模块

职责：
- 意图帧数据类
- 关键帧识别
- LLM 调用与响应解析

注意：
- LLM 客户端已迁移到 core/stage2/llm/config.py
- 轨迹变异相关功能在 core/stage2/mutator.py

使用示例：
    from core.stage2.llm.intention_models import identify_key_frames, parse_intention_response
    from core.stage2.llm.config import UnifiedLLMClient

    # 关键帧识别
    key_frames = identify_key_frames(fragment)
"""

import json
import re
from typing import Dict, Any, List
from dataclasses import dataclass

from core.stage2.llm.config import UnifiedLLMClient


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
        """转换为字典格式"""
        return {
            "frame": self.frame,
            "frame_type": self.frame_type,
            "intention": self.intention,
            "reasoning": self.reasoning,
        }


# =============================================================================
# 意图生成函数
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

    参数：
        fragment: 轨迹片段数据（包含 metadata, interaction_features 等）

    返回：
        关键帧列表，每项包含：
            - frame: 帧索引
            - frame_type: 帧类型
    """
    # 获取 metadata
    frag_meta = fragment.get("metadata", fragment)
    anchor_frame = frag_meta.get("anchor_frame", 0)
    n_before = frag_meta.get("n_before", 0)
    n_after = frag_meta.get("n_after", 0)
    frame_count = frag_meta.get("frame_count", 50)

    # 获取交互特征
    ifeatures = fragment.get("interaction_features", {})
    ttc_long = ifeatures.get("ttc_long", [])
    rel_dist = ifeatures.get("rel_dist", [])
    rel_vel_x = ifeatures.get("rel_vel_x", [])

    key_frames = []

    # 1. anchor - 锚点帧（最危险时刻）
    key_frames.append({
        "frame": anchor_frame,
        "frame_type": "anchor"
    })

    # 2. min_ttc - 最小TTC帧
    if ttc_long:
        valid_ttc = [(i, t) for i, t in enumerate(ttc_long) if t < float('inf')]
        if valid_ttc:
            min_ttc_idx = min(valid_ttc, key=lambda x: x[1])[0]
            if min_ttc_idx != anchor_frame:  # 避免重复
                key_frames.append({
                    "frame": min_ttc_idx,
                    "frame_type": "min_ttc"
                })

    # 3. min_dist - 最近距离帧
    if rel_dist:
        min_dist_idx = min(range(len(rel_dist)), key=lambda i: rel_dist[i])
        if min_dist_idx != anchor_frame:  # 避免重复
            key_frames.append({
                "frame": min_dist_idx,
                "frame_type": "min_dist"
            })

    # 4. max_closing - 最大接近速度帧
    if rel_vel_x:
        # rel_vel_x 为负表示接近，取最小值（最大接近速度）
        min_idx = min(range(len(rel_vel_x)), key=lambda i: rel_vel_x[i])
        if min_idx != anchor_frame:  # 避免重复
            key_frames.append({
                "frame": min_idx,
                "frame_type": "max_closing"
            })

    # 5. start - 起始帧
    if anchor_frame != 0:
        key_frames.append({
            "frame": 0,
            "frame_type": "start"
        })

    # 6. end - 结束帧
    if anchor_frame != frame_count - 1:
        key_frames.append({
            "frame": frame_count - 1,
            "frame_type": "end"
        })

    # 7. pre_mid - 锚点前中点帧
    if n_before > 0:
        pre_mid_frame = anchor_frame - n_before // 2
        if pre_mid_frame > 0 and pre_mid_frame != anchor_frame:
            key_frames.append({
                "frame": pre_mid_frame,
                "frame_type": "pre_mid"
            })

    # 8. post_mid - 锚点后中点帧
    if n_after > 0:
        post_mid_frame = anchor_frame + n_after // 2
        if post_mid_frame < frame_count - 1 and post_mid_frame != anchor_frame:
            key_frames.append({
                "frame": post_mid_frame,
                "frame_type": "post_mid"
            })

    # 按帧索引排序
    key_frames.sort(key=lambda x: x["frame"])

    return key_frames


def generate_intention(
    client: UnifiedLLMClient,
    full_prompt: str,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 4096
) -> str:
    """
    调用 LLM 生成意图

    参数：
        client: UnifiedLLMClient 实例
        full_prompt: 完整的询问提示词（系统提示词 + 轨迹信息 + 用户询问）
        model: 模型名称
        temperature: 温度参数（默认0.3，低温度更确定性）
        max_tokens: 最大token数（默认4096）

    返回：
        LLM 原始响应文本
    """
    return client.chat(
        full_prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )


def extract_json_from_response(text: str) -> str:
    """
    从 LLM 响应文本中提取 JSON 字符串

    支持格式：
        ```json
        {...}
        ```
    或直接返回 {...}

    参数：
        text: LLM 响应文本

    返回：
        JSON 字符串
    """
    # 尝试提取 ```json ... ``` 包裹的JSON
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if match:
        return match.group(1)

    # 尝试直接提取 JSON 对象
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)

    return text


def parse_intention_response(
    response: str,
    key_frames: List[Dict[str, Any]],
    trajectory_prompt: str
) -> Dict[str, Any]:
    """
    解析 LLM 响应为结构化意图数据

    参数：
        response: LLM 响应文本（应包含 JSON 格式的 intention_frames）
        key_frames: 关键帧列表（用于验证）
        trajectory_prompt: 轨迹提示词（原样传递到结果）

    返回：
        意图分析结果字典：
            - intention_frames: List[IntentionFrame]
            - trajectory_prompt: str
    """
    json_str = extract_json_from_response(response)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析失败: {e}\n原始响应: {response}")

    # 提取 intention_frames
    intention_frames = data.get("intention_frames", [])

    # 验证并标准化每帧数据
    result_frames = []
    for item in intention_frames:
        frame = item.get("frame")
        frame_type = item.get("frame_type", "")
        intention = item.get("intention", "")
        reasoning = item.get("reasoning", "")

        # 验证必要字段
        if frame is None or intention is None:
            continue

        result_frames.append(IntentionFrame(
            frame=frame,
            frame_type=frame_type,
            intention=intention,
            reasoning=reasoning
        ))

    # 按帧索引排序
    result_frames.sort(key=lambda x: x.frame)

    return {
        "intention_frames": result_frames,
        "trajectory_prompt": trajectory_prompt
    }


# =============================================================================
# LLM 意图生成器
# =============================================================================

from core.stage2.llm.prompt_builder import (
    TrajectoryPromptBuilder,
    SYSTEM_PROMPT,
    build_intention_query_prompt,
)


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
        response = generate_intention(
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
