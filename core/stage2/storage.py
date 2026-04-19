"""
core/stage2/storage.py - 数据持久化模块

职责：
- 带意图片段的保存
- 轨迹变异的保存

数据类定义在 core/stage2/intention_generator.py (IntentionFrame)
"""

import os
import json
from typing import Dict, Any, List


# =============================================================================
# 片段 + 意图分析 结果保存
# =============================================================================

def save_fragment_with_intention(
    fragment: dict,
    intention_result: Dict[str, Any],
    output_dir: str,
    provider: str,
    model: str,
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

    frag = fragment.get("fragment", fragment) if "fragment" in fragment else fragment
    meta = fragment.get("metadata", frag.get("metadata", {}))
    scenario_id = meta.get("scenario_id", "unknown")
    ego_id = meta.get("ego_vehicle_id", "unknown")
    target_id = frag.get("target_trajectory", {}).get("vehicle_id", "unknown")
    fragment_id = frag.get("fragment_id", f"{scenario_id}_{ego_id}_{target_id}")

    file_name = f"{fragment_id}_with_intention.json"
    file_path = os.path.join(output_dir, file_name)

    intention_frames = intention_result.get("intention_frames", [])

    intention_frames_data = []
    if isinstance(intention_frames, list):
        for f in intention_frames:
            if hasattr(f, 'to_dict'):
                intention_frames_data.append(f.to_dict())
            elif isinstance(f, dict):
                intention_frames_data.append(f)

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
        },
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"已保存到: {file_path}")
    return file_path


def save_variants_to_json(
    variants: List[Dict],
    fragment: dict,
    output_dir: str,
    algorithm: str = "dfs-topk",
) -> str:
    """
    将轨迹变异结果持久化到 JSON 文件

    Args:
        variants: 变异轨迹列表
        fragment: 原始片段数据
        output_dir: 输出目录
        algorithm: 变异算法标识

    Returns:
        保存的文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    frag = fragment.get("original_fragment", fragment)
    meta = frag.get("metadata", {})
    scenario_id = meta.get("scenario_id", "unknown")
    ego_id = meta.get("ego_vehicle_id", "unknown")
    target_id = frag.get("target_trajectory", {}).get("vehicle_id", "unknown")
    fragment_id = frag.get("fragment_id", f"{scenario_id}_{ego_id}_{target_id}")

    file_name = f"{fragment_id}_variants.json"
    file_path = os.path.join(output_dir, file_name)

    ego_trajectory = frag.get("ego_trajectory", {})
    original_target_trajectory = frag.get("target_trajectory", {})

    output_data = {
        "fragment_id": fragment_id,
        "metadata": meta,
        "ego_trajectory": ego_trajectory,
        "original_target_trajectory": original_target_trajectory,
        "variant_count": len(variants),
        "variants": variants,
        "generation_info": {"algorithm": algorithm},
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"已保存 {len(variants)} 条变异轨迹到: {file_path}")
    return file_path