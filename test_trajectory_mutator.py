"""
test_trajectory_mutator.py - 轨迹变异模块测试

用法：
    python test_trajectory_mutator.py [intention_json_path]

示例：
    python test_trajectory_mutator.py data/intention/frag_10135f16cd538e19_1811_1707_87_with_intention.json
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.stage2.mutator import (
    merge_intentions,
    TrajectoryMutator,
    IntentBlock,
    DrivingIntention,
)
from core.stage2.intention_generator import (
    generate_trajectory_variants,
    save_variants_to_json,
)


def test_merge_intentions():
    """测试意图合并（支持变长轨迹）"""
    print("=" * 70)
    print("Test 1: merge_intentions with variable-length trajectories")
    print("=" * 70)

    # 模拟意图帧（数据中有 frame=0,18,26,50）
    intention_frames = [
        {"frame": 0, "intention": "cruise_maintain"},
        {"frame": 18, "intention": "decelerate_to_yield"},
        {"frame": 26, "intention": "accelerate_through"},
        {"frame": 50, "intention": "go_straight"},
    ]

    # 测试固定帧数（旧行为）
    blocks_old = merge_intentions(intention_frames, total_frames=50)
    print(f"[50 frames] Block count: {len(blocks_old)}")
    for b in blocks_old:
        print(f"  {b.intent.value}: frame [{b.start_frame}, {b.end_frame}), duration={b.duration:.1f}s")

    # 测试51帧（实际数据）
    blocks_51 = merge_intentions(intention_frames, total_frames=51)
    print(f"\n[51 frames] Block count: {len(blocks_51)}")
    for b in blocks_51:
        print(f"  {b.intent.value}: frame [{b.start_frame}, {b.end_frame}), duration={b.duration:.1f}s")

    # 验证最后一个Block延伸到总帧数
    last_block = blocks_51[-1]
    assert last_block.end_frame == 51, f"Last block end_frame should be 51, got {last_block.end_frame}"
    print(f"\n[OK] Last block extends to total frames: end_frame={last_block.end_frame}")


def test_mutator_with_real_data(json_path: str):
    """使用真实数据测试轨迹变异"""
    print("\n" + "=" * 70)
    print("Test 2: TrajectoryMutator with real data")
    print("=" * 70)

    if not os.path.exists(json_path):
        print(f"Error: File not found {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 显示数据基本信息
    orig = data.get("original_fragment", data)
    meta = orig.get("metadata", {})
    frame_count = meta.get("frame_count", 0)
    duration = meta.get("duration", 0)

    print(f"Fragment ID: {meta.get('fragment_id', 'unknown')}")
    print(f"Frame count: {frame_count}, Duration: {duration}s")
    print(f"Danger level: {meta.get('danger_level', 'unknown')}")

    # 提取意图帧
    intention_analysis = data.get("intention_analysis", {})
    intention_frames = intention_analysis.get("intention_frames", [])

    intention_sequence = data.get("intention_sequence", {})
    seq_frames = intention_sequence.get("intention_sequence", [])

    print(f"Intention frames: {len(intention_frames)} (format1), {len(seq_frames)} (format2)")

    # 测试变异生成（top_k=10）
    print("\nGenerating trajectory variants (top_k=10%)...")
    try:
        variants = generate_trajectory_variants(data, top_k=10.0, random_seed=42)
        print(f"[OK] Generated {len(variants)} variants")

        if variants:
            v0 = variants[0]
            print(f"  First variant frame count: {len(v0.get('positions', []))}")
            print(f"  Fields: {list(v0.keys())}")

        # 保存到 JSON
        output_path = save_variants_to_json(variants, data)
        print(f"[OK] Saved to: {output_path}")

    except Exception as e:
        print(f"[FAIL] Variant generation failed: {e}")
        import traceback
        traceback.print_exc()


def test_combination_count():
    """测试变异组合数计算"""
    print("\n" + "=" * 70)
    print("Test 3: Mutation combination counts per intention")
    print("=" * 70)

    mutator = TrajectoryMutator()
    mutations = mutator.mutations

    for intent, mutation_list in mutations.items():
        for m in mutation_list:
            if m.is_perturbation:
                combo = 1
            else:
                combo = len(m.delta_a_mag_values) * len(m.delta_omega_values)
            print(f"  {intent.value}: {combo} combinations")


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else (
        "data/intention/frag_10135f16cd538e19_1811_1707_87_with_intention.json"
    )

    test_merge_intentions()
    test_combination_count()
    test_mutator_with_real_data(json_path)

    print("\n" + "=" * 70)
    print("All tests finished!")
    print("=" * 70)
