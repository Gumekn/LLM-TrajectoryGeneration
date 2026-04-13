"""
测试 Step 1: 构造轨迹信息提示词

用法：
    python test_step1_prompt.py [scenario_id] [ego_id]

示例：
    python test_step1_prompt.py 10135f16cd538e19 1811
"""

import sys
import json
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_intention_generator import build_trajectory_prompt


def test_step1(scenario_id: str = "10135f16cd538e19", ego_id: str = "1811"):
    """测试 Step 1：构造轨迹信息提示词"""

    data_dir = "data/processed"
    file_path = os.path.join(data_dir, f"{scenario_id}_{ego_id}.json")

    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"加载数据: {file_path}")
    print(f"场景ID: {data['metadata']['scenario_id']}")
    print(f"片段数量: {len(data['fragments'])}")

    if not data['fragments']:
        print("警告: 没有片段数据")
        return

    fragment = data['fragments'][0]

    print("\n" + "=" * 70)
    print("Step 1: 构造轨迹信息提示词")
    print("=" * 70 + "\n")

    # 构造提示词
    prompt = build_trajectory_prompt(fragment)

    print(prompt)


if __name__ == "__main__":
    scenario_id = sys.argv[1] if len(sys.argv) > 1 else "10135f16cd538e19"
    ego_id = sys.argv[2] if len(sys.argv) > 2 else "1811"

    test_step1(scenario_id, ego_id)
