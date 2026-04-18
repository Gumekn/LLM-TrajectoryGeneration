"""
core/stage2/run_mutation.py - 轨迹变异入口

用法:
    python -m core.stage2.run_mutation --input data/intention/xxx_with_intention.json
    或直接双击运行本文件
"""



import sys
import os

# 确保项目根目录在 sys.path 中（支持直接双击运行，必须在最前面）
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import json
    
from core.stage2.intention_generator import run_mutation


def main():
    # =============================================================
    # 轨迹变异配置（在此修改）
    # =============================================================
    INTENTION_INPUT_DIR = "data/intention"   # run_intention 输出目录
    VARIANT_OUTPUT_DIR = "data/variants"
    TOP_K = 100.0
    RANDOM_SEED = None
    # =============================================================

    parser = argparse.ArgumentParser(description="轨迹变异")
    parser.add_argument("--input", "-i", default=INTENTION_INPUT_DIR, help="输入带意图片段 JSON 文件路径或目录")
    parser.add_argument("--output-dir", default=VARIANT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=float, default=TOP_K, help="保留危险分数前K%%的分支")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子")
    args = parser.parse_args()

    # 支持目录输入：遍历目录下所有 _with_intention.json 文件
    input_path = args.input
    if os.path.isdir(input_path):
        json_files = sorted([
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith("_with_intention.json")
        ])
        if not json_files:
            print(f"目录为空或无 *_with_intention.json 文件: {input_path}")
            return
        print(f"检测到 {len(json_files)} 个文件，开始逐一变异...\n")
    else:
        json_files = [input_path]

    total_variants = 0
    for file_path in json_files:
        print(f"处理文件: {file_path}")
        with open(file_path, encoding="utf-8") as f:
            fragment_with_intention = json.load(f)

        variants = run_mutation(
            fragment_with_intention,
            top_k=args.top_k,
            output_dir=args.output_dir,
            save=True,
            random_seed=args.seed,
        )
        print(f"  → {len(variants)} 条变异轨迹")
        total_variants += len(variants)

    print(f"\n全部完成，共生成 {total_variants} 条变异轨迹，结果保存至: {args.output_dir}/")


if __name__ == "__main__":
    main()