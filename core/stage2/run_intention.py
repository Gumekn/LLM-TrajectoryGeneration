"""
core/stage2/run_intention.py - 意图生成入口

用法:
    python -m core.stage2.run_intention --input data/processed/xxx.json
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

from core.stage2.intention_generator import (
    run_intention_analysis,
    run_intention_analysis_with_fallback,
)


def main():
    # =============================================================
    # 意图生成配置（在此修改）
    # =============================================================
    PROVIDER = "qwen"
    MODEL = "qwen3.6-plus"
    INPUT_DIR = "data/processed"      # stage1 输出目录
    INTENTION_OUTPUT_DIR = "data/intention"
    # =============================================================

    parser = argparse.ArgumentParser(description="意图生成")
    parser.add_argument("--input", "-i", default=INPUT_DIR, help="输入 JSON 文件路径或目录（支持单个片段或包含 fragments 数组）")
    parser.add_argument("--provider", default=PROVIDER)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output-dir", default=INTENTION_OUTPUT_DIR)
    parser.add_argument("--fallback", action="store_true", help="启用降级模式（LLM失败时返回空意图）")
    args = parser.parse_args()

    # 支持目录输入：遍历目录下所有 .json 文件
    input_path = args.input
    if os.path.isdir(input_path):
        json_files = sorted([
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".json")
        ])
        if not json_files:
            print(f"目录为空: {input_path}")
            return
        print(f"检测到 {len(json_files)} 个文件，开始逐一生成意图...\n")
    else:
        json_files = [input_path]

    for file_path in json_files:
        print(f"\n处理文件: {file_path}")
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if "fragments" in data:
            fragments = data["fragments"]
            top_meta = data.get("metadata", {})
            print(f"  检测到 {len(fragments)} 个片段...")
            for i, frag in enumerate(fragments):
                combined = {**top_meta, **frag}
                combined["fragment"] = frag
                print(f"  [{i+1}/{len(fragments)}] {frag.get('fragment_id', 'unknown')}")
                run_fn = run_intention_analysis_with_fallback if args.fallback else run_intention_analysis
                result = run_fn(
                    combined,
                    provider=args.provider,
                    model=args.model,
                    output_dir=args.output_dir,
                )
                print(f"    → {len(result['intention_frames'])} 个意图帧")
        else:
            run_fn = run_intention_analysis_with_fallback if args.fallback else run_intention_analysis
            result = run_fn(
                data,
                provider=args.provider,
                model=args.model,
                output_dir=args.output_dir,
            )
            print(f"  → {len(result['intention_frames'])} 个意图帧")

    print(f"\n全部完成，结果保存至: {args.output_dir}/")


if __name__ == "__main__":
    main()