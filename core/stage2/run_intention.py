"""
core/stage2/run_intention.py - 意图生成入口

用法:
    python -m core.stage2.run_intention -i data/processed/10135f16cd538e19_1811.json
"""

import argparse
import json

from core.stage2.intention_generator import (
    run_intention_analysis,
    run_intention_analysis_with_fallback,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    DEFAULT_INTENTION_DIR,
)


def main():
    parser = argparse.ArgumentParser(description="意图生成")
    parser.add_argument("--input", "-i", required=True, help="输入 JSON 文件路径（支持单个片段或包含 fragments 数组）")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_INTENTION_DIR)
    parser.add_argument("--fallback", action="store_true", help="启用降级模式（LLM失败时返回空意图）")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    # 判断是单个片段还是数组格式
    if "fragments" in data:
        # 数组格式：遍历每个片段逐一生成意图
        fragments = data["fragments"]
        top_meta = data.get("metadata", {})
        print(f"检测到 {len(fragments)} 个片段，开始逐一生成意图...")
        for i, frag in enumerate(fragments):
            # 片段元信息优先，顶级元信息作补充
            combined = {**top_meta, **frag}
            combined["fragment"] = frag
            print(f"\n[{i+1}/{len(fragments)}] 处理片段: {frag.get('fragment_id', 'unknown')}")
            run_fn = run_intention_analysis_with_fallback if args.fallback else run_intention_analysis
            result = run_fn(
                combined,
                provider=args.provider,
                model=args.model,
                output_dir=args.output_dir,
            )
            print(f"  → {len(result['intention_frames'])} 个意图帧")
    else:
        # 单片段格式：直接处理
        run_fn = run_intention_analysis_with_fallback if args.fallback else run_intention_analysis
        result = run_fn(
            data,
            provider=args.provider,
            model=args.model,
            output_dir=args.output_dir,
        )
        print(f"意图生成完成: {len(result['intention_frames'])} 个意图帧")
        print(f"结果保存至: {args.output_dir}/")


if __name__ == "__main__":
    main()