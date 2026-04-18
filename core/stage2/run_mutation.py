"""
core/stage2/run_mutation.py - 轨迹变异入口

用法:
    python -m core.stage2.run_mutation --input data/intention/xxx_with_intention.json
"""

import argparse
import json

from core.stage2.intention_generator import (
    run_mutation,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    DEFAULT_VARIANT_DIR,
)


def main():
    parser = argparse.ArgumentParser(description="轨迹变异")
    parser.add_argument("--input", "-i", required=True, help="输入带意图片段 JSON 文件路径")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_VARIANT_DIR)
    parser.add_argument("--top-k", type=float, default=10.0, help="保留危险分数前K%%的分支")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        fragment_with_intention = json.load(f)

    variants = run_mutation(
        fragment_with_intention,
        top_k=args.top_k,
        output_dir=args.output_dir,
        provider=args.provider,
        model=args.model,
        save=True,
        random_seed=args.seed,
    )

    print(f"变异生成完成: {len(variants)} 条轨迹")


if __name__ == "__main__":
    main()