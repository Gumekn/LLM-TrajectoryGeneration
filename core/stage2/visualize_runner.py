"""
轨迹变异可视化主入口

参数配置和路径设置
"""

from core.stage2.variant_visualizer import process_single_json, process_all_jsons

# ============== 配置参数 ==============

# 输入数据目录
INPUT_DIR = "data/variants"

# 输出可视化结果目录
OUTPUT_ROOT = "visualization_results"

# 坐标轴范围（米）
AXIS_RANGE = 80

# 视频帧率
FPS = 10

# ============== 运行 ==============

if __name__ == '__main__':
    output_dirs = process_all_jsons(
        input_dir=INPUT_DIR,
        output_root=OUTPUT_ROOT,
        axis_range=AXIS_RANGE,
        fps=FPS
    )

    print(f"\nCompleted! Generated {len(output_dirs)} scenario visualizations.")
