"""
main.py - 危险轨迹生成系统 第一阶段主程序

本程序实现从Waymo原始数据中提取危险轨迹片段的完整流程。

处理流程:
    1. 加载Waymo场景数据
    2. 检索场景中的移动车辆，排除静止车辆
    3. 用户选择主车
    4. 分析主车与所有交互车的风险
    5. 截取危险轨迹片段
    6. 提取11维特征
    7. 保存到JSON文件

使用方法:
    python main.py
"""

import os
import sys
import argparse
from typing import List, Dict, Optional

from core import (
    WaymoDataLoader,
    ScenarioRiskAnalyzer,
    ScenarioProcessor,
)


# =============================================================================
# 配置参数 - 直接在代码中指定
# =============================================================================

# 数据路径配置
DATA_DIR = "data/waymo-open"          # Waymo原始数据目录
OUTPUT_DIR = "data/processed"         # 处理后数据输出目录

# 截取参数
N_BEFORE_SEC = 2.0                   # 锚点前截取时长（秒）
N_AFTER_SEC = 3.0                    # 锚点后截取时长（秒）
MIN_BEFORE_FRAMES = 10               # 锚点前最小帧数阈值（帧），低于此值则跳过
MIN_AFTER_FRAMES = 15                # 锚点后最小帧数阈值（帧），低于此值则跳过

# 风险分析参数
RISK_THRESHOLD = 3                   # 风险分数阈值，低于此值的车辆被选为关键车
SPEED_THRESHOLD = 0.5                # 速度阈值 (m/s)，低于此值的车辆被视为静止

# 场景ID（可修改为其他场景）
SCENARIO_ID = "10135f16cd538e19"


# =============================================================================
# 命令行参数解析
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="危险轨迹生成系统 - 第一阶段",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--scenario", "-s",
        type=str,
        default=SCENARIO_ID,
        help=f"场景ID (默认: {SCENARIO_ID})"
    )

    parser.add_argument(
        "--ego-id", "-e",
        type=str,
        default=None,
        help="主车ID " \
        ""
    )

    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default=DATA_DIR,
        help=f"数据目录 (默认: {DATA_DIR})"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=OUTPUT_DIR,
        help=f"输出目录 (默认: {OUTPUT_DIR})"
    )

    parser.add_argument(
        "--n-before",
        type=float,
        default=N_BEFORE_SEC,
        help=f"锚点前时长(秒) (默认: {N_BEFORE_SEC})"
    )

    parser.add_argument(
        "--n-after",
        type=float,
        default=N_AFTER_SEC,
        help=f"锚点后时长(秒) (默认: {N_AFTER_SEC})"
    )

    parser.add_argument(
        "--risk-threshold", "-r",
        type=int,
        default=RISK_THRESHOLD,
        help=f"风险阈值 (默认: {RISK_THRESHOLD})"
    )

    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=SPEED_THRESHOLD,
        help=f"速度阈值(m/s) (默认: {SPEED_THRESHOLD})"
    )

    parser.add_argument(
        "--min-before-frames",
        type=int,
        default=MIN_BEFORE_FRAMES,
        help=f"锚点前最小帧数阈值 (默认: {MIN_BEFORE_FRAMES})"
    )

    parser.add_argument(
        "--min-after-frames",
        type=int,
        default=MIN_AFTER_FRAMES,
        help=f"锚点后最小帧数阈值 (默认: {MIN_AFTER_FRAMES})"
    )

    return parser.parse_args()


# =============================================================================
# 主程序
# =============================================================================

def print_header(title: str) -> None:
    """打印标题栏"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_config() -> None:
    """打印当前配置参数"""
    print_header("当前配置参数")
    print(f"  数据目录:        {DATA_DIR}")
    print(f"  输出目录:        {OUTPUT_DIR}")
    print(f"  场景ID:         {SCENARIO_ID}")
    print(f"  锚点前时长:     {N_BEFORE_SEC} 秒")
    print(f"  锚点后时长:     {N_AFTER_SEC} 秒")
    print(f"  锚点前最小帧:   {MIN_BEFORE_FRAMES} 帧")
    print(f"  锚点后最小帧:   {MIN_AFTER_FRAMES} 帧")
    print(f"  风险阈值:       {RISK_THRESHOLD}")
    print(f"  速度阈值:       {SPEED_THRESHOLD} m/s")


def get_user_selected_ego_id(loader: WaymoDataLoader, scenario_data: Dict,
                             cmd_ego_id: str = None) -> str:
    """获取用户选择的主车ID

    显示场景中所有移动车辆的信息（速度、帧数等），供用户选择。
    如果命令行指定了ego_id，则直接使用；否则让用户交互选择。

    Args:
        loader: WaymoDataLoader实例
        scenario_data: 场景数据
        cmd_ego_id: 命令行指定的主车ID

    Returns:
        用户选择的车辆ID
    """
    print_header("步骤1: 选择主车")

    # 获取移动车辆列表
    moving_vehicles = loader.get_moving_vehicle_list(scenario_data, SPEED_THRESHOLD)

    if not moving_vehicles:
        print("错误: 场景中没有检测到移动的车辆！")
        sys.exit(1)

    # 默认选择sdc_id（自车ID），如果它不在移动车辆列表中，则选择速度最快的车辆
    sdc_id = loader.get_sdc_id(scenario_data)
    moving_ids = [v['vehicle_id'] for v in moving_vehicles]
    if sdc_id and sdc_id in moving_ids:
        default_id = sdc_id
    else:
        default_id = moving_vehicles[0]['vehicle_id']

    # 显示车辆信息
    print(f"\n{'=' * 60}")
    print(f"场景ID: {SCENARIO_ID}")
    print(f"{'=' * 60}")
    print(f"{'编号':<6} {'车辆ID':<12} {'平均速度(m/s)':<18} {'最大速度(m/s)':<15} {'有效帧数':<10}")
    print("-" * 60)

    for i, v in enumerate(moving_vehicles):
        marker = " *" if v['vehicle_id'] == default_id else ""
        print(f"{i:<6} {v['vehicle_id']:<12} {v['mean_speed']:<18.2f} "
              f"{v['max_speed']:<15.2f} {v['valid_frames']:<10}{marker}")

    print("-" * 60)
    print(f"共 {len(moving_vehicles)} 辆移动车辆")
    print(f"(* 表示默认主车选择: {default_id})")

    # 如果命令行指定了ego_id，直接使用
    if cmd_ego_id:
        valid_ids = [v['vehicle_id'] for v in moving_vehicles]
        if cmd_ego_id in valid_ids:
            selected_id = cmd_ego_id
            print(f"\n使用命令行指定的主车: {selected_id}")
        else:
            print(f"\n错误: 命令行指定的主车ID '{cmd_ego_id}' 不是有效的移动车辆")
            print(f"将使用默认选择: {default_id}")
            selected_id = default_id
    else:
        # 交互式选择
        default_label = "SDC自车" if default_id == sdc_id else "速度最快车辆"
        print(f"\n默认选择: {default_label} (ID: {default_id})")

        try:
            user_input = input(">>> 请输入主车ID（直接回车使用默认）: ").strip()

            if user_input:
                valid_ids = [v['vehicle_id'] for v in moving_vehicles]
                while user_input not in valid_ids:
                    print(f"错误: '{user_input}' 不是有效的移动车辆ID")
                    user_input = input(">>> 请重新输入主车ID: ").strip()
                selected_id = user_input
            else:
                selected_id = default_id
        except EOFError:
            # 非交互式环境，使用默认值
            print("\n非交互式环境，使用默认选择")
            selected_id = default_id

    # 获取选中车辆的信息
    selected_info = next(v for v in moving_vehicles if v['vehicle_id'] == selected_id)

    print(f"\n已选择主车: {selected_id}")
    print(f"  平均速度: {selected_info['mean_speed']:.2f} m/s")
    print(f"  最大速度: {selected_info['max_speed']:.2f} m/s")

    return selected_id


def run_pipeline(scenario_id: str, ego_id: str) -> None:
    """运行完整的处理流程

    Args:
        scenario_id: 场景ID
        ego_id: 主车ID
    """
    print_header("步骤2: 风险分析")

    # 初始化数据加载器
    loader = WaymoDataLoader(DATA_DIR)

    # 加载场景数据
    print(f"加载场景: {scenario_id}")
    scenario_data = loader.load_scenario(scenario_id)

    # 获取主车轨迹
    print(f"提取主车轨迹: {ego_id}")
    ego_trajectory = loader.extract_vehicle_trajectory(scenario_data, ego_id)
    print(f"  主车轨迹帧数: {ego_trajectory.frame_count}")

    # 风险分析
    print(f"\n分析主车与所有交互车的风险（阈值<={RISK_THRESHOLD}）...")
    analyzer = ScenarioRiskAnalyzer(scenario_data, ego_id)
    risk_results = analyzer.analyze_all(risk_threshold=RISK_THRESHOLD)

    print(f"发现 {len(risk_results)} 个关键车辆")

    # 显示关键车辆信息
    if risk_results:
        print(f"\n{'关键车ID':<12} {'危险类型':<12} {'危险等级':<10} {'最低风险分':<12} {'最危险帧':<10}")
        print("-" * 60)
        for target_id, result in risk_results.items():
            print(f"{target_id:<12} {result.danger_type:<12} {result.danger_level:<10} "
                  f"{result.min_risk_score:<12.1f} {result.anchor_frame:<10}")


def process_and_save(scenario_id: str, ego_id: str) -> None:
    """处理场景并保存结果

    Args:
        scenario_id: 场景ID
        ego_id: 主车ID
    """
    print_header("步骤3: 片段截取与特征提取")

    # 初始化处理器
    processor = ScenarioProcessor(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR
    )

    # 处理场景
    print(f"\n处理场景: {scenario_id}")
    print(f"  主车ID: {ego_id}")
    print(f"  截取参数: 锚点前{N_BEFORE_SEC}秒({MIN_BEFORE_FRAMES}帧), 锚点后{N_AFTER_SEC}秒({MIN_AFTER_FRAMES}帧)")
    print(f"  风险阈值: {RISK_THRESHOLD}")

    scenario = processor.process_scenario(
        scenario_id=scenario_id,
        ego_id=ego_id,
        n_before_sec=N_BEFORE_SEC,
        n_after_sec=N_AFTER_SEC,
        risk_threshold=RISK_THRESHOLD,
        min_before_frames=MIN_BEFORE_FRAMES,
        min_after_frames=MIN_AFTER_FRAMES
    )

    # 显示结果摘要
    print(f"\n处理完成！")
    print(f"  场景ID: {scenario.metadata.scenario_id}")
    print(f"  总帧数: {scenario.metadata.total_frames}")
    print(f"  采样率: {scenario.metadata.sampling_rate} Hz")
    print(f"  主车ID: {scenario.metadata.ego_vehicle_id}")
    print(f"  关键车数量: {scenario.metadata.num_key_vehicles}")
    print(f"  片段数量: {len(scenario.fragments)}")

    # 显示每个片段的详细信息
    if scenario.fragments:
        print(f"\n{'=' * 70}")
        print(f"{'片段ID':<8} {'目标车ID':<12} {'危险类型':<12} {'等级':<8} {'风险分':<10} {'帧数':<6}")
        print("-" * 70)

        for frag in scenario.fragments:
            print(f"{frag.metadata.fragment_id:<8} {frag.metadata.target_vehicle_id:<12} "
                  f"{frag.metadata.danger_type:<12} {frag.metadata.danger_level:<8} "
                  f"{frag.metadata.min_risk_score:<10.1f} {frag.metadata.frame_count:<6}")

        # 显示第一个片段的11维特征
        print(f"\n{'=' * 70}")
        print("第一个片段的11维特征:")
        print("-" * 70)
        for name, value in zip(scenario.fragments[0].feature_names,
                               scenario.fragments[0].feature_vector):
            print(f"  {name:<20}: {value:.4f}")

        print(f"\n结果已保存到: {OUTPUT_DIR}/{scenario_id}_{ego_id}.json")
    else:
        print(f"\n未提取到任何片段，未生成JSON文件")


def main() -> None:
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 使用命令行参数覆盖配置
    global DATA_DIR, OUTPUT_DIR, SCENARIO_ID, N_BEFORE_SEC, N_AFTER_SEC, RISK_THRESHOLD, SPEED_THRESHOLD, MIN_BEFORE_FRAMES, MIN_AFTER_FRAMES
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    SCENARIO_ID = args.scenario
    N_BEFORE_SEC = args.n_before
    N_AFTER_SEC = args.n_after
    RISK_THRESHOLD = args.risk_threshold
    SPEED_THRESHOLD = args.speed_threshold
    MIN_BEFORE_FRAMES = args.min_before_frames
    MIN_AFTER_FRAMES = args.min_after_frames

    # 打印配置
    print_config()

    # 初始化数据加载器
    loader = WaymoDataLoader(DATA_DIR)

    # 加载场景数据
    print_header("加载场景数据")
    print(f"从 {DATA_DIR} 加载场景: {SCENARIO_ID}")

    if not os.path.exists(os.path.join(DATA_DIR, f"{SCENARIO_ID}.pkl")):
        print(f"错误: 场景文件不存在 - {DATA_DIR}/{SCENARIO_ID}.pkl")
        sys.exit(1)

    scenario_data = loader.load_scenario(SCENARIO_ID)
    print(f"场景加载成功!")
    print(f"  场景ID: {SCENARIO_ID}")
    print(f"  数据键: {list(scenario_data.keys())}")

    # 获取用户选择的主车（传入命令行指定的ego_id）
    ego_id = get_user_selected_ego_id(loader, scenario_data, args.ego_id)

    # 运行风险分析
    run_pipeline(SCENARIO_ID, ego_id)

    # 处理并保存
    process_and_save(SCENARIO_ID, ego_id)

    # 完成
    print_header("处理完成!")
    print("\n第一阶段轨迹生成已完成！")


if __name__ == "__main__":
    main()
