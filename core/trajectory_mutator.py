"""
core/trajectory_mutator.py - 意图驱动轨迹变异器（穷举版本）

职责：
基于驾驶意图的穷举变异，生成所有有效的轨迹变体

变异变量：
    Delta_a: 纵向加速度增量 (m/s^2)，颗粒度 0.1
    Delta_omega: 横摆角速度增量 (rad/s)，颗粒度 0.0087 (0.5度/s)

运动学状态向量：
    S_t = [v_t, a_t, theta_t, omega_t, x_t, y_t]

状态更新公式：
    a_t = clamp(a_{t-1} + Delta_a, -8.0, 3.0)
    omega_t = clamp(omega_{t-1} + Delta_omega, -0.349, 0.349)  # rad/s
    v_t = max(0, v_{t-1} + a_t * dt)
    theta_t = theta_{t-1} + omega_t * dt
    x_t = x_{t-1} + v_t * cos(theta_t) * dt
    y_t = y_{t-1} + v_t * sin(theta_t) * dt

使用示例：
    from core.trajectory_mutator import IntentionDrivenTrajectoryMutator

    mutator = IntentionDrivenTrajectoryMutator()
    variants = mutator.mutate(fragment)
"""

import math
import itertools
from typing import Dict, List, Optional, Any, Iterator, Tuple

import numpy as np

from core.llm.prompt_builder import (
    IntentionSequence,
    IntentionPhase,
    DrivingIntention,
)


# =============================================================================
# 物理常数与边界约束
# =============================================================================

DT = 0.1  # 时间步长，10Hz

# 纵向加速度边界 (m/s^2)
A_MIN = -8.0
A_MAX = 3.0

# 横摆角速度边界 (rad/s)
# 20 deg/s = 20 * pi / 180 rad
OMEGA_MIN = -20 * math.pi / 180  # -0.349 rad/s
OMEGA_MAX = 20 * math.pi / 180   # 0.349 rad/s


# =============================================================================
# 变异变量离散值定义
# =============================================================================

# Delta_a: 纵向加速度增量 (m/s^2)
# 变异全集: -0.5, -0.2, -0.1, 0, 0.1, 0.2
DELTA_A_VALUES = [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2]

# Delta_omega: 横摆角速度增量 (rad/s)
# 变异全集: -1.0, -0.5, 0, 0.5, 1.0 deg/s
# 转换为弧度: -0.01745, -0.00873, 0, 0.00873, 0.01745 rad
DELTA_OMEGA_DEG = [-1.0, -0.5, 0.0, 0.5, 1.0]  # deg/s
DELTA_OMEGA_VALUES = [d * math.pi / 180 for d in DELTA_OMEGA_DEG]  # rad/s


# =============================================================================
# 意图剪枝规则
# =============================================================================

class IntentionPruningRules:
    """
    意图剪枝规则 - 定义每个意图类型的有效变异组合

    属性：
        delta_a_values: 有效的 Delta_a 列表
        delta_omega_values: 有效的 Delta_omega 列表
        special_stop: 是否需要停车特判
        lock_steering: 是否锁死方向盘
    """

    def __init__(
        self,
        delta_a_values: List[float],
        delta_omega_values: List[float],
        special_stop: bool = False,
        lock_steering: bool = False,
    ):
        self.delta_a_values = delta_a_values
        self.delta_omega_values = delta_omega_values
        self.special_stop = special_stop
        self.lock_steering = lock_steering

    def generate_combinations(self) -> Iterator[Tuple[float, float]]:
        """
        生成该意图的所有有效变异组合

        穷举所有 Delta_a 和 Delta_omega 的笛卡尔积
        """
        for da in self.delta_a_values:
            for dw in self.delta_omega_values:
                yield (da, dw)


# 各意图的剪枝规则
INTENTION_PRUNING: Dict[DrivingIntention, IntentionPruningRules] = {
    # 类别1：纯纵向意图（限制 Delta_omega 约等于 0）

    DrivingIntention.CRUISE_MAINTAIN: IntentionPruningRules(
        delta_a_values=[-0.1, 0.0, 0.1],  # 趋零收敛
        delta_omega_values=DELTA_OMEGA_VALUES[1:4],  # -0.5, 0, 0.5 deg/s
    ),

    DrivingIntention.ACCELERATE_THROUGH: IntentionPruningRules(
        delta_a_values=[0.0, 0.1, 0.2],  # 强制非负
        delta_omega_values=DELTA_OMEGA_VALUES[1:4],  # -0.5, 0, 0.5 deg/s
    ),

    DrivingIntention.DECELERATE_TO_YIELD: IntentionPruningRules(
        delta_a_values=[-0.2, -0.1, 0.0],  # 平缓减速
        delta_omega_values=DELTA_OMEGA_VALUES[1:4],  # -0.5, 0, 0.5 deg/s
    ),

    DrivingIntention.DECELERATE_TO_STOP: IntentionPruningRules(
        delta_a_values=[-0.2, -0.1, 0.0],  # 减速
        delta_omega_values=DELTA_OMEGA_VALUES[1:4],  # -0.5, 0, 0.5 deg/s
        special_stop=True,  # 需要停车特判
    ),

    DrivingIntention.EMERGENCY_BRAKE: IntentionPruningRules(
        delta_a_values=[-0.5, -0.2],  # 强制深度减速
        delta_omega_values=[0.0],  # 锁死方向盘
        lock_steering=True,
    ),

    # 类别2：横向/综合意图

    DrivingIntention.LANE_CHANGE_LEFT: IntentionPruningRules(
        delta_a_values=[-0.1, 0.0, 0.1],
        delta_omega_values=DELTA_OMEGA_VALUES[2:5],  # 0, 0.5, 1.0 deg/s
    ),

    DrivingIntention.LANE_CHANGE_RIGHT: IntentionPruningRules(
        delta_a_values=[-0.1, 0.0, 0.1],
        delta_omega_values=DELTA_OMEGA_VALUES[0:3],  # -1.0, -0.5, 0 deg/s
    ),

    DrivingIntention.TURN_LEFT: IntentionPruningRules(
        delta_a_values=[-0.2, -0.1, 0.0, 0.1],  # 允许入弯降速
        delta_omega_values=DELTA_OMEGA_VALUES[3:5],  # 0.5, 1.0 deg/s
    ),

    DrivingIntention.TURN_RIGHT: IntentionPruningRules(
        delta_a_values=[-0.2, -0.1, 0.0, 0.1],
        delta_omega_values=DELTA_OMEGA_VALUES[0:2],  # -1.0, -0.5 deg/s
    ),

    DrivingIntention.GO_STRAIGHT: IntentionPruningRules(
        delta_a_values=[-0.2, -0.1, 0.0, 0.1, 0.2],  # 自由跟车
        delta_omega_values=DELTA_OMEGA_VALUES[1:4],  # -0.5, 0, 0.5 deg/s
    ),

    DrivingIntention.UNKNOWN: IntentionPruningRules(
        delta_a_values=[-0.1, 0.0, 0.1],
        delta_omega_values=[0.0],
    ),
}


# =============================================================================
# 运动学计算函数
# =============================================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """将值限制在 [min_val, max_val] 范围内"""
    return max(min_val, min(max_val, value))


def compute_speed_from_velocity(velocity: List[float]) -> float:
    """从速度向量计算速度标量"""
    if not velocity:
        return 0.0
    return math.sqrt(velocity[0]**2 + velocity[1]**2)


def compute_initial_state(trajectory: Dict) -> Dict:
    """
    从原始轨迹数据计算初始状态

    参数：
        trajectory: VehicleTrajectory 结构，包含 positions, headings, velocities, accelerations

    返回：
        初始状态字典 {v, a, theta, omega, x, y}
    """
    velocities = trajectory.get("velocities", [[0.0, 0.0]])
    headings = trajectory.get("headings", [0.0])
    positions = trajectory.get("positions", [[0.0, 0.0, 0.0]])

    # 速度标量
    v = compute_speed_from_velocity(velocities[0]) if velocities else 0.0

    # 纵向加速度（取局部坐标系X分量）
    a = velocities[0][0] if velocities else 0.0  # 默认用速度近似

    # 航向角
    theta = headings[0] if headings else 0.0

    # 横摆角速度：通过相邻帧航向差分计算
    if len(headings) >= 2:
        omega = (headings[1] - headings[0]) / DT
    else:
        omega = 0.0

    # 位置
    x = positions[0][0] if positions else 0.0
    y = positions[0][1] if positions else 0.0

    return {
        "v": v,
        "a": a,
        "theta": theta,
        "omega": omega,
        "x": x,
        "y": y,
    }


def integrate_state(
    state: Dict,
    delta_a: float,
    delta_omega: float,
    dt: float = DT
) -> Dict:
    """
    运动学状态积分 - 根据变异量计算下一帧状态

    公式：
        a_t = clamp(a_{t-1} + Delta_a, -8.0, 3.0)
        omega_t = clamp(omega_{t-1} + Delta_omega, -0.349, 0.349)
        v_t = max(0, v_{t-1} + a_t * dt)
        theta_t = theta_{t-1} + omega_t * dt
        x_t = x_{t-1} + v_t * cos(theta_t) * dt
        y_t = y_{t-1} + v_t * sin(theta_t) * dt

    参数：
        state: 当前状态字典 {v, a, theta, omega, x, y}
        delta_a: 纵向加速度增量 (m/s^2)
        delta_omega: 横摆角速度增量 (rad/s)
        dt: 时间步长

    返回：
        下一帧状态字典
    """
    v_prev = state["v"]
    a_prev = state["a"]
    theta_prev = state["theta"]
    omega_prev = state["omega"]
    x_prev = state["x"]
    y_prev = state["y"]

    # 加速度更新
    a = clamp(a_prev + delta_a, A_MIN, A_MAX)

    # 横摆角速度更新
    omega = clamp(omega_prev + delta_omega, OMEGA_MIN, OMEGA_MAX)

    # 速度更新（确保非负）
    v = max(0.0, v_prev + a * dt)

    # 航向角更新
    theta = theta_prev + omega * dt

    # 位置更新
    x = x_prev + v * math.cos(theta) * dt
    y = y_prev + v * math.sin(theta) * dt

    return {
        "v": v,
        "a": a,
        "theta": theta,
        "omega": omega,
        "x": x,
        "y": y,
    }


# =============================================================================
# 轨迹变异器
# =============================================================================

class IntentionDrivenTrajectoryMutator:
    """
    意图驱动轨迹变异器（穷举版本）

    根据意图序列穷举生成所有有效的轨迹变体

    使用示例：
        mutator = IntentionDrivenTrajectoryMutator()
        variants = mutator.mutate(fragment)
    """

    def mutate(self, fragment: Dict) -> List[Dict]:
        """
        穷举生成所有轨迹变体

        参数：
            fragment: 原始轨迹片段数据（包含意图序列和轨迹）

        返回：
            所有变体轨迹列表
        """
        # 解析片段数据
        frag = fragment.get("fragment", fragment) if "fragment" in fragment else fragment
        target = frag.get("target_trajectory", {})
        frame_count = frag.get("frame_count", 50)

        # 获取意图序列
        intention_seq = self._build_intention_sequence(frag)

        # 收集每个意图段的变异组合
        segment_combinations = []
        for phase in intention_seq.phases:
            rules = INTENTION_PRUNING.get(
                phase.intention,
                INTENTION_PRUNING[DrivingIntention.UNKNOWN]
            )
            combos = list(rules.generate_combinations())
            segment_combinations.append({
                "phase": phase,
                "combinations": combos,
                "rules": rules,
            })

        # 计算总变体数
        total_variants = 1
        for seg in segment_combinations:
            total_variants *= len(seg["combinations"])

        print(f"意图段数量: {len(segment_combinations)}")
        print(f"各段组合数: {[len(seg['combinations']) for seg in segment_combinations]}")
        print(f"总变体数量: {total_variants}")

        # 穷举生成所有变体
        variants = []
        variant_id = 0

        # 递归穷举所有段的组合
        for combo_tuple in self._enumerate_combinations(segment_combinations, 0, []):
            variant = self._generate_trajectory(
                target, intention_seq, combo_tuple, variant_id, frame_count
            )
            variants.append(variant)
            variant_id += 1

            if variant_id % 1000 == 0:
                print(f"已生成 {variant_id}/{total_variants} 个变体...")

        print(f"完成！共生成 {len(variants)} 个变体")
        return variants

    def _build_intention_sequence(self, frag: Dict) -> IntentionSequence:
        """
        构建意图序列

        如果片段中包含意图数据，使用片段中的意图
        否则使用默认的单一意图段
        """
        # 检查是否有意图分析结果
        intention_analysis = frag.get("intention_analysis", {})

        if intention_analysis and "intention_frames" in intention_analysis:
            # 从意图帧构建意图序列
            frames = intention_analysis["intention_frames"]
            if frames:
                # 使用第一个意图帧的意图作为整个轨迹的意图
                first_intention = frames[0].get("intention", "cruise_maintain")
                try:
                    intention = DrivingIntention(first_intention)
                except ValueError:
                    intention = DrivingIntention.CRUISE_MAINTAIN

                return IntentionSequence(
                    phases=[
                        IntentionPhase(
                            start_frame=0,
                            end_frame=frag.get("frame_count", 50),
                            intention=intention,
                            reasoning="从意图帧推断"
                        )
                    ],
                    overall_strategy="意图驱动"
                )

        # 默认意图序列
        frame_count = frag.get("frame_count", 50)
        return IntentionSequence(
            phases=[
                IntentionPhase(
                    start_frame=0,
                    end_frame=frame_count,
                    intention=DrivingIntention.CRUISE_MAINTAIN,
                    reasoning="默认意图"
                )
            ],
            overall_strategy="默认策略"
        )

    def _enumerate_combinations(
        self,
        segments: List[Dict],
        idx: int,
        current: List[Tuple[float, float]]
    ) -> Iterator[List[Tuple[float, float]]]:
        """
        递归穷举所有意图段的参数组合

        参数：
            segments: 意图段列表
            idx: 当前处理的段索引
            current: 当前累积的组合列表

        返回：
            完整组合列表的迭代器
        """
        if idx >= len(segments):
            yield current.copy()
            return

        for combo in segments[idx]["combinations"]:
            current.append(combo)
            yield from self._enumerate_combinations(segments, idx + 1, current)
            current.pop()

    def _generate_trajectory(
        self,
        target: Dict,
        intention_seq: IntentionSequence,
        combo_list: List[Tuple[float, float]],
        variant_id: int,
        frame_count: int,
    ) -> Dict:
        """
        根据变异组合生成一条轨迹

        参数：
            target: 原始轨迹数据
            intention_seq: 意图序列
            combo_list: 各意图段对应的 (Delta_a, Delta_omega) 组合列表
            variant_id: 变体编号
            frame_count: 总帧数

        返回：
            变体字典
        """
        # 计算初始状态
        state = compute_initial_state(target)

        # 初始化输出数组
        positions = np.zeros((frame_count, 3))
        headings = np.zeros(frame_count)
        velocities = np.zeros((frame_count, 2))
        accelerations = np.zeros((frame_count, 2))

        # 特殊处理：停车标志
        stopped = False

        # 获取每个意图段对应的组合索引
        combo_idx = 0

        for t in range(frame_count):
            # 存储当前状态
            positions[t] = [state["x"], state["y"], target.get("positions", [[0, 0, 0]])[0][2]]
            headings[t] = state["theta"]
            velocities[t] = [state["v"], 0.0]  # 局部坐标系：纵向速度 v，横向 0
            accelerations[t] = [state["a"], state["v"] * state["omega"]]  # 纵向加速度，向心加速度

            # 确定当前帧对应的变异组合
            delta_a, delta_omega = self._get_combo_for_frame(
                intention_seq, combo_list, t, combo_idx
            )

            # 更新组合索引
            for i, phase in enumerate(intention_seq.phases):
                if t == phase.start_frame and i > combo_idx:
                    combo_idx = i

            # 特殊判断：停车
            if stopped:
                state = {
                    "v": 0.0,
                    "a": 0.0,
                    "theta": state["theta"],
                    "omega": 0.0,
                    "x": state["x"],
                    "y": state["y"],
                }
            else:
                # 运动学积分
                state = integrate_state(state, delta_a, delta_omega)

                # 检查是否需要停车特判
                if self._should_stop(state, intention_seq, t):
                    stopped = True

        return {
            "original_fragment_id": target.get("fragment_id", "unknown"),
            "intention_sequence": intention_seq.to_dict(),
            "variant_id": variant_id,
            "params": [
                {"phase": phase.start_frame, "delta_a": combo[0], "delta_omega": combo[1]}
                for phase, combo in zip(intention_seq.phases, combo_list)
            ],
            "trajectory": {
                "positions": positions.tolist(),
                "headings": headings.tolist(),
                "velocities": velocities.tolist(),
                "accelerations": accelerations.tolist(),
                "valid": [True] * frame_count,
            }
        }

    def _get_combo_for_frame(
        self,
        intention_seq: IntentionSequence,
        combo_list: List[Tuple[float, float]],
        frame: int,
        current_idx: int
    ) -> Tuple[float, float]:
        """获取指定帧对应的变异组合"""
        for i, phase in enumerate(intention_seq.phases):
            if phase.start_frame <= frame < phase.end_frame:
                if i < len(combo_list):
                    return combo_list[i]
                else:
                    return (0.0, 0.0)
        return (0.0, 0.0)

    def _should_stop(
        self,
        state: Dict,
        intention_seq: IntentionSequence,
        frame: int
    ) -> bool:
        """判断是否应该停车"""
        for phase in intention_seq.phases:
            if phase.start_frame <= frame < phase.end_frame:
                rules = INTENTION_PRUNING.get(
                    phase.intention,
                    INTENTIONPruningRules([-0.1, 0.0, 0.1], [0.0])
                )
                # decelerate_to_stop 特判：速度为0时强制停车
                if rules.special_stop and state["v"] <= 0.01:
                    return True
        return False


def mutate_trajectories(fragment: Dict) -> List[Dict]:
    """
    便捷函数：穷举生成所有轨迹变体

    参数：
        fragment: 原始轨迹片段

    返回：
        所有变异后的轨迹列表
    """
    mutator = IntentionDrivenTrajectoryMutator()
    return mutator.mutate(fragment)


def get_combination_count(intention: DrivingIntention) -> int:
    """
    获取指定意图类型的有效组合数

    参数：
        intention: 驾驶意图类型

    返回：
        有效变异组合数量
    """
    rules = INTENTION_PRUNING.get(intention)
    if rules is None:
        return 0
    return len(list(rules.generate_combinations()))
