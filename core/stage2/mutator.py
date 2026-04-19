"""
core/stage2/mutator.py - 意图驱动轨迹变异穷举核心算法

职责：
- 意图区块合并（游程编码）
- 意图变异规则定义
- 运动学递推模型
- 简化风险分数计算
- DFS + 逐 Block Top-K% 剪枝穷举
- 轨迹变异器类

物理常数：
- DT = 0.1s（时间步长，10Hz采样率）
- 合加速度范围: [0.0, 10.0] m/s²
- 横摆角速度范围: [-0.35, 0.35] rad/s
- 加速度随机扰动标准差: 0.05 m/s²
- 角速度随机扰动标准差: 0.004 rad/s

使用示例：
    from core.stage2.mutator import TrajectoryMutator

    mutator = TrajectoryMutator()
    variants = mutator.mutate(fragment, top_k=10)
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from enum import Enum


# =============================================================================
# 物理常数
# =============================================================================

DT = 0.1  # 时间步长 (s)
A_MAG_MIN = 0.0  # 合加速度最小值 (m/s²)
A_MAG_MAX = 10.0  # 合加速度最大值 (m/s²)
OMEGA_MIN = -0.35  # 横摆角速度最小值 (rad/s)
OMEGA_MAX = 0.35  # 横摆角速度最大值 (rad/s)
SIGMA_A = 0.05  # 加速度随机扰动标准差 (m/s²)
SIGMA_OMEGA = 0.004  # 角速度随机扰动标准差 (rad/s)


# =============================================================================
# 驾驶意图类型
# =============================================================================

class DrivingIntention(Enum):
    """驾驶意图类型"""
    CRUISE_MAINTAIN = "cruise_maintain"
    ACCELERATE_THROUGH = "accelerate_through"
    DECELERATE_TO_YIELD = "decelerate_to_yield"
    DECELERATE_TO_STOP = "decelerate_to_stop"
    EMERGENCY_BRAKE = "emergency_brake"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    GO_STRAIGHT = "go_straight"
    UNKNOWN = "unknown"


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class IntentBlock:
    """意图区块 - 合并后的变异单元"""
    intent: DrivingIntention
    start_frame: int
    end_frame: int
    duration: float  # 秒

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class TrajectoryState:
    """运动学状态"""
    v: float      # 速度标量 (m/s)
    a_x: float    # 纵向加速度 (m/s²)
    a_y: float    # 横向加速度 (m/s²)
    theta: float  # 航向角 (rad)
    omega: float  # 横摆角速度 (rad/s)
    x: float      # X坐标 (m)
    y: float      # Y坐标 (m)

    @property
    def a_mag(self) -> float:
        """合加速度大小"""
        return math.sqrt(self.a_x**2 + self.a_y**2)


@dataclass
class IntentionMutation:
    """意图变异规格"""
    delta_a_mag_values: List[float]  # 合加速度增量选择列表
    delta_omega_values: List[float]   # 航向角增量选择列表
    is_perturbation: bool = False    # 是否为随机扰动模式


# =============================================================================
# 工具函数
# =============================================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """限制值在范围内"""
    return max(min_val, min(max_val, value))


# =============================================================================
# 意图变异规则定义
# =============================================================================

def get_intention_mutations() -> Dict[DrivingIntention, List[IntentionMutation]]:
    """
    获取所有意图的变异配置

    每种意图3×3=9种组合（除cruise_maintain为随机扰动外）
    """
    return {
        DrivingIntention.CRUISE_MAINTAIN: [
            IntentionMutation(delta_a_mag_values=[0.0], delta_omega_values=[0.0], is_perturbation=True)
        ],
        DrivingIntention.ACCELERATE_THROUGH: [
            IntentionMutation(delta_a_mag_values=[0.2, 0.4, 0.6], delta_omega_values=[0.0])
        ],
        DrivingIntention.DECELERATE_TO_YIELD: [
            IntentionMutation(delta_a_mag_values=[-0.3, -0.6, -1.0], delta_omega_values=[0.0])
        ],
        DrivingIntention.DECELERATE_TO_STOP: [
            IntentionMutation(delta_a_mag_values=[-0.3, -0.6, -1.0], delta_omega_values=[0.0])
        ],
        DrivingIntention.EMERGENCY_BRAKE: [
            IntentionMutation(delta_a_mag_values=[-3.0, -5.0, -8.0], delta_omega_values=[0.0])
        ],
        DrivingIntention.LANE_CHANGE_LEFT: [
            IntentionMutation(delta_a_mag_values=[-0.2, 0.0, 0.2], delta_omega_values=[0.01, 0.02, 0.03])
        ],
        DrivingIntention.LANE_CHANGE_RIGHT: [
            IntentionMutation(delta_a_mag_values=[-0.2, 0.0, 0.2], delta_omega_values=[-0.03, -0.02, -0.01])
        ],
        DrivingIntention.TURN_LEFT: [
            IntentionMutation(delta_a_mag_values=[-0.2, 0.0, 0.2], delta_omega_values=[0.02, 0.04, 0.06])
        ],
        DrivingIntention.TURN_RIGHT: [
            IntentionMutation(delta_a_mag_values=[-0.2, 0.0, 0.2], delta_omega_values=[-0.06, -0.04, -0.02])
        ],
        DrivingIntention.GO_STRAIGHT: [
            IntentionMutation(delta_a_mag_values=[-0.2, 0.0, 0.2], delta_omega_values=[0.0])
        ],
        DrivingIntention.UNKNOWN: [
            IntentionMutation(delta_a_mag_values=[0.0], delta_omega_values=[0.0], is_perturbation=True)
        ],
    }


INTENTION_MUTATIONS = get_intention_mutations()


# =============================================================================
# 意图合并（游程编码）
# =============================================================================

def merge_intentions(intention_frames: List[Dict], total_frames: int = 50) -> List[IntentBlock]:
    """
    将意图帧序列合并为意图区块（游程编码）

    相邻相同意图合并为一个 Block，每个 Block 的 end_frame 取到下一个意图帧的 frame。
    最后一个 Block 的 end_frame 延伸到 total_frames。

    Args:
        intention_frames: 意图帧列表，每项包含 frame, intention
        total_frames: 轨迹总帧数（由实际数据决定，不再是固定50）

    Returns:
        IntentBlock列表
    """
    if not intention_frames:
        return [IntentBlock(
            intent=DrivingIntention.CRUISE_MAINTAIN,
            start_frame=0,
            end_frame=total_frames,
            duration=total_frames * DT
        )]

    sorted_frames = sorted(intention_frames, key=lambda x: x.get("frame", 0))

    blocks = []
    current_intent_str = sorted_frames[0].get("intention", "unknown")
    current_start = sorted_frames[0].get("frame", 0)

    try:
        current_intent = DrivingIntention(current_intent_str)
    except ValueError:
        current_intent = DrivingIntention.UNKNOWN

    for frame_data in sorted_frames[1:]:
        intent_str = frame_data.get("intention", "unknown")
        try:
            intent = DrivingIntention(intent_str)
        except ValueError:
            intent = DrivingIntention.UNKNOWN

        if intent != current_intent:
            blocks.append(IntentBlock(
                intent=current_intent,
                start_frame=current_start,
                end_frame=frame_data.get("frame", current_start),
                duration=(frame_data.get("frame", current_start) - current_start) * DT
            ))
            current_intent = intent
            current_start = frame_data.get("frame", current_start)

    # 保存最后一个Block，end_frame 延伸到 total_frames
    blocks.append(IntentBlock(
        intent=current_intent,
        start_frame=current_start,
        end_frame=total_frames,
        duration=(total_frames - current_start) * DT
    ))

    return blocks


# =============================================================================
# 初始状态提取
# =============================================================================

def extract_initial_state(trajectory: Dict) -> TrajectoryState:
    """从轨迹数据提取初始状态"""
    velocities = trajectory.get("velocities", [[0.0, 0.0]])
    headings = trajectory.get("headings", [0.0])
    positions = trajectory.get("positions", [[0.0, 0.0, 0.0]])
    accelerations = trajectory.get("accelerations", [[0.0, 0.0]])

    v = math.sqrt(velocities[0][0]**2 + velocities[0][1]**2) if velocities else 0.0
    a_x = accelerations[0][0] if accelerations else 0.0
    a_y = accelerations[0][1] if accelerations else 0.0
    theta = headings[0] if headings else 0.0
    omega = (headings[1] - headings[0]) / DT if len(headings) >= 2 else 0.0
    x = positions[0][0] if positions else 0.0
    y = positions[0][1] if positions else 0.0

    return TrajectoryState(v=v, a_x=a_x, a_y=a_y, theta=theta, omega=omega, x=x, y=y)


# =============================================================================
# 加速度变异
# =============================================================================

def mutate_vector_acceleration(current_state: TrajectoryState, delta_a_mag: float) -> Tuple[float, float]:
    """变异合加速度，并还原为分量"""
    a_mag = current_state.a_mag

    if a_mag > 1e-6:
        cos_theta = math.cos(current_state.theta)
        sin_theta = math.sin(current_state.theta)
        a_mag_new = clamp(a_mag + delta_a_mag, A_MAG_MIN, A_MAG_MAX)
        a_x_new = a_mag_new * cos_theta
        a_y_new = a_mag_new * sin_theta
    else:
        a_mag_new = clamp(delta_a_mag, A_MAG_MIN, A_MAG_MAX)
        cos_theta = math.cos(current_state.theta)
        sin_theta = math.sin(current_state.theta)
        a_x_new = a_mag_new * cos_theta
        a_y_new = a_mag_new * sin_theta

    return a_x_new, a_y_new


# =============================================================================
# 运动学积分
# =============================================================================

def kinematically_integrate(state: TrajectoryState, a_x: float, a_y: float, omega: float, dt: float = DT) -> TrajectoryState:
    """运动学积分一步"""
    v = max(0.0, state.v + a_x * dt)
    cos_theta = math.cos(state.theta)
    sin_theta = math.sin(state.theta)
    v_x_global = v * cos_theta - a_y * sin_theta * dt
    v_y_global = v * sin_theta + a_y * cos_theta * dt
    x = state.x + v_x_global * dt
    y = state.y + v_y_global * dt
    theta = state.theta + omega * dt

    return TrajectoryState(v=v, a_x=a_x, a_y=a_y, theta=theta, omega=omega, x=x, y=y)


# =============================================================================
# 变异组合穷举
# =============================================================================

def get_mutation_combinations(mutation: IntentionMutation) -> List[Tuple[float, float]]:
    """穷举一个意图变异配置的所有 (delta_a_mag, delta_omega) 组合"""
    if mutation.is_perturbation:
        return [(0.0, 0.0)]
    return [
        (da, dw)
        for da in mutation.delta_a_mag_values
        for dw in mutation.delta_omega_values
    ]


# =============================================================================
# 简化风险分数计算
# =============================================================================

def compute_block_risk_score(state: TrajectoryState, ego_trajectory: Dict, frame_idx: int) -> float:
    """基于当前帧状态与 ego 轨迹，计算简化风险分数（用于逐 Block 剪枝）"""
    ego_positions = ego_trajectory.get("positions", [])
    ego_velocities = ego_trajectory.get("velocities", [])

    if not ego_positions or frame_idx >= len(ego_positions):
        return 0.0

    ego_pos = ego_positions[frame_idx]
    ego_v = ego_velocities[frame_idx] if frame_idx < len(ego_velocities) else [0.0, 0.0]

    dx = state.x - ego_pos[0]
    dy = state.y - ego_pos[1]
    dist = math.sqrt(dx * dx + dy * dy)

    target_vx = state.v * math.cos(state.theta)
    target_vy = state.v * math.sin(state.theta)
    rel_vx = target_vx - ego_v[0]
    rel_vy = target_vy - ego_v[1]

    if dist > 1e-6:
        closing_speed = -(rel_vx * dx + rel_vy * dy) / dist
    else:
        closing_speed = math.sqrt(rel_vx * rel_vx + rel_vy * rel_vy)

    if closing_speed > 0.1 and dist > 0.1:
        approx_ttc = dist / closing_speed
    else:
        approx_ttc = 999.0

    risk = 50.0 / (dist + 0.5) + max(0.0, closing_speed) * 5.0 + 30.0 / (approx_ttc + 0.1)
    return risk


# =============================================================================
# 最终风险分数计算
# =============================================================================

def compute_risk_score(variant_states: List[TrajectoryState], ego_trajectory: Dict) -> float:
    """计算轨迹最终风险分数（复用 processor 的 RiskCalculator）"""
    try:
        from core.stage1.processor import RiskCalculator, VehicleTrajectory
        import numpy as np

        n = len(variant_states)

        target_positions = np.array([[s.x, s.y, 0.0] for s in variant_states])
        target_headings = np.array([s.theta for s in variant_states])
        target_velocities = np.array([[s.v, 0.0] for s in variant_states])
        target_accelerations = np.array([[s.a_x, s.a_y] for s in variant_states])

        target_traj = VehicleTrajectory(
            vehicle_id="target",
            positions=target_positions,
            headings=target_headings,
            velocities=target_velocities,
            accelerations=target_accelerations,
            valid=np.array([True] * n)
        )

        ego_pos_list = ego_trajectory.get("positions", [])
        ego_vel_list = ego_trajectory.get("velocities", [])
        ego_acc_list = ego_trajectory.get("accelerations", [])
        ego_hdg_list = ego_trajectory.get("headings", [0.0] * n)

        ego_positions = np.array([[p[0], p[1], p[2] if len(p) > 2 else 0.0] for p in ego_pos_list])
        ego_headings = np.array(ego_hdg_list)
        ego_velocities = np.array([[v[0], v[1]] for v in ego_vel_list])
        ego_accelerations = np.array([[a[0], a[1]] for a in ego_acc_list])

        ego_traj = VehicleTrajectory(
            vehicle_id="ego",
            positions=ego_positions,
            headings=ego_headings,
            velocities=ego_velocities,
            accelerations=ego_accelerations,
            valid=np.array([True] * n)
        )

        calculator = RiskCalculator(ego_traj, target_traj)
        calculator.compute_relative_metrics()
        calculator.compute_risk_scores()

        risk_scores = calculator.risk_scores
        min_risk = float(np.min(risk_scores[:, 6]))

        return -min_risk

    except Exception:
        return 0.0


# =============================================================================
# DFS + 逐 Block Top-K% 剪枝
# =============================================================================

def dfs_mutate_with_pruning(
    block_idx: int,
    blocks: List[IntentBlock],
    current_state: TrajectoryState,
    trajectory: List[TrajectoryState],
    results: List[Tuple[List[TrajectoryState], float]],
    ego_trajectory: Dict,
    top_k: float = 10.0
):
    """深度优先搜索 + 逐 Block Top-K% 剪枝"""
    if block_idx == len(blocks):
        risk = compute_risk_score(trajectory, ego_trajectory)
        results.append((trajectory.copy(), risk))
        return

    block = blocks[block_idx]
    mutations_list = INTENTION_MUTATIONS.get(block.intent, INTENTION_MUTATIONS[DrivingIntention.UNKNOWN])

    all_branches: List[Tuple[List[TrajectoryState], TrajectoryState, float]] = []

    for mutation in mutations_list:
        combinations = get_mutation_combinations(mutation)
        for delta_a, delta_omega in combinations:
            if mutation.is_perturbation:
                delta_a = random.gauss(0, SIGMA_A)
                delta_omega = random.gauss(0, SIGMA_OMEGA)

            a_x_new, a_y_new = mutate_vector_acceleration(current_state, delta_a)
            omega_new = clamp(current_state.omega + delta_omega, OMEGA_MIN, OMEGA_MAX)

            new_state = current_state
            block_trajectory: List[TrajectoryState] = []

            if block.intent == DrivingIntention.DECELERATE_TO_STOP:
                for _ in range(block.frame_count):
                    if new_state.v < 0.1:
                        new_state = TrajectoryState(v=0.0, a_x=0.0, a_y=0.0, theta=new_state.theta, omega=0.0, x=new_state.x, y=new_state.y)
                    else:
                        new_state = kinematically_integrate(new_state, a_x_new, a_y_new, omega_new)
                    block_trajectory.append(new_state)
            else:
                for _ in range(block.frame_count):
                    new_state = kinematically_integrate(new_state, a_x_new, a_y_new, omega_new)
                    block_trajectory.append(new_state)

            end_frame_idx = len(trajectory) + len(block_trajectory) - 1
            block_risk = compute_block_risk_score(new_state, ego_trajectory, end_frame_idx)

            all_branches.append((trajectory + block_trajectory, new_state, block_risk))

    # 如果不是最后一个 Block：按风险分数排序，Top-K% 剪枝
    if block_idx < len(blocks) - 1 and all_branches:
        all_branches.sort(key=lambda x: x[2], reverse=True)
        cutoff = max(1, int(len(all_branches) * top_k / 100))
        all_branches = all_branches[:cutoff]

    for branch_trajectory, branch_state, _ in all_branches:
        dfs_mutate_with_pruning(
            block_idx + 1, blocks, branch_state, branch_trajectory, results, ego_trajectory, top_k
        )


# =============================================================================
# 轨迹变异器
# =============================================================================

def build_variant_output(variant_states: List[TrajectoryState], variant_id: int) -> Dict:
    """将变异状态序列构建为输出格式"""
    positions = [[s.x, s.y, 0.0] for s in variant_states]
    headings = [s.theta for s in variant_states]
    velocities = [[s.v, 0.0] for s in variant_states]
    accelerations = [[s.a_x, s.a_y] for s in variant_states]

    return {
        "variant_id": variant_id,
        "mutated_target_trajectory": {
            "positions": positions,
            "headings": headings,
            "velocities": velocities,
            "accelerations": accelerations,
            "valid": [True] * len(variant_states)
        }
    }


class TrajectoryMutator:
    """
    意图驱动轨迹变异器（DFS + 逐 Block Top-K% 剪枝）

    使用示例：
        from core.llm.trajectory_mutator import TrajectoryMutator

        mutator = TrajectoryMutator()
        variants = mutator.mutate(fragment, top_k=10)
    """

    def __init__(self, random_seed: int = None):
        if random_seed is not None:
            random.seed(random_seed)
        self.mutations = INTENTION_MUTATIONS

    def _extract_intention_frames(self, fragment: Dict) -> List[Dict]:
        """从 fragment 中统一提取 intention_frames 列表"""
        intention_analysis = fragment.get("intention_analysis", {})
        frames = intention_analysis.get("intention_frames", [])
        if frames:
            return frames

        intention_sequence = fragment.get("intention_sequence", {})
        phases = intention_sequence.get("intention_sequence", [])
        if phases:
            result = []
            for p in phases:
                result.append({
                    "frame": p.get("start_frame", 0),
                    "intention": p.get("intention", "unknown")
                })
            return result

        return []

    def _extract_trajectories(self, fragment: Dict) -> Tuple[Dict, Dict]:
        """从 fragment 中提取 target_trajectory 和 ego_trajectory"""
        target_traj = fragment.get("target_trajectory", {})
        ego_traj = fragment.get("ego_trajectory", {})

        if not target_traj:
            orig = fragment.get("original_fragment", {})
            target_traj = orig.get("target_trajectory", {})
            ego_traj = orig.get("ego_trajectory", {})

        if not target_traj:
            frag = fragment.get("fragment", {})
            target_traj = frag.get("target_trajectory", {})
            ego_traj = frag.get("ego_trajectory", {})

        return target_traj, ego_traj

    def mutate(self, fragment: Dict, top_k: float = 10.0) -> List[Dict]:
        """轨迹变异穷举主入口（DFS + 逐 Block Top-K% 剪枝）"""
        intention_frames = self._extract_intention_frames(fragment)
        target_trajectory, ego_trajectory = self._extract_trajectories(fragment)

        total_frames = len(target_trajectory.get("positions", []))
        if total_frames == 0:
            total_frames = 50

        blocks = merge_intentions(intention_frames, total_frames)
        initial_state = extract_initial_state(target_trajectory)

        results = []
        dfs_mutate_with_pruning(0, blocks, initial_state, [], results, ego_trajectory, top_k)

        return [
            build_variant_output(traj, i)
            for i, (traj, risk) in enumerate(results)
        ]

    def get_combination_count(self, blocks: List[IntentBlock]) -> int:
        """计算给定Block序列的总变异组合数"""
        total = 1
        for block in blocks:
            mutations = self.mutations.get(block.intent, self.mutations[DrivingIntention.UNKNOWN])
            for m in mutations:
                if m.is_perturbation:
                    total *= 1
                else:
                    total *= len(m.delta_a_mag_values) * len(m.delta_omega_values)
        return total
