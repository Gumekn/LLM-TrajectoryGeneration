"""
core/trajectory_mutator.py - 意图驱动轨迹变异器（v2：宏动作+DFS+TopK剪枝）

职责：
基于驾驶意图的穷举变异，生成最危险的轨迹变体

核心算法：
1. 意图合并（游程编码）：8~13个意图帧 → 3~5个IntentBlock
2. 宏动作变异：每个Block起始帧选定(Delta_a, Delta_omega)，Block内保持恒定
3. DFS深度优先遍历：3~5个Block，每Block 3种分支 → 最多243条轨迹
4. Top-K%危险剪枝：按危险分数排序，只保留前K%最危险轨迹

变异变量：
    Delta_a: 纵向加速度增量 (m/s^2)
    Delta_omega: 横摆角速度增量 (rad/s)

运动学状态向量：
    S_t = [v_t, a_t, theta_t, omega_t, x_t, y_t]

宏动作递推公式（Block内恒定控制）：
    a_block = clamp(a_{t-1} + Delta_a, -8.0, 3.0)
    omega_block = clamp(omega_{t-1} + Delta_omega, -0.35, 0.35)
    v_t = max(0, v_{t-1} + a_block * dt)
    theta_t = theta_{t-1} + omega_block * dt
    x_t = x_{t-1} + v_t * cos(theta_t) * dt
    y_t = y_{t-1} + v_t * sin(theta_t) * dt

使用示例：
    from core.trajectory_mutator import IntentionDrivenTrajectoryMutator

    mutator = IntentionDrivenTrajectoryMutator()
    variants = mutator.mutate(fragment, top_k=10)
"""

import math
from typing import Dict, List, Optional, Any, Iterator, Tuple, NamedTuple
from dataclasses import dataclass

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
OMEGA_MIN = -0.35  # rad/s
OMEGA_MAX = 0.35   # rad/s


# =============================================================================
# 数据结构定义
# =============================================================================

class IntentBlock(NamedTuple):
    """意图区块：合并后的变异单元"""
    intent: DrivingIntention
    start_frame: int
    end_frame: int
    duration: float  # 秒

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class MutationCombo:
    """变异组合"""
    delta_a: float
    delta_omega: float
    description: str


@dataclass
class TrajectoryState:
    """运动学状态"""
    v: float      # 速度标量 (m/s)
    a: float      # 纵向加速度 (m/s^2)
    theta: float  # 航向角 (rad)
    omega: float  # 横摆角速度 (rad/s)
    x: float      # X坐标
    y: float      # Y坐标

    def to_dict(self) -> Dict:
        return {
            "v": self.v,
            "a": self.a,
            "theta": self.theta,
            "omega": self.omega,
            "x": self.x,
            "y": self.y,
        }


@dataclass
class RiskScore:
    """危险分数"""
    trajectory_id: int
    risk_value: float
    trajectory: Dict


# =============================================================================
# 意图钦定变异组合（每种意图3种最优组合）
# =============================================================================

INTENTION_MUTATIONS: Dict[DrivingIntention, List[MutationCombo]] = {
    # 纯纵向意图
    DrivingIntention.CRUISE_MAINTAIN: [
        MutationCombo(-0.1, 0.0, "轻微减速"),
        MutationCombo(0.0, 0.0, "保持不变"),
        MutationCombo(0.1, 0.0, "轻微加速"),
    ],

    DrivingIntention.ACCELERATE_THROUGH: [
        MutationCombo(0.0, 0.0087, "保持加速+微右转"),
        MutationCombo(0.1, 0.0, "匀加速"),
        MutationCombo(0.2, 0.0, "强加速"),
    ],

    DrivingIntention.DECELERATE_TO_YIELD: [
        MutationCombo(-0.1, 0.0, "平缓减速"),
        MutationCombo(-0.2, 0.0, "中等减速"),
        MutationCombo(-0.5, 0.0, "急减速"),
    ],

    DrivingIntention.DECELERATE_TO_STOP: [
        MutationCombo(-0.1, 0.0, "平缓停车"),
        MutationCombo(-0.2, 0.0, "中等停车"),
        MutationCombo(-0.5, 0.0, "快速停车"),
    ],

    DrivingIntention.EMERGENCY_BRAKE: [
        MutationCombo(-0.5, 0.0, "轻度急刹"),
        MutationCombo(-1.0, 0.0, "中度急刹"),
        MutationCombo(-2.0, 0.0, "重度急刹"),
    ],

    # 横向/综合意图
    DrivingIntention.LANE_CHANGE_LEFT: [
        MutationCombo(0.0, 0.0087, "慢速左变道"),
        MutationCombo(0.0, 0.0175, "中速左变道"),
        MutationCombo(0.0, 0.0349, "快速左变道"),
    ],

    DrivingIntention.LANE_CHANGE_RIGHT: [
        MutationCombo(0.0, -0.0087, "慢速右变道"),
        MutationCombo(0.0, -0.0175, "中速右变道"),
        MutationCombo(0.0, -0.0349, "快速右变道"),
    ],

    DrivingIntention.TURN_LEFT: [
        MutationCombo(-0.1, 0.0175, "入弯降速左转"),
        MutationCombo(-0.2, 0.0175, "深度降速左转"),
        MutationCombo(0.0, 0.0349, "保持速度左转"),
    ],

    DrivingIntention.TURN_RIGHT: [
        MutationCombo(-0.1, -0.0175, "入弯降速右转"),
        MutationCombo(-0.2, -0.0175, "深度降速右转"),
        MutationCombo(0.0, -0.0349, "保持速度右转"),
    ],

    DrivingIntention.GO_STRAIGHT: [
        MutationCombo(-0.1, 0.0, "减速直行"),
        MutationCombo(0.0, 0.0, "匀速直行"),
        MutationCombo(0.1, 0.0, "加速直行"),
    ],

    DrivingIntention.UNKNOWN: [
        MutationCombo(-0.1, 0.0, "轻微减速"),
        MutationCombo(0.0, 0.0, "保持不变"),
        MutationCombo(0.1, 0.0, "轻微加速"),
    ],
}


# =============================================================================
# 意图合并（游程编码）
# =============================================================================

def merge_intentions(intention_seq: IntentionSequence) -> List[IntentBlock]:
    """
    将意图序列合并为意图区块（游程编码）

    相邻且相同的意图合并为一个IntentBlock

    参数：
        intention_seq: 意图序列

    返回：
        IntentBlock列表
    """
    if not intention_seq.phases:
        # 默认单一Block
        return [IntentBlock(
            intent=DrivingIntention.CRUISE_MAINTAIN,
            start_frame=0,
            end_frame=50,
            duration=5.0
        )]

    blocks = []
    current_intent = intention_seq.phases[0].intention
    current_start = intention_seq.phases[0].start_frame

    for phase in intention_seq.phases:
        if phase.intention != current_intent:
            # 意图变更，保存当前Block
            blocks.append(IntentBlock(
                intent=current_intent,
                start_frame=current_start,
                end_frame=phase.start_frame,
                duration=(phase.start_frame - current_start) * DT
            ))
            current_intent = phase.intention
            current_start = phase.start_frame

    # 保存最后一个Block
    last_phase = intention_seq.phases[-1]
    blocks.append(IntentBlock(
        intent=current_intent,
        start_frame=current_start,
        end_frame=last_phase.end_frame,
        duration=(last_phase.end_frame - current_start) * DT
    ))

    return blocks


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


def compute_initial_state(trajectory: Dict) -> TrajectoryState:
    """
    从原始轨迹数据计算初始状态
    """
    velocities = trajectory.get("velocities", [[0.0, 0.0]])
    headings = trajectory.get("headings", [0.0])
    positions = trajectory.get("positions", [[0.0, 0.0, 0.0]])
    accelerations = trajectory.get("accelerations", [[0.0, 0.0]])

    # 速度标量
    v = compute_speed_from_velocity(velocities[0]) if velocities else 0.0

    # 纵向加速度
    a = accelerations[0][0] if accelerations else 0.0

    # 航向角
    theta = headings[0] if headings else 0.0

    # 横摆角速度
    if len(headings) >= 2:
        omega = (headings[1] - headings[0]) / DT
    else:
        omega = 0.0

    # 位置
    x = positions[0][0] if positions else 0.0
    y = positions[0][1] if positions else 0.0

    return TrajectoryState(v=v, a=a, theta=theta, omega=omega, x=x, y=y)


def compute_block_control(
    current_state: TrajectoryState,
    delta_a: float,
    delta_omega: float
) -> Tuple[float, float]:
    """
    计算区块恒定控制量

    参数：
        current_state: 当前状态
        delta_a: 纵向加速度增量
        delta_omega: 横摆角速度增量

    返回：
        (a_block, omega_block)
    """
    a_block = clamp(current_state.a + delta_a, A_MIN, A_MAX)
    omega_block = clamp(current_state.omega + delta_omega, OMEGA_MIN, OMEGA_MAX)
    return a_block, omega_block


def kinematically_integrate(
    state: TrajectoryState,
    a_block: float,
    omega_block: float,
    z: float = 0.0
) -> TrajectoryState:
    """
    运动学积分一步

    公式：
        v_t = max(0, v_{t-1} + a_block * dt)
        theta_t = theta_{t-1} + omega_block * dt
        x_t = x_{t-1} + v_t * cos(theta_t) * dt
        y_t = y_{t-1} + v_t * sin(theta_t) * dt

    参数：
        state: 当前状态
        a_block: 区块恒定纵向加速度
        omega_block: 区块恒定横摆角速度
        z: Z坐标（固定）

    返回：
        下一帧状态
    """
    v = max(0.0, state.v + a_block * DT)
    theta = state.theta + omega_block * DT
    x = state.x + v * math.cos(theta) * DT
    y = state.y + v * math.sin(theta) * DT

    return TrajectoryState(
        v=v,
        a=a_block,
        theta=theta,
        omega=omega_block,
        x=x,
        y=y
    )


# =============================================================================
# 危险分数计算
# =============================================================================

def compute_risk_score(
    trajectory: Dict,
    target_trajectory: Optional[Dict] = None
) -> float:
    """
    计算轨迹危险分数

    基于与目标轨迹的接近程度和碰撞时间TTC

    参数：
        trajectory: 变异后的轨迹
        target_trajectory: 原始轨迹（用于计算相对指标）

    返回：
        危险分数（越大越危险）
    """
    positions = trajectory.get("positions", [])
    velocities = trajectory.get("velocities", [])

    if not positions or len(positions) < 2:
        return 0.0

    # 如果没有目标轨迹，使用速度变化率和加速度作为危险指标
    if target_trajectory is None:
        risk = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            # 使用加速度变化作为危险指标
            if velocities and i < len(velocities):
                accel = velocities[i][0]
                risk += abs(accel) * 0.1
        return risk / len(positions)

    # 计算与目标轨迹的偏离程度
    target_positions = target_trajectory.get("positions", [])
    if not target_positions:
        return 0.0

    risk = 0.0
    min_dist = float('inf')

    for i in range(min(len(positions), len(target_positions))):
        dx = positions[i][0] - target_positions[i][0]
        dy = positions[i][1] - target_positions[i][1]
        dist = math.sqrt(dx*dx + dy*dy)

        # 距离越近越危险
        risk += 1.0 / (dist + 0.1)

        # 记录最小距离
        if dist < min_dist:
            min_dist = dist

    # 考虑速度差异
    target_velocities = target_trajectory.get("velocities", [])
    if target_velocities:
        for i in range(min(len(velocities), len(target_velocities))):
            dv = velocities[i][0] - target_velocities[i][0]
            risk += abs(dv) * 0.5

    return risk


# =============================================================================
# DFS + Top-K% 剪枝算法
# =============================================================================

class DFSMutator:
    """
    深度优先搜索轨迹变异器

    使用DFS遍历所有Block的变异组合，
    生成轨迹后按危险分数排序，保留Top-K%
    """

    def __init__(
        self,
        blocks: List[IntentBlock],
        target_trajectory: Dict,
        top_k: float = 10.0
    ):
        self.blocks = blocks
        self.target_trajectory = target_trajectory
        self.top_k = top_k
        self.variant_id = 0
        self.results: List[RiskScore] = []

    def mutate(self) -> List[Dict]:
        """执行DFS变异，返回Top-K%危险轨迹"""
        # 计算初始状态
        initial_state = compute_initial_state(self.target_trajectory)
        frame_count = self.target_trajectory.get("frame_count", 50)

        # DFS遍历
        self._dfs(0, initial_state, [], [])

        # 按危险分数排序
        self.results.sort(key=lambda x: x.risk_value, reverse=True)

        # Top-K%筛选
        k = max(1, int(len(self.results) * self.top_k / 100))
        top_results = self.results[:k]

        print(f"DFS完成：共生成 {len(self.results)} 条轨迹，保留 Top-{self.top_k}% = {k} 条")

        return [r.trajectory for r in top_results]

    def _dfs(
        self,
        block_idx: int,
        current_state: TrajectoryState,
        trajectory_states: List,
        mutation_log: List[Dict]
    ):
        """深度优先搜索"""
        if block_idx >= len(self.blocks):
            # 到达叶子节点，生成完整轨迹
            trajectory = self._build_trajectory(
                trajectory_states,
                mutation_log,
                len(self.blocks)
            )
            risk = compute_risk_score(trajectory, self.target_trajectory)
            self.results.append(RiskScore(
                trajectory_id=self.variant_id,
                risk_value=risk,
                trajectory=trajectory
            ))
            self.variant_id += 1
            return

        block = self.blocks[block_idx]

        # 获取该意图的变异组合
        mutations = INTENTION_MUTATIONS.get(
            block.intent,
            INTENTION_MUTATIONS[DrivingIntention.UNKNOWN]
        )

        for mutation in mutations:
            # 计算区块恒定控制量
            a_block, omega_block = compute_block_control(
                current_state, mutation.delta_a, mutation.delta_omega
            )

            # 递推该Block内所有帧
            new_states = []
            state = current_state
            for _ in range(block.frame_count):
                state = kinematically_integrate(state, a_block, omega_block)
                new_states.append(state)

            # 记录变异日志
            new_log = mutation_log + [{
                "block_idx": block_idx,
                "intent": block.intent.value,
                "start_frame": block.start_frame,
                "end_frame": block.end_frame,
                "delta_a": mutation.delta_a,
                "delta_omega": mutation.delta_omega,
                "a_block": a_block,
                "omega_block": omega_block,
                "description": mutation.description,
            }]

            # 递归处理下一Block
            self._dfs(
                block_idx + 1,
                state,
                trajectory_states + new_states,
                new_log
            )

    def _build_trajectory(
        self,
        states: List[TrajectoryState],
        mutation_log: List[Dict],
        block_count: int
    ) -> Dict:
        """根据状态列表构建轨迹字典"""
        frame_count = len(states)
        positions = np.zeros((frame_count, 3))
        headings = np.zeros(frame_count)
        velocities = np.zeros((frame_count, 2))
        accelerations = np.zeros((frame_count, 2))

        # 获取原始Z坐标
        orig_positions = self.target_trajectory.get("positions", [[0, 0, 0]] * frame_count)

        for i, state in enumerate(states):
            positions[i] = [state.x, state.y, orig_positions[i][2] if i < len(orig_positions) else 0.0]
            headings[i] = state.theta
            velocities[i] = [state.v, 0.0]  # 局部坐标系
            accelerations[i] = [state.a, state.v * state.omega]  # 纵向+向心加速度

        return {
            "original_fragment_id": self.target_trajectory.get("fragment_id", "unknown"),
            "variant_id": self.variant_id - 1,  # 已在叶子节点递增
            "mutation_log": mutation_log,
            "trajectory": {
                "positions": positions.tolist(),
                "headings": headings.tolist(),
                "velocities": velocities.tolist(),
                "accelerations": accelerations.tolist(),
                "valid": [True] * frame_count,
            }
        }


# =============================================================================
# 轨迹变异器主类
# =============================================================================

class IntentionDrivenTrajectoryMutator:
    """
    意图驱动轨迹变异器（v2）

    算法：
    1. 意图合并（游程编码）
    2. 宏动作变异（Block内恒定控制量）
    3. DFS深度优先遍历
    4. Top-K%危险剪枝

    使用示例：
        mutator = IntentionDrivenTrajectoryMutator()
        variants = mutator.mutate(fragment, top_k=10)
    """

    def __init__(self, top_k: float = 10.0):
        """
        参数：
            top_k: 保留最危险轨迹的比例（默认10%）
        """
        self.top_k = top_k

    def mutate(self, fragment: Dict, top_k: Optional[float] = None) -> List[Dict]:
        """
        穷举生成Top-K%最危险轨迹

        参数：
            fragment: 原始轨迹片段（包含意图序列）
            top_k: 保留最危险轨迹的比例（默认10%）

        返回：
            危险轨迹变体列表
        """
        if top_k is None:
            top_k = self.top_k

        # 解析片段数据
        frag = fragment.get("fragment", fragment) if "fragment" in fragment else fragment
        target = frag.get("target_trajectory", frag)
        frame_count = frag.get("frame_count", 50)

        # 构建意图序列
        intention_seq = self._build_intention_sequence(frag)

        # 意图合并（游程编码）
        blocks = merge_intentions(intention_seq)

        print(f"意图合并后Block数量: {len(blocks)}")
        for i, block in enumerate(blocks):
            print(f"  Block {i}: {block.intent.value}, [{block.start_frame}, {block.end_frame}], {block.duration}s")

        # DFS + Top-K%变异
        dfs_mutator = DFSMutator(blocks, target, top_k=top_k)
        variants = dfs_mutator.mutate()

        return variants

    def _build_intention_sequence(self, frag: Dict) -> IntentionSequence:
        """
        从片段数据构建意图序列

        如果有意图分析结果，使用分析结果
        否则使用默认意图序列
        """
        intention_analysis = frag.get("intention_analysis", {})

        if intention_analysis and "intention_frames" in intention_analysis:
            # 从意图帧构建意图序列
            frames = intention_analysis["intention_frames"]
            if frames:
                # 按frame排序
                sorted_frames = sorted(frames, key=lambda x: x.get("frame", 0))

                # 构建phase列表
                phases = []
                for f in sorted_frames:
                    frame_num = f.get("frame", 0)
                    intent_str = f.get("intention", "cruise_maintain")
                    try:
                        intent = DrivingIntention(intent_str)
                    except ValueError:
                        intent = DrivingIntention.CRUISE_MAINTAIN

                    # 假设每帧持续0.5s（关键帧间隔）
                    phases.append(IntentionPhase(
                        start_frame=int(frame_num),
                        end_frame=min(int(frame_num) + 5, 50),
                        intention=intent,
                        reasoning=f.get("reasoning", "")
                    ))

                if phases:
                    return IntentionSequence(
                        phases=phases,
                        overall_strategy=intention_analysis.get("overall_strategy", "意图驱动")
                    )

        # 默认意图序列（cruise_maintain整个轨迹）
        return IntentionSequence(
            phases=[
                IntentionPhase(
                    start_frame=0,
                    end_frame=50,
                    intention=DrivingIntention.CRUISE_MAINTAIN,
                    reasoning="默认意图"
                )
            ],
            overall_strategy="默认策略"
        )


def mutate_trajectories(fragment: Dict, top_k: float = 10.0) -> List[Dict]:
    """
    便捷函数：穷举生成Top-K%最危险轨迹

    参数：
        fragment: 原始轨迹片段
        top_k: 保留最危险轨迹的比例（默认10%）

    返回：
        危险轨迹变体列表
    """
    mutator = IntentionDrivenTrajectoryMutator(top_k=top_k)
    return mutator.mutate(fragment, top_k=top_k)


def get_combination_count(intention: DrivingIntention) -> int:
    """
    获取指定意图类型的有效组合数（固定为3）

    参数：
        intention: 驾驶意图类型

    返回：
        有效变异组合数量（固定3）
    """
    return len(INTENTION_MUTATIONS.get(intention, []))
