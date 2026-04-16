"""
core/trajectory_mutator.py - 意图驱动轨迹变异器（v3：矢量加速度+随机扰动+DFS+TopK）

职责：
基于驾驶意图的穷举变异，生成最危险的轨迹变体

核心算法：
1. 意图合并（游程编码）：8~13个意图帧 → 3~5个IntentBlock
2. 矢量加速度变异：ax, ay 合并为标量，统一变异
3. 随机扰动保持：cruise_maintain等意图使用随机扰动而非固定值
4. DFS深度优先遍历 + Top-K%危险剪枝
5. 危险分数复用 processor.py 的 RiskCalculator

使用示例：
    from core.trajectory_mutator import IntentionDrivenTrajectoryMutator

    mutator = IntentionDrivenTrajectoryMutator()
    variants = mutator.mutate(fragment, top_k=10)
"""

import math
import random
from typing import Dict, List, Optional, Any, Iterator, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum

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
A_X_MIN = -8.0
A_X_MAX = 3.0

# 横向加速度边界 (m/s^2)
A_Y_MIN = -5.0
A_Y_MAX = 5.0

# 横摆角速度边界 (rad/s)
OMEGA_MIN = -0.35
OMEGA_MAX = 0.35

# 随机扰动标准差
SIGMA_A = 0.05  # 加速度标准差
SIGMA_OMEGA = 0.004  # 角速度标准差 (rad/s)


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
class MutationRange:
    """变异范围（连续区间）"""
    low: float
    high: float

    def sample(self) -> float:
        """从区间采样"""
        return random.uniform(self.low, self.high)


@dataclass
class MutationSpec:
    """
    变异规格：描述单个变量的变异方式

    mutation_type: "fixed" | "range" | "perturbation"
    - fixed: 固定值
    - range: 连续区间（采样）
    - perturbation: 随机扰动（相对于当前值）
    """
    mutation_type: str = "fixed"  # "fixed", "range", "perturbation"
    delta: float = 0.0  # 固定值或扰动系数
    range_low: float = 0.0  # 区间下界
    range_high: float = 0.0  # 区间上界

    @staticmethod
    def fixed(value: float) -> 'MutationSpec':
        return MutationSpec(mutation_type="fixed", delta=value)

    @staticmethod
    def range(low: float, high: float) -> 'MutationSpec':
        return MutationSpec(mutation_type="range", range_low=low, range_high=high)

    @staticmethod
    def perturbation(std: float) -> 'MutationSpec':
        return MutationSpec(mutation_type="perturbation", delta=std)


# 变异类型枚举
class MutationType(Enum):
    FIXED = "fixed"
    RANGE = "range"
    PERTURBATION = "perturbation"


@dataclass
class IntentionMutation:
    """
    意图变异规格：分别定义加速度和角速度的变异

    delta_a_spec: 加速度变异规格
    delta_omega_spec: 角速度变异规格
    """
    delta_a_spec: MutationSpec
    delta_omega_spec: MutationSpec


@dataclass
class TrajectoryState:
    """运动学状态"""
    v: float      # 速度标量 (m/s)
    a_x: float    # 纵向加速度 (m/s^2)
    a_y: float    # 横向加速度 (m/s^2)
    theta: float  # 航向角 (rad)
    omega: float  # 横摆角速度 (rad/s)
    x: float      # X坐标
    y: float      # Y坐标

    @property
    def a_mag(self) -> float:
        """矢量加速度大小"""
        return math.sqrt(self.a_x**2 + self.a_y**2)

    def to_dict(self) -> Dict:
        return {
            "v": self.v,
            "a_x": self.a_x,
            "a_y": self.a_y,
            "theta": self.theta,
            "omega": self.omega,
            "x": self.x,
            "y": self.y,
        }


@dataclass
class RiskScore:
    """危险分数结果"""
    trajectory_id: int
    risk_value: float
    trajectory: Dict


# =============================================================================
# 意图变异规格定义（v3：变量分开，随机扰动）
# =============================================================================

def make_intention_mutations() -> Dict[DrivingIntention, List[IntentionMutation]]:
    """
    创建所有意图的变异规格

    格式：每个意图有多个变异方案，每个方案定义：
    - delta_a_spec: 加速度变异
    - delta_omega_spec: 角速度变异
    """
    perturbations = {
        "a": MutationSpec.perturbation(SIGMA_A),
        "omega": MutationSpec.perturbation(SIGMA_OMEGA),
    }

    return {
        # ==================== 纯纵向意图 ====================
        DrivingIntention.CRUISE_MAINTAIN: [
            # 保持状态：随机扰动
            IntentionMutation(
                delta_a_spec=MutationSpec.perturbation(SIGMA_A),
                delta_omega_spec=MutationSpec.perturbation(SIGMA_OMEGA),
            ),
        ],

        DrivingIntention.ACCELERATE_THROUGH: [
            # 温和加速
            IntentionMutation(
                delta_a_spec=MutationSpec.range(0.1, 0.2),
                delta_omega_spec=MutationSpec.perturbation(SIGMA_OMEGA),
            ),
            # 中度加速
            IntentionMutation(
                delta_a_spec=MutationSpec.range(0.2, 0.5),
                delta_omega_spec=MutationSpec.perturbation(SIGMA_OMEGA),
            ),
            # 强力加速
            IntentionMutation(
                delta_a_spec=MutationSpec.range(0.5, 1.0),
                delta_omega_spec=MutationSpec.perturbation(SIGMA_OMEGA),
            ),
        ],

        DrivingIntention.DECELERATE_TO_YIELD: [
            # 平缓减速
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-0.3, -0.1),
                delta_omega_spec=MutationSpec.perturbation(SIGMA_OMEGA),
            ),
            # 中度减速
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-0.6, -0.3),
                delta_omega_spec=MutationSpec.perturbation(SIGMA_OMEGA),
            ),
            # 急减速
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-1.0, -0.6),
                delta_omega_spec=MutationSpec.perturbation(SIGMA_OMEGA),
            ),
        ],

        DrivingIntention.DECELERATE_TO_STOP: [
            # 平缓停车
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-0.3, -0.1),
                delta_omega_spec=MutationSpec.perturbation(SIGMA_OMEGA),
            ),
            # 中度停车
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-0.6, -0.3),
                delta_omega_spec=MutationSpec.perturbation(SIGMA_OMEGA),
            ),
            # 快速停车
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-1.0, -0.6),
                delta_omega_spec=MutationSpec.perturbation(SIGMA_OMEGA),
            ),
        ],

        DrivingIntention.EMERGENCY_BRAKE: [
            # 轻度急刹
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-2.0, -1.0),
                delta_omega_spec=MutationSpec.fixed(0.0),
            ),
            # 中度急刹
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-4.0, -2.0),
                delta_omega_spec=MutationSpec.fixed(0.0),
            ),
            # 重度急刹
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-8.0, -4.0),
                delta_omega_spec=MutationSpec.fixed(0.0),
            ),
        ],

        # ==================== 横向/综合意图 ====================
        DrivingIntention.LANE_CHANGE_LEFT: [
            # 保持速度左变道 - 慢速
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-0.1, 0.1),
                delta_omega_spec=MutationSpec.range(0.0087, 0.0175),
            ),
            # 加速左变道 - 中速
            IntentionMutation(
                delta_a_spec=MutationSpec.range(0.0, 0.2),
                delta_omega_spec=MutationSpec.range(0.0175, 0.0262),
            ),
            # 减速左变道 - 快速
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-0.2, 0.0),
                delta_omega_spec=MutationSpec.range(0.0262, 0.0349),
            ),
        ],

        DrivingIntention.LANE_CHANGE_RIGHT: [
            # 保持速度右变道 - 慢速
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-0.1, 0.1),
                delta_omega_spec=MutationSpec.range(-0.0349, -0.0262),
            ),
            # 加速右变道 - 中速
            IntentionMutation(
                delta_a_spec=MutationSpec.range(0.0, 0.2),
                delta_omega_spec=MutationSpec.range(-0.0262, -0.0175),
            ),
            # 减速右变道 - 快速
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-0.2, 0.0),
                delta_omega_spec=MutationSpec.range(-0.0175, -0.0087),
            ),
        ],

        DrivingIntention.TURN_LEFT: [
            # 入弯降速左转
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-0.2, -0.1),
                delta_omega_spec=MutationSpec.range(0.0175, 0.0262),
            ),
            # 保持速度左转
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-0.1, 0.0),
                delta_omega_spec=MutationSpec.range(0.0262, 0.0349),
            ),
            # 加速出弯左转
            IntentionMutation(
                delta_a_spec=MutationSpec.range(0.0, 0.2),
                delta_omega_spec=MutationSpec.range(0.0349, 0.0524),
            ),
        ],

        DrivingIntention.TURN_RIGHT: [
            # 入弯降速右转
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-0.2, -0.1),
                delta_omega_spec=MutationSpec.range(-0.0524, -0.0349),
            ),
            # 保持速度右转
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-0.1, 0.0),
                delta_omega_spec=MutationSpec.range(-0.0349, -0.0262),
            ),
            # 加速出弯右转
            IntentionMutation(
                delta_a_spec=MutationSpec.range(0.0, 0.2),
                delta_omega_spec=MutationSpec.range(-0.0262, -0.0175),
            ),
        ],

        DrivingIntention.GO_STRAIGHT: [
            # 减速直行
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-0.2, -0.1),
                delta_omega_spec=MutationSpec.perturbation(SIGMA_OMEGA),
            ),
            # 保持速度直行
            IntentionMutation(
                delta_a_spec=MutationSpec.range(-0.05, 0.05),
                delta_omega_spec=MutationSpec.perturbation(SIGMA_OMEGA),
            ),
            # 加速直行
            IntentionMutation(
                delta_a_spec=MutationSpec.range(0.1, 0.2),
                delta_omega_spec=MutationSpec.perturbation(SIGMA_OMEGA),
            ),
        ],

        DrivingIntention.UNKNOWN: [
            IntentionMutation(
                delta_a_spec=MutationSpec.perturbation(SIGMA_A),
                delta_omega_spec=MutationSpec.perturbation(SIGMA_OMEGA),
            ),
        ],
    }


# 全局变异规格
INTENTION_MUTATIONS = make_intention_mutations()


# =============================================================================
# 意图合并（游程编码）
# =============================================================================

def merge_intentions(intention_seq: IntentionSequence) -> List[IntentBlock]:
    """
    将意图序列合并为意图区块（游程编码）
    """
    if not intention_seq.phases:
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

    # 纵向/横向加速度
    a_x = accelerations[0][0] if accelerations else 0.0
    a_y = accelerations[0][1] if accelerations else 0.0

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

    return TrajectoryState(v=v, a_x=a_x, a_y=a_y, theta=theta, omega=omega, x=x, y=y)


def sample_delta(spec: MutationSpec, current_value: float = 0.0) -> float:
    """
    根据变异规格采样Delta值

    Args:
        spec: 变异规格
        current_value: 当前值（用于扰动计算）

    Returns:
        采样的Delta值
    """
    if spec.mutation_type == "fixed":
        return spec.delta
    elif spec.mutation_type == "range":
        return spec.sample()
    elif spec.mutation_type == "perturbation":
        # 随机扰动：围绕current_value添加高斯噪声
        return random.gauss(0, spec.delta)
    return 0.0


def compute_block_control(
    current_state: TrajectoryState,
    delta_a: float,
    delta_omega: float
) -> Tuple[float, float, float]:
    """
    计算区块恒定控制量

    Returns:
        (a_x_block, a_y_block, omega_block)
    """
    # 加速度更新（矢量形式）
    a_x_block = clamp(current_state.a_x + delta_a, A_X_MIN, A_X_MAX)
    a_y_block = clamp(current_state.a_y, A_Y_MIN, A_Y_MAX)  # 横向加速度保持

    # 横摆角速度更新
    omega_block = clamp(current_state.omega + delta_omega, OMEGA_MIN, OMEGA_MAX)

    return a_x_block, a_y_block, omega_block


def kinematically_integrate(
    state: TrajectoryState,
    a_x_block: float,
    a_y_block: float,
    omega_block: float,
    z: float = 0.0
) -> TrajectoryState:
    """
    运动学积分一步

    公式：
        v_t = max(0, v_{t-1} + a_x_block * dt)
        theta_t = theta_{t-1} + omega_block * dt
        x_t = x_{t-1} + v_t * cos(theta_t) * dt
        y_t = y_{t-1} + v_t * sin(theta_t) * dt
    """
    v = max(0.0, state.v + a_x_block * DT)
    theta = state.theta + omega_block * DT
    x = state.x + v * math.cos(theta) * DT
    y = state.y + v * math.sin(theta) * DT

    return TrajectoryState(
        v=v,
        a_x=a_x_block,
        a_y=a_y_block,
        theta=theta,
        omega=omega_block,
        x=x,
        y=y
    )


# =============================================================================
# 危险分数计算（复用processor.py）
# =============================================================================

def compute_risk_score(
    variant_trajectory: Dict,
    ego_trajectory: Dict,
    target_trajectory: Dict
) -> float:
    """
    计算轨迹危险分数（复用processor.py的RiskCalculator）

    Args:
        variant_trajectory: 变异后的轨迹
        ego_trajectory: 主车轨迹
        target_trajectory: 目标车轨迹

    Returns:
        危险分数（越大越危险，取负的最小风险评分）
    """
    try:
        from core.processor import RiskCalculator, VehicleTrajectory

        # 构建VehicleTrajectory对象
        n = len(variant_trajectory.get("positions", []))

        # 创建目标车轨迹（使用变异后的轨迹）
        target_traj = VehicleTrajectory(
            positions=np.array([[p[0], p[1], p[2] if len(p) > 2 else 0.0] for p in variant_trajectory.get("positions", [])]),
            headings=np.array(variant_trajectory.get("headings", [0.0] * n)),
            velocities=np.array([[v[0], v[1]] for v in variant_trajectory.get("velocities", [[0, 0]] * n)]),
            accelerations=np.array([[a[0], a[1]] for a in variant_trajectory.get("accelerations", [[0, 0]] * n)]),
            valid=np.array(variant_trajectory.get("valid", [True] * n))
        )

        # 创建主车轨迹
        ego_traj = VehicleTrajectory(
            positions=np.array([[p[0], p[1], p[2] if len(p) > 2 else 0.0] for p in ego_trajectory.get("positions", [])]),
            headings=np.array(ego_trajectory.get("headings", [0.0] * n)),
            velocities=np.array([[v[0], v[1]] for v in ego_trajectory.get("velocities", [[0, 0]] * n)]),
            accelerations=np.array([[a[0], a[1]] for a in ego_trajectory.get("accelerations", [[0, 0]] * n)]),
            valid=np.array(ego_trajectory.get("valid", [True] * n))
        )

        # 计算风险
        calculator = RiskCalculator(ego_traj, target_traj)
        calculator.compute_relative_metrics()
        calculator.compute_risk_scores()

        # 取最小风险分数（越危险分数越低）
        risk_scores = calculator.risk_scores
        min_risk = float(np.min(risk_scores[:, 6]))

        # 返回负值（使得风险越高值越大，方便排序）
        return -min_risk

    except Exception as e:
        # 如果导入或计算失败，使用简化风险评估
        return _fallback_risk_score(variant_trajectory, ego_trajectory)


def _fallback_risk_score(variant_trajectory: Dict, ego_trajectory: Dict) -> float:
    """
    备用风险分数计算（当RiskCalculator不可用时）

    基于与原始轨迹的偏离程度
    """
    positions = variant_trajectory.get("positions", [])
    ego_positions = ego_trajectory.get("positions", [])

    if not positions or not ego_positions:
        return 0.0

    total_deviation = 0.0
    min_dist = float('inf')

    for i in range(min(len(positions), len(ego_positions))):
        dx = positions[i][0] - ego_positions[i][0]
        dy = positions[i][1] - ego_positions[i][1]
        dist = math.sqrt(dx*dx + dy*dy)
        total_deviation += dist
        if dist < min_dist:
            min_dist = dist

    # 偏离越大、最小距离越近，风险越高
    avg_deviation = total_deviation / len(positions) if positions else 0
    return avg_deviation + 1.0 / (min_dist + 0.1)


# =============================================================================
# DFS + Top-K% 剪枝算法
# =============================================================================

class DFSMutator:
    """
    深度优先搜索轨迹变异器（v3）
    """

    def __init__(
        self,
        blocks: List[IntentBlock],
        target_trajectory: Dict,
        ego_trajectory: Optional[Dict],
        top_k: float = 10.0
    ):
        self.blocks = blocks
        self.target_trajectory = target_trajectory
        self.ego_trajectory = ego_trajectory or target_trajectory
        self.top_k = top_k
        self.variant_id = 0
        self.results: List[RiskScore] = []

    def mutate(self) -> List[Dict]:
        """执行DFS变异，返回Top-K%危险轨迹"""
        initial_state = compute_initial_state(self.target_trajectory)
        frame_count = self.target_trajectory.get("frame_count", 50)

        print(f"意图合并后Block数量: {len(self.blocks)}")
        total_combinations = 1
        for block in self.blocks:
            mutations = INTENTION_MUTATIONS.get(
                block.intent,
                INTENTION_MUTATIONS[DrivingIntention.UNKNOWN]
            )
            total_combinations *= len(mutations)
            print(f"  Block {block.intent.value}: {len(mutations)} 种变异")
        print(f"总变异组合数: {total_combinations}")

        # DFS遍历
        self._dfs(0, initial_state, [])

        # 按危险分数排序
        self.results.sort(key=lambda x: x.risk_value, reverse=True)

        # Top-K%筛选
        k = max(1, int(len(self.results) * self.top_k / 100))
        top_results = self.results[:k]

        print(f"DFS完成：共生成 {len(self.results)} 条轨迹，Top-{self.top_k}% = {k} 条")

        return [r.trajectory for r in top_results]

    def _dfs(
        self,
        block_idx: int,
        current_state: TrajectoryState,
        trajectory_states: List[TrajectoryState]
    ):
        """深度优先搜索"""
        if block_idx >= len(self.blocks):
            # 到达叶子节点
            trajectory = self._build_trajectory(trajectory_states)
            risk = compute_risk_score(
                trajectory,
                self.ego_trajectory,
                self.target_trajectory
            )
            self.results.append(RiskScore(
                trajectory_id=self.variant_id,
                risk_value=risk,
                trajectory=trajectory
            ))
            self.variant_id += 1
            return

        block = self.blocks[block_idx]

        # 获取该意图的变异规格
        mutations = INTENTION_MUTATIONS.get(
            block.intent,
            INTENTION_MUTATIONS[DrivingIntention.UNKNOWN]
        )

        for mutation in mutations:
            # 采样Delta值
            delta_a = sample_delta(mutation.delta_a_spec, current_state.a_x)
            delta_omega = sample_delta(mutation.delta_omega_spec, current_state.omega)

            # 计算区块恒定控制量
            a_x_block, a_y_block, omega_block = compute_block_control(
                current_state, delta_a, delta_omega
            )

            # 递推该Block内所有帧
            new_states = []
            state = current_state
            for _ in range(block.frame_count):
                state = kinematically_integrate(state, a_x_block, a_y_block, omega_block)
                new_states.append(state)

            # 记录变异日志（附加到前一个状态）
            self._dfs(block_idx + 1, state, trajectory_states + new_states)

    def _build_trajectory(self, states: List[TrajectoryState]) -> Dict:
        """根据状态列表构建轨迹字典"""
        if not states:
            return {}

        frame_count = len(states)
        positions = np.zeros((frame_count, 3))
        headings = np.zeros(frame_count)
        velocities = np.zeros((frame_count, 2))
        accelerations = np.zeros((frame_count, 2))

        # 获取原始Z坐标
        orig_positions = self.target_trajectory.get("positions", [[0, 0, 0]] * frame_count)

        for i, state in enumerate(states):
            positions[i] = [
                state.x,
                state.y,
                orig_positions[i][2] if i < len(orig_positions) else 0.0
            ]
            headings[i] = state.theta
            velocities[i] = [state.v, 0.0]
            accelerations[i] = [state.a_x, state.a_y]

        return {
            "original_fragment_id": self.target_trajectory.get("fragment_id", "unknown"),
            "variant_id": self.variant_id - 1,
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
    意图驱动轨迹变异器（v3）

    算法：
    1. 意图合并（游程编码）
    2. 矢量加速度变异 + 随机扰动
    3. DFS深度优先遍历
    4. Top-K%危险剪枝（使用RiskCalculator）
    """

    def __init__(self, top_k: float = 10.0, seed: Optional[int] = None):
        """
        参数：
            top_k: 保留最危险轨迹的比例（默认10%）
            seed: 随机种子（用于 reproducibility）
        """
        self.top_k = top_k
        if seed is not None:
            random.seed(seed)

    def mutate(self, fragment: Dict, top_k: Optional[float] = None) -> List[Dict]:
        """
        穷举生成Top-K%最危险轨迹

        参数：
            fragment: 原始轨迹片段
            top_k: 保留最危险轨迹的比例

        返回：
            危险轨迹变体列表
        """
        if top_k is None:
            top_k = self.top_k

        # 解析片段数据
        frag = fragment.get("fragment", fragment) if "fragment" in fragment else fragment
        target = frag.get("target_trajectory", frag)
        ego = frag.get("ego_trajectory", target)
        frame_count = frag.get("frame_count", 50)

        # 构建意图序列
        intention_seq = self._build_intention_sequence(frag)

        # 意图合并
        blocks = merge_intentions(intention_seq)

        # DFS + Top-K%变异
        dfs_mutator = DFSMutator(blocks, target, ego, top_k=top_k)
        variants = dfs_mutator.mutate()

        return variants

    def _build_intention_sequence(self, frag: Dict) -> IntentionSequence:
        """从片段数据构建意图序列"""
        intention_analysis = frag.get("intention_analysis", {})

        if intention_analysis and "intention_frames" in intention_analysis:
            frames = intention_analysis["intention_frames"]
            if frames:
                sorted_frames = sorted(frames, key=lambda x: x.get("frame", 0))
                phases = []
                for f in sorted_frames:
                    frame_num = f.get("frame", 0)
                    intent_str = f.get("intention", "cruise_maintain")
                    try:
                        intent = DrivingIntention(intent_str)
                    except ValueError:
                        intent = DrivingIntention.CRUISE_MAINTAIN

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

        # 默认意图序列
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


def mutate_trajectories(fragment: Dict, top_k: float = 10.0, seed: Optional[int] = None) -> List[Dict]:
    """
    便捷函数：穷举生成Top-K%最危险轨迹

    参数：
        fragment: 原始轨迹片段
        top_k: 保留最危险轨迹的比例
        seed: 随机种子

    返回：
        危险轨迹变体列表
    """
    mutator = IntentionDrivenTrajectoryMutator(top_k=top_k, seed=seed)
    return mutator.mutate(fragment, top_k=top_k)


def get_combination_count(intention: DrivingIntention) -> int:
    """
    获取指定意图类型的变异方案数
    """
    return len(INTENTION_MUTATIONS.get(intention, []))
