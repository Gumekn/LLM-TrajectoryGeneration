"""
core/processor.py - 危险轨迹生成系统核心处理模块

本模块整合了轨迹生成阶段的所有核心功能，按功能分为以下几类：

1. 数据类型定义 (types)         - 第20-400行
   定义系统中使用的数据结构，供其他所有模块使用。

2. Waymo数据加载 (data_loader)  - 第400-700行
   负责从Waymo pkl格式数据中加载场景、提取车辆轨迹。
   被 Main.py 调用。

3. 风险计算 (risk_calculator)    - 第700-1100行
   基于Data_Processor.py重构，实现TTC计算、风险分数计算、危险类型判断。
   被 Main.py 和 ScenarioProcessor 调用。

4. 片段截取与特征提取 (fragment_extractor) - 第1100-1800行
   实现危险轨迹片段截取、11维特征提取、JSON持久化。
   被 Main.py 调用。

使用示例:
    from core.processor import WaymoDataLoader, ScenarioRiskAnalyzer, ScenarioProcessor

    processor = ScenarioProcessor("data/waymo-open", output_dir="data/processed")
    scenario = processor.process_scenario("10135f16cd538e19", ego_id="1811")
"""

import os
import json
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

import numpy as np


# =============================================================================
# 第一部分：数据类型定义 (types.py)
# =============================================================================
# 用途：定义系统中使用的数据结构
# 被使用：所有其他模块使用这些类型进行数据传递和类型标注
# =============================================================================

# -----------------------------------------------------------------------------
# 基础数据类型
# -----------------------------------------------------------------------------

@dataclass
class VehicleTrajectory:
    """车辆轨迹数据

    Attributes:
        vehicle_id: 车辆唯一标识
        positions: 全局位置序列 (N, 3) - [x, y, z]
        headings: 航向角序列 (N,) - rad
        velocities: 局部速度序列 (N, 2) - [vx, vy] m/s
        accelerations: 局部加速度序列 (N, 2) - [ax, ay] m/s²
        sizes: 尺寸序列 (N, 3) - [length, width, height] m
        valid: 有效标志序列 (N,)

    被使用于:
        - WaymoDataLoader.extract_vehicle_trajectory() 返回轨迹
        - WaymoDataLoader.extract_ego_trajectory() 返回轨迹
        - RiskCalculator.__init__() 接收主车和交互车轨迹
        - compute_interaction_features_series() 计算相对运动
        - compute_target_speed_features() 提取速度特征
        - compute_target_accel_features() 提取加速度特征
        - compute_trajectory_length() 计算轨迹长度
        - compute_max_curvature() 计算曲率
        - create_trajectory_fragment() 创建片段
        - slice_trajectory() 切片轨迹
    """
    vehicle_id: str
    positions: np.ndarray
    headings: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    sizes: np.ndarray
    valid: np.ndarray

    def __post_init__(self):
        """验证数据形状一致性"""
        n = len(self.valid)
        assert self.positions.shape == (n, 3), f"positions shape mismatch: {self.positions.shape}"
        assert self.headings.shape == (n,), f"headings shape mismatch: {self.headings.shape}"
        assert self.velocities.shape == (n, 2), f"velocities shape mismatch: {self.velocities.shape}"
        assert self.accelerations.shape == (n, 2), f"accelerations shape mismatch: {self.accelerations.shape}"
        assert self.sizes.shape == (n, 3), f"sizes shape mismatch: {self.sizes.shape}"

    @property
    def frame_count(self) -> int:
        """返回轨迹帧数"""
        return len(self.valid)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于JSON序列化）"""
        return {
            'vehicle_id': self.vehicle_id,
            'positions': self.positions.tolist(),
            'headings': self.headings.tolist(),
            'velocities': self.velocities.tolist(),
            'accelerations': self.accelerations.tolist(),
            'sizes': self.sizes.tolist(),
            'valid': self.valid.tolist()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VehicleTrajectory':
        """从字典格式创建（用于JSON反序列化）"""
        return cls(
            vehicle_id=data['vehicle_id'],
            positions=np.array(data['positions']),
            headings=np.array(data['headings']),
            velocities=np.array(data['velocities']),
            accelerations=np.array(data['accelerations']),
            sizes=np.array(data['sizes']),
            valid=np.array(data['valid'])
        )


@dataclass
class InteractionFeatures:
    """交互特征序列（每帧计算）

    Attributes:
        rel_pos_x: 相对位置X (N,)
        rel_pos_y: 相对位置Y (N,)
        rel_dist: 相对距离 (N,)
        rel_vel_x: 相对速度X (N,)
        rel_vel_y: 相对速度Y (N,)
        rel_speed: 相对速度大小 (N,)
        ttc_long: 纵向TTC (N,) - 秒
        ttc_lat: 横向TTC (N,) - 秒
        local_angle: 局部坐标系下相对角度 (N,) - rad
        lateral_offset: 横向偏移 (N,) - m
        longitudinal_offset: 纵向偏移 (N,) - m
        heading_diff: 航向差 (N,) - rad

    被使用于:
        - compute_interaction_features_series() 返回交互特征
        - create_trajectory_fragment() 组装片段数据
    """
    rel_pos_x: np.ndarray
    rel_pos_y: np.ndarray
    rel_dist: np.ndarray
    rel_vel_x: np.ndarray
    rel_vel_y: np.ndarray
    rel_speed: np.ndarray
    ttc_long: np.ndarray
    ttc_lat: np.ndarray
    local_angle: np.ndarray
    lateral_offset: np.ndarray
    longitudinal_offset: np.ndarray
    heading_diff: np.ndarray

    def to_dict(self) -> Dict[str, List]:
        """转换为字典格式"""
        return {
            'rel_pos_x': self.rel_pos_x.tolist(),
            'rel_pos_y': self.rel_pos_y.tolist(),
            'rel_dist': self.rel_dist.tolist(),
            'rel_vel_x': self.rel_vel_x.tolist(),
            'rel_vel_y': self.rel_vel_y.tolist(),
            'rel_speed': self.rel_speed.tolist(),
            'ttc_long': self.ttc_long.tolist(),
            'ttc_lat': self.ttc_lat.tolist(),
            'local_angle': self.local_angle.tolist(),
            'lateral_offset': self.lateral_offset.tolist(),
            'longitudinal_offset': self.longitudinal_offset.tolist(),
            'heading_diff': self.heading_diff.tolist()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, List]) -> 'InteractionFeatures':
        """从字典创建"""
        return cls(
            rel_pos_x=np.array(data['rel_pos_x']),
            rel_pos_y=np.array(data['rel_pos_y']),
            rel_dist=np.array(data['rel_dist']),
            rel_vel_x=np.array(data['rel_vel_x']),
            rel_vel_y=np.array(data['rel_vel_y']),
            rel_speed=np.array(data['rel_speed']),
            ttc_long=np.array(data['ttc_long']),
            ttc_lat=np.array(data['ttc_lat']),
            local_angle=np.array(data['local_angle']),
            lateral_offset=np.array(data['lateral_offset']),
            longitudinal_offset=np.array(data['longitudinal_offset']),
            heading_diff=np.array(data['heading_diff'])
        )


@dataclass
class InteractionStats:
    """交互统计特征（从序列计算）

    Attributes:
        min_ttc_long: 最小纵向TTC - 秒
        min_ttc_lat: 最小横向TTC - 秒
        min_rel_dist: 最小相对距离 - m
        mean_rel_dist: 平均相对距离 - m
        max_closing_speed: 最大纵向接近速度 - m/s
        mean_closing_speed: 平均纵向接近速度 - m/s
        approach_time_ratio: 接近时间占比
        mean_heading_diff: 平均航向差 - rad

    被使用于:
        - compute_interaction_stats() 返回统计特征
        - extract_11d_features() 使用统计特征计算11维向量
        - create_trajectory_fragment() 组装片段数据
    """
    min_ttc_long: float
    min_ttc_lat: float
    min_rel_dist: float
    mean_rel_dist: float
    max_closing_speed: float
    mean_closing_speed: float
    approach_time_ratio: float
    mean_heading_diff: float

    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            'min_ttc_long': self.min_ttc_long,
            'min_ttc_lat': self.min_ttc_lat,
            'min_rel_dist': self.min_rel_dist,
            'mean_rel_dist': self.mean_rel_dist,
            'max_closing_speed': self.max_closing_speed,
            'mean_closing_speed': self.mean_closing_speed,
            'approach_time_ratio': self.approach_time_ratio,
            'mean_heading_diff': self.mean_heading_diff
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'InteractionStats':
        """从字典创建"""
        return cls(**data)


# -----------------------------------------------------------------------------
# 元数据类型
# -----------------------------------------------------------------------------

@dataclass
class FragmentMetadata:
    """片段元数据

    Attributes:
        fragment_id: 片段唯一标识
        scenario_id: 原始场景ID
        anchor_frame: 最危险帧在原始场景中的索引
        ego_vehicle_id: 主车ID
        target_vehicle_id: 交互车ID
        danger_type: 危险类型
        danger_level: 危险等级
        min_risk_score: 最低风险分数
        frame_count: 总帧数
        duration: 时长（秒）
        n_before: 锚点前帧数
        n_after: 锚点后帧数

    被使用于:
        - create_trajectory_fragment() 创建片段元数据
        - TrajectoryFragment.metadata 存储片段元信息
    """
    fragment_id: str
    scenario_id: str
    anchor_frame: int
    ego_vehicle_id: str
    target_vehicle_id: str
    danger_type: str
    danger_level: str
    min_risk_score: float
    frame_count: int
    duration: float
    n_before: int
    n_after: int

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'fragment_id': self.fragment_id,
            'scenario_id': self.scenario_id,
            'anchor_frame': self.anchor_frame,
            'ego_vehicle_id': self.ego_vehicle_id,
            'target_vehicle_id': self.target_vehicle_id,
            'danger_type': self.danger_type,
            'danger_level': self.danger_level,
            'min_risk_score': self.min_risk_score,
            'frame_count': self.frame_count,
            'duration': self.duration,
            'n_before': self.n_before,
            'n_after': self.n_after
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FragmentMetadata':
        """从字典创建"""
        return cls(**data)


@dataclass
class ScenarioMetadata:
    """场景元数据

    Attributes:
        scenario_id: 场景唯一标识
        source_file: 原始数据文件路径
        total_frames: 总帧数
        sampling_rate: 采样率 (Hz)
        ego_vehicle_id: 主车ID
        num_key_vehicles: 关键车数量
        processed_at: 处理时间

    被使用于:
        - ScenarioProcessor.process_scenario() 创建场景元数据
        - ProcessedScenario.metadata 存储场景元信息
    """
    scenario_id: str
    source_file: str
    total_frames: int
    sampling_rate: float
    ego_vehicle_id: str
    num_key_vehicles: int
    processed_at: str

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'scenario_id': self.scenario_id,
            'source_file': self.source_file,
            'total_frames': self.total_frames,
            'sampling_rate': self.sampling_rate,
            'ego_vehicle_id': self.ego_vehicle_id,
            'num_key_vehicles': self.num_key_vehicles,
            'processed_at': self.processed_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScenarioMetadata':
        """从字典创建"""
        return cls(**data)


# -----------------------------------------------------------------------------
# 轨迹片段数据类型
# -----------------------------------------------------------------------------

@dataclass
class TrajectoryFragment:
    """截取的轨迹片段（存储用）

    包含一个危险交互的完整数据：主车轨迹、交互车轨迹、交互特征、11维特征向量。

    Attributes:
        metadata: 片段元数据
        target_trajectory: 交互车轨迹
        ego_trajectory: 主车轨迹
        interaction_features: 交互特征序列（每帧）
        interaction_stats: 交互统计特征
        feature_vector: 11维特征向量
        feature_names: 特征名称列表

    被使用于:
        - create_trajectory_fragment() 创建片段对象
        - ScenarioProcessor.process_scenario() 返回片段列表
        - extract_features_from_fragment() 提取特征字典
    """
    metadata: FragmentMetadata
    target_trajectory: VehicleTrajectory
    ego_trajectory: VehicleTrajectory
    interaction_features: InteractionFeatures
    interaction_stats: InteractionStats
    feature_vector: np.ndarray
    feature_names: List[str] = field(default_factory=lambda: [
        'mean_speed', 'max_speed', 'speed_std',
        'mean_accel', 'max_accel',
        'min_ttc_long', 'mean_ttc_long',
        'min_rel_dist', 'max_closing_speed',
        'trajectory_length', 'max_curvature'
    ])

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于JSON序列化）"""
        return {
            'fragment_id': self.metadata.fragment_id,
            'metadata': self.metadata.to_dict(),
            'target_trajectory': self.target_trajectory.to_dict(),
            'ego_trajectory': self.ego_trajectory.to_dict(),
            'interaction_features': self.interaction_features.to_dict(),
            'interaction_stats': self.interaction_stats.to_dict(),
            'feature_vector': self.feature_vector.tolist(),
            'feature_names': self.feature_names
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrajectoryFragment':
        """从字典创建（用于JSON反序列化）"""
        return cls(
            metadata=FragmentMetadata.from_dict(data['metadata']),
            target_trajectory=VehicleTrajectory.from_dict(data['target_trajectory']),
            ego_trajectory=VehicleTrajectory.from_dict(data['ego_trajectory']),
            interaction_features=InteractionFeatures.from_dict(data['interaction_features']),
            interaction_stats=InteractionStats.from_dict(data['interaction_stats']),
            feature_vector=np.array(data['feature_vector']),
            feature_names=data.get('feature_names', [
                'mean_speed', 'max_speed', 'speed_std',
                'mean_accel', 'max_accel',
                'min_ttc_long', 'mean_ttc_long',
                'min_rel_dist', 'max_closing_speed',
                'trajectory_length', 'max_curvature'
            ])
        )


@dataclass
class ProcessedScenario:
    """处理后的场景数据（JSON文件根对象）

    Attributes:
        metadata: 场景元数据
        fragments: 该场景下所有危险片段列表

    被使用于:
        - ScenarioProcessor.process_scenario() 返回处理结果
        - ScenarioProcessor.save_to_json() 序列化到JSON
        - ScenarioProcessor.load_from_json() 从JSON反序列化
    """
    metadata: ScenarioMetadata
    fragments: List[TrajectoryFragment]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'metadata': self.metadata.to_dict(),
            'fragments': [f.to_dict() for f in self.fragments]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessedScenario':
        """从字典创建"""
        return cls(
            metadata=ScenarioMetadata.from_dict(data['metadata']),
            fragments=[TrajectoryFragment.from_dict(f) for f in data['fragments']]
        )


# -----------------------------------------------------------------------------
# 风险计算相关类型
# -----------------------------------------------------------------------------

@dataclass
class RiskAnalysisResult:
    """风险分析结果

    Attributes:
        anchor_frame: 最危险帧索引
        max_risk_score: 最大风险分数
        risk_scores: 每帧风险分数序列 (N, 7) - [s_dl, s_da, s_tl, s_ta, dsc, tsc, total]
        ttc_long: 纵向TTC序列 (N,)
        ttc_lat: 横向TTC序列 (N,)
        rel_dist: 相对距离序列 (N,)
        rel_direction: 语义方向序列 (N,)
        danger_type: 推断的危险类型
        danger_level: 危险等级

    被使用于:
        - RiskCalculator.analyze() 返回分析结果
        - ScenarioRiskAnalyzer.analyze_pair() 返回分析结果
        - ScenarioRiskAnalyzer.analyze_all() 返回结果字典
        - Main.py run_pipeline() 显示关键车辆信息
        - create_trajectory_fragment() 使用风险结果创建片段
        - ScenarioProcessor.process_scenario() 遍历风险结果
    """
    anchor_frame: int
    max_risk_score: float
    risk_scores: np.ndarray
    ttc_long: np.ndarray
    ttc_lat: np.ndarray
    rel_dist: np.ndarray
    rel_direction: np.ndarray
    danger_type: str
    danger_level: str

    @property
    def min_risk_score(self) -> float:
        """返回最低风险分数"""
        return float(np.min(self.risk_scores[:, 6]))


# -----------------------------------------------------------------------------
# 特征名称常量
# -----------------------------------------------------------------------------

FEATURE_NAMES = [
    'mean_speed',      # 0: 平均速度 (m/s)
    'max_speed',       # 1: 最大速度 (m/s)
    'speed_std',       # 2: 速度标准差 (m/s)
    'mean_accel',      # 3: 平均加速度绝对值 (m/s²)
    'max_accel',       # 4: 最大加速度绝对值 (m/s²)
    'min_ttc_long',    # 5: 最小纵向TTC (s)
    'mean_ttc_long',   # 6: 平均纵向TTC (s)
    'min_rel_dist',    # 7: 最小相对距离 (m)
    'max_closing_speed', # 8: 最大纵向接近速度 (m/s)
    'trajectory_length', # 9: 行驶距离 (m)
    'max_curvature'    # 10: 最大曲率 (1/m)
]
"""11维特征向量特征名称列表

被使用于:
    - extract_11d_features() 返回特征向量时附带特征名
    - create_trajectory_fragment() 创建片段时附带特征名
    - extract_features_from_fragment() 将特征名与特征值配对
"""


# =============================================================================
# 第二部分：Waymo数据加载 (data_loader.py)
# =============================================================================
# 用途：从Waymo pkl格式数据中加载场景、提取车辆轨迹
# 被使用：Main.py 调用 WaymoDataLoader 类
# =============================================================================

class WaymoDataLoader:
    """Waymo数据加载器

    加载Waymo pkl格式数据，提取场景信息和车辆轨迹。
    支持获取移动车辆列表、提取单个车辆轨迹等功能。

    被使用于:
        - Main.py main() 创建 WaymoDataLoader 实例
        - Main.py get_user_selected_ego_id() 获取移动车辆列表
        - Main.py run_pipeline() 加载场景数据和主车轨迹
        - ScenarioProcessor.__init__() 创建数据加载器

    Example:
        loader = WaymoDataLoader("data/waymo-open")
        scenario = loader.load_scenario("10135f16cd538e19")
        vehicles = loader.get_vehicle_list(scenario)
        traj = loader.extract_vehicle_trajectory(scenario, "312")
    """

    def __init__(self, data_dir: str):
        """初始化数据加载器

        Args:
            data_dir: Waymo数据目录路径
        """
        self.data_dir = data_dir
        self.scenarios: Dict[str, Dict] = {}

    def load_scenario(self, scenario_id: str) -> Dict:
        """加载单个场景

        Args:
            scenario_id: 场景ID（不含.pkl后缀）

        Returns:
            场景数据字典

        Raises:
            FileNotFoundError: 当场景文件不存在时

        被使用于:
            - Main.py run_pipeline() 加载场景数据
            - ScenarioProcessor.process_scenario() 加载场景数据
        """
        file_path = os.path.join(self.data_dir, f"{scenario_id}.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"场景文件不存在: {file_path}")

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        self.scenarios[scenario_id] = data
        return data

    def get_scenario(self, scenario_id: str) -> Optional[Dict]:
        """获取已加载的场景

        Args:
            scenario_id: 场景ID

        Returns:
            场景数据字典，若未加载则返回None

        被使用于: (当前未在主程序中直接调用)
        """
        return self.scenarios.get(scenario_id)

    def get_vehicle_list(self, scenario_data: Dict) -> List[str]:
        """获取场景中所有车辆ID

        Args:
            scenario_data: 场景数据字典

        Returns:
            车辆ID列表

        被使用于: (当前未在主程序中直接调用)
        """
        tracks = scenario_data.get('object_tracks', {})
        return [vid for vid, info in tracks.items() if info.get('type') == 'VEHICLE']

    def get_moving_vehicle_list(self, scenario_data: Dict, speed_threshold: float = 0.5) -> List[Dict]:
        """获取移动中的车辆列表，排除静止车辆

        通过计算车辆在有效帧内的平均速度来判断是否移动。
        如果平均速度大于阈值，则认为车辆在移动。

        Args:
            scenario_data: 场景数据字典
            speed_threshold: 速度阈值 (m/s)，默认0.5m/s

        Returns:
            移动车辆信息列表，每个元素包含:
            - vehicle_id: 车辆ID
            - mean_speed: 平均速度
            - max_speed: 最大速度
            - valid_frames: 有效帧数

        被使用于:
            - Main.py get_user_selected_ego_id() 显示并选择主车
        """
        tracks = scenario_data.get('object_tracks', {})
        moving_vehicles = []

        for vid, info in tracks.items():
            if info.get('type') != 'VEHICLE':
                continue

            # 提取速度数据
            velocities = np.array(info['state']['local_velocity'])
            valid_mask = np.array(info['state']['valid'])

            if len(velocities) == 0 or np.sum(valid_mask) == 0:
                continue

            # 计算速度大小
            speeds = np.linalg.norm(velocities, axis=1)

            # 只在有效帧内计算
            valid_speeds = speeds[valid_mask]

            if len(valid_speeds) == 0:
                continue

            mean_speed = np.mean(valid_speeds)
            max_speed = np.max(valid_speeds)

            # 判断是否移动
            if mean_speed > speed_threshold:
                moving_vehicles.append({
                    'vehicle_id': vid,
                    'mean_speed': float(mean_speed),
                    'max_speed': float(max_speed),
                    'valid_frames': int(np.sum(valid_mask))
                })

        # 按平均速度降序排列
        moving_vehicles.sort(key=lambda x: x['mean_speed'], reverse=True)

        return moving_vehicles

    def get_vehicle_info(self, scenario_data: Dict, vehicle_id: str) -> Optional[Dict]:
        """获取指定车辆的详细信息

        Args:
            scenario_data: 场景数据字典
            vehicle_id: 车辆ID

        Returns:
            车辆信息字典，包含type和state

        被使用于:
            - extract_vehicle_trajectory() 获取车辆信息
        """
        tracks = scenario_data.get('object_tracks', {})
        return tracks.get(vehicle_id)

    def get_sdc_id(self, scenario_data: Dict) -> Optional[str]:
        """获取自车(SDC)ID

        Args:
            scenario_data: 场景数据字典

        Returns:
            自车ID

        被使用于:
            - Main.py get_user_selected_ego_id() 获取默认主车ID
            - extract_ego_trajectory() 提取自车轨迹
        """
        return scenario_data.get('extra_information', {}).get('sdc_id')

    def get_scene_length(self, scenario_data: Dict) -> int:
        """获取场景帧数

        Args:
            scenario_data: 场景数据字典

        Returns:
            帧数

        被使用于:
            - ScenarioProcessor.process_scenario() 获取总帧数
        """
        return scenario_data.get('extra_information', {}).get('scene_length', 0)

    def extract_vehicle_trajectory(self, scenario_data: Dict, vehicle_id: str) -> VehicleTrajectory:
        """提取车辆轨迹

        Args:
            scenario_data: 场景数据字典
            vehicle_id: 车辆ID

        Returns:
            VehicleTrajectory对象

        Raises:
            ValueError: 当车辆ID不存在时

        被使用于:
            - Main.py run_pipeline() 提取主车轨迹
            - ScenarioProcessor.process_scenario() 提取主车和交互车轨迹
        """
        vehicle_info = self.get_vehicle_info(scenario_data, vehicle_id)
        if vehicle_info is None:
            raise ValueError(f"车辆ID不存在: {vehicle_id}")

        state = vehicle_info['state']

        return VehicleTrajectory(
            vehicle_id=vehicle_id,
            positions=np.array(state['global_center']),
            headings=np.array(state['heading']),
            velocities=np.array(state['local_velocity']),
            accelerations=np.array(state['local_acceleration']),
            sizes=np.array(state['size']),
            valid=np.array(state['valid'])
        )

    def extract_ego_trajectory(self, scenario_data: Dict) -> VehicleTrajectory:
        """提取自车轨迹

        Args:
            scenario_data: 场景数据字典

        Returns:
            VehicleTrajectory对象

        被使用于: (当前未在主程序中直接调用)
        """
        sdc_id = self.get_sdc_id(scenario_data)
        return self.extract_vehicle_trajectory(scenario_data, sdc_id)

    def get_scenario_summary(self, scenario_data: Dict, scenario_id: str) -> Dict:
        """获取场景摘要信息

        Args:
            scenario_data: 场景数据字典
            scenario_id: 场景ID

        Returns:
            摘要信息字典

        被使用于: (当前未在主程序中直接调用)
        """
        tracks = scenario_data.get('object_tracks', {})
        types = [info.get('type') for info in tracks.values()]
        unique_types = {t: types.count(t) for t in set(types)}

        sample_id = list(tracks.keys())[0]
        total_frames = len(tracks[sample_id]['state']['valid'])

        return {
            'scenario_id': scenario_id,
            'total_frames': total_frames,
            'object_count': len(tracks),
            'type_distribution': unique_types,
            'sdc_id': self.get_sdc_id(scenario_data),
            'vehicle_ids': self.get_vehicle_list(scenario_data)
        }


# =============================================================================
# 第三部分：风险计算 (risk_calculator.py)
# =============================================================================
# 用途：基于Data_Processor.py重构，实现TTC计算、风险分数计算、危险类型判断
# 被使用：Main.py 调用 ScenarioRiskAnalyzer 类，ScenarioProcessor 内部调用 RiskCalculator
# =============================================================================

# -----------------------------------------------------------------------------
# 风险分数计算相关函数
# -----------------------------------------------------------------------------

def score_distance(d: float) -> float:
    """距离评分函数

    基于相对距离计算评分（0-5分）

    Args:
        d: 相对距离（m）

    Returns:
        评分（0-5）

    被使用于:
        - RiskCalculator.compute_risk_scores() 计算风险分数
    """
    d_abs = abs(d)
    if d_abs < 0.3:
        return 0
    if d_abs < 0.8:
        return 1
    if d_abs < 1.3:
        return 2
    if d_abs < 3.0:
        return 3
    if d_abs < 5.0:
        return 4
    return 5


def score_ttc(ttc: float) -> float:
    """TTC评分函数

    基于TTC值计算评分（0-5分）

    Args:
        ttc: TTC值（秒）

    Returns:
        评分（0-5）

    被使用于:
        - RiskCalculator.compute_risk_scores() 计算风险分数
    """
    if ttc < 0.15:
        return 0
    if np.isnan(ttc) or np.isinf(ttc) or ttc < 0:
        return 5
    if ttc <= 0.65:
        return 1
    if ttc <= 1.15:
        return 2
    if ttc <= 3.0:
        return 3
    if ttc <= 5.0:
        return 4
    return 5


# -----------------------------------------------------------------------------
# 坐标系转换函数
# -----------------------------------------------------------------------------

def local_to_global_velocity(local_vel: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """局部速度转全局速度

    Args:
        local_vel: 局部速度 (N, 2) - [vx, vy]
        heading: 航向角 (N,) - rad

    Returns:
        全局速度 (N, 2) - [vx, vy]

    被使用于:
        - RiskCalculator.compute_relative_metrics() 转换速度到全局坐标系
    """
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    gx = local_vel[:, 0] * cos_h - local_vel[:, 1] * sin_h
    gy = local_vel[:, 0] * sin_h + local_vel[:, 1] * cos_h
    return np.stack([gx, gy], axis=-1)


# -----------------------------------------------------------------------------
# 语义方向判断
# -----------------------------------------------------------------------------

def compute_relative_direction(rel_pos: np.ndarray,
                                ego_heading: float,
                                ego_size: np.ndarray) -> Tuple[str, str, str]:
    """计算语义方向（细粒度）

    基于相对位置计算详细的语义方向描述。

    Args:
        rel_pos: 相对位置 [x, y] - m
        ego_heading: 主车航向角 - rad
        ego_size: 主车尺寸 [l, w, h] - m

    Returns:
        (direction, sub_direction, distance_level)
        - direction: front/rear/left/right
        - sub_direction: front_left/front_right/rear_left/rear_right/directly_ahead/behind/lateral/adjacent
        - distance_level: close/medium/far

    被使用于:
        - RiskCalculator.compute_relative_metrics() 计算每帧的语义方向
    """
    dx, dy = rel_pos[0], rel_pos[1]
    dist = np.sqrt(dx**2 + dy**2)

    # 转换到主车局部坐标系
    local_angle = np.arctan2(dy, dx) - ego_heading
    while local_angle > np.pi:
        local_angle -= 2 * np.pi
    while local_angle < -np.pi:
        local_angle += 2 * np.pi

    # 主方向判断
    if abs(local_angle) < np.pi / 8:
        direction = "front"
    elif abs(local_angle) > 7 * np.pi / 8:
        direction = "rear"
    elif local_angle > 0:
        direction = "left"
    else:
        direction = "right"

    # 子方向判断（基于车辆尺寸）
    ego_length, ego_width = ego_size[0], ego_size[1]
    dist_threshold_l = ego_length * 1.5
    dist_threshold_w = ego_width * 1.5

    is_longitudinal = abs(dx) > dist_threshold_l
    is_lateral = abs(dy) > dist_threshold_w

    if is_longitudinal and is_lateral:
        sub = f"{'front' if dx > 0 else 'rear'}_{'left' if dy > 0 else 'right'}"
    elif is_longitudinal:
        sub = "directly_ahead" if dx > 0 else "directly_behind"
    elif is_lateral:
        sub = "lateral"
    else:
        sub = "adjacent"

    # 距离级别
    if dist < 5:
        dist_level = "close"
    elif dist < 15:
        dist_level = "medium"
    else:
        dist_level = "far"

    return direction, sub, dist_level


def get_direction_weight(d_long: float, d_lat: float) -> float:
    """获取方向权重

    用于综合风险计算时判断交互的主要方向。

    Args:
        d_long: 纵向相对距离 - m
        d_lat: 横向相对距离 - m

    Returns:
        权重值（0.0-1.0）

    被使用于:
        - RiskCalculator.compute_risk_scores() 计算综合风险分数
    """
    angle = np.arctan2(d_lat, d_long)
    if (-np.pi / 4 <= angle <= np.pi / 4) or (angle > 3 * np.pi / 4) or (angle <= -3 * np.pi / 4):
        return 1.0
    return 0.0


# -----------------------------------------------------------------------------
# 风险分析类
# -----------------------------------------------------------------------------

class RiskCalculator:
    """风险计算器

    计算两车之间的风险分数、TTC、危险类型等。

    被使用于:
        - ScenarioRiskAnalyzer.analyze_pair() 分析主车与目标车辆的风险

    Example:
        calculator = RiskCalculator(ego_trajectory, target_trajectory)
        result = calculator.analyze()
    """

    def __init__(self,
                 ego_trajectory: VehicleTrajectory,
                 target_trajectory: VehicleTrajectory):
        """初始化风险计算器

        Args:
            ego_trajectory: 主车轨迹
            target_trajectory: 交互车轨迹
        """
        self.ego = ego_trajectory
        self.target = target_trajectory
        self.n = ego_trajectory.frame_count

        # 预分配结果数组
        self.ttc_long = np.full(self.n, np.inf)
        self.ttc_lat = np.full(self.n, np.inf)
        self.rel_dist = np.full(self.n, np.inf)
        self.rel_direction = np.array(["unknown"] * self.n)
        self.risk_scores = np.full((self.n, 7), 5.0)  # [s_dl, s_da, s_tl, s_ta, dsc, tsc, total]

    def compute_relative_metrics(self) -> None:
        """计算相对度量（位置、速度、距离）

        被使用于:
            - analyze() 执行完整分析时调用
        """
        # 转换速度到全局坐标系
        ego_vel_g = local_to_global_velocity(self.ego.velocities, self.ego.headings)
        target_vel_g = local_to_global_velocity(self.target.velocities, self.target.headings)

        # 相对位置（取xy平面）
        dp = self.target.positions[:, :2] - self.ego.positions[:, :2]

        # 相对速度
        dv = target_vel_g - ego_vel_g

        # 转换到主车局部坐标系
        cos_h = np.cos(self.ego.headings)
        sin_h = np.sin(self.ego.headings)
        d_l = dp[:, 0] * cos_h + dp[:, 1] * sin_h
        d_a = -dp[:, 0] * sin_h + dp[:, 1] * cos_h
        v_l = dv[:, 0] * cos_h + dv[:, 1] * sin_h
        v_a = -dv[:, 0] * sin_h + dv[:, 1] * cos_h

        # TTC计算（与Data_Processor.py一致，使用np.where）
        with np.errstate(divide='ignore', invalid='ignore'):
            self.ttc_long = np.where((d_l * v_l) < 0, -d_l / v_l, np.inf)
            self.ttc_lat = np.where((d_a * v_a) < 0, -d_a / v_a, np.inf)

        # 逐帧计算其他度量
        for t in range(self.n):
            if not (self.ego.valid[t] and self.target.valid[t]):
                continue

            ego_l, ego_w = self.ego.sizes[t, 0], self.ego.sizes[t, 1]

            # 调整后距离
            adj_dl = d_l[t] if abs(d_l[t]) > ego_l else 0
            adj_da = d_a[t] if abs(d_a[t]) > ego_w else 0
            self.rel_dist[t] = np.sqrt(adj_dl**2 + adj_da**2)

            # 语义方向
            direction, sub, _ = compute_relative_direction(
                dp[t], self.ego.headings[t], self.ego.sizes[t]
            )
            self.rel_direction[t] = direction

    def compute_risk_scores(self) -> None:
        """计算风险分数

        被使用于:
            - analyze() 执行完整分析时调用
        """
        for t in range(self.n):
            if not (self.ego.valid[t] and self.target.valid[t]):
                continue

            # 获取当前帧的相对量
            cos_h = np.cos(self.ego.headings[t])
            sin_h = np.sin(self.ego.headings[t])
            dp = self.target.positions[t, :2] - self.ego.positions[t, :2]
            d_l = dp[0] * cos_h + dp[1] * sin_h
            d_a = -dp[0] * sin_h + dp[1] * cos_h

            # 评分
            s_dl = score_distance(d_l)
            s_da = score_distance(d_a)
            s_tl = score_ttc(abs(self.ttc_long[t]))
            s_ta = score_ttc(abs(self.ttc_lat[t]))

            # 方向权重
            w = get_direction_weight(d_l, d_a)

            # 综合评分
            dsc = s_dl * w + s_da * (1 - w)
            tsc = s_tl * w + s_ta * (1 - w)

            self.risk_scores[t] = [s_dl, s_da, s_tl, s_ta, dsc, tsc, int((dsc + tsc) / 2)]

    def infer_danger_type(self) -> str:
        """推断危险类型

        基于交互特征判断危险类型。

        Returns:
            危险类型字符串

        被使用于:
            - analyze() 执行完整分析时调用
        """
        # 找到有效TTC的帧
        valid_ttc_long = self.ttc_long[np.isfinite(self.ttc_long)]
        valid_ttc_lat = self.ttc_lat[np.isfinite(self.ttc_lat)]

        if len(valid_ttc_long) == 0 and len(valid_ttc_lat) == 0:
            return "unknown"

        min_ttc_long = np.min(valid_ttc_long) if len(valid_ttc_long) > 0 else np.inf
        min_ttc_lat = np.min(valid_ttc_lat) if len(valid_ttc_lat) > 0 else np.inf

        # 计算平均航向差
        heading_diff = self.target.headings - self.ego.headings
        valid_heading_diff = heading_diff[self.ego.valid & self.target.valid]
        mean_heading_diff = np.mean(valid_heading_diff) if len(valid_heading_diff) > 0 else 0

        # 判断逻辑
        # 1. 纵向接近为主
        if min_ttc_long < min_ttc_lat * 0.5 and abs(mean_heading_diff) < np.pi / 6:
            if mean_heading_diff > 0:
                return "rear_end"  # 追尾
            else:
                return "head_on"  # 对向

        # 2. 横向接近为主
        if min_ttc_lat < min_ttc_long * 0.5:
            if abs(mean_heading_diff) > np.pi / 4:
                return "cut_in"  # 变道切入
            else:
                return "crossing"  # 交叉穿行

        # 3. 混合型
        return "mixed"

    def infer_danger_level(self, min_risk_score: float) -> str:
        """推断危险等级

        基于最低风险分数划分危险等级。

        Args:
            min_risk_score: 最低风险分数

        Returns:
            危险等级字符串

        被使用于:
            - analyze() 执行完整分析时调用
        """
        if min_risk_score <= 1:
            return "high"
        elif min_risk_score <= 3:
            return "medium"
        else:
            return "low"

    def analyze(self) -> RiskAnalysisResult:
        """执行完整风险分析

        Returns:
            RiskAnalysisResult对象

        被使用于:
            - ScenarioRiskAnalyzer.analyze_pair() 返回分析结果
        """
        self.compute_relative_metrics()
        self.compute_risk_scores()

        # 找到最危险帧
        min_risk_idx = np.argmin(self.risk_scores[:, 6])
        max_risk_score = float(np.min(self.risk_scores[:, 6]))

        danger_type = self.infer_danger_type()
        danger_level = self.infer_danger_level(max_risk_score)

        return RiskAnalysisResult(
            anchor_frame=int(min_risk_idx),
            max_risk_score=max_risk_score,
            risk_scores=self.risk_scores,
            ttc_long=self.ttc_long,
            ttc_lat=self.ttc_lat,
            rel_dist=self.rel_dist,
            rel_direction=self.rel_direction,
            danger_type=danger_type,
            danger_level=danger_level
        )


# -----------------------------------------------------------------------------
# 批量风险分析
# -----------------------------------------------------------------------------

class ScenarioRiskAnalyzer:
    """场景风险分析器

    分析场景中所有车辆对的风险，选择最危险的关键车。

    被使用于:
        - Main.py run_pipeline() 分析所有车辆风险
        - ScenarioProcessor.process_scenario() 分析关键车辆

    Example:
        analyzer = ScenarioRiskAnalyzer(scenario_data, ego_id)
        results = analyzer.analyze_all(risk_threshold=3)
    """

    def __init__(self, scenario_data: Dict, ego_id: str):
        """初始化场景风险分析器

        Args:
            scenario_data: Waymo场景数据字典
            ego_id: 主车ID
        """
        self.data = scenario_data
        self.ego_id = ego_id
        self.scene_len = scenario_data['extra_information']['scene_length']
        self.tracks = scenario_data['object_tracks']

    def get_vehicle_ids(self) -> List[str]:
        """获取所有车辆ID（除主车外）

        被使用于:
            - analyze_all() 遍历所有车辆
        """
        return [
            vid for vid, info in self.tracks.items()
            if info['type'] == 'VEHICLE' and vid != self.ego_id
        ]

    def analyze_pair(self, target_id: str) -> Optional[RiskAnalysisResult]:
        """分析主车与目标车辆的风险

        Args:
            target_id: 目标车辆ID

        Returns:
            RiskAnalysisResult或None

        被使用于:
            - analyze_all() 遍历每个车辆
        """
        try:
            # 提取轨迹
            ego_state = self.tracks[self.ego_id]['state']
            target_state = self.tracks[target_id]['state']

            ego_trajectory = VehicleTrajectory(
                vehicle_id=self.ego_id,
                positions=np.array(ego_state['global_center']),
                headings=np.array(ego_state['heading']),
                velocities=np.array(ego_state['local_velocity']),
                accelerations=np.array(ego_state['local_acceleration']),
                sizes=np.array(ego_state['size']),
                valid=np.array(ego_state['valid'])
            )

            target_trajectory = VehicleTrajectory(
                vehicle_id=target_id,
                positions=np.array(target_state['global_center']),
                headings=np.array(target_state['heading']),
                velocities=np.array(target_state['local_velocity']),
                accelerations=np.array(target_state['local_acceleration']),
                sizes=np.array(target_state['size']),
                valid=np.array(target_state['valid'])
            )

            calculator = RiskCalculator(ego_trajectory, target_trajectory)
            return calculator.analyze()

        except Exception as e:
            print(f"分析车辆对 {self.ego_id}-{target_id} 失败: {e}")
            return None

    def analyze_all(self, risk_threshold: int = 3) -> Dict[str, RiskAnalysisResult]:
        """分析所有车辆对的风险

        Args:
            risk_threshold: 风险分数阈值，只返回低于此阈值的结果

        Returns:
            目标ID到RiskAnalysisResult的字典

        被使用于:
            - Main.py run_pipeline() 获取关键车辆列表
            - ScenarioProcessor.process_scenario() 获取关键车辆列表
        """
        vehicle_ids = self.get_vehicle_ids()
        results = {}

        for vid in vehicle_ids:
            result = self.analyze_pair(vid)
            if result is not None and result.max_risk_score <= risk_threshold:
                results[vid] = result

        return results


# -----------------------------------------------------------------------------
# 便捷函数
# -----------------------------------------------------------------------------

def compute_interaction_features(ego_trajectory: VehicleTrajectory,
                                  target_trajectory: VehicleTrajectory) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """便捷函数：计算交互特征

    Args:
        ego_trajectory: 主车轨迹
        target_trajectory: 交互车轨迹

    Returns:
        (ttc_long, ttc_lat, rel_dist) - TTC序列和相对距离序列

    被使用于: (当前未在主程序中直接调用)
    """
    calculator = RiskCalculator(ego_trajectory, target_trajectory)
    calculator.compute_relative_metrics()
    return calculator.ttc_long, calculator.ttc_lat, calculator.rel_dist


# =============================================================================
# 第四部分：片段截取与特征提取 (fragment_extractor.py)
# =============================================================================
# 用途：实现危险轨迹片段截取、11维特征提取、JSON持久化
# 被使用：Main.py 调用 ScenarioProcessor 类和顶层函数
# =============================================================================

# -----------------------------------------------------------------------------
# 轨迹片段截取
# -----------------------------------------------------------------------------

def slice_trajectory(trajectory: VehicleTrajectory,
                     start_frame: int, end_frame: int) -> VehicleTrajectory:
    """切片轨迹

    Args:
        trajectory: 原始轨迹
        start_frame: 起始帧（含）
        end_frame: 结束帧（不含）

    Returns:
        切片后的轨迹

    被使用于:
        - create_trajectory_fragment() 截取主车和交互车轨迹
    """
    return VehicleTrajectory(
        vehicle_id=trajectory.vehicle_id,
        positions=trajectory.positions[start_frame:end_frame],
        headings=trajectory.headings[start_frame:end_frame],
        velocities=trajectory.velocities[start_frame:end_frame],
        accelerations=trajectory.accelerations[start_frame:end_frame],
        sizes=trajectory.sizes[start_frame:end_frame],
        valid=trajectory.valid[start_frame:end_frame]
    )


def extract_fragment_slice(anchor_frame: int,
                          n_before_frames: int,
                          n_after_frames: int,
                          total_frames: int) -> Tuple[int, int, int, int]:
    """计算片段截取的边界（带边界处理）

    Args:
        anchor_frame: 锚点帧索引
        n_before_frames: 需要的锚点前帧数
        n_after_frames: 需要的锚点后帧数
        total_frames: 场景总帧数

    Returns:
        (start_frame, end_frame, actual_before, actual_after)

    被使用于:
        - create_trajectory_fragment() 计算截取边界
    """
    # 计算理想边界
    start_frame = anchor_frame - n_before_frames
    end_frame = anchor_frame + n_after_frames + 1

    # 边界处理
    if start_frame < 0:
        end_frame -= start_frame  # 补偿左侧
        start_frame = 0

    if end_frame > total_frames:
        start_frame -= (end_frame - total_frames)  # 补偿右侧
        start_frame = max(0, start_frame)
        end_frame = total_frames

    actual_before = anchor_frame - start_frame
    actual_after = end_frame - anchor_frame - 1

    return start_frame, end_frame, actual_before, actual_after


def should_skip_fragment(anchor_frame: int,
                         n_before_frames: int,
                         n_after_frames: int,
                         total_frames: int,
                         min_before_frames: int = 10,
                         min_after_frames: int = 15) -> Tuple[bool, str]:
    """判断是否应跳过片段计算

    当锚点帧太靠近场景边界，导致截取的片段过短时，跳过该交互车。

    Args:
        anchor_frame: 锚点帧索引
        n_before_frames: 需要的锚点前帧数
        n_after_frames: 需要的锚点后帧数
        total_frames: 场景总帧数
        min_before_frames: 锚点前最小帧数阈值
        min_after_frames: 锚点后最小帧数阈值

    Returns:
        (should_skip, reason) - 是否跳过及原因

    被使用于:
        - ScenarioProcessor.process_scenario() 检查锚点帧边界
    """
    # 检查锚点帧到开始边界
    if anchor_frame < min_before_frames:
        reason = (f"跳过关键车: 锚点帧{anchor_frame}距开始仅{anchor_frame}帧 "
                  f"< 最小要求{min_before_frames}帧，截取片段过短")
        return True, reason

    # 检查锚点帧到结束边界
    frames_after_anchor = total_frames - anchor_frame - 1
    if frames_after_anchor < min_after_frames:
        reason = (f"跳过关键车: 锚点帧{anchor_frame}距结束仅{frames_after_anchor}帧 "
                  f"< 最小要求{min_after_frames}帧，截取片段过短")
        return True, reason

    return False, ""


# -----------------------------------------------------------------------------
# 交互特征计算
# -----------------------------------------------------------------------------

def compute_interaction_features_series(
    ego_trajectory: VehicleTrajectory,
    target_trajectory: VehicleTrajectory
) -> InteractionFeatures:
    """计算交互特征序列

    计算两车之间每帧的相对运动特征。

    Args:
        ego_trajectory: 主车轨迹
        target_trajectory: 交互车轨迹

    Returns:
        InteractionFeatures对象

    被使用于:
        - create_trajectory_fragment() 计算截取片段的交互特征
    """
    n = ego_trajectory.frame_count

    # 预分配数组
    rel_pos_x = np.zeros(n)
    rel_pos_y = np.zeros(n)
    rel_dist = np.zeros(n)
    rel_vel_x = np.zeros(n)
    rel_vel_y = np.zeros(n)
    rel_speed = np.zeros(n)
    ttc_long = np.full(n, np.inf)
    ttc_lat = np.full(n, np.inf)
    local_angle = np.zeros(n)
    lateral_offset = np.zeros(n)
    longitudinal_offset = np.zeros(n)
    heading_diff = np.zeros(n)

    for t in range(n):
        if not (ego_trajectory.valid[t] and target_trajectory.valid[t]):
            continue

        # 相对位置
        rel_pos = target_trajectory.positions[t, :2] - ego_trajectory.positions[t, :2]
        rel_pos_x[t] = rel_pos[0]
        rel_pos_y[t] = rel_pos[1]
        rel_dist[t] = np.linalg.norm(rel_pos)

        # 转换速度到全局坐标系
        ego_heading = ego_trajectory.headings[t]
        cos_h, sin_h = np.cos(ego_heading), np.sin(ego_heading)

        # 全局速度
        ego_vx = ego_trajectory.velocities[t, 0] * cos_h - ego_trajectory.velocities[t, 1] * sin_h
        ego_vy = ego_trajectory.velocities[t, 0] * sin_h + ego_trajectory.velocities[t, 1] * cos_h
        target_vx = target_trajectory.velocities[t, 0] * cos_h - target_trajectory.velocities[t, 1] * sin_h
        target_vy = target_trajectory.velocities[t, 0] * sin_h + target_trajectory.velocities[t, 1] * cos_h

        # 相对速度
        rel_vel_x[t] = target_vx - ego_vx
        rel_vel_y[t] = target_vy - ego_vy
        rel_speed[t] = np.linalg.norm([rel_vel_x[t], rel_vel_y[t]])

        # 局部坐标系投影
        longitudinal_offset[t] = rel_pos[0] * cos_h + rel_pos[1] * sin_h
        lateral_offset[t] = -rel_pos[0] * sin_h + rel_pos[1] * cos_h

        # 局部角度
        local_angle[t] = np.arctan2(rel_pos[1], rel_pos[0]) - ego_heading
        while local_angle[t] > np.pi:
            local_angle[t] -= 2 * np.pi
        while local_angle[t] < -np.pi:
            local_angle[t] += 2 * np.pi

        # 航向差
        heading_diff[t] = target_trajectory.headings[t] - ego_heading

        # TTC计算
        ego_l, ego_w = ego_trajectory.sizes[t, 0], ego_trajectory.sizes[t, 1]

        # 纵向TTC
        if longitudinal_offset[t] > ego_l:
            rel_vel_long = -rel_vel_x[t] * cos_h - rel_vel_y[t] * sin_h
            if rel_vel_long < 0:
                adj_d = longitudinal_offset[t] - ego_l
                ttc_long[t] = adj_d / abs(rel_vel_long)
        elif longitudinal_offset[t] < -ego_l:
            rel_vel_long = -rel_vel_x[t] * cos_h - rel_vel_y[t] * sin_h
            if rel_vel_long > 0:
                adj_d = abs(longitudinal_offset[t] + ego_l)
                ttc_long[t] = adj_d / rel_vel_long

        # 横向TTC
        if lateral_offset[t] > ego_w:
            rel_vel_lat = -rel_vel_x[t] * sin_h + rel_vel_y[t] * cos_h
            if rel_vel_lat < 0:
                adj_d = lateral_offset[t] - ego_w
                ttc_lat[t] = abs(adj_d / rel_vel_lat)
        elif lateral_offset[t] < -ego_w:
            rel_vel_lat = -rel_vel_x[t] * sin_h + rel_vel_y[t] * cos_h
            if rel_vel_lat > 0:
                adj_d = abs(lateral_offset[t] + ego_w)
                ttc_lat[t] = abs(adj_d / rel_vel_lat)

    return InteractionFeatures(
        rel_pos_x=rel_pos_x,
        rel_pos_y=rel_pos_y,
        rel_dist=rel_dist,
        rel_vel_x=rel_vel_x,
        rel_vel_y=rel_vel_y,
        rel_speed=rel_speed,
        ttc_long=ttc_long,
        ttc_lat=ttc_lat,
        local_angle=local_angle,
        lateral_offset=lateral_offset,
        longitudinal_offset=longitudinal_offset,
        heading_diff=heading_diff
    )


def compute_interaction_stats(features: InteractionFeatures) -> InteractionStats:
    """计算交互统计特征

    从交互特征序列计算统计量。

    Args:
        features: 交互特征序列

    Returns:
        InteractionStats对象

    被使用于:
        - create_trajectory_fragment() 计算截取片段的统计特征
    """
    # 有效TTC
    valid_ttc_long = features.ttc_long[np.isfinite(features.ttc_long)]
    valid_ttc_lat = features.ttc_lat[np.isfinite(features.ttc_lat)]
    valid_dist = features.rel_dist[features.rel_dist > 0]
    valid_closing = -features.rel_vel_x[np.where(features.rel_vel_x < 0)]

    # TTC统计
    min_ttc_long = float(np.min(valid_ttc_long)) if len(valid_ttc_long) > 0 else np.inf
    min_ttc_lat = float(np.min(valid_ttc_lat)) if len(valid_ttc_lat) > 0 else np.inf

    # 距离统计
    min_rel_dist = float(np.min(valid_dist)) if len(valid_dist) > 0 else 0
    mean_rel_dist = float(np.mean(valid_dist)) if len(valid_dist) > 0 else 0

    # 接近速度统计
    max_closing_speed = float(np.max(valid_closing)) if len(valid_closing) > 0 else 0
    mean_closing_speed = float(np.mean(valid_closing)) if len(valid_closing) > 0 else 0

    # 接近时间占比
    is_approaching = features.rel_vel_x < 0
    approach_time_ratio = float(np.sum(is_approaching) / len(is_approaching))

    # 平均航向差
    mean_heading_diff = float(np.mean(np.abs(features.heading_diff)))

    return InteractionStats(
        min_ttc_long=min_ttc_long,
        min_ttc_lat=min_ttc_lat,
        min_rel_dist=min_rel_dist,
        mean_rel_dist=mean_rel_dist,
        max_closing_speed=max_closing_speed,
        mean_closing_speed=mean_closing_speed,
        approach_time_ratio=approach_time_ratio,
        mean_heading_diff=mean_heading_diff
    )


# -----------------------------------------------------------------------------
# 11维特征提取
# -----------------------------------------------------------------------------

def compute_target_speed_features(target_trajectory: VehicleTrajectory) -> Tuple[float, float, float]:
    """计算交互车速度特征

    Args:
        target_trajectory: 交互车轨迹

    Returns:
        (mean_speed, max_speed, speed_std)

    被使用于:
        - extract_11d_features() 提取11维特征向量
    """
    # 计算速度大小
    speeds = np.linalg.norm(target_trajectory.velocities, axis=1)
    valid_speeds = speeds[target_trajectory.valid]

    mean_speed = float(np.mean(valid_speeds)) if len(valid_speeds) > 0 else 0
    max_speed = float(np.max(valid_speeds)) if len(valid_speeds) > 0 else 0
    speed_std = float(np.std(valid_speeds)) if len(valid_speeds) > 0 else 0

    return mean_speed, max_speed, speed_std


def compute_target_accel_features(target_trajectory: VehicleTrajectory) -> Tuple[float, float]:
    """计算交互车加速度特征

    Args:
        target_trajectory: 交互车轨迹

    Returns:
        (mean_accel, max_accel)

    被使用于:
        - extract_11d_features() 提取11维特征向量
    """
    # 计算加速度大小
    accels = np.linalg.norm(target_trajectory.accelerations, axis=1)
    valid_accels = accels[target_trajectory.valid]

    mean_accel = float(np.mean(valid_accels)) if len(valid_accels) > 0 else 0
    max_accel = float(np.max(valid_accels)) if len(valid_accels) > 0 else 0

    return mean_accel, max_accel


def compute_trajectory_length(trajectory: VehicleTrajectory) -> float:
    """计算轨迹长度

    Args:
        trajectory: 轨迹

    Returns:
        轨迹长度（米）

    被使用于:
        - extract_11d_features() 提取11维特征向量
    """
    positions = trajectory.positions[:, :2]
    diffs = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return float(np.sum(segment_lengths))


def compute_max_curvature(trajectory: VehicleTrajectory) -> float:
    """计算最大曲率

    基于航向角变化率计算曲率。

    Args:
        trajectory: 轨迹

    Returns:
        最大曲率 (1/m)

    被使用于:
        - extract_11d_features() 提取11维特征向量
    """
    valid = trajectory.valid
    headings = trajectory.headings
    positions = trajectory.positions[:, :2]

    # 计算航向角变化
    dtheta = np.diff(headings[valid])
    # 归一化角度差
    dtheta = np.mod(dtheta + np.pi, 2 * np.pi) - np.pi

    # 计算弧长
    ds = np.linalg.norm(np.diff(positions[valid], axis=0), axis=1)
    ds = np.where(ds > 0.001, ds, 0.001)  # 避免除零

    # 曲率 = dtheta / ds
    curvatures = np.abs(dtheta) / ds

    return float(np.max(curvatures)) if len(curvatures) > 0 else 0


def extract_11d_features(target_trajectory: VehicleTrajectory,
                         interaction_stats: InteractionStats) -> np.ndarray:
    """提取11维特征向量

    特征顺序：
    0-4: 交互车轨迹特征
    5-10: 交互特征

    Args:
        target_trajectory: 交互车轨迹
        interaction_stats: 交互统计特征

    Returns:
        11维特征向量

    被使用于:
        - create_trajectory_fragment() 创建片段时提取特征
    """
    # 速度特征
    mean_speed, max_speed, speed_std = compute_target_speed_features(target_trajectory)

    # 加速度特征
    mean_accel, max_accel = compute_target_accel_features(target_trajectory)

    # 轨迹长度
    traj_length = compute_trajectory_length(target_trajectory)

    # 最大曲率
    max_curvature = compute_max_curvature(target_trajectory)

    # 组装特征向量
    feature_vector = np.array([
        mean_speed,      # 0: 平均速度 (m/s)
        max_speed,       # 1: 最大速度 (m/s)
        speed_std,       # 2: 速度标准差 (m/s)
        mean_accel,      # 3: 平均加速度 (m/s²)
        max_accel,       # 4: 最大加速度 (m/s²)
        interaction_stats.min_ttc_long,      # 5: 最小纵向TTC (s)
        np.mean(interaction_stats.min_ttc_long),  # 6: 平均TTC (s) - 复用min值作为近似
        interaction_stats.min_rel_dist,      # 7: 最小相对距离 (m)
        interaction_stats.max_closing_speed,  # 8: 最大纵向接近速度 (m/s)
        traj_length,     # 9: 行驶距离 (m)
        max_curvature     # 10: 最大曲率 (1/m)
    ])

    # 修正mean_ttc_long（之前用的是min值）
    feature_vector[6] = interaction_stats.mean_rel_dist / (interaction_stats.max_closing_speed + 0.001)

    return feature_vector


# -----------------------------------------------------------------------------
# 片段创建
# -----------------------------------------------------------------------------

def create_trajectory_fragment(
    scenario_id: str,
    ego_trajectory: VehicleTrajectory,
    target_trajectory: VehicleTrajectory,
    anchor_frame: int,
    risk_result: RiskAnalysisResult,
    n_before_sec: float,
    n_after_sec: float,
    sampling_rate: float = 10.0
) -> TrajectoryFragment:
    """创建轨迹片段

    Args:
        scenario_id: 场景ID
        ego_trajectory: 主车轨迹
        target_trajectory: 交互车轨迹
        anchor_frame: 最危险帧索引
        risk_result: 风险分析结果
        n_before_sec: 锚点前时长（秒）
        n_after_sec: 锚点后时长（秒）
        sampling_rate: 采样率

    Returns:
        TrajectoryFragment对象

    被使用于:
        - ScenarioProcessor.process_scenario() 创建每个关键车的片段
    """
    # 计算截取边界
    n_before_frames = int(n_before_sec * sampling_rate)
    n_after_frames = int(n_after_sec * sampling_rate)

    start_frame, end_frame, actual_before, actual_after = extract_fragment_slice(
        anchor_frame, n_before_frames, n_after_frames,
        ego_trajectory.frame_count
    )

    # 截取轨迹
    ego_fragment = slice_trajectory(ego_trajectory, start_frame, end_frame)
    target_fragment = slice_trajectory(target_trajectory, start_frame, end_frame)

    # 计算交互特征
    interaction_features = compute_interaction_features_series(ego_fragment, target_fragment)
    interaction_stats = compute_interaction_stats(interaction_features)

    # 提取11维特征
    feature_vector = extract_11d_features(target_fragment, interaction_stats)

    # 生成片段ID
    fragment_id = f"frag_{scenario_id}_{ego_trajectory.vehicle_id}_{target_trajectory.vehicle_id}_{anchor_frame}"

    # 创建元数据
    metadata = FragmentMetadata(
        fragment_id=fragment_id,
        scenario_id=scenario_id,
        anchor_frame=anchor_frame,
        ego_vehicle_id=ego_trajectory.vehicle_id,
        target_vehicle_id=target_trajectory.vehicle_id,
        danger_type=risk_result.danger_type,
        danger_level=risk_result.danger_level,
        min_risk_score=risk_result.min_risk_score,
        frame_count=end_frame - start_frame,
        duration=float(end_frame - start_frame) / sampling_rate,
        n_before=actual_before,
        n_after=actual_after
    )

    return TrajectoryFragment(
        metadata=metadata,
        target_trajectory=target_fragment,
        ego_trajectory=ego_fragment,
        interaction_features=interaction_features,
        interaction_stats=interaction_stats,
        feature_vector=feature_vector,
        feature_names=FEATURE_NAMES
    )


# -----------------------------------------------------------------------------
# 场景处理
# -----------------------------------------------------------------------------

class ScenarioProcessor:
    """场景处理器

    处理单个Waymo场景，提取所有危险轨迹片段。

    被使用于:
        - Main.py process_and_save() 创建处理器并处理场景

    Example:
        processor = ScenarioProcessor("data/waymo-open", output_dir="data/processed")
        processor.process_scenario("10135f16cd538e19", n_before_sec=2.0, n_after_sec=3.0)
    """

    def __init__(self, data_dir: str, output_dir: str):
        """初始化场景处理器

        Args:
            data_dir: Waymo数据目录
            output_dir: 输出目录
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.loader = WaymoDataLoader(data_dir)

    def process_scenario(self,
                         scenario_id: str,
                         ego_id: str,
                         n_before_sec: float = 2.0,
                         n_after_sec: float = 3.0,
                         risk_threshold: int = 3,
                         min_before_frames: int = 10,
                         min_after_frames: int = 15) -> ProcessedScenario:
        """处理单个场景

        Args:
            scenario_id: 场景ID
            ego_id: 主车ID（用户指定）
            n_before_sec: 锚点前截取时长（秒）
            n_after_sec: 锚点后截取时长（秒）
            risk_threshold: 风险分数阈值

        Returns:
            ProcessedScenario对象

        被使用于:
            - Main.py process_and_save() 处理并保存场景
        """
        print(f"\n处理场景: {scenario_id}")

        # 加载数据
        scenario_data = self.loader.load_scenario(scenario_id)
        total_frames = self.loader.get_scene_length(scenario_data)

        print(f"  主车ID: {ego_id}, 总帧数: {total_frames}")

        # 提取主车轨迹
        ego_trajectory = self.loader.extract_vehicle_trajectory(scenario_data, ego_id)

        # 风险分析
        analyzer = ScenarioRiskAnalyzer(scenario_data, ego_id)
        risk_results = analyzer.analyze_all(risk_threshold=risk_threshold)

        print(f"  发现 {len(risk_results)} 个关键车辆（风险分数<={risk_threshold}）")

        # 处理每个关键车
        fragments = []
        skipped_due_to_boundary = 0
        for target_id, risk_result in risk_results.items():
            anchor_frame = risk_result.anchor_frame

            # 检查锚点帧是否太靠近边界
            skip, reason = should_skip_fragment(
                anchor_frame=anchor_frame,
                n_before_frames=int(n_before_sec * 10),
                n_after_frames=int(n_after_sec * 10),
                total_frames=total_frames,
                min_before_frames=min_before_frames,
                min_after_frames=min_after_frames
            )
            if skip:
                print(f"  {reason}")
                skipped_due_to_boundary += 1
                continue

            print(f"  处理关键车: {target_id}, 风险类型: {risk_result.danger_type}")

            # 提取交互车轨迹
            target_trajectory = self.loader.extract_vehicle_trajectory(scenario_data, target_id)

            # 创建片段
            fragment = create_trajectory_fragment(
                scenario_id=scenario_id,
                ego_trajectory=ego_trajectory,
                target_trajectory=target_trajectory,
                anchor_frame=anchor_frame,
                risk_result=risk_result,
                n_before_sec=n_before_sec,
                n_after_sec=n_after_sec
            )

            fragments.append(fragment)

        # 分析未提取到片段的原因
        if len(fragments) == 0:
            print(f"\n  === 未提取到片段的原因分析 ===")
            if len(risk_results) == 0:
                print(f"  原因1: 场景中没有找到风险分数<={risk_threshold}的关键车")
            elif skipped_due_to_boundary == len(risk_results):
                print(f"  原因2: 所有{len(risk_results)}个关键车的锚点帧都太靠近场景边界，被跳过")
            else:
                print(f"  原因3: {len(risk_results)}个关键车中有{skipped_due_to_boundary}个因锚点帧太靠近边界被跳过")

        # 创建场景元数据
        metadata = ScenarioMetadata(
            scenario_id=scenario_id,
            source_file=f"{scenario_id}.pkl",
            total_frames=total_frames,
            sampling_rate=10.0,
            ego_vehicle_id=ego_id,
            num_key_vehicles=len(fragments),
            processed_at=datetime.now().isoformat()
        )

        processed_scenario = ProcessedScenario(metadata=metadata, fragments=fragments)

        # 只有在有片段时才保存JSON
        if len(fragments) > 0:
            self.save_to_json(processed_scenario)
        else:
            print(f"  警告: 未提取到任何片段，跳过JSON保存")

        print(f"  完成！提取了 {len(fragments)} 个片段")

        return processed_scenario

    def save_to_json(self, scenario: ProcessedScenario) -> str:
        """保存场景到JSON文件

        Args:
            scenario: ProcessedScenario对象

        Returns:
            保存的文件路径

        被使用于:
            - process_scenario() 保存处理结果
        """
        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(self.output_dir, f"{scenario.metadata.scenario_id}_{scenario.metadata.ego_vehicle_id}.json")

        data = scenario.to_dict()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  已保存到: {file_path}")
        return file_path

    def load_from_json(self, scenario_id: str, ego_id: str) -> ProcessedScenario:
        """从JSON文件加载场景

        Args:
            scenario_id: 场景ID
            ego_id: 主车ID

        Returns:
            ProcessedScenario对象

        被使用于: (当前未在主程序中直接调用)
        """
        file_path = os.path.join(self.output_dir, f"{scenario_id}_{ego_id}.json")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return ProcessedScenario.from_dict(data)


# -----------------------------------------------------------------------------
# 批量处理
# -----------------------------------------------------------------------------

def process_scenario_batch(data_dir: str,
                           output_dir: str,
                           scenario_ids: List[str],
                           **kwargs) -> List[ProcessedScenario]:
    """批量处理场景

    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        scenario_ids: 场景ID列表
        **kwargs: 传递给ScenarioProcessor.process_scenario的参数

    Returns:
        处理后的场景列表

    被使用于: (当前未在主程序中直接调用，用于批量处理)
    """
    processor = ScenarioProcessor(data_dir, output_dir)
    results = []

    for s_id in scenario_ids:
        try:
            scenario = processor.process_scenario(s_id, **kwargs)
            results.append(scenario)
        except Exception as e:
            print(f"处理场景 {s_id} 失败: {e}")

    return results


# -----------------------------------------------------------------------------
# 便捷函数
# -----------------------------------------------------------------------------

def extract_features_from_fragment(fragment: TrajectoryFragment) -> Dict[str, float]:
    """便捷函数：从片段提取特征字典

    Args:
        fragment: 轨迹片段

    Returns:
        特征名字典

    被使用于: (当前未在主程序中直接调用)
    """
    return dict(zip(fragment.feature_names, fragment.feature_vector.tolist()))


# =============================================================================
# 模块导出
# =============================================================================

__all__ = [
    # 类型
    'VehicleTrajectory',
    'InteractionFeatures',
    'InteractionStats',
    'FragmentMetadata',
    'ScenarioMetadata',
    'TrajectoryFragment',
    'ProcessedScenario',
    'RiskAnalysisResult',
    'FEATURE_NAMES',

    # 数据加载
    'WaymoDataLoader',

    # 风险计算
    'RiskCalculator',
    'ScenarioRiskAnalyzer',
    'compute_interaction_features',
    'score_distance',
    'score_ttc',
    'local_to_global_velocity',
    'compute_relative_direction',

    # 片段处理
    'ScenarioProcessor',
    'process_scenario_batch',
    'create_trajectory_fragment',
    'extract_11d_features',
    'extract_features_from_fragment',

    # 工具函数
    'slice_trajectory',
    'extract_fragment_slice',
    'should_skip_fragment',
    'compute_interaction_features_series',
    'compute_interaction_stats',
    'compute_target_speed_features',
    'compute_target_accel_features',
    'compute_trajectory_length',
    'compute_max_curvature',
]
