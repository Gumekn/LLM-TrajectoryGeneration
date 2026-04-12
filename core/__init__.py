"""
core/ - 核心处理模块

本模块整合了危险轨迹生成与评估系统的所有核心功能。
建议直接导入 processor 模块以获得完整功能。

导入方式（推荐）:
    from core.processor import (
        WaymoDataLoader,
        ScenarioRiskAnalyzer,
        ScenarioProcessor,
        VehicleTrajectory,
        ...
    )

或者从 core 包导入（保持向后兼容）:
    from core import (
        WaymoDataLoader,
        ScenarioRiskAnalyzer,
        ScenarioProcessor,
    )
"""

# 从 processor 模块导入所有公开接口
from .processor import (
    # 类型
    VehicleTrajectory,
    InteractionFeatures,
    InteractionStats,
    FragmentMetadata,
    ScenarioMetadata,
    TrajectoryFragment,
    ProcessedScenario,
    RiskAnalysisResult,
    FEATURE_NAMES,

    # 数据加载
    WaymoDataLoader,

    # 风险计算
    RiskCalculator,
    ScenarioRiskAnalyzer,
    compute_interaction_features,
    score_distance,
    score_ttc,
    local_to_global_velocity,
    compute_relative_direction,

    # 片段处理
    ScenarioProcessor,
    process_scenario_batch,
    create_trajectory_fragment,
    extract_11d_features,
    extract_features_from_fragment,

    # 工具函数
    slice_trajectory,
    extract_fragment_slice,
    should_skip_fragment,
    compute_interaction_features_series,
    compute_interaction_stats,
    compute_target_speed_features,
    compute_target_accel_features,
    compute_trajectory_length,
    compute_max_curvature,
)

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
