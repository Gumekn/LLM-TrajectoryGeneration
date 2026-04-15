"""
core/ - 核心处理模块

本模块整合了危险轨迹生成与评估系统的所有核心功能。

建议导入方式：
    from core.processor import WaymoDataLoader, ScenarioProcessor
    from core import build_trajectory_prompt, TrajectoryPromptBuilder
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

# 轨迹提示词构造和意图生成
from .llm_intention_generator import (
    build_trajectory_prompt,
    LLMIntentionGenerator,
    save_fragment_with_intention,
)

# 轨迹变异
from .trajectory_mutator import (
    IntentionDrivenTrajectoryMutator,
    mutate_trajectories,
    get_combination_count,
)

# LLM 统一客户端和提示词构造
from .llm import (
    UnifiedLLMClient,
    list_providers,
    DrivingIntention,
    IntentionPhase,
    IntentionSequence,
    TrajectoryPromptBuilder,
    SYSTEM_PROMPT,
    identify_key_frames,
    generate_intention,
    parse_intention_response,
    IntentionFrame,
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

    # 片段处理
    'ScenarioProcessor',
    'process_scenario_batch',
    'create_trajectory_fragment',
    'extract_11d_features',
    'extract_features_from_fragment',

    # 轨迹提示词构造
    'build_trajectory_prompt',
    'LLMIntentionGenerator',
    'save_fragment_with_intention',

    # 轨迹变异
    'IntentionDrivenTrajectoryMutator',
    'mutate_trajectories',
    'get_combination_count',

    # LLM 客户端
    'UnifiedLLMClient',
    'list_providers',

    # LLM 数据模型
    'DrivingIntention',
    'IntentionPhase',
    'IntentionSequence',
    'IntentionFrame',

    # 提示词构造
    'TrajectoryPromptBuilder',
    'SYSTEM_PROMPT',

    # 意图生成
    'identify_key_frames',
    'generate_intention',
    'parse_intention_response',
]
