"""
core/trajectory_mutator.py - 基于意图的轨迹变异器

根据 LLM 生成的驾驶意图序列，对轨迹进行变异。
变异通过以下步骤：
1. 意图序列 → 关键帧变异参数
2. 三次样条插值得到全帧参数
3. 运动学积分计算 x, y 坐标
"""

import math
import json
from typing import Dict, List, Optional, Any

import numpy as np
from scipy.interpolate import CubicSpline

from core.llm.prompt_builder import (
    IntentionSequence,
    IntentionPhase,
    DrivingIntention,
    MutationParams,
)


# 意图参数模板
# 每个意图定义关键帧的运动学参数曲线
INTENTION_TEMPLATES = {
    DrivingIntention.CRUISE_MAINTAIN: {
        "speed_scale": {"type": "constant", "value": 1.0, "noise": 0.05},
        "accel_long": {"type": "zero", "noise": 0.1},
        "accel_lat": {"type": "zero", "noise": 0.05},
        "heading_rate": 0.0,
    },

    DrivingIntention.DECELERATE_TO_YIELD: {
        "speed_scale": {
            "type": "exponential_decay",
            "initial": 1.0,
            "final_range": [0.3, 0.6],
            "decay_rate_range": [0.2, 0.5],
        },
        "accel_long": {"type": "constant", "range": [-3.0, -1.5]},
        "accel_lat": {"type": "zero"},
        "heading_rate": 0.0,
    },

    DrivingIntention.DECELERATE_TO_STOP: {
        "speed_scale": {
            "type": "linear_decay",
            "initial": 1.0,
            "final": 0.0,
        },
        "accel_long": {"type": "constant", "range": [-4.0, -2.0]},
        "accel_lat": {"type": "zero"},
        "heading_rate": 0.0,
    },

    DrivingIntention.ACCELERATE_THROUGH: {
        "speed_scale": {
            "type": "sigmoid_growth",
            "initial_range": [0.7, 0.9],
            "final_range": [1.2, 1.5],
        },
        "accel_long": {"type": "bell_curve", "peak_range": [2.0, 4.0], "timing": 0.3},
        "accel_lat": {"type": "zero"},
        "heading_rate": 0.0,
    },

    DrivingIntention.EMERGENCY_BRAKE: {
        "speed_scale": {
            "type": "sharp_decay",
            "initial": 1.0,
            "final": 0.0,
        },
        "accel_long": {"type": "constant", "range": [-8.0, -5.0]},
        "accel_lat": {"type": "zero"},
        "heading_rate": 0.0,
    },

    DrivingIntention.LANE_CHANGE_LEFT: {
        "speed_scale": {"type": "slight_increase", "scale_range": [1.0, 1.2]},
        "accel_long": {"type": "small_positive"},
        "accel_lat": {"type": "bell_curve", "peak_range": [1.0, 2.0], "timing": 0.5},
        "heading_rate": {"type": "bell_curve", "peak_range": [0.05, 0.15], "timing": 0.4},
        "lateral_offset": {"type": "ramp", "start": 0.0, "end_range": [2.0, 3.5]},
    },

    DrivingIntention.LANE_CHANGE_RIGHT: {
        "speed_scale": {"type": "slight_increase", "scale_range": [1.0, 1.2]},
        "accel_long": {"type": "small_positive"},
        "accel_lat": {"type": "bell_curve", "peak_range": [-2.0, -1.0], "timing": 0.5},
        "heading_rate": {"type": "bell_curve", "peak_range": [-0.15, -0.05], "timing": 0.4},
        "lateral_offset": {"type": "ramp", "start": 0.0, "end_range": [-3.5, -2.0]},
    },

    DrivingIntention.TURN_LEFT: {
        "speed_scale": {"type": "constant", "value": 0.8, "noise": 0.05},
        "accel_long": {"type": "small_negative"},
        "accel_lat": {"type": "constant", "range": [1.0, 2.0]},
        "heading_rate": {"type": "constant", "range": [0.1, 0.2]},
    },

    DrivingIntention.TURN_RIGHT: {
        "speed_scale": {"type": "constant", "value": 0.8, "noise": 0.05},
        "accel_long": {"type": "small_negative"},
        "accel_lat": {"type": "constant", "range": [-2.0, -1.0]},
        "heading_rate": {"type": "constant", "range": [-0.2, -0.1]},
    },

    DrivingIntention.GO_STRAIGHT: {
        "speed_scale": {"type": "constant", "value": 1.0, "noise": 0.05},
        "accel_long": {"type": "zero", "noise": 0.1},
        "accel_lat": {"type": "zero"},
        "heading_rate": 0.0,
    },

    DrivingIntention.UNKNOWN: {
        "speed_scale": {"type": "constant", "value": 1.0, "noise": 0.1},
        "accel_long": {"type": "zero", "noise": 0.2},
        "accel_lat": {"type": "zero", "noise": 0.1},
        "heading_rate": 0.0,
    },
}


def _compute_speed(velocity: List[float]) -> float:
    """计算速度标量"""
    return math.sqrt(velocity[0]**2 + velocity[1]**2) if velocity else 0


class IntentionDrivenTrajectoryMutator:
    """基于 LLM 意图的轨迹变异器"""

    def __init__(self, intention_generator=None, seed: Optional[int] = None):
        """
        初始化变异器

        Args:
            intention_generator: LLM 意图生成器，如果为 None 则使用随机意图
            seed: 随机种子，用于可重现性
        """
        self.intention_generator = intention_generator
        if seed is not None:
            np.random.seed(seed)

    def mutate(self, fragment: Dict, n_variants: int = 10) -> List[Dict]:
        """
        生成变异轨迹

        Args:
            fragment: 原始轨迹片段（来自 JSON）
            n_variants: 生成的变体数量

        Returns:
            变异后的轨迹列表，每个元素包含:
                - original_fragment_id: 原始片段ID
                - intention_sequence: 意图序列
                - trajectory: 变异后的轨迹数据
                - variant_id: 变体编号
        """
        # 1. 生成意图序列
        if self.intention_generator:
            intention_seq = self.intention_generator.generate(fragment)
        else:
            # 随机生成意图序列
            intention_seq = self._generate_random_intention_sequence(fragment)

        # 2. 为每个变体生成变异参数
        variants = []
        for i in range(n_variants):
            variant = self._generate_variant(fragment, intention_seq, variant_id=i)
            variants.append(variant)

        return variants

    def _generate_random_intention_sequence(self, fragment: Dict) -> IntentionSequence:
        """生成随机意图序列（当无 LLM 时使用）"""
        meta = fragment.get("metadata", {})
        frag = fragment.get("fragment", fragment) if "fragment" in fragment else fragment
        frame_count = frag.get("frame_count", 50)

        # 随机选择意图
        intentions = [
            DrivingIntention.CRUISE_MAINTAIN,
            DrivingIntention.DECELERATE_TO_YIELD,
            DrivingIntention.ACCELERATE_THROUGH,
        ]
        intention = np.random.choice(intentions)

        return IntentionSequence(
            phases=[
                IntentionPhase(
                    start_frame=0,
                    end_frame=frame_count,
                    intention=intention,
                    confidence=0.8,
                    reasoning="随机生成"
                )
            ],
            overall_strategy="默认策略"
        )

    def _generate_variant(
        self,
        fragment: Dict,
        intention_seq: IntentionSequence,
        variant_id: int
    ) -> Dict:
        """生成单条变异轨迹"""
        np.random.seed(variant_id)

        meta = fragment.get("metadata", {})
        frag = fragment.get("fragment", fragment) if "fragment" in fragment else fragment
        target = frag.get("target_trajectory", {})

        frame_count = frag.get("frame_count", 50)
        dt = 0.1  # 10Hz

        # 1. 生成关键帧参数
        key_frames, key_params = self._generate_key_params(
            intention_seq, frame_count, variant_id
        )

        # 2. 插值得到全帧参数
        indicators = self._interpolate_params(key_frames, key_params, frame_count)

        # 3. 运动学积分
        trajectory = self._integrate_trajectory(target, indicators, dt)

        return {
            "original_fragment_id": frag.get("fragment_id", "unknown"),
            "intention_sequence": intention_seq.to_dict(),
            "trajectory": trajectory,
            "variant_id": variant_id
        }

    def _generate_key_params(
        self,
        intention_seq: IntentionSequence,
        frame_count: int,
        variant_id: int
    ) -> tuple:
        """为每个意图阶段生成关键帧参数"""
        key_frames = set()
        key_params = {
            "speed_scale": [],
            "accel_long": [],
            "accel_lat": [],
            "heading_rate": [],
            "lateral_offset": []
        }

        for phase in intention_seq.phases:
            template = INTENTION_TEMPLATES.get(
                phase.intention,
                INTENTION_TEMPLATES[DrivingIntention.CRUISE_MAINTAIN]
            )

            # 添加关键帧
            key_frames.add(phase.start_frame)
            key_frames.add(phase.end_frame)

            # 从模板采样参数
            phase_params = self._sample_from_template(template, phase, variant_id)

            key_params["speed_scale"].append(phase_params["speed_scale"])
            key_params["accel_long"].append(phase_params["accel_long"])
            key_params["accel_lat"].append(phase_params["accel_lat"])
            key_params["heading_rate"].append(phase_params["heading_rate"])
            key_params["lateral_offset"].append(phase_params.get("lateral_offset", 0.0))

        # 去重并排序
        key_frames = sorted(key_frames)

        # 确保关键帧参数与关键帧数量匹配
        for key in key_params:
            while len(key_params[key]) < len(key_frames):
                key_params[key].append(key_params[key][-1] if key_params[key] else 0.0)

        return key_frames, key_params

    def _sample_from_template(
        self,
        template: Dict,
        phase: IntentionPhase,
        variant_id: int
    ) -> Dict:
        """从意图模板采样参数"""
        result = {}

        # speed_scale
        speed_cfg = template.get("speed_scale", {})
        speed_type = speed_cfg.get("type", "constant")

        if speed_type == "constant":
            value = speed_cfg.get("value", 1.0)
            noise = speed_cfg.get("noise", 0)
            result["speed_scale"] = value * (1 + np.random.uniform(-noise, noise))
        elif speed_type == "exponential_decay":
            initial = speed_cfg.get("initial", 1.0)
            final_range = speed_cfg.get("final_range", [0.3, 0.6])
            final = np.random.uniform(*final_range)
            result["speed_scale"] = [initial, final]
        elif speed_type == "sigmoid_growth":
            initial_range = speed_cfg.get("initial_range", [0.7, 0.9])
            final_range = speed_cfg.get("final_range", [1.2, 1.5])
            initial = np.random.uniform(*initial_range)
            final = np.random.uniform(*final_range)
            result["speed_scale"] = [initial, final]
        elif speed_type == "linear_decay" or speed_type == "sharp_decay":
            initial = speed_cfg.get("initial", 1.0)
            final = speed_cfg.get("final", 0.0)
            result["speed_scale"] = [initial, final]
        elif speed_type == "slight_increase":
            scale_range = speed_cfg.get("scale_range", [1.0, 1.2])
            scale = np.random.uniform(*scale_range)
            result["speed_scale"] = [1.0, scale]
        else:
            result["speed_scale"] = 1.0

        # accel_long
        accel_cfg = template.get("accel_long", {})
        accel_type = accel_cfg.get("type", "zero")

        if accel_type == "zero":
            noise = accel_cfg.get("noise", 0.1)
            result["accel_long"] = np.random.uniform(-noise, noise)
        elif accel_type == "constant":
            range_vals = accel_cfg.get("range", [-3.0, -1.5])
            result["accel_long"] = np.random.uniform(*range_vals)
        elif accel_type == "small_positive":
            result["accel_long"] = np.random.uniform(0.5, 1.5)
        elif accel_type == "small_negative":
            result["accel_long"] = np.random.uniform(-1.5, -0.5)
        else:
            result["accel_long"] = 0.0

        # accel_lat
        lat_cfg = template.get("accel_lat", {})
        lat_type = lat_cfg.get("type", "zero")

        if lat_type == "zero":
            noise = lat_cfg.get("noise", 0.05)
            result["accel_lat"] = np.random.uniform(-noise, noise)
        elif lat_type == "constant":
            range_vals = lat_cfg.get("range", [1.0, 2.0])
            result["accel_lat"] = np.random.uniform(*range_vals)
        elif lat_type == "bell_curve":
            # 钟形曲线参数在插值时处理，这里只返回峰值
            peak_range = lat_cfg.get("peak_range", [1.0, 2.0])
            result["accel_lat"] = np.random.uniform(*peak_range)
        else:
            result["accel_lat"] = 0.0

        # heading_rate
        hr = template.get("heading_rate", 0)
        if isinstance(hr, dict):
            if hr.get("type") == "bell_curve":
                peak_range = hr.get("peak_range", [0.05, 0.15])
                result["heading_rate"] = np.random.uniform(*peak_range)
            else:
                range_vals = hr.get("range", [0.1, 0.2])
                result["heading_rate"] = np.random.uniform(*range_vals)
        elif isinstance(hr, (int, float)):
            result["heading_rate"] = hr * (1 + np.random.uniform(-0.1, 0.1))
        else:
            result["heading_rate"] = 0.0

        # lateral_offset
        if "lateral_offset" in template:
            lo_cfg = template["lateral_offset"]
            if lo_cfg.get("type") == "ramp":
                start = lo_cfg.get("start", 0.0)
                end_range = lo_cfg.get("end_range", [2.0, 3.5])
                end = np.random.uniform(*end_range)
                result["lateral_offset"] = [start, end]
            else:
                result["lateral_offset"] = [0.0, 0.0]
        else:
            result["lateral_offset"] = 0.0

        return result

    def _interpolate_params(
        self,
        key_frames: List[int],
        key_params: Dict,
        frame_count: int
    ) -> Dict:
        """三次样条插值"""
        if len(key_frames) < 2:
            # 常数值
            return {
                key: np.full(frame_count, val[0] if isinstance(val, list) else val)
                for key, val in key_params.items()
            }

        t_key = np.array(key_frames)
        t_full = np.arange(frame_count)

        result = {}
        for param_key, values in key_params.items():
            # 处理特殊格式 [start, end]
            processed_values = []
            for v in values:
                if isinstance(v, list):
                    processed_values.append(v[0])  # 取起始值
                else:
                    processed_values.append(v)

            if len(processed_values) < len(key_frames):
                processed_values.extend([processed_values[-1]] * (len(key_frames) - len(processed_values)))

            try:
                cs = CubicSpline(t_key, processed_values)
                result[param_key] = cs(t_full)
            except Exception:
                result[param_key] = np.full(frame_count, processed_values[0] if processed_values else 0.0)

        return result

    def _integrate_trajectory(
        self,
        original_trajectory: Dict,
        indicators: Dict,
        dt: float
    ) -> Dict:
        """运动学积分"""
        frame_count = len(indicators["speed_scale"])

        # 初始状态
        velocities = original_trajectory.get("velocities", [[0, 0]])
        headings = original_trajectory.get("headings", [0])
        positions = original_trajectory.get("positions", [[0, 0, 0]])

        initial_speed = _compute_speed(velocities[0]) if velocities else 0
        initial_heading = headings[0] if headings else 0
        initial_pos = positions[0] if positions else [0, 0, 0]

        # 积分
        new_positions = np.zeros((frame_count, 3))
        new_headings = np.zeros(frame_count)
        new_velocities = np.zeros((frame_count, 2))
        new_valid = [True] * frame_count

        new_positions[0] = initial_pos
        new_headings[0] = initial_heading

        for t in range(frame_count - 1):
            # 当前状态
            speed_scale = indicators["speed_scale"][t]
            v = initial_speed * speed_scale

            h = new_headings[t]

            # 加速度（全局坐标系）
            a_long = indicators["accel_long"][t]
            a_lat = indicators["accel_lat"][t]

            # 积分速度（考虑加速度）
            v_next = v + a_long * dt
            v_next = max(v_next, 0)  # 速度非负

            # 积分航向
            h_rate = indicators["heading_rate"][t]
            h_next = h + h_rate * dt

            # 积分位置
            x_next = new_positions[t, 0] + v * math.cos(h) * dt
            y_next = new_positions[t, 1] + v * math.sin(h) * dt

            new_positions[t + 1] = [x_next, y_next, new_positions[t, 2]]
            new_headings[t + 1] = h_next
            new_velocities[t] = [v * math.cos(h), v * math.sin(h)]

        return {
            "positions": new_positions.tolist(),
            "headings": new_headings.tolist(),
            "velocities": new_velocities.tolist(),
            "valid": new_valid
        }


def mutate_trajectories(
    fragment: Dict,
    n_variants: int = 10,
    model_name: str = "qwen-plus",
    seed: Optional[int] = None
) -> List[Dict]:
    """
    便捷函数：变异轨迹

    Args:
        fragment: 原始轨迹片段
        n_variants: 变体数量
        model_name: LLM 模型名称
        seed: 随机种子

    Returns:
        变异后的轨迹列表
    """
    from core.llm_intention_generator import LLMIntentionGenerator

    generator = LLMIntentionGenerator(model_name)
    mutator = IntentionDrivenTrajectoryMutator(generator, seed=seed)
    return mutator.mutate(fragment, n_variants)
