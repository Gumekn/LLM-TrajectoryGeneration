"""
core/stage2/llm/prompt_builder.py - 提示词构造器

包含：
1. 用户提示词构造（build_intention_query_prompt）
2. 系统提示词（SYSTEM_PROMPT）
3. 轨迹信息提示词构造器（TrajectoryPromptBuilder）
4. 驾驶意图类型定义（DrivingIntention）
5. 意图数据结构（IntentionPhase, IntentionSequence）

使用示例：
    from core.stage2.llm.prompt_builder import TrajectoryPromptBuilder, SYSTEM_PROMPT

    # Step 1: 构造轨迹提示词
    builder = TrajectoryPromptBuilder(sample_interval=2)
    prompt = builder.build_prompt(fragment)

    # Step 2: 构造完整 LLM 提示词
    from core.stage2.llm.prompt_builder import build_intention_query_prompt
    full_prompt = build_intention_query_prompt(trajectory_prompt, key_frames, SYSTEM_PROMPT)
"""

import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any


# =============================================================================
# 第一步：用户提示词构造
# =============================================================================

def build_intention_query_prompt(
    trajectory_prompt: str,
    key_frames: list,
    system_prompt: str
) -> str:
    """构造完整的意图询问提示词（用户部分）"""
    frames_list = "\n".join([
        f"- frame {f['frame']}: {f['frame_type']}"
        for f in key_frames
    ])

    query = f"""## 关键帧列表
{frames_list}

## 分析要求
请分析上述关键帧，结合整个轨迹的全局信息（空间关系演变、速度变化趋势、TTC变化等），为每个关键帧生成驾驶意图。

意图应基于：
1. 该帧的瞬时状态（位置、速度、TTC等）
2. 该帧前后帧的变化趋势（接近还是远离）
3. 整体轨迹的交互模式

请以JSON格式输出意图分析结果。"""

    return f"{system_prompt}\n\n{trajectory_prompt}\n\n{query}"


# =============================================================================
# 第二步：系统提示词
# =============================================================================

SYSTEM_PROMPT = """你是一个自动驾驶领域的驾驶意图预测专家。

## 你的任务
根据提供的轨迹信息，为交互车在关键帧生成驾驶意图。

## 分析原则
1. **全局分析**：应从整个轨迹的全局角度分析，理解交互车的整体行为模式和演变趋势
2. **因果推理**：结合空间关系、速度变化、TTC趋势等因素，综合判断意图
3. **科学判断**：意图应与当前运动状态相符，符合物理规律

## 意图类型定义
1. cruise_maintain: 保持当前速度直行
2. accelerate_through: 加速通过（风险区域/路口）
3. decelerate_to_yield: 减速让行（让主车先行）
4. decelerate_to_stop: 减速至停止
5. emergency_brake: 紧急制动
6. lane_change_left: 向左变道
7. lane_change_right: 向右变道
8. turn_left: 左转
9. turn_right: 右转
10. go_straight: 直行（无变道）

## 意图碰撞影响
- decelerate_to_yield: 大幅减速 → 碰撞可能性降低
- accelerate_through: 加速 → 若在主车前方则碰撞可能性增加
- emergency_brake: 急刹 → 碰撞可能性降低
- lane_change_left/right: 换道 → 改变空间关系
- turn_left/right: 转弯 → 改变轨迹路径
- cruise_maintain: 保持当前速度 → 取决于当前空间关系
- go_straight: 直行 → 无横向运动

## 输出格式
```json
{
    "intention_frames": [
        {"frame": 18, "frame_type": "min_ttc", "intention": "decelerate_to_yield", "reasoning": "两车距最近"},
        {"frame": 26, "frame_type": "min_dist", "intention": "accelerate_through", "reasoning": "加速制造碰撞"}
    ]
}
```

## 重要说明
- reasoning 必须不超过10个字
- 每行一个帧的意图分析
- frame_type 说明：min_ttc(最小TTC帧)、min_dist(最近距离帧)、max_closing(最大接近速度帧)、anchor(锚点帧)、start(起始帧)、end(结束帧)、pre_mid(锚点前中点)、post_mid(锚点后中点)
3. 风险感知: 高风险时选择能导致碰撞的意图

## 关键分析要点
- 分析两车的空间关系：谁在前/后、左/右
- 分析速度对比：谁更快、快多少
- 分析 TTC 和距离趋势：正在接近还是远离
- 选择能最大化碰撞概率的意图"""


# =============================================================================
# 第三步：轨迹信息提示词构造器
# =============================================================================

class TrajectoryPromptBuilder:
    """轨迹信息提示词构造器 - 从轨迹片段数据提取并格式化轨迹信息"""

    # 意图类型列表（供LLM参考）
    INTENTION_TYPES = [
        "cruise_maintain", "accelerate_through", "decelerate_to_yield",
        "decelerate_to_stop", "emergency_brake", "lane_change_left",
        "lane_change_right", "turn_left", "turn_right", "go_straight",
    ]

    def __init__(self, sample_interval: int = 2):
        self.sample_interval = sample_interval

    def build_prompt(self, fragment: Dict) -> str:
        """构造轨迹信息提示词（7部分结构）"""
        if "fragment" in fragment:
            meta = fragment.get("metadata", {})
            frag = fragment.get("fragment", {})
        elif "metadata" in fragment and "ego_trajectory" in fragment:
            meta = fragment.get("metadata", {})
            frag = fragment
        else:
            meta = {}
            frag = fragment

        scene_info = self._build_scene_info(meta, frag)
        spatial_rel = self._build_spatial_relationship(meta, frag)
        key_frames = self._build_key_frame_analysis(meta, frag)
        trajectory_evo = self._build_vehicle_trajectory_evolution(meta, frag)
        interaction_stats = self._build_interaction_stats(meta, frag)
        trajectory_profile = self._build_trajectory_profile(frag)

        prompt_parts = [
            scene_info, "", spatial_rel, "", key_frames, "",
            trajectory_evo, "", interaction_stats, "", trajectory_profile,
        ]
        return "\n".join(prompt_parts)

    def _build_scene_info(self, meta: Dict, frag: Dict) -> str:
        """构造场景基本信息"""
        frag_meta = frag.get("metadata", frag)
        scenario_id = frag_meta.get("scenario_id", meta.get("scenario_id", "unknown"))
        frame_count = frag_meta.get("frame_count", 50)
        duration = frame_count / 10.0
        anchor_frame = frag_meta.get("anchor_frame", 0)
        danger_type = frag_meta.get("danger_type", "unknown")
        danger_level = frag_meta.get("danger_level", "unknown")
        min_risk_score = frag_meta.get("min_risk_score", 0)

        if min_risk_score <= 1:
            collision_potential = "CRITICAL - 极高碰撞风险"
        elif min_risk_score <= 2:
            collision_potential = "HIGH - 高碰撞风险"
        elif min_risk_score <= 3:
            collision_potential = "MEDIUM - 中等碰撞风险"
        else:
            collision_potential = "LOW - 低碰撞风险"

        return f"""## Scene Context & Danger Profile

场景ID: {scenario_id}
片段时长: {duration:.1f} 秒 ({frame_count} 帧 @ 10Hz)
锚点帧: {anchor_frame} (最危险时刻，锚点前{frag_meta.get('n_before', 0)}帧，锚点后{frag_meta.get('n_after', 0)}帧)
危险类型: {danger_type}
危险等级: {danger_level}
最低风险分数: {min_risk_score}
碰撞风险评估: {collision_potential}"""

    def _build_spatial_relationship(self, meta: Dict, frag: Dict) -> str:
        """构造两车空间关系"""
        ifeatures = frag.get("interaction_features", {})
        ego_traj = frag.get("ego_trajectory", {})
        target_traj = frag.get("target_trajectory", {})
        frag_meta = frag.get("metadata", frag)

        ego_positions = ego_traj.get("positions", [])
        target_positions = target_traj.get("positions", [])

        if not ego_positions or not target_positions:
            return "## Spatial Relationship\n空间关系: 无轨迹数据"

        anchor_frame = frag_meta.get("anchor_frame", 0)
        frame_count = frag_meta.get("frame_count", 50)

        init_idx = 0
        anchor_idx = min(anchor_frame, len(ego_positions) - 1, len(target_positions) - 1)
        end_idx = min(frame_count - 1, len(ego_positions) - 1, len(target_positions) - 1)

        init_spatial = self._describe_spatial_at(ego_positions[init_idx], target_positions[init_idx])
        anchor_spatial = self._describe_spatial_at(ego_positions[anchor_idx], target_positions[anchor_idx])
        end_spatial = self._describe_spatial_at(ego_positions[end_idx], target_positions[end_idx])

        init_dist = self._compute_distance(ego_positions[init_idx], target_positions[init_idx])
        anchor_dist = self._compute_distance(ego_positions[anchor_idx], target_positions[anchor_idx])
        end_dist = self._compute_distance(ego_positions[end_idx], target_positions[end_idx])

        if anchor_dist < init_dist and anchor_dist < end_dist:
            pattern = "CLOSING then PASSING - 先接近后远离"
        elif anchor_dist < init_dist:
            pattern = "CONSISTENT CLOSING - 持续接近"
        elif end_dist < init_dist:
            pattern = "DIVERGING - 逐渐远离"
        else:
            pattern = "STABLE - 相对稳定"

        return f"""## Spatial Relationship

初始 (frame 0): {init_spatial} (距离 {init_dist:.2f}m)
锚点 (frame {anchor_idx}): {anchor_spatial} (距离 {anchor_dist:.2f}m)
结束 (frame {end_idx}): {end_spatial} (距离 {end_dist:.2f}m)

空间模式: {pattern}"""

    def _describe_spatial_at(self, ego_pos: List, target_pos: List) -> str:
        """描述某一时刻两车的空间关系"""
        dx = target_pos[0] - ego_pos[0]
        dy = target_pos[1] - ego_pos[1]

        long_desc = ""
        lat_desc = ""

        if dx > 5:
            long_desc = "前方"
        elif dx < -5:
            long_desc = "后方"
        elif dx > 1:
            long_desc = "略前方"
        elif dx < -1:
            long_desc = "略后方"
        else:
            long_desc = "同位置"

        if dy > 3:
            lat_desc = "右侧"
        elif dy < -3:
            lat_desc = "左侧"
        elif dy > 1:
            lat_desc = "略右侧"
        elif dy < -1:
            lat_desc = "略左侧"
        else:
            lat_desc = "同横向"

        return f"target位于ego{long_desc}{lat_desc} (纵向{dx:.2f}m, 横向{dy:.2f}m)"

    def _build_key_frame_analysis(self, meta: Dict, frag: Dict) -> str:
        """构造4个关键帧分析"""
        ifeatures = frag.get("interaction_features", {})
        frag_meta = frag.get("metadata", frag)
        anchor_frame = frag_meta.get("anchor_frame", 0)
        frame_count = frag_meta.get("frame_count", 50)

        key_frames = self._identify_key_frames(ifeatures, anchor_frame)

        ego_traj = frag.get("ego_trajectory", {})
        target_traj = frag.get("target_trajectory", {})

        ego_positions = ego_traj.get("positions", [])
        target_positions = target_traj.get("positions", [])
        ego_velocities = ego_traj.get("velocities", [])
        target_velocities = target_traj.get("velocities", [])

        lines = ["## Key Frame Analysis (4 Critical Moments)"]

        af = key_frames.get("anchor")
        if af is not None and af < len(ego_positions) and af < len(target_positions):
            ego_pos = ego_positions[af]
            target_pos = target_positions[af]
            ego_spd = self._compute_speed(ego_velocities[af]) if af < len(ego_velocities) else 0
            target_spd = self._compute_speed(target_velocities[af]) if af < len(target_velocities) else 0
            spatial = self._describe_spatial_at(ego_pos, target_pos)
            dist = self._compute_distance(ego_pos, target_pos)

            lines.append(f"""
>>> FRAME {af}: ANCHOR (Most Dangerous Moment) <<<
  帧位置: frame {af} of {frame_count}
  风险等级: ANCHOR (最危险时刻)
  ego位置: ({ego_pos[0]:.2f}, {ego_pos[1]:.2f}), 速度: {ego_spd:.2f} m/s
  target位置: ({target_pos[0]:.2f}, {target_pos[1]:.2f}), 速度: {target_spd:.2f} m/s
  空间关系: {spatial}
  相对距离: {dist:.2f} m""")

        mtf = key_frames.get("min_ttc")
        if mtf is not None and mtf < len(ego_positions) and mtf < len(target_positions):
            ttc_values = ifeatures.get("ttc_long", [])
            ttc_val = ttc_values[mtf] if mtf < len(ttc_values) else float('inf')

            ego_pos = ego_positions[mtf]
            target_pos = target_positions[mtf]
            spatial = self._describe_spatial_at(ego_pos, target_pos)
            dist = self._compute_distance(ego_pos, target_pos)

            ttc_label = "CRITICAL" if ttc_val < 0.5 else ("HIGH" if ttc_val < 1.5 else "MEDIUM")

            lines.append(f"""
>>> FRAME {mtf}: MINIMUM TTC ({ttc_label}) <<<
  帧位置: frame {mtf} of {frame_count}
  TTC_long: {ttc_val:.3f}s - {ttc_label}
  上下文: 以当前接近速度，约{ttc_val:.2f}秒后可能发生碰撞
  空间关系: {spatial}
  相对距离: {dist:.2f} m""")

        mdf = key_frames.get("min_dist")
        if mdf is not None and mdf < len(ego_positions) and mdf < len(target_positions):
            rel_dists = ifeatures.get("rel_dist", [])
            dist_val = rel_dists[mdf] if mdf < len(rel_dists) else float('inf')

            ego_pos = ego_positions[mdf]
            target_pos = target_positions[mdf]
            spatial = self._describe_spatial_at(ego_pos, target_pos)

            dist_label = "EXTREMELY CLOSE" if dist_val < 5 else ("VERY CLOSE" if dist_val < 10 else "CLOSE")

            lines.append(f"""
>>> FRAME {mdf}: MINIMUM DISTANCE ({dist_label}) <<<
  帧位置: frame {mdf} of {frame_count}
  相对距离: {dist_val:.2f}m - {dist_label}
  上下文: 两车距离仅{dist_val:.2f}米，低于典型车辆长度
  空间关系: {spatial}""")

        mcsf = key_frames.get("max_closing")
        if mcsf is not None and mcsf < len(ego_positions) and mcsf < len(target_positions):
            rel_vels = ifeatures.get("rel_vel_x", [])
            closing_val = abs(rel_vels[mcsf]) if mcsf < len(rel_vels) else 0

            ego_pos = ego_positions[mcsf]
            target_pos = target_positions[mcsf]
            spatial = self._describe_spatial_at(ego_pos, target_pos)

            lines.append(f"""
>>> FRAME {mcsf}: MAXIMUM CLOSING SPEED <<<
  帧位置: frame {mcsf} of {frame_count}
  接近速度: {closing_val:.2f} m/s ({closing_val*3.6:.1f} km/h) - VERY AGGRESSIVE
  上下文: target正以{closing_val:.2f}m/s的相对速度接近ego
  空间关系: {spatial}""")

        return "\n".join(lines)

    def _identify_key_frames(self, ifeatures: Dict, anchor_frame: int) -> Dict:
        """识别4个关键帧索引"""
        result = {"anchor": anchor_frame}

        ttc_long = ifeatures.get("ttc_long", [])
        rel_dist = ifeatures.get("rel_dist", [])
        rel_vel_x = ifeatures.get("rel_vel_x", [])

        if ttc_long:
            valid_ttc = [(i, t) for i, t in enumerate(ttc_long) if t < float('inf')]
            if valid_ttc:
                min_ttc_idx = min(valid_ttc, key=lambda x: x[1])[0]
                result["min_ttc"] = min_ttc_idx

        if rel_dist:
            min_dist_idx = min(range(len(rel_dist)), key=lambda i: rel_dist[i])
            result["min_dist"] = min_dist_idx

        if rel_vel_x:
            max_closing_idx = min(range(len(rel_vel_x)), key=lambda i: abs(rel_vel_x[i]))
            result["max_closing"] = max_closing_idx

        return result

    def _build_vehicle_trajectory_evolution(self, meta: Dict, frag: Dict) -> str:
        """构造两车轨迹演变"""
        ego_info = self._build_vehicle_info(meta, frag, is_ego=True)
        target_info = self._build_vehicle_info(meta, frag, is_ego=False)
        speed_comparison = self._build_speed_comparison(meta, frag)

        return f"""## Vehicle Trajectory Evolution

### Ego Vehicle (主车)
{ego_info}

### Interaction Vehicle (交互车)
{target_info}

### Speed Comparison
{speed_comparison}"""

    def _build_vehicle_info(self, meta: Dict, frag: Dict, is_ego: bool) -> str:
        """构造单车的轨迹演变信息"""
        if is_ego:
            traj = frag.get("ego_trajectory", {})
            vehicle_id = meta.get("ego_vehicle_id", "unknown")
            label = "Ego"
        else:
            traj = frag.get("target_trajectory", {})
            vehicle_id = traj.get("vehicle_id", "unknown")
            label = "Target"

        positions = traj.get("positions", [])
        velocities = traj.get("velocities", [])
        headings = traj.get("headings", [])
        sizes = traj.get("sizes", [])

        if not positions:
            return f"  无轨迹数据"

        init_pos = positions[0]
        init_heading = headings[0] if headings else 0
        init_size = sizes[0] if sizes else [4.0, 1.8, 1.5]
        init_speed = self._compute_speed(velocities[0]) if velocities else 0

        final_pos = positions[-1]
        final_heading = headings[-1] if headings else 0
        final_speed = self._compute_speed(velocities[-1]) if velocities else init_speed

        speed_trend, speed_pct = self._describe_speed_trend(velocities)

        heading_change = final_heading - init_heading
        heading_change_deg = math.degrees(heading_change)

        if abs(heading_change_deg) < 5:
            movement_pattern = "GOING STRAIGHT"
        elif heading_change_deg > 0:
            movement_pattern = f"TURNING LEFT ({heading_change_deg:.1f}°)"
        else:
            movement_pattern = f"TURNING RIGHT ({abs(heading_change_deg):.1f}°)"

        traj_length = self._compute_trajectory_length(positions)

        return f"""初始: pos({init_pos[0]:.2f}, {init_pos[1]:.2f}), speed {init_speed:.2f} m/s, heading {init_heading:.3f} rad
  最终: pos({final_pos[0]:.2f}, {final_pos[1]:.2f}), speed {final_speed:.2f} m/s, heading {final_heading:.3f} rad
  尺寸: 长{init_size[0]:.2f}m x 宽{init_size[1]:.2f}m x 高{init_size[2]:.2f}m
  速度趋势: {speed_trend} ({speed_pct:+.1f}%)
  航向变化: {heading_change_deg:+.1f}° - {movement_pattern}
  轨迹长度: {traj_length:.1f} m"""

    def _describe_speed_trend(self, velocities: List) -> Tuple[str, float]:
        """描述速度趋势"""
        if len(velocities) < 2:
            return "STABLE", 0.0

        speeds = [self._compute_speed(v) for v in velocities]
        init_speed = speeds[0]
        final_speed = speeds[-1]

        if init_speed > 0:
            pct_change = (final_speed - init_speed) / init_speed * 100
        else:
            pct_change = 0.0

        if abs(pct_change) < 5:
            return "STABLE", pct_change
        elif pct_change > 0:
            return "INCREASING", pct_change
        else:
            return "DECREASING", pct_change

    def _compute_trajectory_length(self, positions: List) -> float:
        """计算轨迹总长度"""
        if len(positions) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(positions)):
            total += self._compute_distance(positions[i-1], positions[i])
        return total

    def _build_speed_comparison(self, _: Dict, frag: Dict) -> str:
        """构造两车速度对比"""
        ego_traj = frag.get("ego_trajectory", {})
        target_traj = frag.get("target_trajectory", {})

        ego_velocities = ego_traj.get("velocities", [])
        target_velocities = target_traj.get("velocities", [])

        if not ego_velocities or not target_velocities:
            return "无法计算速度对比"

        ego_speeds = [self._compute_speed(v) for v in ego_velocities]
        target_speeds = [self._compute_speed(v) for v in target_velocities]

        min_ego = min(ego_speeds)
        max_ego = max(ego_speeds)
        min_target = min(target_speeds)
        max_target = max(target_speeds)

        avg_ego = sum(ego_speeds) / len(ego_speeds)
        avg_target = sum(target_speeds) / len(target_speeds)

        speed_diff = avg_target - avg_ego

        if speed_diff > 2:
            comparison = f"target比ego平均快 {speed_diff:.2f} m/s"
        elif speed_diff < -2:
            comparison = f"target比ego平均慢 {abs(speed_diff):.2f} m/s"
        else:
            comparison = "两车速度相近"

        return f"""Ego速度范围: {min_ego:.2f} - {max_ego:.2f} m/s (平均 {avg_ego:.2f} m/s)
Target速度范围: {min_target:.2f} - {max_target:.2f} m/s (平均 {avg_target:.2f} m/s)
速度对比: {comparison}"""

    def _build_interaction_stats(self, _: Dict, frag: Dict) -> str:
        """构造交互统计信息"""
        stats = frag.get("interaction_stats", {})
        ifeatures = frag.get("interaction_features", {})

        min_ttc_long = stats.get("min_ttc_long", float('inf'))
        min_ttc_lat = stats.get("min_ttc_lat", float('inf'))
        min_rel_dist = stats.get("min_rel_dist", 0)
        mean_rel_dist = stats.get("mean_rel_dist", 0)
        max_closing_speed = stats.get("max_closing_speed", 0)
        mean_heading_diff = stats.get("mean_heading_diff", 0)

        if min_ttc_long < 0.5:
            ttc_label = "CRITICAL - 极可能碰撞"
        elif min_ttc_long < 1.0:
            ttc_label = "HIGH - 高度危险"
        elif min_ttc_long < 2.0:
            ttc_label = "MEDIUM - 中度危险"
        else:
            ttc_label = "LOW - 低度危险"

        if min_rel_dist < 5:
            dist_label = "EXTREMELY CLOSE - 极度危险"
        elif min_rel_dist < 10:
            dist_label = "VERY CLOSE - 非常危险"
        elif min_rel_dist < 20:
            dist_label = "CLOSE - 较危险"
        else:
            dist_label = "SAFE - 安全"

        ttc_long = ifeatures.get("ttc_long", [])
        valid_ttc_count = sum(1 for t in ttc_long if t < float('inf'))
        ttc_valid_ratio = valid_ttc_count / len(ttc_long) if ttc_long else 0

        return f"""## Interaction Context Summary

### TTC Analysis:
  最小纵向TTC: {min_ttc_long:.3f}s - {ttc_label}
  最小横向TTC: {min_ttc_lat:.2f}s
  TTC有效帧比例: {ttc_valid_ratio:.1%}

### Distance Analysis:
  最小相对距离: {min_rel_dist:.2f}m - {dist_label}
  平均相对距离: {mean_rel_dist:.2f}m

### Closing Pattern:
  最大接近速度: {max_closing_speed:.2f} m/s ({max_closing_speed*3.6:.1f} km/h)
  平均航向差: {mean_heading_diff:.3f} rad ({math.degrees(mean_heading_diff):.1f}°)"""

    def _build_trajectory_profile(self, frag: Dict) -> str:
        """构造轨迹剖面（双车，每N帧采样，关键帧标记）"""
        ego_traj = frag.get("ego_trajectory", {})
        target_traj = frag.get("target_trajectory", {})
        ifeatures = frag.get("interaction_features", {})
        frag_meta = frag.get("metadata", frag)
        anchor_frame = frag_meta.get("anchor_frame", 0)

        ego_velocities = ego_traj.get("velocities", [])
        target_velocities = target_traj.get("velocities", [])
        ego_positions = ego_traj.get("positions", [])
        target_positions = target_traj.get("positions", [])

        if not target_velocities:
            return "## Trajectory Profile\n轨迹特征: 无数据"

        key_frames = self._identify_key_frames(ifeatures, anchor_frame)

        lines = [f"## Trajectory Profile (每 {self.sample_interval} 帧采样)"]
        header = f"{'帧':<5} {'t_spd':<8} {'e_spd':<8} {'rel_dist':<10} {'TTC_long':<10} {'Spatial'}"
        lines.append(header)
        lines.append("-" * 70)

        for i in range(0, len(target_velocities), self.sample_interval):
            target_spd = self._compute_speed(target_velocities[i])
            ego_spd = self._compute_speed(ego_velocities[i]) if i < len(ego_velocities) else 0

            rel_dists = ifeatures.get("rel_dist", [])
            ttc_values = ifeatures.get("ttc_long", [])

            rel_dist = rel_dists[i] if i < len(rel_dists) else 0
            ttc = ttc_values[i] if i < len(ttc_values) else float('inf')

            if i < len(ego_positions) and i < len(target_positions):
                spatial = self._describe_spatial_at(ego_positions[i], target_positions[i])
            else:
                spatial = "N/A"

            markers = []
            if i == key_frames.get("anchor"):
                markers.append("ANCHOR[KEY]")
            if i == key_frames.get("min_ttc"):
                markers.append("MIN_TTC[KEY]")
            if i == key_frames.get("min_dist"):
                markers.append("MIN_DIST[KEY]")
            if i == key_frames.get("max_closing"):
                markers.append("MAX_CLOSE[KEY]")

            marker_str = " " + " ".join(markers) if markers else ""

            ttc_str = f"{ttc:.2f}s" if ttc < float('inf') else "inf"

            lines.append(f"{i:<5} {target_spd:<8.2f} {ego_spd:<8.2f} {rel_dist:<10.2f} {ttc_str:<10} {spatial}{marker_str}")

        return "\n".join(lines)

    def _compute_speed(self, velocity: List[float]) -> float:
        """计算速度标量"""
        if not velocity:
            return 0
        return math.sqrt(velocity[0]**2 + velocity[1]**2)

    def _compute_distance(self, pos1: List, pos2: List) -> float:
        """计算两点距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


# =============================================================================
# 驾驶意图类型定义
# =============================================================================
# 注意: DrivingIntention, IntentionPhase, IntentionSequence 已迁移到
# core/llm/intention_models.py 中统一管理，此处仅保留给 LLM 的文本定义（SYSTEM_PROMPT）
# 以避免重复定义导致的类型不一致问题
