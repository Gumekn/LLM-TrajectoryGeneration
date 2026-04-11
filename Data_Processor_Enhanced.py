"""
Data_Processor_Enhanced.py
基于学术版框架"五、数据流与处理流程"增强版数据处理模块
新增：轨迹向量化、LLM生成、物理约束验证、RAG检索、合理性评估

数据流与处理流程：
[Step 1] 场景加载与车辆选择
[Step 2] 风险计算与危险片段定位
[Step 3] 轨迹特征提取与向量化
[Step 4] LLM轨迹变异生成
[Step 5] 物理约束验证（硬过滤）
[Step 6] RAG检索相似案例
[Step 7] LLM合理性评估
[Step 8] 人工审核与知识库更新
[Step 9] CARLA格式导出与闭环测试
"""

import os
import pickle
import numpy as np
import copy
import json
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

# =================================================================
# 功能类 1：场景数据准备 (ScenarioDataLoader) - 原有功能
# =================================================================
class ScenarioDataLoader:
    """负责从物理地址读取原始 pkl 数据"""

    def __init__(self, data_dir, scenario_id):
        # 使用 os.path.abspath 确保路径在 Windows 系统下的健壮性
        self.scenario_id = scenario_id
        self.file_path = os.path.abspath(os.path.join(data_dir, f"{scenario_id}.pkl"))
        self.data = None

    def load_data(self):
        """读取原始 pkl 文件并返回数据字典"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"无法找到文件，请核对路径: {self.file_path}")

        with open(self.file_path, 'rb') as f:
            self.data = pickle.load(f)
        return self.data

    def select_observer(self):
        """交互式选择观察车 ID"""
        sdc_id = self.data['extra_information']['sdc_id']
        tracks = self.data['object_tracks']
        vehicle_ids = [vid for vid, info in tracks.items() if info['type'] == 'VEHICLE']

        print("\n" + "=" * 60)
        print(f"场景 ID: {self.scenario_id}")
        print(f"自车 (SDC) ID: {sdc_id}")
        print(f"场景内所有车辆 ID: \n{vehicle_ids}")
        print("=" * 60)

        user_input = input(f">>> 请输入观察车 ID (默认 {sdc_id}): ").strip()
        return user_input if user_input else str(sdc_id)


# =================================================================
# 功能类 2：风险分数计算类 (RiskCalculator) - 原有功能
# =================================================================
class RiskCalculator:
    """在 pkl 字典中创建 processing-result -> RiskScore 并注入结果"""

    def __init__(self, scenario_data):
        self.data = scenario_data
        self.scene_len = self.data['extra_information']['scene_length']
        self.tracks = self.data['object_tracks']

    @staticmethod
    def _score_dist(d):
        d_abs = abs(d)
        if d_abs < 0.3: return 0
        if d_abs < 0.8: return 1
        if d_abs < 1.3: return 2
        if d_abs < 3.0: return 3
        if d_abs < 5.0: return 4
        return 5

    @staticmethod
    def _score_ttc(ttc):
        if ttc < 0.15: return 0
        if np.isnan(ttc) or np.isinf(ttc) or ttc < 0: return 5
        if ttc <= 0.65: return 1
        if ttc <= 1.15: return 2
        if ttc <= 3.0: return 3
        if ttc <= 5.0: return 4
        return 5

    def _local_to_global_vel(self, local_vel, heading):
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        gx = local_vel[:, 0] * cos_h - local_vel[:, 1] * sin_h
        gy = local_vel[:, 0] * sin_h + local_vel[:, 1] * cos_h
        return np.stack([gx, gy], axis=-1)

    def _get_direction_weight(self, d_long, d_lat):
        angle = np.arctan2(d_lat, d_long)
        if (-np.pi / 4 <= angle <= np.pi / 4) or (angle > 3 * np.pi / 4) or (angle <= -3 * np.pi / 4):
            return 1.0
        return 0.0

    def compute_relative_metrics(self, obs_id, tar_id):
        obs = self.tracks[obs_id]['state']
        tar = self.tracks[tar_id]['state']
        v_obs_g = self._local_to_global_vel(obs['local_velocity'], obs['heading'])
        v_tar_g = self._local_to_global_vel(tar['local_velocity'], tar['heading'])
        dp_g = tar['global_center'][:, :2] - obs['global_center'][:, :2]
        dv_g = v_tar_g - v_obs_g
        cos_h, sin_h = np.cos(obs['heading']), np.sin(obs['heading'])
        d_l = dp_g[:, 0] * cos_h + dp_g[:, 1] * sin_h
        v_l = dv_g[:, 0] * cos_h + dv_g[:, 1] * sin_h
        d_a = -dp_g[:, 0] * sin_h + dp_g[:, 1] * cos_h
        v_a = -dv_g[:, 0] * sin_h + dv_g[:, 1] * cos_h
        with np.errstate(divide='ignore', invalid='ignore'):
            tl = np.where((d_l * v_l) < 0, -d_l / v_l, np.inf)
            ta = np.where((d_a * v_a) < 0, -d_a / v_a, np.inf)
        return d_l, d_a, tl, ta

    def run(self, obs_id):
        vehicle_ids = [vid for vid, info in self.tracks.items() if info['type'] == 'VEHICLE' and vid != obs_id]
        risk_map = {}
        global_min_score = 5.0

        for vid in vehicle_ids:
            scores = np.full((self.scene_len, 7), 5.0)
            dl, da, tl, ta = self.compute_relative_metrics(obs_id, vid)
            valid_mask = self.tracks[obs_id]['state']['valid'] & self.tracks[vid]['state']['valid']

            for t in range(self.scene_len):
                if not valid_mask[t]: continue
                s_dl, s_da = self._score_dist(dl[t]), self._score_dist(da[t])
                s_tl, s_ta = self._score_ttc(abs(tl[t])), self._score_ttc(abs(ta[t]))
                w = self._get_direction_weight(dl[t], da[t])
                dsc, tsc = (s_dl * w + s_da * (1 - w)), (s_tl * w + s_ta * (1 - w))
                scores[t] = [s_dl, s_da, s_tl, s_ta, dsc, tsc, int((dsc + tsc) / 2)]

            risk_map[vid] = scores
            global_min_score = min(global_min_score, np.min(scores[:, 6]))

        print("\n" + "-" * 40)
        print(f"风险分数计算完成 (观察车: {obs_id})")
        print(f"全场景最低风险总分: {int(global_min_score)}")
        print("-" * 40)

        # 在 pkl 字典根目录下新建两级字典存储结果
        updated_data = copy.deepcopy(self.data)
        if 'processing-result' not in updated_data:
            updated_data['processing-result'] = {}

        # 注入计算结果与元数据
        updated_data['processing-result']['RiskScore'] = risk_map
        updated_data['processing-result']['obs_id'] = obs_id

        return updated_data


# =================================================================
# 功能类 3：关键车筛选类 (KeyVehicleFilter) - 原有功能
# =================================================================
class KeyVehicleFilter:
    """读取 pkl 字典中的 RiskScore 并注入 KeyCarID_Select 结果"""

    def __init__(self, scenario_data):
        self.data = scenario_data
        self.risk_results = self.data.get('processing-result', {}).get('RiskScore', {})

    def run(self, threshold):
        key_vehicle_list = []
        raw_key_ids = []

        for vid, risk_matrix in self.risk_results.items():
            total_scores = risk_matrix[:, 6]
            min_score = np.min(total_scores)

            if min_score <= threshold:
                min_frame_idx = np.argmin(total_scores)
                # 1. 存入 list 供转为 Numpy 数组 (用于 details)
                key_vehicle_list.append([int(vid), int(min_score), int(min_frame_idx)])
                # 2. 存入原生 ID
                raw_key_ids.append(vid)

        updated_data = copy.deepcopy(self.data)

        if 'processing-result' not in updated_data:
            updated_data['processing-result'] = {}

        # 转换为要求的 N*3 Numpy 数组
        details_array = np.array(key_vehicle_list) if key_vehicle_list else np.empty((0, 3))

        updated_data['processing-result']['KeyCarID_Select'] = {
            'threshold_used': threshold,
            'key_ids': raw_key_ids,  # 使用原生的 ID 列表
            'details': details_array
        }

        print("\n" + "-" * 40)
        print(f"关键车筛选完成 (阈值: {threshold})")
        print(f"发现目标: {len(raw_key_ids)} 辆")
        print(f"关键车列表 [ID, 最低风险分, 首次出现帧]:")
        if len(key_vehicle_list) > 0:
            print(details_array)
        print("-" * 40)

        return updated_data


# =================================================================
# 功能类 4：关键车轨迹提取模块 (KeyVehicleTrajectoryExtractor) - 原有功能
# =================================================================
class KeyVehicleTrajectoryExtractor:
    """提取关键车轨迹并计算相对于观察车的交互特征(相对距离、速度、加速度、TTC)"""

    def __init__(self, scenario_data):
        self.data = scenario_data
        # 从 processing-result 检索必要信息
        proc_res = self.data.get('processing-result', {})
        self.obs_id = proc_res.get('obs_id')
        self.key_ids = proc_res.get('KeyCarID_Select', {}).get('key_ids', [])
        self.tracks = self.data['object_tracks']
        self.scene_len = self.data['extra_information']['scene_length']

    def run(self):
        if not self.key_ids or not self.obs_id:
            print("未发现关键车或观察车，跳过轨迹提取。")
            return self.data

        # 在字典中初始化存储位置
        updated_data = copy.deepcopy(self.data)
        if 'KeyVehicle_Trajectory' not in updated_data['processing-result']:
            updated_data['processing-result']['KeyVehicle_Trajectory'] = {}

        traj_container = updated_data['processing-result']['KeyVehicle_Trajectory']
        obs_state = self.tracks[self.obs_id]['state']

        for kid in self.key_ids:
            tar_state = self.tracks[kid]['state']
            # 1. 完整提取原始 7 项数据
            kid_traj = {
                'action': tar_state['action'],
                'global_center': tar_state['global_center'],
                'heading': tar_state['heading'],
                'local_acceleration': tar_state['local_acceleration'],
                'local_velocity': tar_state['local_velocity'],
                'size': tar_state['size'],
                'valid': tar_state['valid']
            }

            # 2. 初始化新增交互变量的容器 (Numpy Array)
            interaction_data = {
                'rel_dist_long': np.zeros(self.scene_len),
                'rel_dist_lat': np.zeros(self.scene_len),
                'adj_dist_long': np.zeros(self.scene_len),
                'adj_dist_lat': np.zeros(self.scene_len),
                'rel_velocity_long': np.zeros(self.scene_len),
                'rel_velocity_lat': np.zeros(self.scene_len),
                'rel_accel_long': np.zeros(self.scene_len),
                'rel_accel_lat': np.zeros(self.scene_len),
                'relative_angle': np.zeros(self.scene_len),
                'relative_direction': [None] * self.scene_len,
                'ttc_long': np.full(self.scene_len, np.inf),
                'ttc_lat': np.full(self.scene_len, np.inf),
            }

            # 3. 逐帧计算交互特征
            for t in range(self.scene_len):
                if not (obs_state['valid'][t] and tar_state['valid'][t]):
                    continue

                # --- 基础参数准备 ---
                x_ego, y_ego, theta_ego = obs_state['global_center'][t, 0], obs_state['global_center'][t, 1], \
                obs_state['heading'][t]
                x_tar, y_tar, theta_tar = tar_state['global_center'][t, 0], tar_state['global_center'][t, 1], \
                tar_state['heading'][t]
                v_ego, v_tar = np.linalg.norm(obs_state['local_velocity'][t]), np.linalg.norm(
                    tar_state['local_velocity'][t])
                a_ego, a_tar = np.linalg.norm(obs_state['local_acceleration'][t]), np.linalg.norm(
                    tar_state['local_acceleration'][t])

                # 动态获取当前帧观察车尺寸
                ego_length, ego_width = obs_state['size'][t, 0], obs_state['size'][t, 1]

                # --- (1) 相对距离与坐标变换 ---
                dx, dy = x_tar - x_ego, y_tar - y_ego
                d_long = dx * np.cos(-theta_ego) - dy * np.sin(-theta_ego)
                d_lat = dx * np.sin(-theta_ego) + dy * np.cos(-theta_ego)

                # --- (2) 考虑尺寸的调整后的距离 ---
                # 使用 ego 车当前帧尺寸作为阈值
                if -ego_length < d_long < ego_length:
                    adj_dl = 0
                else:
                    adj_dl = d_long - ego_length if d_long > 0 else d_long + ego_length

                if -ego_width < d_lat < ego_width:
                    adj_dt = 0
                else:
                    adj_dt = d_lat - ego_width if d_lat > 0 else d_lat + ego_width

                # --- (3) 相对速度与加速度 ---
                v_rel_long = v_tar * np.cos(theta_tar - theta_ego) - v_ego
                v_rel_lat = v_tar * np.sin(theta_tar - theta_ego)

                a_rel_long = a_tar * np.cos(theta_tar) - a_ego * np.cos(theta_ego)
                a_rel_lat = a_tar * np.sin(theta_tar) - a_ego * np.sin(theta_ego)

                # --- (4) 相对方向角与语义方向 ---
                rel_angle = np.arctan2(dy, dx)

                # 语义方位判定
                if d_long > ego_length:
                    if d_lat > ego_width:
                        r_dir = "Front-left"
                    elif d_lat < -ego_width:
                        r_dir = "Front-right"
                    else:
                        r_dir = "Front"
                elif d_long < -ego_length:
                    if d_lat > ego_width:
                        r_dir = "Rear-left"
                    elif d_lat < -ego_width:
                        r_dir = "Rear-right"
                    else:
                        r_dir = "Behind"
                else:
                    if d_lat > ego_width:
                        r_dir = "Left"
                    elif d_lat < -ego_width:
                        r_dir = "Right"
                    else:
                        r_dir = "Collision"

                # --- (5) TTC 计算逻辑 ---
                ttc_l, ttc_t = np.inf, np.inf


                # 纵向 TTC
                if r_dir in ["Front", "Front-left", "Front-right"]:
                    if v_rel_long < 0: ttc_l = adj_dl / abs(v_rel_long)
                elif r_dir in ["Behind", "Rear-left", "Rear-right"]:
                    if v_rel_long > 0: ttc_l = abs(adj_dl) / v_rel_long

                # 横向 TTC
                if r_dir in ["Left", "Front-left", "Rear-left"]:
                    if v_rel_lat < 0: ttc_t = abs(adj_dt / v_rel_lat)
                elif r_dir in ["Right", "Front-right", "Rear-right"]:
                    if v_rel_lat > 0: ttc_t = abs(adj_dt / v_rel_lat)

                # 填充结果
                interaction_data['rel_dist_long'][t], interaction_data['rel_dist_lat'][t] = d_long, d_lat
                interaction_data['adj_dist_long'][t], interaction_data['adj_dist_lat'][t] = adj_dl, adj_dt
                interaction_data['rel_velocity_long'][t], interaction_data['rel_velocity_lat'][
                    t] = v_rel_long, v_rel_lat
                interaction_data['rel_accel_long'][t], interaction_data['rel_accel_lat'][t] = a_rel_long, a_rel_lat
                interaction_data['relative_angle'][t] = rel_angle
                interaction_data['relative_direction'][t] = r_dir
                interaction_data['ttc_long'][t], interaction_data['ttc_lat'][t] = ttc_l, ttc_t

            # 将新增交互变量合并到该关键车 ID 下的字典中
            kid_traj.update(interaction_data)
            traj_container[kid] = kid_traj

        print("\n" + "-" * 40)
        print(f"关键车轨迹提取与交互计算完成")
        print(f"已提取关键车总数: {len(self.key_ids)}")
        print("-" * 40)

        return updated_data


# =================================================================
# 功能类 5：危险轨迹截取类 (KeyVehicleTrajectoryClipper) - 原有功能
# =================================================================
class KeyVehicleTrajectoryClipper:
    """根据用户指定的 ID、间隔和范围，截取关键车及对应自车的危险轨迹片段"""

    def __init__(self, scenario_data):
        self.data = scenario_data
        self.proc_res = self.data.get('processing-result', {})  # 提前提取以方便后续使用
        self.details = self.proc_res.get('KeyCarID_Select', {}).get('details', np.empty((0, 3)))
        self.key_ids = self.proc_res.get('KeyCarID_Select', {}).get('key_ids', [])
        self.full_trajectories = self.proc_res.get('KeyVehicle_Trajectory', {})

    def show_candidate_info(self):
        """在控制台格式化显示所有候选关键车信息"""
        if self.details.size == 0:
            print("⚠无关键车数据可供截取。")
            return False

        print("\n" + "=" * 60)
        print(f"{'关键车 ID':<15} | {'最低风险分':<12} | {'起始危险帧':<10}")

        min_score = np.min(self.details[:, 1])
        best_candidates = []

        for row in self.details:
            vid, score, frame = int(row[0]), int(row[1]), int(row[2])
            mark = "[最关键]" if score == min_score else ""
            if score == min_score:
                best_candidates.append(str(vid))
            print(f"{vid:<17} | {score:<15} | {frame:<12} {mark}")

        print("-" * 60)
        print(f"建议优先处理 ID: {', '.join(best_candidates)}")
        print("=" * 60)
        return True

    def run(self, time_interval, n_back, n_forward):
        if not self.show_candidate_info():
            return self.data

        # 1. 用户输入目标 ID
        target_input = input(">>> 请输入要截取的关键车 ID: ").strip()

        target_id = None
        for kid in self.key_ids:
            if str(kid) == target_input:
                target_id = kid
                break

        if target_id is None:
            print(f"输入错误：ID {target_input} 不在关键车列表中。")
            return self.data

        # 2. 定位该车的危险锚点帧
        idx_in_details = np.where(self.details[:, 0].astype(int) == int(target_id))[0][0]
        anchor_f = int(self.details[idx_in_details, 2])

        # 3. 计算采样步长
        step = int(round(time_interval / 0.1))

        # 获取原始轨迹全量字典
        raw_dict = self.full_trajectories.get(target_id, {})
        if not raw_dict:
            print(f"错误：未找到 ID {target_id} 的轨迹数据。")
            return self.data

        # 4. 执行多变量自适应切片
        clipped_trajectory = {}
        actual_back_count = 0
        actual_forward_count = 0
        total_frames = 0
        final_indices = []  # 提取出来，供自车使用

        for var_name, var_data in raw_dict.items():
            current_len = len(var_data) if isinstance(var_data, list) else var_data.shape[0]

            back_indices = [anchor_f - i * step for i in range(1, n_back + 1)]
            forward_indices = [anchor_f + i * step for i in range(1, n_forward + 1)]

            valid_back = sorted([i for i in back_indices if 0 <= i < current_len])
            valid_forward = sorted([i for i in forward_indices if 0 <= i < current_len])

            actual_back_count = len(valid_back)
            actual_forward_count = len(valid_forward)

            # 合并总索引序列
            final_indices = valid_back + [anchor_f] + valid_forward
            total_frames = len(final_indices)

            if isinstance(var_data, list):
                clipped_trajectory[var_name] = [var_data[i] for i in final_indices]
            else:
                clipped_trajectory[var_name] = var_data[final_indices]

        # --- [截取观察车对应帧下的风险轨迹] ---
        obs_id = self.proc_res.get('obs_id')
        obs_car_trajectory_risk = {}
        # 索引 Root > object_tracks > obsID > state
        ego_raw_state = self.data.get('object_tracks', {}).get(obs_id, {}).get('state', {})

        if ego_raw_state:
            for var_name, var_data in ego_raw_state.items():
                if isinstance(var_data, list):
                    obs_car_trajectory_risk[var_name] = [var_data[i] for i in final_indices]
                else:
                    # 确保是 Numpy 数组时使用索引提取
                    obs_car_trajectory_risk[var_name] = var_data[final_indices]

        # 5. 封装结果
        updated_data = copy.deepcopy(self.data)
        # 关键车风险轨迹数据
        updated_data['processing-result']['KeyVehicleTrajectory_Risk'] = {
            'selected_id': target_id,
            'anchor_frame_raw': anchor_f,
            'sampling_config': {
                'interval': time_interval,
                'actual_n_back': actual_back_count,
                'actual_n_forward': actual_forward_count,
                'total_frames': total_frames
            },
            'clipped_trajectory': clipped_trajectory
        }
        # 新增项：观察车风险轨迹数据 (ObsCarTrajectory_risk)
        updated_data['processing-result']['ObsCarTrajectory_risk'] = obs_car_trajectory_risk

        print("\n" + "-" * 40)
        print(f"危险轨迹片段截取完成！")
        print(f"ID: {target_id} (关键车) & ID: {obs_id} (观察车)")
        print(f"锚点帧: {anchor_f} | 总输出帧数: {total_frames}")
        print(f"已同步提取自车轨迹至 processing-result > ObsCarTrajectory_risk")
        print("-" * 40)

        return updated_data


# =================================================================
# 功能类 6：轨迹文本化类 (TrajectoryTextualizer) - 原有功能
# =================================================================
class TrajectoryTextualizer:
    """轨迹文本化功能类：直接处理功能类5截取的同步轨迹，生成结构化 Prompt"""

    def __init__(self, scenario_data):
        self.data = scenario_data
        proc_res = self.data.get('processing-result', {})

        # 1. 获取关键车风险轨迹数据节点
        self.key_risk_node = proc_res.get('KeyVehicleTrajectory_Risk', {})
        if not self.key_risk_node:
            raise ValueError("未发现关键车危险轨迹截取数据，请先执行功能类 5。")

        # 2. 获取自车风险轨迹数据节点
        self.obs_traj = proc_res.get('ObsCarTrajectory_risk', {})
        if not self.obs_traj:
            raise ValueError("未发现自车风险轨迹数据。")

        # 3. 基础变量初始化
        self.key_traj = self.key_risk_node['clipped_trajectory']
        self.tar_id = self.key_risk_node['selected_id']
        self.ego_id = proc_res.get('obs_id')
        self.config = self.key_risk_node['sampling_config']

    def _infer_traffic_motion(self, i):
        """基于交通工程术语生成状态动作描述（保持逻辑不变）"""
        beta = self.key_traj['relative_angle'][i]
        rel_dir = self.key_traj['relative_direction'][i]
        rv_long = self.key_traj['rel_velocity_long'][i]
        rv_lat = self.key_traj['rel_velocity_lat'][i]
        rd_lat = self.key_traj['rel_dist_lat'][i]
        abs_beta = abs(beta)

        if abs_beta < np.pi / 4:
            topology = "同向追随"
        elif abs_beta > 3 * np.pi / 4:
            topology = "对向对冲"
        else:
            topology = "侧向交叉"

        status_header = f"关键车 {self.tar_id} 处于自车 {rel_dir}，当前表现为{topology}交互态势。"

        intent_desc = ""
        if topology == "同向追随":
            if rv_long < -0.5:
                intent_desc += "其纵向间距正在快速缩减，表现为后方高速逼近。" if "Rear" in rel_dir else "自车正快速闭合与前方障碍物的纵向间距。"
            if (rd_lat > 0 and rv_lat < -0.3) or (rd_lat < 0 and rv_lat > 0.3):
                intent_desc += "同时，检测到明显的横向侵入行为，存在切入侧碰风险。"
        elif topology == "对向对冲":
            if rv_long < -2.0:
                intent_desc += "两车相对闭合速度极高，存在正面碰撞及车道偏离冲突。"
            else:
                intent_desc += "当前处于对向错车阶段，横向安全净距需重点关注。"
        elif topology == "侧向交叉":
            if abs(rv_lat) > 0.8:
                intent_desc += "关键车正在横穿自车行驶路径，冲突点存在时空重叠风险。"
            else:
                intent_desc += "侧向位置相对稳定，但处于交叉冲突敏感区域。"

        return f"{status_header} {intent_desc}"

    def run(self):
        """全量遍历同步轨迹，生成文本描述"""
        # interval 取自 sampling_config
        t_interval = self.config['interval']
        total_frames = self.config['total_frames']
        prompt_list = []

        # 全量遍历，不再使用外部 sample_step
        for i in range(total_frames):
            t_curr = i * t_interval

            # 自车数据提取 (来自 ObsCarTrajectory_risk)
            ego_gc = self.obs_traj['global_center'][i]
            ego_lv = self.obs_traj['local_velocity'][i]
            ego_la = self.obs_traj['local_acceleration'][i]
            ego_h = self.obs_traj['heading'][i]

            # 关键车数据提取 (来自 clipped_trajectory)
            key_rd = self.key_traj['relative_direction'][i]
            key_ra = self.key_traj['relative_angle'][i]

            motion_trend = self._infer_traffic_motion(i)

            # 填充文本模板
            frame_text = (
                f"在 {t_curr:.2f}s 时刻，\n"
                f"   自车 {self.ego_id} 位置：({ego_gc[0]:.2f}, {ego_gc[1]:.2f})，"
                f"纵向速度 {ego_lv[0]:.2f}，横向速度 {ego_lv[1]:.2f}，"
                f"纵向加速度 {ego_la[0]:.2f}，横向加速度 {ego_la[1]:.2f}，"
                f"车身方向：{ego_h:.3f}。\n"
                f"   关键车 {self.tar_id} 在自车的 {key_rd} 方向，"
                f"相对于自车的方向角：{key_ra:.3f}，"
                f"横向相对距离：{self.key_traj['rel_dist_lat'][i]:.2f}，纵向相对距离：{self.key_traj['rel_dist_long'][i]:.2f}，"
                f"横向相对速度：{self.key_traj['rel_velocity_lat'][i]:.2f}，纵向相对速度：{self.key_traj['rel_velocity_long'][i]:.2f}，"
                f"横向相对加速度：{self.key_traj['rel_accel_lat'][i]:.2f}，纵向相对加速度：{self.key_traj['rel_accel_long'][i]:.2f}，"
                f"纵向ttc：{self.key_traj['ttc_long'][i]:.2f}，横向ttc：{self.key_traj['ttc_lat'][i]:.2f}。"
                f"运动趋势：{motion_trend}\n"
            )
            prompt_list.append(frame_text)

        final_prompt = "\n".join(prompt_list)

        # --- [新建字典存储生成的提示词] ---
        updated_data = copy.deepcopy(self.data)
        updated_data['processing-result']['LLM_Prompt_Result'] = {
            'final_prompt': final_prompt,
            'timestamp_count': total_frames,
            'target_vehicle_id': self.tar_id
        }

        return updated_data, final_prompt


# =================================================================
# 【新增】Step 3: 轨迹特征提取与向量化 (TrajectoryFeatureExtractor)
# =================================================================
@dataclass
class TrajectoryFeatures:
    """轨迹特征数据结构"""
    # 速度统计特征 (3维)
    mean_speed: float
    max_speed: float
    speed_std: float

    # 加速度统计特征 (2维)
    mean_accel: float
    max_accel: float

    # 横向运动特征 (2维)
    max_lateral_offset: float
    lateral_std: float

    # 安全指标特征 (2维)
    min_ttc: float
    mean_ttc: float

    # 几何特征 (2维)
    trajectory_length: float
    max_curvature: float

    def to_vector(self) -> np.ndarray:
        """转换为12维特征向量"""
        return np.array([
            self.mean_speed, self.max_speed, self.speed_std,
            self.mean_accel, self.max_accel,
            self.max_lateral_offset, self.lateral_std,
            self.min_ttc, self.mean_ttc,
            self.trajectory_length, self.max_curvature
        ])

    @classmethod
    def from_trajectory(cls, trajectory: Dict) -> 'TrajectoryFeatures':
        """从轨迹字典提取特征"""
        # 提取速度信息
        velocity = trajectory.get('local_velocity', np.array([[0, 0]]))
        speed = np.linalg.norm(velocity, axis=1)

        # 提取加速度信息
        acceleration = trajectory.get('local_acceleration', np.array([[0, 0]]))
        accel_mag = np.linalg.norm(acceleration, axis=1)

        # 提取位置信息
        global_center = trajectory.get('global_center', np.array([[0, 0, 0]]))

        # 计算轨迹长度
        diffs = np.diff(global_center[:, :2], axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        trajectory_length = np.sum(segment_lengths)

        # 计算曲率
        heading = trajectory.get('heading', np.array([0]))
        heading_diff = np.diff(heading)
        curvature = np.abs(heading_diff) / (segment_lengths + 1e-6)

        # TTC信息
        ttc_long = trajectory.get('ttc_long', np.array([np.inf]))
        ttc_lat = trajectory.get('ttc_lat', np.array([np.inf]))
        valid_ttc = np.minimum(ttc_long, ttc_lat)
        valid_ttc = valid_ttc[np.isfinite(valid_ttc)]

        # 横向偏移（相对距离）
        rel_dist_lat = trajectory.get('rel_dist_lat', np.array([0]))

        return cls(
            mean_speed=float(np.mean(speed)),
            max_speed=float(np.max(speed)),
            speed_std=float(np.std(speed)),
            mean_accel=float(np.mean(accel_mag)),
            max_accel=float(np.max(accel_mag)),
            max_lateral_offset=float(np.max(np.abs(rel_dist_lat))),
            lateral_std=float(np.std(rel_dist_lat)),
            min_ttc=float(np.min(valid_ttc)) if len(valid_ttc) > 0 else np.inf,
            mean_ttc=float(np.mean(valid_ttc)) if len(valid_ttc) > 0 else np.inf,
            trajectory_length=float(trajectory_length),
            max_curvature=float(np.max(curvature)) if len(curvature) > 0 else 0.0
        )


class TrajectoryFeatureExtractor:
    """
    【Step 3】轨迹特征提取与向量化
    提取12维统计特征，用于RAG检索
    """

    def __init__(self, scenario_data: Dict):
        self.data = scenario_data
        self.proc_res = self.data.get('processing-result', {})

    def extract_features(self, trajectory_key: str = 'KeyVehicleTrajectory_Risk') -> Dict[str, np.ndarray]:
        """
        提取所有关键车轨迹的特征向量

        Returns:
            Dict[str, np.ndarray]: {vehicle_id: 12维特征向量}
        """
        trajectory_data = self.proc_res.get(trajectory_key, {})
        clipped_traj = trajectory_data.get('clipped_trajectory', {})

        if not clipped_traj:
            return {}

        features = TrajectoryFeatures.from_trajectory(clipped_traj)

        # 存储特征到processing-result
        updated_data = copy.deepcopy(self.data)
        if 'TrajectoryFeatures' not in updated_data['processing-result']:
            updated_data['processing-result']['TrajectoryFeatures'] = {}

        vehicle_id = trajectory_data.get('selected_id', 'unknown')
        updated_data['processing-result']['TrajectoryFeatures'][vehicle_id] = {
            'vector': features.to_vector().tolist(),
            'details': {
                'mean_speed': features.mean_speed,
                'max_speed': features.max_speed,
                'speed_std': features.speed_std,
                'mean_accel': features.mean_accel,
                'max_accel': features.max_accel,
                'max_lateral_offset': features.max_lateral_offset,
                'lateral_std': features.lateral_std,
                'min_ttc': features.min_ttc,
                'mean_ttc': features.mean_ttc,
                'trajectory_length': features.trajectory_length,
                'max_curvature': features.max_curvature
            }
        }

        print("\n" + "-" * 40)
        print(f"轨迹特征提取完成")
        print(f"车辆ID: {vehicle_id}")
        print(f"12维特征向量: {features.to_vector()}")
        print("-" * 40)

        return updated_data, features


# =================================================================
# 【新增】Step 4: LLM轨迹变异生成 (LLMTrajectoryGenerator)
# =================================================================
@dataclass
class LLMConfig:
    """LLM配置参数"""
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4096
    danger_type: str = "rear_end"  # rear_end, cut_in, head_on, side_swipe
    variation_strength: float = 0.5  # 变异强度 0-1


class LLMTrajectoryGenerator:
    """
    【Step 4】LLM轨迹变异生成
    基于提示词生成危险轨迹变体
    """

    # 危险类型模板
    DANGER_TYPE_TEMPLATES = {
        "rear_end": {
            "description": "追尾碰撞风险",
            "modification_focus": "增加后方车辆的接近速度，缩短车头时距",
            "constraints": ["纵向加速度 < 6 m/s²", "保持车道内行驶", "避免横向偏移过大"]
        },
        "cut_in": {
            "description": "切入碰撞风险",
            "modification_focus": "增加横向切入速度和幅度，减少与自车的横向间距",
            "constraints": ["横向速度 < 3 m/s", "切入角度 < 45°", "避免与自车侧向碰撞"]
        },
        "head_on": {
            "description": "对向碰撞风险",
            "modification_focus": "增加对向车辆的接近速度，偏离正常行驶轨迹",
            "constraints": ["相对速度 < 30 m/s", "偏离角度 < 15°", "避免直接正面碰撞"]
        },
        "side_swipe": {
            "description": "侧面剐蹭风险",
            "modification_focus": "减少横向间距，增加并行行驶时的相对速度",
            "constraints": ["横向间距 > 0.5m", "横向相对速度 < 2 m/s", "避免直接接触"]
        }
    }

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self.generated_trajectories = []

    def build_system_prompt(self) -> str:
        """构建系统提示词"""
        danger_info = self.DANGER_TYPE_TEMPLATES.get(
            self.config.danger_type,
            self.DANGER_TYPE_TEMPLATES["rear_end"]
        )

        return f"""你是一个专业的自动驾驶危险场景生成专家。
你的任务是基于给定的真实驾驶场景，生成交互车的危险轨迹变体。

【当前危险类型】{danger_info['description']}
【修改重点】{danger_info['modification_focus']}

【物理约束】
{chr(10).join(['- ' + c for c in danger_info['constraints']])}
- 纵向加速度 < 6 m/s²
- 纵向减速度 < 8 m/s²
- 横向加速度 < 4 m/s²
- 最大速度 < 35 m/s (126 km/h)
- 车辆尺寸保持不变

【生成要求】
1. 保持自车轨迹不变，只修改交互车轨迹
2. 生成的轨迹必须满足上述物理约束
3. 修改后的轨迹应该与原始轨迹有一定连续性
4. 危险程度应该逐步增加，而非突然发生
5. 输出格式必须是结构化的JSON

【输出格式】
```json
{{
    "generated_trajectory": [
        {{
            "timestamp": 0.0,
            "global_center": [x, y, z],
            "heading": 0.0,
            "local_velocity": [vx, vy],
            "local_acceleration": [ax, ay],
            "size": [length, width, height]
        }},
        ...
    ],
    "modification_summary": "简要说明对原始轨迹的修改",
    "danger_level": "high/medium/low",
    "confidence": 0.85
}}
```
"""

    def build_user_prompt(self, original_prompt: str, target_frames: int) -> str:
        """构建用户提示词"""
        danger_info = self.DANGER_TYPE_TEMPLATES.get(
            self.config.danger_type,
            self.DANGER_TYPE_TEMPLATES["rear_end"]
        )

        return f"""【原始轨迹信息】
{original_prompt}

【生成要求】
- 危险类型: {danger_info['description']}
- 目标帧数: {target_frames}
- 变异强度: {self.config.variation_strength * 100}%

请基于上述信息，生成交互车的危险轨迹变体。
输出必须是有效的JSON格式，包含完整的轨迹点序列。
"""

    def generate_trajectory(self, scenario_data: Dict, num_variants: int = 3) -> Tuple[Dict, List[Dict]]:
        """
        生成多个轨迹变体

        Args:
            scenario_data: 场景数据字典
            num_variants: 生成变体数量

        Returns:
            Tuple[updated_data, list_of_generated_trajectories]
        """
        proc_res = scenario_data.get('processing-result', {})
        prompt_result = proc_res.get('LLM_Prompt_Result', {})
        original_prompt = prompt_result.get('final_prompt', '')
        target_frames = prompt_result.get('timestamp_count', 20)

        if not original_prompt:
            raise ValueError("未找到LLM提示词，请先执行TrajectoryTextualizer")

        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(original_prompt, target_frames)

        # 模拟生成的轨迹变体（实际项目中应调用LLM API）
        generated_variants = []

        for i in range(num_variants):
            variant = self._simulate_llm_generation(
                scenario_data,
                variant_id=i,
                total_frames=target_frames
            )
            generated_variants.append(variant)

        # 存储生成结果
        updated_data = copy.deepcopy(scenario_data)
        if 'GeneratedTrajectories' not in updated_data['processing-result']:
            updated_data['processing-result']['GeneratedTrajectories'] = []

        updated_data['processing-result']['GeneratedTrajectories'] = generated_variants

        # 存储LLM调用信息
        updated_data['processing-result']['LLM_Generation_Info'] = {
            'model': self.config.model,
            'temperature': self.config.temperature,
            'danger_type': self.config.danger_type,
            'variation_strength': self.config.variation_strength,
            'num_variants': num_variants,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'timestamp': datetime.now().isoformat()
        }

        print("\n" + "=" * 40)
        print(f"LLM轨迹生成完成")
        print(f"生成变体数量: {num_variants}")
        print(f"危险类型: {self.config.danger_type}")
        print(f"变异强度: {self.config.variation_strength}")
        print("=" * 40)

        return updated_data, generated_variants

    def _simulate_llm_generation(self, scenario_data: Dict, variant_id: int, total_frames: int) -> Dict:
        """
        模拟LLM生成轨迹（实际项目中应替换为真实LLM调用）

        Returns:
            生成的轨迹变体字典
        """
        proc_res = scenario_data.get('processing-result', {})
        key_risk_node = proc_res.get('KeyVehicleTrajectory_Risk', {})
        original_traj = key_risk_node.get('clipped_trajectory', {})

        if not original_traj:
            raise ValueError("未找到原始轨迹数据")

        # 获取原始轨迹数据
        global_center = original_traj.get('global_center', np.array([[0, 0, 0]]))
        heading = original_traj.get('heading', np.array([0]))
        velocity = original_traj.get('local_velocity', np.array([[0, 0]]))
        acceleration = original_traj.get('local_acceleration', np.array([[0, 0]]))
        size = original_traj.get('size', np.array([[4.5, 2.0, 1.5]]))

        # 生成变体（添加随机扰动）
        generated_points = []
        variation_factor = self.config.variation_strength * (0.8 + variant_id * 0.2)

        for i in range(total_frames):
            # 根据危险类型添加扰动
            if self.config.danger_type == "rear_end":
                # 追尾：增加纵向速度
                vel_perturb = [velocity[i][0] * (1 + variation_factor * 0.3), velocity[i][1]]
                pos_perturb = [
                    global_center[i][0] + variation_factor * (i * 0.5),
                    global_center[i][1],
                    global_center[i][2]
                ]
            elif self.config.danger_type == "cut_in":
                # 切入：增加横向偏移
                lat_offset = variation_factor * 2.0 * np.sin(i * np.pi / total_frames)
                vel_perturb = [velocity[i][0], velocity[i][1] + variation_factor * 1.5]
                pos_perturb = [
                    global_center[i][0],
                    global_center[i][1] + lat_offset,
                    global_center[i][2]
                ]
            else:
                # 默认：随机扰动
                vel_perturb = [
                    velocity[i][0] * (1 + variation_factor * 0.1),
                    velocity[i][1] + variation_factor * 0.5
                ]
                pos_perturb = [
                    global_center[i][0] + variation_factor * (i * 0.2),
                    global_center[i][1] + variation_factor * 0.3,
                    global_center[i][2]
                ]

            point = {
                "timestamp": i * 0.1,
                "global_center": [float(x) for x in pos_perturb],
                "heading": float(heading[i]) if i < len(heading) else 0.0,
                "local_velocity": [float(v) for v in vel_perturb],
                "local_acceleration": [float(a) for a in acceleration[i]] if i < len(acceleration) else [0.0, 0.0],
                "size": [float(s) for s in size[i]] if i < len(size) else [4.5, 2.0, 1.5]
            }
            generated_points.append(point)

        return {
            "variant_id": variant_id,
            "generated_trajectory": generated_points,
            "modification_summary": f"基于{self.config.danger_type}危险类型的轨迹变体 #{variant_id + 1}",
            "danger_level": "high" if variation_factor > 0.7 else "medium" if variation_factor > 0.4 else "low",
            "confidence": 0.8 + variant_id * 0.05,
            "parent_trajectory_id": key_risk_node.get('selected_id', 'unknown')
        }


# =================================================================
# 【新增】Step 5: 物理约束验证 (PhysicsValidator)
# =================================================================
@dataclass
class ValidationResult:
    """验证结果数据结构"""
    is_valid: bool
    failed_rules: List[str]
    warnings: List[str]
    metrics: Dict[str, float]


class PhysicsValidator:
    """
    【Step 5】物理约束验证（硬过滤）
    检查生成的轨迹是否符合物理约束
    """

    # 约束规则定义
    CONSTRAINTS = {
        'max_longitudinal_accel': {'threshold': 6.0, 'unit': 'm/s²', 'severity': 'hard'},
        'max_longitudinal_decel': {'threshold': 8.0, 'unit': 'm/s²', 'severity': 'hard'},
        'max_lateral_accel': {'threshold': 4.0, 'unit': 'm/s²', 'severity': 'hard'},
        'max_speed': {'threshold': 35.0, 'unit': 'm/s', 'severity': 'hard'},
        'max_jerk': {'threshold': 10.0, 'unit': 'm/s³', 'severity': 'soft'},
        'curvature_continuity': {'threshold': 0.1, 'unit': 'rad/m', 'severity': 'soft'},
    }

    def __init__(self, constraints: Dict = None):
        self.constraints = constraints or self.CONSTRAINTS

    def validate_trajectory(self, trajectory: List[Dict]) -> ValidationResult:
        """
        验证单条轨迹

        Args:
            trajectory: 轨迹点列表

        Returns:
            ValidationResult: 验证结果
        """
        failed_rules = []
        warnings = []
        metrics = {}

        # 提取轨迹数据
        velocities = np.array([p['local_velocity'] for p in trajectory])
        accelerations = np.array([p['local_acceleration'] for p in trajectory])
        headings = np.array([p['heading'] for p in trajectory])
        positions = np.array([p['global_center'] for p in trajectory])

        # 计算速度大小
        speeds = np.linalg.norm(velocities, axis=1)
        metrics['max_speed'] = float(np.max(speeds))
        metrics['mean_speed'] = float(np.mean(speeds))

        # 计算加速度大小
        long_accels = accelerations[:, 0]
        lat_accels = accelerations[:, 1]

        metrics['max_long_accel'] = float(np.max(long_accels))
        metrics['min_long_accel'] = float(np.min(long_accels))
        metrics['max_lat_accel'] = float(np.max(np.abs(lat_accels)))

        # 检查纵向加速度
        if metrics['max_long_accel'] > self.constraints['max_longitudinal_accel']['threshold']:
            failed_rules.append(
                f"纵向加速度超限: {metrics['max_long_accel']:.2f} > "
                f"{self.constraints['max_longitudinal_accel']['threshold']} m/s²"
            )

        # 检查纵向减速度
        if abs(metrics['min_long_accel']) > self.constraints['max_longitudinal_decel']['threshold']:
            failed_rules.append(
                f"纵向减速度超限: {abs(metrics['min_long_accel']):.2f} > "
                f"{self.constraints['max_longitudinal_decel']['threshold']} m/s²"
            )

        # 检查横向加速度
        if metrics['max_lat_accel'] > self.constraints['max_lateral_accel']['threshold']:
            failed_rules.append(
                f"横向加速度超限: {metrics['max_lat_accel']:.2f} > "
                f"{self.constraints['max_lateral_accel']['threshold']} m/s²"
            )

        # 检查最大速度
        if metrics['max_speed'] > self.constraints['max_speed']['threshold']:
            failed_rules.append(
                f"速度超限: {metrics['max_speed']:.2f} > "
                f"{self.constraints['max_speed']['threshold']} m/s"
            )

        # 计算Jerk（加速度变化率）
        if len(accelerations) > 1:
            jerk = np.diff(accelerations, axis=0) / 0.1  # 假设0.1s间隔
            jerk_mag = np.linalg.norm(jerk, axis=1)
            metrics['max_jerk'] = float(np.max(jerk_mag))

            if metrics['max_jerk'] > self.constraints['max_jerk']['threshold']:
                warnings.append(
                    f"Jerk过高: {metrics['max_jerk']:.2f} > "
                    f"{self.constraints['max_jerk']['threshold']} m/s³"
                )

        # 检查曲率连续性
        if len(headings) > 1:
            heading_diff = np.diff(headings)
            position_diff = np.diff(positions[:, :2], axis=0)
            distances = np.linalg.norm(position_diff, axis=1) + 1e-6
            curvature = np.abs(heading_diff) / distances

            metrics['max_curvature'] = float(np.max(curvature))
            metrics['curvature_std'] = float(np.std(curvature))

            if metrics['curvature_std'] > self.constraints['curvature_continuity']['threshold']:
                warnings.append(
                    f"曲率变化不连续: std={metrics['curvature_std']:.3f}"
                )

        is_valid = len(failed_rules) == 0

        return ValidationResult(
            is_valid=is_valid,
            failed_rules=failed_rules,
            warnings=warnings,
            metrics=metrics
        )

    def validate_all(self, scenario_data: Dict) -> Tuple[Dict, List[Dict]]:
        """
        验证所有生成的轨迹

        Returns:
            Tuple[updated_data, validated_results]
        """
        proc_res = scenario_data.get('processing-result', {})
        generated_trajs = proc_res.get('GeneratedTrajectories', [])

        if not generated_trajs:
            raise ValueError("未找到生成的轨迹，请先执行LLMTrajectoryGenerator")

        validation_results = []
        passed_trajectories = []

        for traj_data in generated_trajs:
            trajectory = traj_data.get('generated_trajectory', [])
            result = self.validate_trajectory(trajectory)

            validation_entry = {
                'variant_id': traj_data.get('variant_id'),
                'is_valid': result.is_valid,
                'failed_rules': result.failed_rules,
                'warnings': result.warnings,
                'metrics': result.metrics,
                'danger_level': traj_data.get('danger_level'),
                'confidence': traj_data.get('confidence')
            }

            validation_results.append(validation_entry)

            if result.is_valid:
                passed_trajectories.append(traj_data)

        # 存储验证结果
        updated_data = copy.deepcopy(scenario_data)
        updated_data['processing-result']['PhysicsValidation'] = {
            'validation_results': validation_results,
            'passed_count': len(passed_trajectories),
            'failed_count': len(validation_results) - len(passed_trajectories),
            'timestamp': datetime.now().isoformat()
        }

        # 更新通过的轨迹
        updated_data['processing-result']['ValidatedTrajectories'] = passed_trajectories

        print("\n" + "=" * 40)
        print(f"物理约束验证完成")
        print(f"通过验证: {len(passed_trajectories)}/{len(validation_results)}")
        print(f"失败数量: {len(validation_results) - len(passed_trajectories)}")
        print("=" * 40)

        return updated_data, validation_results


# =================================================================
# 【新增】Step 6-7: RAG检索与评估 (RAGEvaluator) - 预留数据库接口
# =================================================================
@dataclass
class SimilarCase:
    """相似案例数据结构"""
    case_id: str
    similarity: float
    label: str
    danger_type: str
    metadata: Dict


class RAGEvaluator:
    """
    【Step 6-7】RAG检索相似案例 + LLM合理性评估

    数据库预留接口说明：
    - 本类预留了与向量数据库（ChromaDB/Milvus/Pinecone等）的交互接口
    - 实际使用时需要实现具体的数据库连接和查询逻辑
    """

    def __init__(self, db_config: Dict = None):
        """
        初始化RAG评估器

        Args:
            db_config: 数据库配置
                {
                    'db_type': 'chromadb',  # 数据库类型
                    'collection_name': 'trajectory_cases',
                    'embedding_dim': 12,
                    'host': 'localhost',
                    'port': 8000,
                    'persist_directory': './chroma_db'
                }
        """
        self.db_config = db_config or {
            'db_type': 'chromadb',
            'collection_name': 'trajectory_cases',
            'embedding_dim': 12,
            'persist_directory': './chroma_db'
        }
        self.db_client = None
        self.collection = None

    def connect_database(self) -> bool:
        """
        连接向量数据库

        Returns:
            bool: 连接是否成功

        预留接口说明：
        - 实际项目中需要根据db_type实现具体的数据库连接逻辑
        - 支持ChromaDB、Milvus、Pinecone等主流向量数据库
        """
        db_type = self.db_config.get('db_type', 'chromadb')

        try:
            if db_type == 'chromadb':
                # 预留：ChromaDB连接
                # import chromadb
                # self.db_client = chromadb.Client(
                #     chromadb.config.Settings(
                #         persist_directory=self.db_config['persist_directory']
                #     )
                # )
                # self.collection = self.db_client.get_or_create_collection(
                #     name=self.db_config['collection_name']
                # )
                print(f"[预留] 连接到 {db_type} 数据库")

            elif db_type == 'milvus':
                # 预留：Milvus连接
                # from pymilvus import connections, Collection
                # connections.connect(
                #     alias="default",
                #     host=self.db_config['host'],
                #     port=self.db_config['port']
                # )
                # self.collection = Collection(self.db_config['collection_name'])
                print(f"[预留] 连接到 {db_type} 数据库")

            elif db_type == 'pinecone':
                # 预留：Pinecone连接
                # import pinecone
                # pinecone.init(api_key=self.db_config.get('api_key'))
                # self.collection = pinecone.Index(self.db_config['collection_name'])
                print(f"[预留] 连接到 {db_type} 数据库")

            return True

        except Exception as e:
            print(f"数据库连接失败: {e}")
            return False

    def query_similar_cases(self, trajectory_vector: np.ndarray, top_k: int = 5) -> List[SimilarCase]:
        """
        检索相似案例

        Args:
            trajectory_vector: 12维轨迹特征向量
            top_k: 返回最相似的K个案例

        Returns:
            List[SimilarCase]: 相似案例列表

        预留接口说明：
        - 实际项目中需要实现具体的数据库查询逻辑
        - 返回的案例应包含案例ID、相似度、标签、危险类型等元数据
        """
        # 模拟相似案例检索（实际项目中应查询真实数据库）
        similar_cases = []

        # 预留：实际数据库查询代码
        # if self.collection:
        #     results = self.collection.query(
        #         query_embeddings=[trajectory_vector.tolist()],
        #         n_results=top_k
        #     )
        #     for i, (id, distance, metadata) in enumerate(zip(
        #         results['ids'][0],
        #         results['distances'][0],
        #         results['metadatas'][0]
        #     )):
        #         similar_cases.append(SimilarCase(
        #             case_id=id,
        #             similarity=1 - distance,  # 距离转相似度
        #             label=metadata.get('label', 'unknown'),
        #             danger_type=metadata.get('danger_type', 'unknown'),
        #             metadata=metadata
        #         ))

        # 模拟数据用于演示
        mock_cases = [
            SimilarCase(
                case_id=f"case_{i:04d}",
                similarity=0.85 - i * 0.1,
                label="reasonable" if i % 2 == 0 else "unreasonable",
                danger_type="rear_end" if i % 3 == 0 else "cut_in",
                metadata={"evaluated_by": "human", "confidence": 0.9}
            )
            for i in range(top_k)
        ]

        return mock_cases

    def llm_reasonableness_eval(self, trajectory: Dict, similar_cases: List[SimilarCase]) -> Dict:
        """
        LLM合理性评估

        Args:
            trajectory: 待评估轨迹
            similar_cases: 相似案例列表

        Returns:
            Dict: 评估结果

        预留接口说明：
        - 实际项目中应调用LLM API进行评估
        - 基于相似案例和轨迹特征，LLM判断轨迹是否合理
        """
        # 计算与相似案例的平均相似度
        avg_similarity = np.mean([c.similarity for c in similar_cases]) if similar_cases else 0
        max_similarity = max([c.similarity for c in similar_cases]) if similar_cases else 0

        # 统计相似案例中的合理/不合理比例
        reasonable_count = sum(1 for c in similar_cases if c.label == "reasonable")
        unreasonable_count = len(similar_cases) - reasonable_count

        # 基于相似度阈值判断（实际项目中应使用LLM）
        if max_similarity > 0.75:
            # 有高度相似的案例，参考其标签
            most_similar = similar_cases[0]
            is_reasonable = most_similar.label == "reasonable"
            confidence = most_similar.similarity
            reasoning = f"与已知{most_similar.label}案例高度相似(相似度{most_similar.similarity:.2f})"
        elif avg_similarity > 0.5:
            # 中度相似，综合判断
            is_reasonable = reasonable_count > unreasonable_count
            confidence = avg_similarity
            reasoning = f"综合{len(similar_cases)}个相似案例判断(合理:{reasonable_count}, 不合理:{unreasonable_count})"
        else:
            # 相似度低，需要LLM深度分析（预留）
            is_reasonable = True  # 默认通过，等待人工审核
            confidence = 0.5
            reasoning = "相似案例不足，需要人工审核"

        return {
            'is_reasonable': is_reasonable,
            'confidence': confidence,
            'reasoning': reasoning,
            'max_similarity': max_similarity,
            'avg_similarity': avg_similarity,
            'similar_cases_count': len(similar_cases),
            'similar_cases_summary': [
                {'id': c.case_id, 'similarity': c.similarity, 'label': c.label}
                for c in similar_cases[:3]
            ]
        }

    def evaluate_trajectories(self, scenario_data: Dict) -> Tuple[Dict, List[Dict]]:
        """
        评估所有通过物理验证的轨迹

        Returns:
            Tuple[updated_data, evaluation_results]
        """
        proc_res = scenario_data.get('processing-result', {})
        validated_trajs = proc_res.get('ValidatedTrajectories', [])

        if not validated_trajs:
            print("警告：未找到通过物理验证的轨迹")
            return scenario_data, []

        # 连接数据库
        self.connect_database()

        # 提取特征向量
        features_data = proc_res.get('TrajectoryFeatures', {})

        evaluation_results = []

        for traj_data in validated_trajs:
            variant_id = traj_data.get('variant_id')
            trajectory = traj_data.get('generated_trajectory', [])

            # 提取轨迹特征向量
            # 实际项目中应从features_data中获取或使用TrajectoryFeatureExtractor重新计算
            vehicle_id = traj_data.get('parent_trajectory_id', 'unknown')
            feature_info = features_data.get(vehicle_id, {})
            feature_vector = np.array(feature_info.get('vector', [0] * 12))

            # 检索相似案例
            similar_cases = self.query_similar_cases(feature_vector, top_k=5)
            max_similarity = max([c.similarity for c in similar_cases]) if similar_cases else 0

            # LLM合理性评估
            eval_result = self.llm_reasonableness_eval(traj_data, similar_cases)

            # 判断是否需要人工审核
            needs_human_review = (
                max_similarity < 0.7 or  # 相似度太低
                eval_result['confidence'] < 0.8  # 置信度不足
            )

            evaluation_entry = {
                'variant_id': variant_id,
                'evaluation': eval_result,
                'similar_cases': [
                    {'id': c.case_id, 'similarity': c.similarity, 'label': c.label}
                    for c in similar_cases
                ],
                'needs_human_review': needs_human_review,
                'timestamp': datetime.now().isoformat()
            }

            evaluation_results.append(evaluation_entry)

        # 存储评估结果
        updated_data = copy.deepcopy(scenario_data)
        updated_data['processing-result']['RAG_Evaluation'] = {
            'evaluation_results': evaluation_results,
            'needs_review_count': sum(1 for e in evaluation_results if e['needs_human_review']),
            'auto_approved_count': sum(1 for e in evaluation_results if not e['needs_human_review']),
            'db_config': self.db_config,
            'timestamp': datetime.now().isoformat()
        }

        print("\n" + "=" * 40)
        print(f"RAG评估完成")
        print(f"自动通过: {sum(1 for e in evaluation_results if not e['needs_human_review'])}")
        print(f"需要人工审核: {sum(1 for e in evaluation_results if e['needs_human_review'])}")
        print("=" * 40)

        return updated_data, evaluation_results

    def add_to_knowledge_base(self, trajectory_data: Dict, label: str, evaluated_by: str = "human") -> bool:
        """
        将轨迹添加到知识库

        Args:
            trajectory_data: 轨迹数据
            label: 标签（reasonable/unreasonable）
            evaluated_by: 评估来源

        Returns:
            bool: 是否添加成功

        预留接口说明：
        - 实际项目中需要实现具体的数据库插入逻辑
        """
        # 提取特征向量
        features = TrajectoryFeatures.from_trajectory(trajectory_data)
        vector = features.to_vector()

        # 生成唯一ID
        case_id = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(vector.tobytes()).hexdigest()[:8]}"

        # 预留：实际数据库插入代码
        # if self.collection:
        #     self.collection.add(
        #         ids=[case_id],
        #         embeddings=[vector.tolist()],
        #         metadatas=[{
        #             'label': label,
        #             'danger_type': trajectory_data.get('danger_type', 'unknown'),
        #             'evaluated_by': evaluated_by,
        #             'timestamp': datetime.now().isoformat()
        #         }],
        #         documents=[json.dumps(trajectory_data)]
        #     )

        print(f"[预留] 添加案例到知识库: {case_id}, 标签: {label}")
        return True


# =================================================================
# 【新增】Step 8: 人工审核接口 (HumanReviewInterface)
# =================================================================
class HumanReviewInterface:
    """
    【Step 8】人工审核与知识库更新接口
    提供人工审核的交互界面和数据记录
    """

    def __init__(self, scenario_data: Dict):
        self.data = scenario_data
        self.reviewed_trajectories = []

    def display_for_review(self, trajectory_data: Dict) -> Dict:
        """
        显示轨迹供人工审核

        Returns:
            审核结果
        """
        variant_id = trajectory_data.get('variant_id')
        danger_level = trajectory_data.get('danger_level')
        confidence = trajectory_data.get('confidence')

        print("\n" + "=" * 60)
        print(f"待审核轨迹 #{variant_id}")
        print(f"危险等级: {danger_level}")
        print(f"LLM置信度: {confidence}")
        print("-" * 60)

        # 显示轨迹概要
        trajectory = trajectory_data.get('generated_trajectory', [])
        if trajectory:
            start_pos = trajectory[0]['global_center']
            end_pos = trajectory[-1]['global_center']
            print(f"起点: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
            print(f"终点: ({end_pos[0]:.2f}, {end_pos[1]:.2f})")
            print(f"轨迹点数: {len(trajectory)}")

        print("-" * 60)

        # 模拟人工审核输入（实际项目中应提供GUI或Web界面）
        # 这里使用简单的命令行交互
        user_input = input("审核结果 [pass/reject/modify/skip]: ").strip().lower()

        review_result = {
            'variant_id': variant_id,
            'review_status': user_input,
            'label': None,
            'reviewer_notes': '',
            'timestamp': datetime.now().isoformat()
        }

        if user_input == 'pass':
            review_result['label'] = 'reasonable'
            review_result['final_status'] = 'approved'
        elif user_input == 'reject':
            review_result['label'] = 'unreasonable'
            review_result['final_status'] = 'rejected'
        elif user_input == 'modify':
            review_result['label'] = 'modified'
            review_result['final_status'] = 'pending'
            review_result['reviewer_notes'] = input("修改建议: ").strip()
        else:
            review_result['final_status'] = 'skipped'

        return review_result

    def run_review(self) -> Tuple[Dict, List[Dict]]:
        """
        执行完整的人工审核流程

        Returns:
            Tuple[updated_data, review_results]
        """
        proc_res = self.data.get('processing-result', {})
        rag_eval = proc_res.get('RAG_Evaluation', {})
        eval_results = rag_eval.get('evaluation_results', [])
        validated_trajs = proc_res.get('ValidatedTrajectories', [])

        review_results = []
        approved_trajectories = []

        for eval_result in eval_results:
            variant_id = eval_result.get('variant_id')
            needs_review = eval_result.get('needs_human_review', True)

            # 找到对应的轨迹数据
            traj_data = None
            for t in validated_trajs:
                if t.get('variant_id') == variant_id:
                    traj_data = t
                    break

            if not traj_data:
                continue

            # 如果需要人工审核或自动评估不确定
            if needs_review or eval_result.get('evaluation', {}).get('confidence', 0) < 0.8:
                review_result = self.display_for_review(traj_data)
            else:
                # 自动通过
                review_result = {
                    'variant_id': variant_id,
                    'review_status': 'auto_pass',
                    'label': 'reasonable' if eval_result['evaluation']['is_reasonable'] else 'unreasonable',
                    'final_status': 'approved' if eval_result['evaluation']['is_reasonable'] else 'rejected',
                    'timestamp': datetime.now().isoformat()
                }

            review_results.append(review_result)

            if review_result['final_status'] == 'approved':
                approved_trajectories.append(traj_data)

        # 存储审核结果
        updated_data = copy.deepcopy(self.data)
        updated_data['processing-result']['HumanReview'] = {
            'review_results': review_results,
            'approved_count': len(approved_trajectories),
            'rejected_count': len(review_results) - len(approved_trajectories),
            'approved_trajectories': approved_trajectories,
            'timestamp': datetime.now().isoformat()
        }

        print("\n" + "=" * 40)
        print(f"人工审核完成")
        print(f"通过: {len(approved_trajectories)}")
        print(f"拒绝: {len(review_results) - len(approved_trajectories)}")
        print("=" * 40)

        return updated_data, review_results


# =================================================================
# 【新增】Step 9: CARLA格式导出 (CarlaExporter)
# =================================================================
@dataclass
class CarlaState:
    """CARLA状态数据结构"""
    timestamp: float
    location: Tuple[float, float, float]
    rotation: Tuple[float, float, float]  # pitch, yaw, roll in degrees
    velocity: Tuple[float, float, float]


class CarlaExporter:
    """
    【Step 9】CARLA格式导出与闭环测试
    将审核通过的轨迹导出为CARLA可用格式
    """

    # CARLA地图映射
    CARLA_MAPS = [
        'Town01', 'Town02', 'Town03', 'Town04', 'Town05',
        'Town06', 'Town07', 'Town10HD'
    ]

    def __init__(self, output_dir: str = './carla_scenarios'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def waymo_to_carla_coords(self, waymo_pos: List[float]) -> Tuple[float, float, float]:
        """
        Waymo坐标系转CARLA坐标系
        Waymo: 右手坐标系，yaw弧度
        CARLA: 左手坐标系，yaw度
        """
        x, y, z = waymo_pos
        # CARLA使用左手坐标系，需要翻转Z轴
        return (x, -y, z)

    def waymo_to_carla_rotation(self, waymo_heading: float) -> Tuple[float, float, float]:
        """
        Waymo航向角转CARLA旋转角
        """
        # Waymo航向角转度并翻转（左手系）
        yaw = -np.degrees(waymo_heading)
        return (0.0, yaw, 0.0)  # pitch, yaw, roll

    def export_to_json(self, trajectory: List[Dict], scenario_id: str, vehicle_id: str) -> str:
        """
        导出为JSON格式

        Returns:
            输出文件路径
        """
        carla_trajectory = []

        for point in trajectory:
            waymo_pos = point['global_center']
            waymo_heading = point['heading']
            waymo_vel = point['local_velocity']

            carla_state = {
                'timestamp': point['timestamp'],
                'location': list(self.waymo_to_carla_coords(waymo_pos)),
                'rotation': list(self.waymo_to_carla_rotation(waymo_heading)),
                'velocity': [waymo_vel[0], -waymo_vel[1], 0.0],  # 翻转Y轴速度
                'size': point.get('size', [4.5, 2.0, 1.5])
            }
            carla_trajectory.append(carla_state)

        output_data = {
            'scenario_id': scenario_id,
            'vehicle_id': vehicle_id,
            'map_name': 'Town04',  # 默认地图，实际应根据场景选择
            'timestep': 0.1,
            'trajectory': carla_trajectory
        }

        output_path = os.path.join(
            self.output_dir,
            f"{scenario_id}_{vehicle_id}_carla.json"
        )

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        return output_path

    def export_to_xosc(self, trajectory: List[Dict], scenario_id: str, vehicle_id: str) -> str:
        """
        导出为OpenSCENARIO格式 (.xosc)

        预留接口：需要实现OpenSCENARIO XML生成逻辑
        """
        # 预留：实现OpenSCENARIO XML生成
        output_path = os.path.join(
            self.output_dir,
            f"{scenario_id}_{vehicle_id}_carla.xosc"
        )

        print(f"[预留] 导出OpenSCENARIO格式: {output_path}")
        return output_path

    def export_all(self, scenario_data: Dict) -> List[str]:
        """
        导出所有审核通过的轨迹

        Returns:
            List[str]: 导出的文件路径列表
        """
        proc_res = scenario_data.get('processing-result', {})
        human_review = proc_res.get('HumanReview', {})
        approved_trajs = human_review.get('approved_trajectories', [])
        scenario_id = scenario_data.get('scenario_id', 'unknown')

        if not approved_trajs:
            print("警告：没有通过审核的轨迹可导出")
            return []

        exported_files = []

        for traj_data in approved_trajs:
            vehicle_id = str(traj_data.get('variant_id', 'unknown'))
            trajectory = traj_data.get('generated_trajectory', [])

            # 导出JSON
            json_path = self.export_to_json(trajectory, scenario_id, vehicle_id)
            exported_files.append(json_path)

            # 预留：导出XOSC
            xosc_path = self.export_to_xosc(trajectory, scenario_id, vehicle_id)
            exported_files.append(xosc_path)

        # 存储导出信息
        updated_data = copy.deepcopy(scenario_data)
        updated_data['processing-result']['CarlaExport'] = {
            'exported_files': exported_files,
            'output_directory': self.output_dir,
            'export_count': len(approved_trajs),
            'timestamp': datetime.now().isoformat()
        }

        print("\n" + "=" * 40)
        print(f"CARLA导出完成")
        print(f"导出文件数: {len(exported_files)}")
        print(f"输出目录: {self.output_dir}")
        print("=" * 40)

        return exported_files


# =================================================================
# 【新增】主流程控制器 (PipelineController)
# =================================================================
class PipelineController:
    """
    全流程控制器
    整合Step 1-9的完整数据处理流程
    """

    def __init__(self, data_dir: str, scenario_id: str):
        self.data_dir = data_dir
        self.scenario_id = scenario_id
        self.data = None

    def run_step1(self) -> Dict:
        """Step 1: 场景加载与车辆选择"""
        print("\n" + "=" * 60)
        print("【Step 1】场景加载与车辆选择")
        print("=" * 60)

        loader = ScenarioDataLoader(self.data_dir, self.scenario_id)
        self.data = loader.load_data()
        obs_id = loader.select_observer()

        return self.data, obs_id

    def run_step2(self, obs_id: str, threshold: float = 2.0) -> Dict:
        """Step 2: 风险计算与危险片段定位"""
        print("\n" + "=" * 60)
        print("【Step 2】风险计算与危险片段定位")
        print("=" * 60)

        # 风险计算
        risk_calc = RiskCalculator(self.data)
        self.data = risk_calc.run(obs_id)

        # 关键车筛选
        key_filter = KeyVehicleFilter(self.data)
        self.data = key_filter.run(threshold)

        # 轨迹提取
        extractor = KeyVehicleTrajectoryExtractor(self.data)
        self.data = extractor.run()

        return self.data

    def run_step3_to_step6(self, time_interval: float = 0.3, n_back: int = 15, n_forward: int = 5) -> Dict:
        """Step 3-6: 危险片段截取、文本化、特征提取"""
        print("\n" + "=" * 60)
        print("【Step 3-6】危险片段截取、文本化、特征提取")
        print("=" * 60)

        # Step 5: 危险片段截取
        clipper = KeyVehicleTrajectoryClipper(self.data)
        self.data = clipper.run(time_interval, n_back, n_forward)

        # Step 6: 轨迹文本化
        textualizer = TrajectoryTextualizer(self.data)
        self.data, prompt_text = textualizer.run()

        # Step 3: 轨迹特征提取
        feature_extractor = TrajectoryFeatureExtractor(self.data)
        self.data, features = feature_extractor.extract_features()

        return self.data

    def run_step4(self, llm_config: LLMConfig = None, num_variants: int = 3) -> Dict:
        """Step 4: LLM轨迹变异生成"""
        print("\n" + "=" * 60)
        print("【Step 4】LLM轨迹变异生成")
        print("=" * 60)

        generator = LLMTrajectoryGenerator(llm_config)
        self.data, generated_trajs = generator.generate_trajectory(self.data, num_variants)

        return self.data

    def run_step5(self) -> Dict:
        """Step 5: 物理约束验证"""
        print("\n" + "=" * 60)
        print("【Step 5】物理约束验证（硬过滤）")
        print("=" * 60)

        validator = PhysicsValidator()
        self.data, validation_results = validator.validate_all(self.data)

        return self.data

    def run_step6_7(self, db_config: Dict = None) -> Dict:
        """Step 6-7: RAG检索相似案例 + LLM合理性评估"""
        print("\n" + "=" * 60)
        print("【Step 6-7】RAG检索与评估")
        print("=" * 60)

        rag_evaluator = RAGEvaluator(db_config)
        self.data, evaluation_results = rag_evaluator.evaluate_trajectories(self.data)

        return self.data

    def run_step8(self) -> Dict:
        """Step 8: 人工审核与知识库更新"""
        print("\n" + "=" * 60)
        print("【Step 8】人工审核与知识库更新")
        print("=" * 60)

        review_interface = HumanReviewInterface(self.data)
        self.data, review_results = review_interface.run_review()

        return self.data

    def run_step9(self, output_dir: str = './carla_scenarios') -> List[str]:
        """Step 9: CARLA格式导出"""
        print("\n" + "=" * 60)
        print("【Step 9】CARLA格式导出")
        print("=" * 60)

        exporter = CarlaExporter(output_dir)
        exported_files = exporter.export_all(self.data)

        return exported_files

    def save_checkpoint(self, step_name: str) -> str:
        """保存处理中间结果"""
        save_path = os.path.join(
            self.data_dir,
            f"{self.scenario_id}_{step_name}.pkl"
        )
        with open(save_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"[检查点保存] -> {save_path}")
        return save_path

    def run_full_pipeline(self,
                         obs_id: str = None,
                         threshold: float = 2.0,
                         time_interval: float = 0.3,
                         n_back: int = 15,
                         n_forward: int = 5,
                         llm_config: LLMConfig = None,
                         num_variants: int = 3,
                         db_config: Dict = None,
                         output_dir: str = './carla_scenarios') -> Dict:
        """
        执行完整的处理流程

        Returns:
            最终处理结果
        """
        try:
            # Step 1: 场景加载
            self.data, selected_obs_id = self.run_step1()
            obs_id = obs_id or selected_obs_id
            self.save_checkpoint("Step1_Loaded")

            # Step 2: 风险计算
            self.data = self.run_step2(obs_id, threshold)
            self.save_checkpoint("Step2_Risk")

            # Step 3-6: 危险片段截取、文本化、特征提取
            self.data = self.run_step3_to_step6(time_interval, n_back, n_forward)
            self.save_checkpoint("Step3to6_Features")

            # Step 4: LLM轨迹生成
            self.data = self.run_step4(llm_config, num_variants)
            self.save_checkpoint("Step4_Generated")

            # Step 5: 物理验证
            self.data = self.run_step5()
            self.save_checkpoint("Step5_Validated")

            # Step 6-7: RAG评估
            self.data = self.run_step6_7(db_config)
            self.save_checkpoint("Step6to7_Evaluated")

            # Step 8: 人工审核
            self.data = self.run_step8()
            self.save_checkpoint("Step8_Reviewed")

            # Step 9: CARLA导出
            exported_files = self.run_step9(output_dir)

            print("\n" + "=" * 60)
            print("全流程处理完成！")
            print(f"导出的CARLA场景文件: {exported_files}")
            print("=" * 60)

            return self.data

        except Exception as e:
            print(f"\n处理流程出错: {e}")
            import traceback
            traceback.print_exc()
            raise


# =================================================================
# 主程序入口
# =================================================================
if __name__ == "__main__":
    # 配置参数
    DATA_DIR = r"D:\PythonProjects\Python_projects_anaconda_zpb\LLM-trajctory\LLM-HYNdatasets\waymo-open"
    SCENARIO_ID = '10135f16cd538e19'

    # LLM配置（预留API接口）
    llm_config = LLMConfig(
        api_key="your-api-key-here",
        model="gpt-4",
        temperature=0.7,
        danger_type="rear_end",
        variation_strength=0.5
    )

    # 数据库配置（预留数据库接口）
    db_config = {
        'db_type': 'chromadb',
        'collection_name': 'trajectory_cases',
        'embedding_dim': 12,
        'persist_directory': './chroma_db'
    }

    # 创建流程控制器并执行完整流程
    controller = PipelineController(DATA_DIR, SCENARIO_ID)

    final_data = controller.run_full_pipeline(
        threshold=2.0,
        time_interval=0.3,
        n_back=15,
        n_forward=5,
        llm_config=llm_config,
        num_variants=3,
        db_config=db_config,
        output_dir='./carla_scenarios'
    )
