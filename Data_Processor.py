import os
import pickle
import numpy as np
import copy


# =================================================================
# 功能类 1：场景数据准备 (ScenarioDataLoader)
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
# 功能类 2：风险分数计算类 (RiskCalculator)
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
# 功能类 3：关键车筛选类 (KeyVehicleFilter)
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
# 功能类 4：关键车轨迹提取模块 (KeyVehicleTrajectoryExtractor)
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
                rel_angle = np.atan2(dy, dx)

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


import numpy as np
import copy


# =================================================================
# 功能类 5：危险轨迹截取类 (KeyVehicleTrajectoryClipper)
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


import numpy as np
import copy

# =================================================================
# 功能类 6：轨迹文本化类
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

