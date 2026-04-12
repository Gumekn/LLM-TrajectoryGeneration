import os
import pickle
import numpy as np


class WaymoDataManager:
    def __init__(self, base_path, subset_name):
        self.data_dir = os.path.join(base_path, subset_name)
        self.scenarios = {}

    def load_scenarios(self, scenario_ids):
        for s_id in scenario_ids:
            file_path = os.path.join(self.data_dir, s_id + '.pkl')
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    self.scenarios[s_id] = pickle.load(f)
                print(f"成功加载场景: {s_id}")
            else:
                print(f"场景文件不存在: {file_path}")

    def get_summary(self, scenario_id):
        if scenario_id not in self.scenarios: return "未加载该场景"
        data = self.scenarios[scenario_id]
        tracks = data.get('object_tracks', {})
        types = [info.get('type') for info in tracks.values()]
        unique_types = {t: types.count(t) for t in set(types)}

        # 获取总帧数基准
        sample_id = list(tracks.keys())[0]
        total_frames = len(tracks[sample_id]['state']['valid'])

        return {
            "场景ID": scenario_id,
            "总帧数": total_frames,
            "物体总数": len(tracks),
            "类型统计": unique_types,
            "根键值": list(data.keys())
        }

    def slice_data(self, scenario_id, start_f, end_f):
        """
        切片方法：递归处理字典，并识别 numpy 数组或列表进行时间轴截取
        """
        if scenario_id not in self.scenarios: return None
        raw_data = self.scenarios[scenario_id]

        # 获取该场景的总帧数作为对齐基准
        sample_id = list(raw_data['object_tracks'].keys())[0]
        max_f = len(raw_data['object_tracks'][sample_id]['state']['valid'])

        def recursive_slice(obj):
            if isinstance(obj, dict):
                # 递归处理字典中的每一个值
                return {k: recursive_slice(v) for k, v in obj.items()}
            elif isinstance(obj, (np.ndarray, list)):
                # 只有长度等于总帧数的数组/列表才被视为“时间序列数据”进行切片
                if len(obj) == max_f:
                    return obj[start_f:end_f]
            return obj

        # 对整个数据集执行递归切片（涵盖了 object_tracks 和 dynamic_map_states）
        sliced = recursive_slice(raw_data)

        # 静态地图特征 map_features 不需要切片，直接覆盖回原始数据（防止被误切）
        sliced['map_features'] = raw_data['map_features']
        sliced['extra_information'] = raw_data.get('extra_information', {})

        return sliced

def print_structure(data, indent=0):
    """递归打印字典结构（不显示数值）"""
    for key, value in data.items():
        print('  ' * indent + f"|-- {key}", end="")
        if isinstance(value, dict):
            print(" (dict)")
            # 限制 object_tracks 只打印一个示例，否则太长
            if key == 'object_tracks' or key == 'map_features':
                sample_key = list(value.keys())[0]
                print('  ' * (indent + 1) + f"|-- [{sample_key}] (sample id)")
                print_structure(value[sample_key], indent + 2)
            else:
                print_structure(value, indent + 1)
        elif isinstance(value, np.ndarray):
            print(f" (ndarray, shape={value.shape})")
        elif isinstance(value, list):
            print(f" (list, len={len(value)})")
        else:
            print(f" ({type(value).__name__}: {value})")


class InteractiveBrowser:
    """
    终极优化版交互式数据浏览器：
    1. 置顶显示当前场景 ID
    2. 支持编号快速选择
    3. 路径追踪与返回
    4. 智能类型预览
    """

    def __init__(self, data, scenario_id="Unknown"):
        self.root = data
        self.current_node = data
        self.scenario_id = scenario_id  # 存储并显示场景 ID
        self.history = []
        self.path_names = ["Root"]
        self.index_map = {}

    def start(self):
        while True:
            print("\n" + "=" * 100)
            current_path = " > ".join(self.path_names)
            header = f"轨迹ID: {self.scenario_id} | 当前路径: {current_path} | [编号]进入 [..]返回 [v]详情 [q]退出"
            print(header)
            print("-" * 100)

            self.index_map = {}

            # --- 核心显示逻辑 ---
            if isinstance(self.current_node, dict):
                keys = sorted(list(self.current_node.keys()))
                print(f"[字典结构] 包含 {len(keys)} 个子项:")

                for i, k in enumerate(keys, 1):
                    self.index_map[str(i)] = k
                    val = self.current_node[k]

                    # 自动提取预览信息
                    if isinstance(val, dict):
                        info = f"{{Dictionary: {len(val)} keys}}"
                    elif hasattr(val, 'shape'):
                        info = f"Numpy Array {val.shape}"
                    elif isinstance(val, list):
                        info = f"List [len={len(val)}]"
                    else:
                        # 限制显示长度防止刷屏
                        info = str(val)[:40] + "..." if len(str(val)) > 40 else str(val)

                    print(f"  ({i:2}) {k:22} -> {info}")

            elif isinstance(self.current_node, (np.ndarray, list)):
                shape_info = self.current_node.shape if hasattr(self.current_node, 'shape') else len(self.current_node)
                print(f"[数组/列表] 规模: {shape_info}")
                print(f"预览前3项: {self.current_node[:3]}")
                print("\n输入 'v' 可查看该数组的完整数值内容")

            else:
                print(f"[具体数值]: {self.current_node}")

            # --- 交互逻辑 ---
            choice = input("\n请输入编号或指令: ").strip()

            if choice.lower() == 'q':
                print("已退出浏览器。")
                break
            elif choice == '..':
                if self.history:
                    self.current_node = self.history.pop()
                    self.path_names.pop()
                else:
                    print("⚠警告: 已在根目录，无法继续返回！")
                continue
            elif choice.lower() == 'v':
                print("\n" + "·" * 30 + " 完整数值详情 " + "·" * 30)
                print(self.current_node)
                print("·" * 74)
                input("\n按回车键继续浏览...")  # 暂停一下，方便用户阅读数值
                continue

            # 编号与 Key 名双向匹配
            target_key = self.index_map.get(choice) or (
                choice if isinstance(self.current_node, dict) and choice in self.current_node else None)

            if target_key is not None:
                self.history.append(self.current_node)
                self.path_names.append(target_key)
                self.current_node = self.current_node[target_key]
            else:
                print(f"错误: 无法找到名为 '{choice}' 的内容，请检查编号或 Key 名。")

# ================= 使用示例 =================

path = r"D:\PythonProjects\Python_projects_anaconda_zpb\LLM-trajctory\LLM-HYNdatasets"
subset = "waymo-open"
target_id = '10135f16cd538e19_Step6_TextPrompt'

manager = WaymoDataManager(path, subset)
manager.load_scenarios([target_id])


fragment = manager.slice_data(target_id, 0,20 )
sdc_id = fragment['extra_information']['sdc_id']
print(f"检测到自车 ID 为: {sdc_id}")

print("\n=== Waymo 完整数据结构树 ===")
print_structure(manager.scenarios[target_id])

print("\n--- [数据概览] ---")
summary = manager.get_summary(target_id)
print(summary)

fragment = manager.slice_data(target_id, 0,199 )
sdc_id = fragment['extra_information']['sdc_id']
print(f"检测到自车 ID 为: {sdc_id}")

browser = InteractiveBrowser(fragment, scenario_id=target_id) # 也可以用 manager.scenarios[target_id] 查看完整数据
browser.start()