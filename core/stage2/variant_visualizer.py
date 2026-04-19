"""
轨迹变异可视化核心模块

提供轨迹可视化相关的函数和类，用于生成动画视频。
"""

import json
import os
import random
import gc
from typing import Dict, List, Any, Optional, Tuple

import imageio
import matplotlib
matplotlib.use('Agg')  # 使用非交互式backend
import matplotlib.pyplot as plt
import numpy as np
import io

from matplotlib.patches import Rectangle, Polygon
from matplotlib.transforms import Affine2D
from tqdm import tqdm


def grab_frame(fig: plt.Figure) -> np.ndarray:
    """从matplotlib figure获取RGB帧"""
    fig.canvas.draw()

    # 使用canvas的renderer获取像素数据
    renderer = fig.canvas.get_renderer()
    if hasattr(renderer, 'buffer_rgba'):
        # 方法1: 从buffer_rgba获取
        img = np.asarray(renderer.buffer_rgba())
        # RGBA转RGB
        img = img[:, :, :3]
    else:
        # 方法2: 使用tobytes获取
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img = img.reshape((h, w, 3))

    return img


def get_random_color(seed: Any) -> Tuple[float, float, float]:
    """生成随机颜色"""
    random.seed(seed)
    r = 0.3 + 0.4 * random.random()
    g = 0.3 + 0.4 * random.random()
    b = 0.3 + 0.4 * random.random()
    return (r, g, b)


def rotate(x: np.ndarray, y: np.ndarray, angle: float) -> np.ndarray:
    """旋转坐标"""
    other_x_trans = np.cos(angle) * x - np.sin(angle) * y
    other_y_trans = np.cos(angle) * y + np.sin(angle) * x
    return np.stack((other_x_trans, other_y_trans), axis=-1)


def load_variant_data(json_path: str) -> Dict[str, Any]:
    """加载变异轨迹JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def preprocess_to_ego_center(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将轨迹数据转换到以主车为中心的坐标系

    以主车起点为原点，主车初始朝向为x轴正方向
    """
    processed = {
        'fragment_id': data['fragment_id'],
        'metadata': data['metadata'],
        'ego_trajectory': {},
        'original_target_trajectory': {},
        'variants': [],
        'generation_info': data.get('generation_info', {})
    }

    ego_traj = data['ego_trajectory']
    target_traj = data['original_target_trajectory']

    ego_positions = np.array(ego_traj['positions'])
    ego_headings = np.array(ego_traj['headings'])
    ego_valid = np.array(ego_traj['valid'])

    target_positions = np.array(target_traj['positions'])
    target_headings = np.array(target_traj['headings'])
    target_valid = np.array(target_traj['valid'])

    origin = ego_positions[0, :2]
    heading_0 = ego_headings[0]

    ego_centered = ego_positions.copy()
    ego_centered[:, 0:2] -= origin
    rotated_ego = rotate(ego_centered[:, 0], ego_centered[:, 1], -heading_0)
    ego_centered[:, 0:2] = rotated_ego

    target_centered = target_positions.copy()
    target_centered[:, 0:2] -= origin
    rotated_target = rotate(target_centered[:, 0], target_centered[:, 1], -heading_0)
    target_centered[:, 0:2] = rotated_target

    processed['ego_trajectory'] = {
        'vehicle_id': ego_traj['vehicle_id'],
        'positions': ego_centered[:, 0:2].tolist(),
        'headings': (ego_headings - heading_0).tolist(),
        'velocities': ego_traj.get('velocities', []),
        'accelerations': ego_traj.get('accelerations', []),
        'valid': ego_valid.tolist()
    }

    processed['original_target_trajectory'] = {
        'vehicle_id': target_traj['vehicle_id'],
        'positions': target_centered[:, 0:2].tolist(),
        'headings': (target_headings - heading_0).tolist(),
        'velocities': target_traj.get('velocities', []),
        'accelerations': target_traj.get('accelerations', []),
        'valid': target_valid.tolist()
    }

    for variant in data.get('variants', []):
        mut_traj = variant['mutated_target_trajectory']
        mut_positions = np.array(mut_traj['positions'])

        mut_centered = mut_positions.copy()
        mut_centered[:, 0:2] -= origin
        rotated_mut = rotate(mut_centered[:, 0], mut_centered[:, 1], -heading_0)
        mut_centered[:, 0:2] = rotated_mut

        processed['variants'].append({
            'variant_id': variant['variant_id'],
            'mutated_target_trajectory': {
                'positions': mut_centered[:, 0:2].tolist(),
                'headings': (np.array(mut_traj['headings']) - heading_0).tolist() if mut_traj.get('headings') else [],
                'velocities': mut_traj.get('velocities', []),
                'accelerations': mut_traj.get('accelerations', []),
                'valid': mut_traj.get('valid', [True] * len(mut_positions))
            }
        })

    return processed


def plot_vehicle_at_frame(ax: plt.Axes, x: float, y: float, heading: float,
                          length: float = 4.5, width: float = 2.0,
                          color: str = 'blue', alpha: float = 0.8,
                          zorder: int = 4) -> None:
    """在指定位置绘制车辆"""
    ax.add_patch(Rectangle(
        xy=(x - 0.5 * length, y - 0.5 * width),
        width=length,
        height=width,
        transform=matplotlib.transforms.Affine2D().rotate_deg_around(x, y, heading * 180 / np.pi) + ax.transData,
        alpha=alpha,
        linewidth=0.5,
        facecolor=color,
        edgecolor='black',
        zorder=zorder
    ))

    point_a = (x + length * np.cos(heading) / 2, y + length * np.sin(heading) / 2)
    point_b = (x + length * np.cos(heading) / 3 - length * np.sin(heading) / (6 * np.sqrt(3)),
               y + length * np.sin(heading) / 3 + length * np.cos(heading) / (6 * np.sqrt(3)))
    point_c = (x + length * np.cos(heading) / 3 + length * np.sin(heading) / (6 * np.sqrt(3)),
               y + length * np.sin(heading) / 3 - length * np.cos(heading) / (6 * np.sqrt(3)))

    ax.add_patch(Polygon(
        xy=(point_a, point_b, point_c),
        linewidth=0.5,
        alpha=0.8,
        facecolor='none',
        edgecolor='black',
        zorder=zorder + 1
    ))


def plot_trajectory_up_to_frame(ax: plt.Axes, positions: List[List[float]],
                                valid: List[bool], current_frame: int,
                                color: str = 'blue', label: str = None,
                                alpha: float = 0.7, zorder: int = 2) -> None:
    """绘制到当前帧为止的轨迹"""
    if not positions or current_frame < 0:
        return

    valid_positions = [(positions[i], i) for i in range(min(current_frame + 1, len(positions)))
                        if i < len(valid) and valid[i]]

    if not valid_positions:
        return

    xs = [p[0] for p, _ in valid_positions]
    ys = [p[1] for p, _ in valid_positions]

    ax.plot(xs, ys, color=color, alpha=alpha, linewidth=1.5, zorder=zorder, label=label)

    if valid_positions:
        last_x, last_i = valid_positions[-1]
        ax.scatter(last_x[0], last_x[1], s=30, color=color, zorder=zorder + 1,
                   edgecolors='white', linewidths=0.5)


class VariantVideoGenerator:
    """变异轨迹视频生成器"""

    def __init__(self, data: Dict[str, Any], output_dir: str,
                 axis_range: float = 80, figure_size: Tuple[int, int] = (10, 10),
                 fps: int = 10):
        """
        初始化视频生成器

        Args:
            data: 预处理后的轨迹数据
            output_dir: 输出目录
            axis_range: 坐标轴范围（米）
            figure_size: 图形大小（英寸）
            fps: 视频帧率
        """
        self.data = data
        self.output_dir = output_dir
        self.axis_range = axis_range
        self.figure_size = figure_size
        self.fps = fps
        self.metadata = data['metadata']
        self.ego_traj = data['ego_trajectory']
        self.target_traj = data['original_target_trajectory']
        self.variants = data['variants']

        self.variant_colors = ['orange', 'lime', 'yellow', 'magenta',
                               'white', 'brown', 'pink', 'cyan']

        os.makedirs(output_dir, exist_ok=True)

    def _setup_figure(self) -> Tuple[plt.Figure, plt.Axes]:
        """创建并配置图形"""
        fig, ax = plt.subplots(figsize=self.figure_size)
        fig.patch.set_facecolor('gray')
        ax.set_facecolor('gray')
        ax.set_aspect('equal')
        ax.set_xlim(-self.axis_range, self.axis_range)
        ax.set_ylim(-self.axis_range, self.axis_range)
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        return fig, ax

    def _draw_frame(self, ax: plt.Axes, frame_idx: int,
                    variant_positions: Optional[List[List[float]]] = None,
                    variant_headings: Optional[List[float]] = None,
                    variant_valid: Optional[List[bool]] = None,
                    draw_history: bool = True,
                    show_info: bool = True) -> None:
        """绘制单帧"""
        ax.clear()

        fig = ax.get_figure()
        fig.patch.set_facecolor('gray')
        ax.set_facecolor('gray')
        ax.set_aspect('equal')
        ax.set_xlim(-self.axis_range, self.axis_range)
        ax.set_ylim(-self.axis_range, self.axis_range)
        ax.grid(True, alpha=0.3, linestyle='--')

        ego_valid = self.ego_traj.get('valid', [True] * len(self.ego_traj['positions']))
        target_valid = self.target_traj.get('valid', [True] * len(self.target_traj['positions']))

        if draw_history:
            plot_trajectory_up_to_frame(
                ax, self.ego_traj['positions'], ego_valid, frame_idx,
                color='red', label=f"Ego ({self.ego_traj['vehicle_id']})", alpha=0.7
            )
            plot_trajectory_up_to_frame(
                ax, self.target_traj['positions'], target_valid, frame_idx,
                color='cyan', label=f"Original ({self.target_traj['vehicle_id']})", alpha=0.7
            )

            if variant_positions and variant_valid:
                for i, (mut_pos, mut_valid) in enumerate(zip(variant_positions, variant_valid)):
                    color = self.variant_colors[i % len(self.variant_colors)]
                    plot_trajectory_up_to_frame(
                        ax, mut_pos, mut_valid, frame_idx,
                        color=color, label=f"Variant {i}", alpha=0.5
                    )

        if frame_idx < len(self.ego_traj['positions']) and (frame_idx < len(ego_valid) and ego_valid[frame_idx]):
            ego_pos = self.ego_traj['positions'][frame_idx]
            ego_heading = self.ego_traj['headings'][frame_idx] if frame_idx < len(self.ego_traj['headings']) else 0
            plot_vehicle_at_frame(ax, ego_pos[0], ego_pos[1], ego_heading, color='red')

        if frame_idx < len(self.target_traj['positions']) and (frame_idx < len(target_valid) and target_valid[frame_idx]):
            target_pos = self.target_traj['positions'][frame_idx]
            target_heading = self.target_traj['headings'][frame_idx] if frame_idx < len(self.target_traj['headings']) else 0
            plot_vehicle_at_frame(ax, target_pos[0], target_pos[1], target_heading, color='cyan')

        if variant_positions and variant_headings and variant_valid:
            for i, (mut_pos, mut_head, mut_valid) in enumerate(zip(variant_positions, variant_headings, variant_valid)):
                if frame_idx < len(mut_pos) and frame_idx < len(mut_valid) and mut_valid[frame_idx]:
                    color = self.variant_colors[i % len(self.variant_colors)]
                    plot_vehicle_at_frame(ax, mut_pos[frame_idx][0], mut_pos[frame_idx][1],
                                          mut_head[frame_idx] if frame_idx < len(mut_head) else 0,
                                          color=color, alpha=0.7)

        if show_info:
            info_text = (
                f"Frame: {frame_idx}/{len(self.ego_traj['positions']) - 1}\n"
                f"Fragment: {self.metadata['fragment_id']}\n"
                f"Danger: {self.metadata['danger_type']} ({self.metadata['danger_level']})"
            )
            ax.text(0.02, 0.98, info_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    fontfamily='monospace',
                    fontsize=8,
                    color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        ax.legend(loc='upper right', fontsize=7, framealpha=0.8)

    def _frames_to_video(self, frames: List[np.ndarray], output_path: str) -> None:
        """将帧列表保存为视频"""
        with imageio.get_writer(output_path, fps=self.fps, macro_block_size=None) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"  Saved: {output_path}")

    def generate_variant_video(self, variant_id: int, max_frames: Optional[int] = None) -> str:
        """
        生成单个变异轨迹的动画视频

        Args:
            variant_id: 变异ID
            max_frames: 最大帧数，None表示全部

        Returns:
            输出视频路径
        """
        if variant_id >= len(self.variants):
            raise ValueError(f"Invalid variant_id: {variant_id}")

        variant = self.variants[variant_id]
        mut_traj = variant['mutated_target_trajectory']

        output_path = os.path.join(self.output_dir, f"variant_{variant_id:03d}.mp4")

        # 跳过已存在的文件
        if os.path.exists(output_path):
            print(f"  Skipping existing: {output_path}")
            return output_path

        n_frames = max_frames if max_frames else len(self.ego_traj['positions'])

        frames = []
        for frame_idx in tqdm(range(n_frames), desc=f"Variant {variant_id}", leave=False):
            fig, ax = self._setup_figure()

            self._draw_frame(
                ax, frame_idx,
                variant_positions=[mut_traj['positions']],
                variant_headings=[mut_traj['headings']],
                variant_valid=[mut_traj.get('valid', [True] * len(mut_traj['positions']))]
            )

            frame = grab_frame(fig)
            frames.append(frame)
            plt.close(fig)
            del fig, ax

        self._frames_to_video(frames, output_path)
        del frames
        gc.collect()
        return output_path

    def generate_all_variant_videos(self) -> List[str]:
        """生成所有变异轨迹的动画视频"""
        output_paths = []
        for variant_id in range(len(self.variants)):
            path = self.generate_variant_video(variant_id)
            output_paths.append(path)
        return output_paths

    def generate_original_trajectory_video(self) -> str:
        """生成主车和原始交互车轨迹的动画视频（无变异）"""
        output_path = os.path.join(self.output_dir, "original_trajectory.mp4")

        # 跳过已存在的文件
        if os.path.exists(output_path):
            print(f"  Skipping existing: {output_path}")
            return output_path

        n_frames = len(self.ego_traj['positions'])
        frames = []

        for frame_idx in tqdm(range(n_frames), desc="Original trajectory", leave=False):
            fig, ax = self._setup_figure()

            self._draw_frame(ax, frame_idx, draw_history=True)

            frame = grab_frame(fig)
            frames.append(frame)
            plt.close(fig)
            del fig, ax

        self._frames_to_video(frames, output_path)
        del frames
        gc.collect()
        return output_path


def process_single_json(json_path: str, output_root: str,
                         axis_range: float = 80, fps: int = 10) -> str:
    """
    处理单个JSON文件，生成所有可视化视频

    Args:
        json_path: JSON文件路径
        output_root: 可视化结果根目录
        axis_range: 坐标轴范围
        fps: 视频帧率

    Returns:
        输出目录路径
    """
    print(f"\nProcessing: {json_path}")

    data = load_variant_data(json_path)
    processed_data = preprocess_to_ego_center(data)

    fragment_id = processed_data['fragment_id']
    output_dir = os.path.join(output_root, fragment_id)
    os.makedirs(output_dir, exist_ok=True)

    generator = VariantVideoGenerator(
        data=processed_data,
        output_dir=output_dir,
        axis_range=axis_range,
        fps=fps
    )

    print(f"  Generating {len(processed_data['variants'])} variant videos...")
    generator.generate_all_variant_videos()

    print(f"  Generating original trajectory video...")
    generator.generate_original_trajectory_video()

    print(f"  Output directory: {output_dir}")
    return output_dir


def process_all_jsons(input_dir: str, output_root: str,
                       axis_range: float = 80, fps: int = 10) -> List[str]:
    """
    处理目录下所有JSON文件

    Args:
        input_dir: 输入JSON目录
        output_root: 可视化结果根目录
        axis_range: 坐标轴范围
        fps: 视频帧率

    Returns:
        处理的输出目录列表
    """
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    if not json_files:
        print(f"No JSON files found in: {input_dir}")
        return []

    print(f"Found {len(json_files)} JSON files")

    output_dirs = []
    for json_file in json_files:
        json_path = os.path.join(input_dir, json_file)
        output_dir = process_single_json(json_path, output_root, axis_range, fps)
        output_dirs.append(output_dir)

    return output_dirs
