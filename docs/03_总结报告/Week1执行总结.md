# Week 1 执行总结

> 执行日期: 2026-04-12
> 阶段: 轨迹生成模块

---

## 一、执行概述

### 1.1 任务目标
完成第一阶段：轨迹生成模块，实现从Waymo原始数据到危险轨迹片段提取的完整流程。

### 1.2 实际完成内容
- core/processor.py - 核心模块整合（数据类型、数据加载、风险计算、片段截取）
- Main.py - 主程序入口
- docs/轨迹生成模块设计.md - 技术文档

### 1.3 处理流程（已实现）

```
Waymo原始pkl文件
      ↓
[Step 1] WaymoDataLoader - 数据加载
         - 加载pkl文件
         - 提取车辆轨迹
         - 获取移动车辆列表
         - 用户选择主车（默认sdc_id）
      ↓
[Step 2] ScenarioRiskAnalyzer - 风险分析
         - 计算TTC（纵向/横向）
         - 计算综合风险分数
         - 定位最危险帧（锚点帧）
      ↓
[Step 3] should_skip_fragment - 边界检查
         - 检查锚点帧是否太靠近场景边界
         - 跳过前/后截取时长不足的交互车
      ↓
[Step 4] FragmentExtractor - 片段截取
         - 截取锚点前后指定时长
         - 边界自动处理
         - 主车+交互车同步
      ↓
[Step 5] 特征提取 - 11维特征向量
         - 交互车轨迹特征(5维)
         - 交互特征(6维)
      ↓
[Step 6] JSON持久化 - data/processed/{scenario_id}_{ego_id}.json
         - 无片段时不生成JSON
         - 输出原因分析
```

---

## 二、问题与解决方案

### 2.1 风险分数计算与原代码不一致

**问题描述**：
用户指出风险分数计算结果与原有 Data_Processor.py 不一致。

**原因分析**：
1. TTC计算逻辑不同
2. 风险分数使用float而非int

**解决方案**：
- TTC计算改为向量化的 `np.where((d_l * v_l) < 0, -d_l / v_l, np.inf)`
- 风险分数改为 `int((dsc + tsc) / 2)`

### 2.2 默认主车选择逻辑

**问题描述**：
用户希望默认选择场景数据中指定的sdc_id，而非速度最快的车辆。

**解决方案**：
- 改为优先使用 sdc_id（自车ID）
- 若sdc_id不在移动车辆列表中，则使用速度最快的车辆
- 在车辆列表中用 `*` 标记默认选择

### 2.3 锚点帧靠近边界导致片段过短

**问题描述**：
当锚点帧太靠近场景起始或结束边界时，截取的片段过短，不利于后续轨迹仿真。

**解决方案**：
- 新增 `should_skip_fragment()` 函数
- 新增参数 `min_before_frames=10` 和 `min_after_frames=15`
- 当锚点帧到边界的距离小于阈值时，跳过该交互车

### 2.4 同一场景不同主车的文件覆盖

**问题描述**：
同一场景可以选择不同主车，原文件名只包含scenario_id，会覆盖。

**解决方案**：
- 输出文件名改为 `{scenario_id}_{ego_id}.json`
- 例如：`10135f16cd538e19_1811.json`

### 2.5 无片段时仍生成空JSON

**问题描述**：
当没有提取到任何片段时，仍会生成空的JSON文件。

**解决方案**：
- 当 `len(fragments) == 0` 时跳过JSON保存
- 输出原因分析：
  - 原因1：场景中没有找到风险分数<=阈值的关键车
  - 原因2：所有关键车的锚点帧都太靠近场景边界
  - 原因3：部分关键车被跳过

### 2.6 代码清理

**问题描述**：
项目中存在未使用的代码（测试文件、未使用的函数和常量）。

**解决方案**：
- 删除 tests/ 目录
- 删除未使用的常量：`ProcessingConfig`, `DANGER_TYPES`, `DANGER_LEVELS`, `RELATIVE_DIRECTIONS`
- 删除未使用的类和函数：`ScenarioDataLoader`, `load_waymo_pkl`, `extract_trajectory_from_data`, `slice_trajectory`, `global_to_local_velocity`, `compute_ttc_series`
- 清理 `__init__.py` 和 `Main.py` 导入

---

## 三、配置参数说明

### 3.1 Main.py 顶部配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| DATA_DIR | data/waymo-open | Waymo原始数据目录 |
| OUTPUT_DIR | data/processed | 处理后数据输出目录 |
| N_BEFORE_SEC | 2.0 | 锚点前截取时长（秒） |
| N_AFTER_SEC | 3.0 | 锚点后截取时长（秒） |
| MIN_BEFORE_FRAMES | 10 | 锚点前最小帧数阈值 |
| MIN_AFTER_FRAMES | 15 | 锚点后最小帧数阈值 |
| RISK_THRESHOLD | 3 | 风险分数阈值，低于此值选为关键车 |
| SPEED_THRESHOLD | 0.5 | 速度阈值 (m/s)，低于此值被视为静止 |
| SCENARIO_ID | 10135f16cd538e19 | 场景ID |

### 3.2 命令行参数

```bash
python main.py [选项]
  --scenario, -s          场景ID
  --ego-id, -e             主车ID
  --data-dir, -d           数据目录
  --output-dir, -o          输出目录
  --n-before               锚点前时长(秒)
  --n-after                锚点后时长(秒)
  --risk-threshold, -r     风险阈值
  --speed-threshold        速度阈值
  --min-before-frames      锚点前最小帧数
  --min-after-frames       锚点后最小帧数
```

---

## 四、核心概念说明

### 4.1 锚点帧（Anchor Frame）

锚点帧 = 最危险帧 = 风险分数最低的那一帧。

**截取示意图**（以帧87为例）：
```
帧80 ----帧85 ----[锚点帧87]---- 帧90 ----帧117
  ↑                        ↑                    ↑
  |------2秒(20帧)---------|------3秒(30帧)-----|
  
  截取范围: [80, 117)，共37帧
```

### 4.2 边界检查逻辑

```python
def should_skip_fragment(anchor_frame, n_before_frames, n_after_frames, total_frames):
    # 检查锚点帧到开始边界
    if anchor_frame < min_before_frames:
        return True  # 跳过
    
    # 检查锚点帧到结束边界  
    frames_after_anchor = total_frames - anchor_frame - 1
    if frames_after_anchor < min_after_frames:
        return True  # 跳过
    
    return False  # 不跳过
```

### 4.3 危险类型动态推断

不预设场景类型，基于实际交互特征动态推断：

```python
def infer_danger_type(ttc_long, ttc_lat, heading_diff):
    # 1. 纵向接近为主
    if min_ttc_long < min_ttc_lat * 0.5 and |mean_heading_diff| < 30°:
        return "rear_end" if heading_diff > 0 else "head_on"
    
    # 2. 横向接近为主
    if min_ttc_lat < min_ttc_long * 0.5:
        return "cut_in" if |heading_diff| > 45° else "crossing"
    
    # 3. 混合型
    return "mixed"
```

---

## 五、11维特征详细定义

| 索引 | 特征名 | 计算方式 | 单位 |
|------|--------|---------|------|
| 0 | mean_speed | mean(\|v\|) | m/s |
| 1 | max_speed | max(\|v\|) | m/s |
| 2 | speed_std | std(\|v\|) | m/s |
| 3 | mean_accel | mean(\|a\|) | m/s² |
| 4 | max_accel | max(\|a\|) | m/s² |
| 5 | min_ttc_long | min(TTC_long) | s |
| 6 | mean_ttc_long | mean(TTC_long) | s |
| 7 | min_rel_dist | min(rel_dist) | m |
| 8 | max_closing_speed | max(-v_long) | m/s |
| 9 | trajectory_length | sum(\|ds\|) | m |
| 10 | max_curvature | max(\|dθ/ds\|) | 1/m |

---

## 六、JSON输出结构

```json
{
  "metadata": {
    "scenario_id": "10135f16cd538e19",
    "source_file": "10135f16cd538e19.pkl",
    "total_frames": 198,
    "sampling_rate": 10.0,
    "ego_vehicle_id": "1811",
    "num_key_vehicles": 1,
    "processed_at": "2026-04-12T10:30:00"
  },
  "fragments": [
    {
      "fragment_id": "frag_10135f16cd538e19_1811_1707_87",
      "metadata": {
        "scenario_id": "10135f16cd538e19",
        "anchor_frame": 87,
        "ego_vehicle_id": "1811",
        "target_vehicle_id": "1707",
        "danger_type": "mixed",
        "danger_level": "medium",
        "min_risk_score": 2.0,
        "frame_count": 51,
        "duration": 5.1,
        "n_before": 20,
        "n_after": 30
      },
      "target_trajectory": { ... },
      "ego_trajectory": { ... },
      "interaction_features": { ... },
      "interaction_stats": { ... },
      "feature_vector": [6.89, 7.61, 0.36, 0.64, 2.12, 0.02, 1.68, 3.86, 6.53, 34.41, 0.007],
      "feature_names": ["mean_speed", "max_speed", "speed_std", "mean_accel", "max_accel", "min_ttc_long", "mean_ttc_long", "min_rel_dist", "max_closing_speed", "trajectory_length", "max_curvature"]
    }
  ]
}
```

---

## 七、测试结果

```
[PASS] main.py 运行正常
[PASS] 数据加载正常
[PASS] 风险分析正常
[PASS] 片段截取正常
[PASS] 特征提取正常
[PASS] JSON持久化正常
[PASS] 边界检查正常
[PASS] 无片段时不生成JSON
```

---

## 八、版本历史

| 版本 | 日期 | 修改内容 |
|------|------|---------|
| v1.0 | 2026-04-12 | 初始实现，包含数据加载、风险计算、片段截取、特征提取 |
| v1.1 | 2026-04-12 | 修复风险分数计算；默认主车改为sdc_id；添加边界检查；文件名包含ego_id；无片段不生成JSON |
