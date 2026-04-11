# 危险轨迹生成与评估系统 - 穷举生成+LLM剪枝版（修订版）

## 项目概述

本项目采用**穷举参数变异 + LLM智能剪枝**的技术路线，通过系统性地遍历参数空间生成大量候选轨迹，再利用LLM的语义理解能力进行合理性评估和筛选，从而高效生成符合物理约束的危险驾驶场景。

**核心思想**：让LLM做它最擅长的事（评估/判断），让计算机做它最擅长的事（批量生成/计算）。

**本次修订重点**：
1. 重新设计RAG知识库结构，完整存储案例（轨迹+评估结果）
2. 明确RAG检索实现机制和数据库存储流程
3. 明确知识库与数据库的关系架构
4. 基于Waymo原始数据变量，设计完整的轨迹变异流程
5. 解决穷举生成数量限制问题，改为动态自适应策略

---

## 一、技术方案对比

| 维度                 | LLM直接生成轨迹        | 穷举生成+LLM剪枝           |
| -------------------- | ---------------------- | -------------------------- |
| **物理一致性** | 难以保证，需大量后处理 | 基于真实轨迹变异，天然连续 |
| **可控性**     | 低，LLM行为难以预测    | 高，参数范围完全可控       |
| **生成速度**   | 慢，API调用耗时        | 快，本地批量计算           |
| **成本**       | 高，每轨迹都调API      | 低，仅评估时调API          |
| **可解释性**   | 差                     | 好，每参数可调             |
| **面试优势**   | 普通                   | 体现算法思维+AI结合        |

---

## 二、系统架构

### 2.1 数据流与处理流程（学术版核心）

```
Waymo原始数据
    ↓
[Step 1] 场景加载与车辆选择
    ↓
[Step 2] 风险计算与危险片段定位
    ↓
[Step 3] 轨迹特征提取与向量化（12维特征）
    ↓
【穷举生成阶段】
    ↓
[Step 4a] 参数空间定义（基于原始数据可变异参数）
    ↓
[Step 4b] 穷举生成候选轨迹（自适应数量，无硬性上限）
    - 关键帧采样 + 平滑插值
    - 物理参数变异（速度、加速度、航向等）
    ↓
[Step 5] 物理约束验证（硬过滤，剔除90%+）
    ↓
【RAG检索与LLM剪枝阶段】
    ↓
[Step 6] RAG检索相似案例（基于12维特征向量）
    - 查询向量编码
    - ChromaDB相似度检索(top_k=5)
    - 返回完整案例信息（轨迹+评估结果）
    ↓
[Step 7] LLM合理性评估（基于相似案例辅助判断）
    - 输入：候选轨迹 + 相似案例评估信息
    - 输出：合理性判断、置信度、改进建议
    - 剪枝策略：物理+RAG+LLM三级过滤
    ↓
[Step 8] 人工审核与知识库更新
    - 审核通过的案例入库（SQLite + ChromaDB）
    ↓
[Step 9] CARLA格式导出与闭环测试
```

### 2.2 Waymo原始数据详细介绍

#### 2.2.1 数据来源与格式

**数据来源**：Waymo Open Dataset（经过预处理为pkl格式）

**文件结构**：
```
data/waymo-open/
├── {scenario_id}.pkl          # 单个场景数据文件
└── ...
```

#### 2.2.2 数据结构详解

**顶层结构**（字典类型）：

| 键名 | 数据类型 | 说明 |
|------|----------|------|
| `object_tracks` | dict | 所有交通参与者的轨迹数据 |
| `map_features` | dict | 静态地图特征（车道线、交通标志等） |
| `dynamic_map_states` | dict | 动态地图状态（红绿灯等） |
| `extra_information` | dict | 场景元数据 |

**`object_tracks`结构**（每个物体ID对应一个字典）：

```python
object_tracks = {
    "vehicle_id_1": {
        "type": "VEHICLE",           # 类型：VEHICLE/PEDESTRIAN/CYCLIST
        "state": {
            "valid": np.ndarray[frames],           # bool，该帧是否有效
            "global_center": np.ndarray[frames, 3], # [x, y, z] 全局坐标
            "heading": np.ndarray[frames],          # 航向角（弧度）
            "local_velocity": np.ndarray[frames, 2], # [vx_long, vy_lat] 局部速度
            "local_acceleration": np.ndarray[frames, 2], # [ax_long, ay_lat] 局部加速度
            "size": np.ndarray[frames, 3],          # [length, width, height] 尺寸
            "action": list[frames]                   # 动作语义标签
        }
    },
    # ... 更多物体
}
```

**`extra_information`结构**：

| 键名 | 类型 | 说明 |
|------|------|------|
| `scene_length` | int | 场景总帧数 |
| `sampling_rate` | float | 采样率（Hz，通常为10Hz） |
| `sdc_id` | str | 自车（SDC）ID |
| `scenario_id` | str | 场景唯一标识 |

#### 2.2.3 可用变量与变异参数设计

**直接可用的原始变量**：

| 变量名 | 维度 | 物理意义 | 可变性 |
|--------|------|----------|--------|
| `global_center` | [x, y, z] | 全局位置坐标 | 通过速度/加速度间接修改 |
| `heading` | scalar | 航向角（rad） | ✅ 可直接变异 |
| `local_velocity` | [vx, vy] | 局部坐标系速度 | ✅ 可直接变异 |
| `local_acceleration` | [ax, ay] | 局部坐标系加速度 | ✅ 可直接变异 |
| `size` | [L, W, H] | 车辆尺寸 | ❌ 一般不变 |

**可计算的派生变量**（用于变异）：

| 派生变量 | 计算方式 | 用途 |
|----------|----------|------|
| `speed` | `sqrt(vx² + vy²)` | 车速大小 |
| `yaw_rate` | `diff(heading) / dt` | 偏航角速度 |
| `curvature` | `yaw_rate / speed` | 轨迹曲率 |
| `jerk` | `diff(acceleration) / dt` | 加速度变化率 |
| `ttc` | `distance / relative_velocity` | 碰撞时间 |

### 2.3 穷举生成模块设计（修订版）

#### 2.3.1 核心变异策略

**策略原则**：不直接修改轨迹的x、y坐标值，而是修改运动学参数（速度、加速度、航向），通过物理公式重新计算轨迹点位置，保证物理一致性和连续性。

**变异方法**：

1. **关键帧采样**：从轨迹中采样关键时间点（如起点、危险点、终点）
2. **参数变异**：对关键帧的运动学参数进行变异
3. **平滑插值**：使用样条曲线或物理模型连接关键帧，生成完整轨迹

#### 2.3.2 变异参数定义

| 参数名 | 物理意义 | 取值范围 | 步长/采样方式 | 影响维度 |
|--------|----------|----------|---------------|----------|
| `speed_scale` | 速度缩放因子 | [0.5, 1.8] | 0.1 | 纵向 |
| `accel_offset` | 纵向加速度偏移 | [-4, 4] m/s² | 0.5 | 纵向 |
| `lateral_offset` | 横向位置偏移 | [-2.5, 2.5] m | 0.25 | 横向 |
| `heading_bias` | 航向角偏移 | [-0.4, 0.4] rad | 0.05 | 方向 |
| `time_shift` | 时间轴偏移 | [-1.5, 1.5] s | 0.3 | 时机 |
| `jerk_scale` | 急动度缩放 | [0.8, 1.5] | 0.1 | 舒适性 |

**参数组合示例**：
```python
# 参数空间配置
param_space = {
    'speed_scale': np.arange(0.5, 1.8, 0.1),      # 13个值
    'accel_offset': np.arange(-4, 4.5, 0.5),       # 17个值  
    'lateral_offset': np.arange(-2.5, 2.75, 0.25), # 21个值
    'heading_bias': np.arange(-0.4, 0.45, 0.05),   # 17个值
}

# 全组合：13 × 17 × 21 × 17 = 78,507 种
# 实际采用智能采样：动态生成，根据计算资源和参数重要性自适应调整
```

#### 2.3.3 自适应采样策略（解决固定数量限制问题）

**问题分析**：原框架固定1000-5000条候选轨迹的限制过于僵化，当：
- 轨迹片段较长（如100帧以上）时，参数组合可能远超5000
- 参数空间维度增加时，组合数指数增长
- 不同场景需要的变异粒度不同

**解决方案：动态自适应采样策略**

| 策略类型 | 触发条件 | 采样逻辑 | 输出数量 |
|----------|----------|----------|----------|
| **全网格采样** | 参数维度≤3，组合数<5000 | 遍历所有参数组合 | 全部组合 |
| **分层随机采样** | 参数维度>3，组合数≥5000 | 按危险类型分层，每层随机采样 | 按计算资源设定 |
| **危险导向采样** | 已知高风险参数区域 | 在风险区域密集采样，其他区域稀疏 | 重点区域密集 |
| **自适应迭代采样** | 多轮生成场景 | 根据上一轮评估结果调整采样密度 | 动态调整 |

**动态数量计算公式**：
```python
def calculate_candidate_count(param_space, config):
    """
    动态计算候选轨迹数量
    
    Args:
        param_space: 参数空间定义
        config: 生成配置
        
    Returns:
        int: 建议生成的候选数量
    """
    # 计算理论组合数
    total_combinations = np.prod([len(v) for v in param_space.values()])
    
    # 根据计算资源设定上限
    max_candidates = config.get('max_candidates', 10000)
    min_candidates = config.get('min_candidates', 1000)
    
    # 根据轨迹长度调整（较长轨迹需要更多样化）
    frame_count = config.get('frame_count', 50)
    length_factor = min(frame_count / 50, 2.0)  # 最多翻倍
    
    # 最终数量
    if total_combinations <= max_candidates:
        return int(total_combinations * length_factor)
    else:
        # 超过上限时采用智能采样
        return int(min(max_candidates * length_factor, total_combinations))
```

### 2.4 轨迹变异完整流程（从原始数据到危险轨迹）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        轨迹变异生成完整流程                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: 原始数据输入                                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  输入：Waymo pkl文件                                                   │ │
│  │  包含：object_tracks, map_features, dynamic_map_states                │ │
│  │  关键数据：目标车辆轨迹（global_center, velocity, acceleration）       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                     ↓                                       │
│  Step 2: 危险片段定位与截取                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  1. 计算风险分数（TTC、距离、速度等多维度）                              │ │
│  │  2. 识别最危险帧（anchor_frame）                                        │ │
│  │  3. 截取片段：[anchor - n_back, anchor + n_forward]                    │ │
│  │  4. 同步截取自车轨迹（用于相对特征计算）                                 │ │
│  │  输出：危险片段轨迹（约30-100帧）                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                     ↓                                       │
│  Step 3: 关键帧采样                                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  策略：非均匀采样，危险区域密集采样                                      │ │
│  │  - 起始帧（t=0）                                                        │ │
│  │  - 危险前兆帧（t=anchor - 10）                                          │ │
│  │  - 危险锚点帧（t=anchor）                                               │ │
│  │  - 结束帧（t=T）                                                        │ │
│  │  输出：关键帧索引列表 + 对应运动学参数                                   │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                     ↓                                       │
│  Step 4: 参数空间定义与采样                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  输入：变异参数配置（speed_scale, accel_offset等）                      │ │
│  │  处理：                                                                │ │
│  │    1. 根据计算资源计算候选数量上限                                       │ │
│  │    2. 选择采样策略（全网格/分层随机/危险导向）                           │ │
│  │    3. 生成参数组合列表                                                  │ │
│  │  输出：参数组合列表 [(p1_v1, p2_v1), (p1_v2, p2_v1), ...]               │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                     ↓                                       │
│  Step 5: 轨迹变异生成（批量处理）                                            │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  对每个参数组合：                                                        │ │
│  │    1. 读取原始关键帧状态（位置、速度、加速度、航向）                       │ │
│  │    2. 应用参数变异：                                                      │ │
│  │       - velocity_new = velocity * speed_scale                          │ │
│  │       - acceleration_new = acceleration + accel_offset                 │ │
│  │       - heading_new = heading + heading_bias                           │ │
│  │    3. 物理积分生成新轨迹：                                                │ │
│  │       - 使用数值积分（Euler/Runge-Kutta）                               │ │
│  │       - 根据新的速度、加速度计算位置序列                                  │ │
│  │       - 考虑车辆动力学约束（最大转向角、加速度限制）                       │ │
│  │    4. 轨迹平滑处理：                                                      │ │
│  │       - 使用B样条或高斯滤波平滑轨迹                                       │ │
│  │       - 确保轨迹C2连续性（位置、速度、加速度连续）                        │ │
│  │    5. 计算相对运动特征（TTC、相对距离等）                                  │ │
│  │  输出：候选轨迹列表（包含完整运动学信息）                                 │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                     ↓                                       │
│  Step 6: 物理约束验证                                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  硬约束检查（失败则剔除）：                                               │ │
│  │    - 最大纵向加速度 ≤ 6 m/s²                                           │ │
│  │    - 最大纵向减速度 ≤ 8 m/s²                                           │ │
│  │    - 最大横向加速度 ≤ 4 m/s²                                           │ │
│  │    - 最大速度 ≤ 35 m/s (126 km/h)                                      │ │
│  │    - 轨迹无断点、无跳变                                                  │ │
│  │  软约束检查（记录警告）：                                                 │ │
│  │    - Jerk变化率 ≤ 10 m/s³                                              │ │
│  │    - 曲率连续性                                                          │ │
│  │  输出：通过验证的候选轨迹列表（剔除率通常>90%）                           │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                     ↓                                       │
│  Step 7: 特征提取与向量化                                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  提取12维特征向量：                                                       │ │
│  │    [mean_speed, max_speed, speed_std, mean_accel, max_accel,           │ │
│  │     max_lateral_offset, lateral_std, min_ttc, mean_ttc,                │ │
│  │     trajectory_length, max_curvature]                                  │ │
│  │  输出：候选轨迹 + 12维特征向量                                           │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                     ↓                                       │
│  Step 8: 数据库存储（候选轨迹池）                                            │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  SQLite trajectory_records表：                                         │ │
│  │    - trajectory_id, scenario_id, vehicle_id                            │ │
│  │    - trajectory_data (BLOB, 序列化轨迹点)                               │ │
│  │    - feature_vector (BLOB, 12维向量)                                    │ │
│  │    - generation_params (JSON, 生成参数)                                 │ │
│  │    - danger_type, danger_level                                         │ │
│  │    - is_validated (物理验证状态)                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、RAG知识库设计（修订版）

### 3.1 知识库与数据库的关系架构

**核心概念**：知识库是数据库的**逻辑子集**，专门用于RAG检索加速和LLM评估辅助。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         数据存储架构                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        SQLite关系型数据库                            │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │   │
│  │  │ scenario_metadata │  │trajectory_records│  │evaluation_results│  │   │
│  │  │ 场景元数据        │  │ 轨迹记录         │  │ 评估结果         │  │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘  │   │
│  │  ┌──────────────────┐  ┌──────────────────┐                        │   │
│  │  │ knowledge_base   │  │ human_reviews    │                        │   │
│  │  │ 知识库索引表     │  │ 人工审核记录     │                        │   │
│  │  └──────────────────┘  └──────────────────┘                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      │ 同步/索引                             │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     ChromaDB向量数据库                               │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                   trajectory_cases集合                         │  │   │
│  │  │  • case_id (id)                                                │  │   │
│  │  │  • embedding (vector[12]) - 轨迹特征向量                       │  │   │
│  │  │  • metadata (dict) - 标签、危险类型、评估来源等                 │  │   │
│  │  │  • document (str) - 案例JSON序列化（轨迹+评估详情）             │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      │ 检索                                  │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          RAG检索层                                   │   │
│  │  • 相似度计算（余弦相似度）                                          │   │
│  │  • 返回：相似案例列表 + 完整评估信息                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**关系说明**：
1. **SQLite**是主数据库，存储所有结构化数据
2. **knowledge_base**表存储知识库的元数据索引
3. **ChromaDB**存储向量数据，用于快速相似度检索
4. 两者通过`case_id`和`trajectory_id`关联

### 3.2 知识库案例结构设计

**一个完整的案例包含**：

```python
class TrajectoryCase:
    """知识库案例完整结构"""
    
    # === 基础标识 ===
    case_id: str                    # 案例唯一ID
    trajectory_id: str              # 关联轨迹ID
    scenario_id: str                # 关联场景ID
    
    # === 轨迹数据（原始+生成信息） ===
    trajectory_data: Dict           # 完整轨迹点序列
    generation_params: Dict         # 生成参数（可解释性）
    parent_trajectory_id: str       # 父轨迹ID（变异追踪）
    
    # === 特征向量（用于检索） ===
    embedding: np.ndarray[12]       # 12维轨迹特征向量
    
    # === 评估结果（核心！用于LLM判断参考） ===
    evaluation: Dict = {
        # 物理验证结果
        "physics_validation": {
            "is_valid": bool,
            "max_long_accel": float,
            "max_lat_accel": float,
            "max_speed": float,
            "failed_rules": List[str],
            "warnings": List[str]
        },
        
        # RAG检索结果
        "rag_evaluation": {
            "max_similarity": float,       # 入库时的最高相似度
            "similar_cases": List[str],    # 相似案例ID列表
            "retrieval_strategy": str      # 检索策略
        },
        
        # LLM评估结果
        "llm_evaluation": {
            "is_reasonable": bool,
            "confidence": float,
            "reasoning": str,              # LLM评估理由（重要！）
            "danger_level": str,           # high/medium/low
            "danger_type": str,            # rear_end/cut_in/head_on/side_swipe
            "suggestions": str             # 改进建议
        },
        
        # 人工审核结果（如有）
        "human_review": {
            "reviewed_by": str,
            "review_status": str,          # approved/rejected/modified
            "corrected_label": str,
            "reviewer_notes": str
        }
    }
    
    # === 标签与分类 ===
    label: str                      # reasonable/unreasonable
    danger_type: str                # rear_end/cut_in/head_on/side_swipe
    danger_level: str               # high/medium/low
    evaluated_by: str               # human/llm_auto/rag
    
    # === 统计信息 ===
    query_count: int                # 被查询次数（热门案例统计）
    created_at: datetime
    updated_at: datetime
```

**为什么需要存储完整的评估结果**：

| 信息类型 | 用途 | 对LLM评估的帮助 |
|----------|------|-----------------|
| 轨迹数据 | 相似度计算、可视化 | 对比轨迹形状、运动模式 |
| 物理验证结果 | 硬性约束检查 | 判断是否符合物理规律 |
| LLM评估理由 | 语义解释 | 提供评估依据和逻辑参考 |
| 危险等级/类型 | 分类信息 | 帮助LLM理解场景性质 |
| 改进建议 | 优化方向 | 为不合理轨迹提供修改思路 |

### 3.3 知识库构建完整流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        知识库构建流程                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  阶段1：冷启动（初始种子库建立）                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  步骤：                                                               │ │
│  │    1. 选择10-20个典型危险场景                                           │ │
│  │    2. 穷举生成候选轨迹（每场景1000-5000条）                              │ │
│  │    3. 物理过滤（剔除率>90%）                                            │ │
│  │    4. 人工审核50-100条高质量案例                                         │ │
│  │    5. 提取特征向量 + 入库                                               │ │
│  │  输出：初始知识库（50-100条人工标注案例）                                │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                     ↓                                       │
│  阶段2：日常入库（系统运行期）                                                │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  触发条件：                                                            │ │
│  │    - LLM评估置信度≥0.8且is_reasonable=True                              │ │
│  │    - 人工审核通过的案例                                                 │ │
│  │    - 系统定期自检发现的代表性案例                                         │ │
│  │                                                                        │ │
│  │  入库流程：                                                             │ │
│  │    1. 提取12维特征向量                                                  │ │
│  │    2. 生成case_id                                                      │ │
│  │    3. 构造metadata（label, danger_type, evaluated_by等）                │ │
│  │    4. 序列化document（轨迹+评估结果JSON）                                │ │
│  │    5. 写入ChromaDB（向量+metadata+document）                            │ │
│  │    6. 写入SQLite knowledge_base表（索引）                               │ │
│  │    7. 更新相关统计信息                                                  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                     ↓                                       │
│  阶段3：知识库维护                                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  定期操作：                                                             │ │
│  │    - 去重检查（基于trajectory_hash）                                     │ │
│  │    - 相似度分布分析（检测聚类质量）                                       │ │
│  │    - 低质量案例清理（unreasonable比例>60%时分析原因）                    │ │
│  │    - 热门案例备份（query_count高的案例）                                  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 RAG检索实现机制

#### 3.4.1 检索流程

```
新候选轨迹
    ↓
┌───────────────────────────────────────────────────────────────────────┐
│ Step 1: 轨迹预处理                                                     │
│  • 提取12维特征向量（与知识库编码方式一致）                              │
│  • L2归一化                                                            │
│  • 转换为ChromaDB查询格式                                               │
└───────────────────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────────────────┐
│ Step 2: 相似度检索（ChromaDB）                                          │
│  • query_embeddings = [normalized_feature_vector]                      │
│  • n_results = top_k (默认5)                                           │
│  • include = ["metadatas", "documents", "distances"]                   │
│  • where = {"label": {"$in": ["reasonable", "unreasonable"]}}          │
└───────────────────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────────────────┐
│ Step 3: 结果解析与增强                                                  │
│  • 解析返回的相似案例列表                                                │
│  • 反序列化document获取完整案例信息                                       │
│  • 计算相似度统计：max_similarity, avg_similarity                       │
│  • 分类统计：合理/不合理案例数量、平均置信度                              │
└───────────────────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────────────────┐
│ Step 4: 检索策略决策                                                    │
│                                                                        │
│  IF max_similarity > 0.75:                                             │
│     → 标准RAG评估（直接参考最相似案例的评估结果）                         │
│                                                                        │
│  ELIF 0.5 < max_similarity ≤ 0.75:                                     │
│     → LLM辅助评估（结合相似案例 + LLM深度分析）                          │
│                                                                        │
│  ELSE:                                                                 │
│     → LLM深度分析（无足够相似案例，需要详细评估）                         │
│                                                                        │
└───────────────────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────────────────┐
│ Step 5: 相似案例格式化（供LLM使用）                                      │
│  • 提取相似案例的关键信息：                                               │
│    - case_id, similarity                                                │
│    - label, danger_type, danger_level                                   │
│    - llm_evaluation.reasoning                                           │
│    - llm_evaluation.suggestions                                         │
│  • 格式化文本（见下方Prompt模板）                                         │
└───────────────────────────────────────────────────────────────────────┘
```

#### 3.4.2 LLM评估Prompt设计（融入相似案例）

```markdown
## 轨迹合理性评估任务（RAG增强版）

### 背景
你正在评估自动驾驶测试场景中的危险轨迹。该轨迹是通过对真实驾驶场景进行参数变异生成的。

### 待评估轨迹信息
1. **场景描述**: {scenario_description}
2. **轨迹特征**:
   - 最大加速度: {max_accel} m/s²
   - 最小TTC: {min_ttc} s
   - 横向偏移范围: [{min_lat}, {max_lat}] m
   - 危险类型: {danger_type}

### 相似案例参考（来自知识库）
【案例1】相似度: {similarity_1}
- 评估标签: {label_1}（{reasoning_1}）
- 危险等级: {danger_level_1}
- 评估理由: {evaluation_reasoning_1}

【案例2】相似度: {similarity_2}
- 评估标签: {label_2}（{reasoning_2}）
- 危险等级: {danger_level_2}
- 评估理由: {evaluation_reasoning_2}

【案例统计】
- 相似案例总数: {total_similar_cases}
- 合理案例数: {reasonable_count}
- 不合理案例数: {unreasonable_count}
- 平均相似度: {avg_similarity}

### 评估标准
1. **物理合理性**: 轨迹是否符合车辆动力学？（参考物理验证结果）
2. **场景合理性**: 这种驾驶行为在现实中是否可能发生？（参考相似案例）
3. **危险程度**: 是否构成有效的安全测试场景？
4. **与相似案例一致性**: 是否与高度相似的已知案例评估结果一致？

### 输出格式
```json
{
    "is_reasonable": true/false,
    "confidence": 0.85,
    "reasoning": "详细评估理由，需说明与相似案例的对比分析",
    "danger_level": "high/medium/low",
    "suggestions": "如不合理，提供具体改进建议（参考相似案例的改进方向）",
    "reference_cases": ["case_001", "case_002"],
    "evaluation_strategy": "rag_standard/rag_assisted/llm_deep"
}
```
```

---

## 四、数据库详细设计（修订版）

### 4.1 数据库选型与架构

| 用途 | 技术选型 | 选型理由 |
|------|----------|----------|
| 关系型数据存储 | SQLite | 本地轻量级、零配置、Python原生支持 |
| 向量检索 | ChromaDB | 专为RAG设计、支持持久化、本地部署 |
| 配置存储 | config.yaml | 结构化配置、人类可读、支持环境变量 |

**知识库与数据库的关系**：
- **知识库** = SQLite中的`knowledge_base`表（索引）+ ChromaDB中的向量数据（检索）
- **完整数据** = SQLite中的`trajectory_records` + `evaluation_results` + `human_reviews`
- **知识库是数据库的子集**，仅包含经评估的高质量案例

### 4.2 修订后的表结构

#### 4.2.1 轨迹记录表（trajectory_records）

| 字段名 | 数据类型 | 约束 | 说明 |
|--------|----------|------|------|
| `trajectory_id` | TEXT | PRIMARY KEY | 轨迹唯一标识 |
| `scenario_id` | TEXT | NOT NULL, FK | 关联场景ID |
| `vehicle_id` | TEXT | NOT NULL | 车辆ID |
| `parent_trajectory_id` | TEXT | FK | 父轨迹ID（变异追踪） |
| `trajectory_data` | BLOB | - | 序列化的轨迹点列表 |
| `feature_vector` | BLOB | - | 12维特征向量（numpy数组） |
| `generation_params` | TEXT | - | 生成参数JSON |
| `danger_type` | TEXT | - | 危险类型枚举 |
| `danger_level` | TEXT | - | 危险等级 |
| `trajectory_hash` | TEXT | UNIQUE | 轨迹数据指纹（去重） |
| `is_validated` | BOOLEAN | DEFAULT 0 | 物理验证状态 |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 创建时间 |

#### 4.2.2 评估结果表（evaluation_results）

| 字段名 | 数据类型 | 约束 | 说明 |
|--------|----------|------|------|
| `evaluation_id` | TEXT | PRIMARY KEY | 评估记录唯一标识 |
| `trajectory_id` | TEXT | NOT NULL, FK | 关联轨迹ID |
| `physics_validation` | TEXT | - | 物理验证详情JSON（含通过状态、指标、失败规则） |
| `is_physics_valid` | BOOLEAN | DEFAULT 0 | 物理验证是否通过 |
| `feature_vector` | BLOB | - | 12维特征向量（冗余存储，便于检索） |
| `max_similarity` | REAL | - | RAG最高相似度 |
| `avg_similarity` | REAL | - | RAG平均相似度 |
| `similar_cases` | TEXT | - | 相似案例列表JSON（含case_id, similarity, label） |
| `llm_is_reasonable` | BOOLEAN | - | LLM评估是否合理 |
| `llm_confidence` | REAL | - | LLM置信度 |
| `llm_reasoning` | TEXT | - | LLM评估理由 |
| `llm_suggestions` | TEXT | - | LLM改进建议 |
| `danger_level` | TEXT | - | 危险等级（LLM评估） |
| `evaluation_strategy` | TEXT | - | 评估策略（rag_standard/rag_assisted/llm_deep） |
| `needs_human_review` | BOOLEAN | DEFAULT 1 | 是否需要人工审核 |
| `final_status` | TEXT | DEFAULT 'pending' | 最终状态 |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 创建时间 |

#### 4.2.3 知识库表（knowledge_base）

| 字段名 | 数据类型 | 约束 | 说明 |
|--------|----------|------|------|
| `case_id` | TEXT | PRIMARY KEY | 案例唯一标识 |
| `trajectory_id` | TEXT | NOT NULL, FK | 关联轨迹ID |
| `evaluation_id` | TEXT | NOT NULL, FK | 关联评估结果ID |
| `embedding` | BLOB | - | 序列化的12维向量（ChromaDB备份） |
| `label` | TEXT | NOT NULL | 标签：reasonable/unreasonable |
| `danger_type` | TEXT | - | 危险类型 |
| `danger_level` | TEXT | - | 危险等级 |
| `evaluated_by` | TEXT | - | 评估来源：human/llm_auto |
| `confidence` | REAL | - | 置信度 |
| `evaluation_reasoning` | TEXT | - | 评估理由摘要 |
| `similar_cases` | TEXT | - | 入库时的相似案例ID列表 |
| `query_count` | INTEGER | DEFAULT 0 | 被查询次数 |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 创建时间 |
| `updated_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 更新时间 |

**ChromaDB集合结构**（`trajectory_cases`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `ids` | List[str] | case_id列表 |
| `embeddings` | List[List[float]] | 12维特征向量列表 |
| `metadatas` | List[Dict] | 元数据字典列表（label, danger_type, evaluated_by等） |
| `documents` | List[str] | 案例JSON序列化（完整轨迹+评估结果） |

### 4.3 数据流转详细设计

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              数据写入流程                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Waymo原始数据                                                                │
│       │                                                                       │
│       ▼                                                                       │
│  ┌─────────────┐    SQLite: scenario_metadata                                 │
│  │ 数据加载模块 │ ────────────────────────────>                                │
│  └─────────────┘                                                              │
│       │                                                                       │
│       ▼                                                                       │
│  ┌─────────────┐    SQLite: trajectory_records                                │
│  │ 穷举生成模块 │ ──候选轨迹（物理验证前）────────>                             │
│  └─────────────┘                                                              │
│       │                                                                       │
│       ▼                                                                       │
│  ┌─────────────┐    SQLite: trajectory_records (更新is_validated)             │
│  │ 物理验证模块 │ ───通过验证的轨迹──────────────>                              │
│  └─────────────┘                                                              │
│       │                                                                       │
│       ▼                                                                       │
│  ┌─────────────┐    SQLite: evaluation_results                                │
│  │ RAG评估模块 │ ──评估结果（含相似案例信息）──────>                            │
│  │             │                                                              │
│  │  ┌───────┐ │    ChromaDB: trajectory_cases (查询)                          │
│  │  │相似度 │ │    ──查询相似案例───────────────>                               │
│  │  │ 检索  │ │    <──返回相似案例───────────────                               │
│  │  └───────┘ │                                                              │
│  └─────────────┘                                                              │
│       │                                                                       │
│       ▼                                                                       │
│  ┌─────────────┐    SQLite: human_reviews                                     │
│  │ 人工审核模块 │ ──审核记录───────────────────>                               │
│  └─────────────┘                                                              │
│       │                                                                       │
│       ├─ 审核通过 ──> SQLite: knowledge_base                                  │
│       │               ChromaDB: trajectory_cases (写入)                       │
│       │               ──案例入库（向量+metadata+document）──>                  │
│       │                                                                       │
│       └─ 审核拒绝 ──> 结束                                                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 五、项目目录结构（修订版）

```
TrajectoryAnalysis/
│
├── docs/                                      # 文档目录
│   ├── README.md                              # 项目说明
│   ├── 工程版框架_穷举剪枝版.md               # 原始框架（保留）
│   ├── 工程版框架_穷举剪枝版_修订版.md        # 本次修订版（新增）
│   ├── 学术版框架.md                          # 学术版框架
│   ├── 项目执行计划_详细版.md                 # 执行计划
│   ├── architecture.md                        # 详细架构设计（新增）
│   ├── api_reference.md                       # API接口文档（新增）
│   ├── data_format_spec.md                    # 数据格式规范（新增）
│   ├── rag_design.md                          # RAG设计文档（新增）
│   └── learning_notes.md                      # 学习笔记
│
├── src/                                       # 源代码目录
│   ├── __init__.py
│   │
│   ├── core/                                  # 核心处理模块
│   │   ├── __init__.py
│   │   ├── data_loader.py                     # Step 1: 数据加载
│   │   ├── risk_calculator.py                 # Step 2: 风险计算
│   │   ├── fragment_extractor.py              # Step 2: 片段截取
│   │   ├── feature_extractor.py               # Step 3: 特征提取
│   │   ├── trajectory_generator.py            # Step 4: 穷举生成器（新增/完善）
│   │   ├── physics_validator.py               # Step 5: 物理验证
│   │   └── carla_adapter.py                   # Step 9: CARLA导出
│   │
│   ├── rag/                                   # RAG知识库模块（新增目录）
│   │   ├── __init__.py
│   │   ├── embedding_encoder.py               # 向量化编码器
│   │   ├── knowledge_base.py                  # ChromaDB管理
│   │   ├── similarity_search.py               # 相似度检索（新增）
│   │   ├── case_formatter.py                  # 案例格式化（新增）
│   │   ├── rag_evaluator.py                   # Step 6-7: RAG评估
│   │   └── llm_interface.py                   # LLM接口封装（新增）
│   │
│   ├── database/                              # 数据库模块（新增目录）
│   │   ├── __init__.py
│   │   ├── models.py                          # 数据模型定义
│   │   ├── sqlite_manager.py                  # SQLite操作封装
│   │   ├── chromadb_manager.py                # ChromaDB操作封装
│   │   ├── repositories.py                    # 数据访问层（Repository模式）
│   │   └── migration.py                       # 数据库迁移脚本
│   │
│   ├── generation/                            # 轨迹生成模块（新增目录）
│   │   ├── __init__.py
│   │   ├── param_space.py                     # 参数空间定义
│   │   ├── sampling_strategies.py             # 采样策略实现
│   │   ├── variation_operators.py             # 变异操作算子
│   │   ├── trajectory_smoothing.py            # 轨迹平滑处理
│   │   └── batch_generator.py                 # 批量生成控制器
│   │
│   ├── ui/                                    # 用户界面
│   │   ├── __init__.py
│   │   ├── app.py                             # Streamlit主应用
│   │   ├── scene_panel.py                     # 场景选择面板
│   │   ├── generation_panel.py                # 生成控制面板
│   │   ├── review_panel.py                    # 审核面板
│   │   └── visualization.py                   # 可视化组件
│   │
│   ├── utils/                                 # 工具函数
│   │   ├── __init__.py
│   │   ├── data_structures.py                 # 数据类定义
│   │   ├── trajectory_utils.py                # 轨迹处理工具
│   │   ├── logger.py                          # 日志管理
│   │   └── config.py                          # 配置管理
│   │
│   ├── Data_INFO.py                           # 数据信息查看（保留）
│   ├── Data_Processor.py                      # 基础数据处理（保留）
│   ├── Data_Processor_Enhanced.py             # 增强版数据处理（保留）
│   ├── database_schema.py                     # 数据库schema（保留，逐步迁移到database/）
│   ├── Main.py                                # 主程序入口
│   └── scenario_visualization.py              # 场景可视化
│
├── tests/                                     # 测试目录
│   ├── __init__.py
│   ├── conftest.py                            # pytest配置
│   ├── unit/                                  # 单元测试
│   │   ├── test_data_loader.py
│   │   ├── test_trajectory_generator.py
│   │   ├── test_physics_validator.py
│   │   └── test_rag_evaluator.py
│   ├── integration/                           # 集成测试（新增）
│   │   ├── test_generation_pipeline.py
│   │   └── test_rag_retrieval.py
│   └── e2e/                                   # E2E测试（新增）
│       └── test_full_pipeline.py
│
├── data/                                      # 数据目录（gitignore）
│   ├── raw/                                   # 原始Waymo数据（pkl）
│   ├── processed/                             # 处理后数据
│   ├── generated/                             # 生成的候选轨迹
│   ├── validated/                             # 物理验证通过
│   ├── evaluated/                             # LLM评估结果
│   ├── reviewed/                              # 人工审核通过
│   ├── knowledge_base/                        # 知识库数据（新增）
│   │   ├── chroma_db/                         # ChromaDB持久化
│   │   └── backups/                           # 知识库备份
│   ├── carla_ready/                           # CARLA可导入格式
│   └── trajectory_database.db                 # SQLite数据库文件
│
├── scripts/                                   # 脚本工具
│   ├── run_pipeline.py                        # 一键运行流水线
│   ├── run_tests.py                           # 运行测试套件
│   ├── setup_db.py                            # 数据库初始化
│   ├── build_knowledge_base.py                # 知识库构建脚本（新增）
│   └── evaluate_trajectories.py               # 批量评估脚本（新增）
│
├── configs/                                   # 配置文件目录（新增）
│   ├── default.yaml                           # 默认配置
│   ├── generation_params.yaml                 # 生成参数配置
│   └── rag_config.yaml                        # RAG配置
│
├── requirements.txt                           # 依赖管理
├── setup.py                                   # 包安装配置（新增）
├── .gitignore                                 # Git忽略规则
└── version.txt                                # 版本号
```

---

## 六、核心模块详细设计（修订版）

### 6.1 穷举生成器（generation/batch_generator.py）

#### 6.1.1 类定义

```python
class ExhaustiveTrajectoryGenerator:
    """
    穷举轨迹生成器
    基于参数空间系统性地生成候选轨迹
    """
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.param_space = ParamSpace(config.param_ranges)
        self.sampler = SamplingStrategyFactory.create(config.sampling_strategy)
        self.variation_ops = VariationOperators()
        self.validator = PhysicsValidator(config.constraints)
        
    def generate(self, base_trajectory: Dict, target_count: Optional[int] = None) -> List[Dict]:
        """
        生成候选轨迹
        
        Args:
            base_trajectory: 原始危险片段轨迹
            target_count: 目标生成数量（None则自动计算）
            
        Returns:
            List[Dict]: 候选轨迹列表（已物理验证）
        """
        # 1. 计算目标数量
        if target_count is None:
            target_count = self._calculate_target_count()
            
        # 2. 关键帧采样
        keyframes = self._sample_keyframes(base_trajectory)
        
        # 3. 参数采样
        param_combinations = self.sampler.sample(
            self.param_space, 
            target_count
        )
        
        # 4. 批量生成与验证
        candidates = []
        for params in param_combinations:
            variant = self._generate_variant(base_trajectory, keyframes, params)
            if self.validator.validate(variant):
                candidates.append(variant)
                
        return candidates
        
    def _calculate_target_count(self) -> int:
        """根据参数空间和计算资源动态计算目标数量"""
        total_combinations = self.param_space.total_combinations()
        max_candidates = self.config.max_candidates
        
        if total_combinations <= max_candidates:
            return int(total_combinations * self.config.length_factor)
        else:
            # 智能采样模式下返回配置的上限
            return max_candidates
```

### 6.2 RAG评估器（rag/rag_evaluator.py）

#### 6.2.1 类定义

```python
class RAGEvaluator:
    """
    RAG检索与评估器
    整合向量检索和LLM评估
    """
    
    def __init__(self, 
                 chroma_manager: ChromaDBManager,
                 llm_interface: LLMInterface,
                 config: RAGConfig):
        self.chroma = chroma_manager
        self.llm = llm_interface
        self.config = config
        self.case_formatter = CaseFormatter()
        
    def evaluate_trajectory(self, trajectory: Dict, feature_vector: np.ndarray) -> EvaluationResult:
        """
        评估单条轨迹
        
        Args:
            trajectory: 候选轨迹数据
            feature_vector: 12维特征向量
            
        Returns:
            EvaluationResult: 评估结果
        """
        # 1. RAG检索相似案例
        similar_cases = self.chroma.query_similar(
            feature_vector,
            top_k=self.config.top_k
        )
        
        # 2. 计算相似度统计
        similarity_stats = self._compute_similarity_stats(similar_cases)
        
        # 3. 确定评估策略
        strategy = self._determine_strategy(similarity_stats)
        
        # 4. 根据策略执行评估
        if strategy == "rag_standard":
            result = self._rag_standard_eval(trajectory, similar_cases)
        elif strategy == "rag_assisted":
            result = self._rag_assisted_eval(trajectory, similar_cases, similarity_stats)
        else:
            result = self._llm_deep_eval(trajectory)
            
        # 5. 组装评估结果
        return EvaluationResult(
            trajectory_id=trajectory['id'],
            similarity_stats=similarity_stats,
            similar_cases=[c.case_id for c in similar_cases],
            llm_result=result,
            evaluation_strategy=strategy,
            needs_human_review=(strategy == "llm_deep" or result.confidence < 0.8)
        )
        
    def _determine_strategy(self, stats: SimilarityStats) -> str:
        """根据相似度统计确定评估策略"""
        if stats.max_similarity > 0.75:
            return "rag_standard"
        elif stats.avg_similarity > 0.5:
            return "rag_assisted"
        else:
            return "llm_deep"
```

---

## 七、总结：本次修订的主要改动

### 7.1 核心改进点

| 方面 | 原框架问题 | 修订版解决方案 |
|------|-----------|---------------|
| **RAG知识库** | 仅存储轨迹向量和简单标签 | 存储完整案例（轨迹+详细评估结果），为LLM提供充足上下文 |
| **RAG检索机制** | 仅描述概念，无实现细节 | 完整的检索流程：向量编码→ChromaDB查询→结果解析→策略决策 |
| **知识库与数据库关系** | 关系不明确 | 明确知识库是数据库的逻辑子集，SQLite存储元数据，ChromaDB存储向量 |
| **原始数据介绍** | 缺少数据格式说明 | 详细说明Waymo数据结构、可用变量、可计算的派生变量 |
| **穷举数量限制** | 固定1000-5000条 | 动态自适应策略，根据参数空间和计算资源自动调整 |
| **轨迹变异流程** | 概念描述，步骤不详细 | 从原始数据输入到危险轨迹输出的8步完整流程，细致到每个操作 |
| **参数空间配置** | 简单表格 | 完整的参数定义、组合计算、采样策略、动态数量计算 |

### 7.2 新增内容清单

1. **文档新增**：
   - `docs/工程版框架_穷举剪枝版_修订版.md`（本文档）
   - `docs/architecture.md`（详细架构设计）
   - `docs/data_format_spec.md`（数据格式规范）
   - `docs/rag_design.md`（RAG设计文档）

2. **代码模块新增**：
   - `src/generation/`（轨迹生成专用模块）
   - `src/database/`（数据库操作专用模块）
   - `src/rag/`（RAG专用模块，从core中分离）
   - `tests/integration/`（集成测试）
   - `tests/e2e/`（E2E测试）
   - `configs/`（配置文件目录）

3. **脚本新增**：
   - `scripts/build_knowledge_base.py`（知识库构建）
   - `scripts/evaluate_trajectories.py`（批量评估）

### 7.3 保持不变的原有设计

- 整体技术路线（穷举+LLM剪枝）
- 学术版核心数据流（Step 1-9）
- 12维特征向量设计
- 物理约束规则
- 项目根目录结构的基本逻辑

---

*文档版本: v2.1（修订版）*
*技术方案: 穷举生成 + LLM剪枝 + RAG增强*
*最后更新: 2024年*
