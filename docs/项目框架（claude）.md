# 危险轨迹生成与评估系统 - 穷举生成+LLM剪枝版

## 项目概述

本项目采用**穷举参数变异 + LLM智能剪枝**的技术路线，通过系统性地遍历参数空间生成大量候选轨迹，再利用LLM的语义理解能力进行合理性评估和筛选，从而高效生成符合物理约束的危险驾驶场景。

**核心思想**：让LLM做它最擅长的事（评估/判断），让计算机做它最擅长的事（批量生成/计算）。

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
[Step 3] 轨迹特征提取与向量化（11维特征）
    ↓
【穷举生成阶段】
    ↓
[Step 4a] 参数空间定义（加速度偏移、速度偏移等）
    ↓
[Step 4b] 穷举生成候选轨迹（1000-5000条）
变异方法：
1-不逐帧修改，而只修改轨迹中的几个关键时间点，然后用平滑曲线链接各点的轨迹
2-不直接修改轨迹x和y值，而是修改方向、速度、加速度等，通过物理公式计算轨迹点位置
    ↓
[Step 5] 物理约束验证（硬过滤，剔除90%）
    ↓
【LLM剪枝阶段】
    ↓
[Step 6] RAG检索相似案例（向量化检索）
    ↓
[Step 7] LLM合理性评估（剪枝至10-50条）
    ↓
[Step 8] 人工审核与知识库更新
    ↓
[Step 9] CARLA格式导出与闭环测试
```

#### 2.2.1 Waymo原始数据格式说明

本系统处理的原始数据来自Waymo开放数据集，存储格式为`.pkl`（Python pickle）文件。每个场景文件包含一个场景的完整时序数据。

**数据结构定义**：

Waymo原始数据结构包含以下主要部分：

| 顶层字段 | 说明 |
|----------|------|
| object_tracks | 场景中所有动态物体的轨迹字典，key为物体ID |
| map_features | 静态地图数据（车道线、交通标志等） |
| dynamic_map_states | 动态地图状态（交通灯等） |
| extra_information | 额外信息 |

**object_tracks中每个物体的数据结构**：

| 字段 | 类型 | 说明 |
|------|------|------|
| type | enum | 物体类型：VEHICLE（车辆）、PEDESTRIAN（行人）、CYCLIST（骑车人）等 |
| state | dict | 物体状态数据 |
| state.action | ndarray | 动作数据 |
| state.global_center | ndarray (N, 3) | 全局位置 [x, y, z]，单位：米 |
| state.heading | ndarray (N,) | 航向角，单位：弧度 |
| state.local_acceleration | ndarray (N, 2) | 局部加速度 [ax, ay]，单位：m/s² |
| state.local_velocity | ndarray (N, 2) | 局部速度 [vx, vy]，单位：m/s |
| state.size | ndarray (N, 3) | 物体尺寸 [长, 宽, 高]，单位：米 |
| state.valid | ndarray (N,) bool | 该帧数据是否有效 |

**extra_information包含**：

| 字段 | 类型 | 说明 |
|------|------|------|
| sdc_id | str | 自车（Self-Driving Car）的ID |
| scene_length | int | 场景总帧数 |
| timestamp | ndarray | 时间戳序列 |

**关键变量说明**：

| 变量名 | 数据类型 | 形状 | 单位 | 物理意义 |
|--------|----------|------|------|----------|
| `global_center` | float | (N, 3) | m | 物体在全局坐标系下的位置 |
| `heading` | float | (N,) | rad | 航向角（车头方向与X轴夹角） |
| `local_velocity` | float | (N, 2) | m/s | 物体在自身坐标系下的速度 [纵向, 横向] |
| `local_acceleration` | float | (N, 2) | m/s² | 物体在自身坐标系下的加速度 [纵向, 横向] |
| `size` | float | (N, 3) | m | 物体尺寸 [长, 宽, 高] |
| `valid` | bool | (N,) | - | 该帧数据是否有效 |

**坐标系说明**：
- Waymo使用右手坐标系（X轴向前，Y轴向左，Z轴向上）
- `local_velocity` 和 `local_acceleration` 是在物体自身坐标系下的表示
- 自身坐标系的X轴与物体的heading方向对齐

**可用于轨迹变异的原始变量**：

| 变量 | 计算衍生方式 | 变异操作 |
|------|-------------|----------|
| 速度 `speed` | `norm(local_velocity)` | speed_scale (缩放) |
| 加速度 `accel` | `norm(local_acceleration)` | accel_offset (偏移) |
| 位置 `(x, y)` | `global_center[:, 0:2]` | lateral_offset (横向偏移) |
| 航向角 `heading` | 直接使用 | heading_bias (角度偏移) |
| 时间 `timestamp` | 直接使用 | time_shift (时间偏移) |

**场景基本信息**：
- 采样率：10 Hz（每帧0.1秒）
- 典型场景长度：100-200帧（10-20秒）
- 物体类型：VEHICLE（车辆）、PEDESTRIAN（行人）、CYCLIST（骑车人）等

---

### 2.2.2 穷举生成模块设计（参数空间与采样策略）

#### 2.2.2.1 变异参数定义

| 参数名             | 物理意义       | 取值范围        | 步长 | 说明           | 变异公式 |
| ------------------ | -------------- | --------------- | ---- | -------------- | -------- |
| `accel_offset`   | 纵向加速度偏移 | [-4, 4] m/s²   | 0.5  | 模拟加减速变化 | `accel_new = accel_base + offset` |
| `speed_scale`    | 速度缩放因子   | [0.7, 1.5]      | 0.1  | 模拟速度变化   | `vel_new = vel_base × scale` |
| `lateral_offset` | 横向位置偏移   | [-2, 2] m       | 0.3  | 模拟换道/切入  | `y_new = y_base + offset` |
| `time_shift`     | 时间轴偏移     | [-1, 1] s       | 0.2  | 模拟时机变化   | 插值重采样 |
| `heading_bias`   | 航向角偏移     | [-0.3, 0.3] rad | 0.1  | 模拟角度变化   | `heading_new = heading_base + bias` |

#### 2.2.2.2 参数空间与采样数量计算

**参数空间大小计算**：

**参数空间定义**：

| 参数名 | 取值范围 | 步长 | 值数量 | 变异公式 |
|--------|----------|------|--------|----------|
| accel_offset | [-4, 4] m/s² | 0.5 | 17 | accel_new = accel_base + offset |
| speed_scale | [0.7, 1.5] | 0.1 | 9 | vel_new = vel_base × scale |
| lateral_offset | [-2, 2] m | 0.3 | 14 | y_new = y_base + offset |
| time_shift | [-1, 1] s | 0.2 | 11 | 插值重采样 |
| heading_bias | [-0.3, 0.3] rad | 0.1 | 6 | heading_new = heading_base + bias |

**全参数空间组合数**：17 × 9 × 14 × 11 × 6 = **141,372** 种

**采样数量确定原则**：

| 原则 | 说明 | 计算方式 |
|------|------|---------|
| 参数空间覆盖率 | 确保每个参数维度都被采样到 | `n_samples ≥ max(各参数值数量) × 采样率` |
| 统计显著性 | 确保生成数量满足统计分析需求 | `n_samples ≥ 30`（中心极限定理） |
| 物理可行性过滤 | 过滤后需保留足够样本 | `n_candidates = n_samples / 物理过滤率(约10%)` |

**推荐采样数量计算公式**：

```
n_recommended = ceil(参数空间大小 / 降采样率)
             = ceil(141,372 / 降采样率)

降采样率选择：
- 保守策略（降采样率=10）：约 14,000 条候选轨迹
- 平衡策略（降采样率=20）：约 7,000 条候选轨迹  
- 激进策略（降采样率=30）：约 4,700 条候选轨迹
```

**物理约束预过滤率估算**：
- 基于物理规则（加速度、速度、碰撞检测）预期过滤：70-90%
- RAG+LLM评估预期过滤：80-95%

**最终期望通过数量**：
```
n_final = n_candidates × (1 - 物理过滤率) × (1 - LLM过滤率)
        ≈ 7,000 × 0.2 × 0.2 ≈ 280 条
```

#### 2.2.2.3 采样策略

| 策略         | 说明                     | 适用场景       | 采样比例 |
| ------------ | ------------------------ | -------------- | -------- |
| 网格采样     | 均匀覆盖参数空间每个维度 | 冷启动阶段     | 100%     |
| 危险导向采样 | 优先TTC<1s等高风险区域   | 已知危险模式后 | 150%     |
| 随机采样     | 随机选取参数组合         | 快速验证       | 50%      |
| 分层采样     | 按危险类型分层后再采样   | 多样性要求高   | 80%      |
| 自适应采样   | 根据评估结果动态调整密度 | 迭代优化阶段   | 可变     |

**采样策略说明**：

1. **网格采样法（推荐作为基线）**：
   - 原理：对参数空间进行网格划分，确保每个参数维度都被均匀覆盖
   - 优点：参数空间覆盖完整，可解释性强
   - 适用场景：冷启动阶段，作为基线策略

2. **分层采样法（推荐用于多样性）**：
   - 原理：先按危险类型（追尾、切入、正面碰撞等）对参数空间分层，再在各层内进行网格采样
   - 优点：保证生成轨迹的多样性
   - 适用场景：需要覆盖多种危险类型时

3. **自适应采样法（推荐用于迭代优化）**：
   - 原理：根据前一轮评估结果动态调整采样密度，高通过率区域增加采样，低通过率区域减少采样
   - 优点：提高生成效率，聚焦于更有希望的参数区域
   - 适用场景：迭代优化阶段

4. **危险导向采样**：
   - 原理：优先在TTC<1s等高风险参数区域进行采样
   - 适用场景：已知危险模式后，提高危险轨迹生成概率

5. **随机采样**：
   - 原理：从参数空间中随机选取参数组合
   - 优点：实现简单，快速
   - 适用场景：快速验证

---

## 三、项目目录结构

```
LLM-2/
├── docs/                           # 文档目录（版本控制）
│   ├── README.md                   # 项目说明
│   ├── CHANGELOG.md                # 版本变更日志
│   ├── architecture.md             # 架构设计文档
│   ├── api_reference.md            # API接口文档
│   └── learning_notes.md           # 学习笔记（面试准备）
│
├── tests/                          # 测试目录（每功能必测）
│   ├── __init__.py
│   ├── test_data_loader.py         # 数据加载测试
│   ├── test_trajectory_generator.py # 轨迹生成测试
│   ├── test_physics_validator.py   # 物理验证测试
│   ├── test_rag_evaluator.py       # RAG评估测试
│   └── conftest.py                 # pytest配置
│
├── core/                           # 核心处理模块（已实现✓）
│   ├── __init__.py                 # 模块导出
│   └── processor.py                # 整合模块：数据类型、数据加载、风险计算、片段截取
│   # 以下模块待实现
│   ├── trajectory_generator.py     # Step 5: 穷举生成器
│   ├── physics_validator.py        # Step 6: 物理验证
│   └── carla_adapter.py            # Step 9: CARLA导出
│
├── rag/                            # RAG知识库模块
│   ├── __init__.py
│   ├── embedding_encoder.py        # 向量化编码器
│   ├── knowledge_base.py           # ChromaDB管理
│   ├── rag_evaluator.py            # Step 6-7: RAG评估
│   └── llm_evaluator.py            # LLM评估接口
│
├── ui/                             # 用户界面（Streamlit）
│   ├── __init__.py
│   ├── app.py                      # 主应用入口
│   ├── scene_panel.py              # 场景选择面板
│   ├── generation_panel.py         # 生成控制面板
│   ├── review_panel.py             # Step 8: 审核面板
│   └── visualization.py            # 可视化组件
│
├── utils/                          # 工具函数
│   ├── __init__.py
│   ├── data_structures.py          # 数据类定义
│   ├── trajectory_utils.py         # 轨迹处理工具
│   ├── logger.py                   # 日志管理
│   └── config.py                   # 配置管理
│
├── data/                           # 数据目录（gitignore）
│   ├── waymo-open/                # 原始Waymo数据（pkl）
│   ├── processed/                  # 处理后数据（JSON，第一阶段）
│   ├── generated/                  # 生成的候选轨迹
│   ├── validated/                  # 物理验证通过
│   ├── evaluated/                  # LLM评估结果
│   ├── reviewed/                   # 人工审核通过
│   └── chroma_db/                  # 向量数据库
│
├── scripts/                        # 脚本工具
│   ├── run_pipeline.py             # 一键运行流水线
│   ├── run_tests.py                # 运行测试套件
│   └── setup_db.py                 # 数据库初始化
│
├── requirements.txt                # 依赖管理
├── config.yaml                     # 配置文件
├── .gitignore                      # Git忽略规则
└── version.txt                     # 版本号
```

---

## 四、核心模块详细设计

### 4.1 穷举生成器 (core/trajectory_generator.py)

#### 4.1.1 输入输出

| 属性           | 说明                            |
| -------------- | ------------------------------- |
| **输入** | 原始危险片段轨迹 + 参数空间配置 |
| **输出** | 候选轨迹列表（1000-5000条）     |
| **性能** | 本地生成，1000条约需10秒        |

#### 4.1.2 生成算法

轨迹生成的核心逻辑包括以下步骤：

1. **参数采样**：根据选定的采样策略（如网格采样、分层采样等），从参数空间中选取一组变异参数

2. **变异应用**：将采样得到的参数应用到原始轨迹上，生成变异后的轨迹。变异操作包括：
   - 速度缩放：改变轨迹的速度大小
   - 加速度偏移：在原有加速度基础上增加偏移量
   - 横向偏移：改变轨迹的横向位置
   - 时间偏移：调整轨迹的时间轴
   - 航向偏移：改变轨迹的行驶方向

3. **轨迹平滑**：对变异后的轨迹进行平滑处理，确保轨迹的连续性和物理合理性

4. **结果收集**：将生成的候选轨迹加入候选集，重复直到达到目标数量

#### 4.1.3 变异操作

**变异方法：**

1. 不逐帧修改，而只修改轨迹中的几个关键时间点，然后用平滑曲线链接各点的轨迹
2. 不直接修改轨迹x和y值，而是修改方向、速度、加速度等，通过物理公式计算轨迹点位置

| 变异类型   | 操作                                  | 物理意义       |
| ---------- | ------------------------------------- | -------------- |
| 加速度偏移 | `accel_new = accel_base + offset`   | 改变加减速行为 |
| 速度缩放   | `vel_new = vel_base * scale`        | 改变车速       |
| 横向偏移   | `y_new = y_base + offset`           | 改变车道位置   |
| 时间偏移   | 时间轴平移                            | 改变交互时机   |
| 航向偏移   | `heading_new = heading_base + bias` | 改变行驶方向   |

#### 4.1.4 轨迹长度与碰撞点设计

**轨迹长度配置**：

轨迹长度由用户在生成时选择，不做硬性约束。用户可根据测试需求选择合适的轨迹长度：

| 轨迹长度 | 适用场景 |
|----------|----------|
| 短（3-5秒） | 快速验证避障反应 |
| 中（6-10秒） | 标准测试场景 |
| 长（10-15秒） | 复杂场景、多次交互 |

**碰撞点控制机制**：

碰撞点位置由轨迹生成算法根据变异参数自动控制，而非固定在某个时间点：

1. **变异影响碰撞时机**：通过速度缩放、时间偏移等参数控制两车何时接近到危险距离
2. **物理约束确保有效性**：碰撞点必须在轨迹有效范围内，且碰撞后有足够帧数展示后效
3. **用户可配置危险程度**：通过参数空间配置控制TTC的最小值出现时间

**碰撞点后轨迹必须保留**：

碰撞点后的轨迹是评估算法避障能力的关键数据，必须保留：

| 方案 | 碰撞后行为 | 适用场景 |
|------|------------|----------|
| 保守型 | 继续行驶或缓慢减速 | 给主车让出避障空间 |
| 激进型 | 保持威胁或加速离开 | 测试极限避障能力 |

**关键原则**：
- 碰撞点前：制造威胁（加速、切入、靠近）
- 碰撞点后：给空间（继续行驶或缓慢减速）
- 目的：测试算法能否利用这个空间完成避障

---

### 4.2 物理验证器 (core/physics_validator.py)

#### 4.2.1 硬约束规则

| 规则ID   | 检查项         | 阈值         | 失败处理               |
| -------- | -------------- | ------------ | ---------------------- |
| PHYS-001 | 最大纵向加速度 | ≤ 6 m/s²   | 剔除                   |
| PHYS-002 | 最大纵向减速度 | ≤ 8 m/s²   | 剔除                   |
| PHYS-003 | 最大横向加速度 | ≤ 4 m/s²   | 剔除                   |
| PHYS-004 | 最大速度       | ≤ 35 m/s    | 剔除                   |
| PHYS-005 | 最小速度       | ≥ 0 m/s     | 剔除（倒车需特殊处理） |
| PHYS-006 | 轨迹连续性     | 无断点       | 剔除                   |
| PHYS-007 | 碰撞检测       | 与自车无碰撞 | 剔除（除非故意）       |

#### 4.2.2 软约束规则（警告）

| 规则ID   | 检查项     | 阈值        | 处理         |
| -------- | ---------- | ----------- | ------------ |
| WARN-001 | Jerk变化率 | ≤ 10 m/s³ | 标记但不剔除 |
| WARN-002 | 曲率变化   | 连续        | 标记但不剔除 |

#### 4.2.3 碰撞检测算法

采用**点距离检测法**进行精确碰撞检测：

**原理**：将车辆简化为4个角点，检测任意两车角点之间的最小距离。

```
     车辆角点
        Front
          ↑
    p1 ──────── p2
     │    ●●●●  │
     │    ●●●●  │  ← 4个角点(p1,p2,p3,p4)
    p4 ──────── p3
```

**检测逻辑**：

碰撞检测采用点距离检测法，核心步骤如下：

1. **角点计算**：根据车辆尺寸（长、宽）和当前帧的航向角，计算车辆在全局坐标系下的4个角点位置

2. **距离计算**：遍历两车所有对应帧的角点组合，计算每对角点之间的欧氏距离，找出最小距离

3. **碰撞判定**：将最小距离与阈值比较，若小于阈值则判定为碰撞

**阈值选择建议**：
| 阈值 | 适用场景 |
|------|---------|
| 0.2m | 高精度要求，严格检测 |
| 0.3m | **推荐默认值** |
| 0.5m | 宽松检测，减少误报 |

---

### 4.3 LLM评估器 (rag/llm_evaluator.py)

#### 4.3.1 评估策略

| 评估阶段 | 方法     | 目标数量    | 说明               |
| -------- | -------- | ----------- | ------------------ |
| 初筛     | 物理规则 | 1000 → 200 | 硬约束过滤         |
| 粗筛     | RAG检索  | 200 → 50   | 基于历史案例相似度 |
| 精筛     | LLM评估  | 50 → 10    | 语义合理性判断     |

#### 4.3.2 LLM评估Prompt设计

```markdown
## 轨迹合理性评估任务

### 背景
你正在评估自动驾驶测试场景中的危险轨迹。该轨迹是通过对真实驾驶场景进行参数变异生成的。

### 输入信息
1. **场景描述**: {scenario_description}
2. **轨迹特征**:
   - 最大加速度: {max_accel} m/s²
   - 最小TTC: {min_ttc} s
   - 横向偏移范围: [{min_lat}, {max_lat}] m
   - 危险类型: {danger_type}

### 评估标准
1. **物理合理性**: 轨迹是否符合车辆动力学？
2. **场景合理性**: 这种驾驶行为在现实中是否可能发生？
3. **危险程度**: 是否构成有效的安全测试场景？

### 输出格式
```json
{
    "is_reasonable": true/false,
    "confidence": 0.85,
    "reasoning": "评估理由",
    "danger_level": "high/medium/low",
    "suggestions": "改进建议（如不合理）"
}
```

```

#### 4.3.3 评估结果处理

| 评估结果 | 后续动作 |
|---------|---------|
| `is_reasonable=true, confidence≥0.8` | 自动通过，进入人工审核 |
| `is_reasonable=true, confidence<0.8` | 标记需要人工审核 |
| `is_reasonable=false` | 直接剔除，记录理由 |

---

### 4.4 特征提取器 (core/feature_extractor.py)

#### 4.4.1 功能说明

将轨迹数据转换为11维特征向量，用于RAG检索和相似度计算。

> **⚠️ 已更新**：原设计特征定义与实际实现不符，已于Week 1修正。

#### 4.4.2 特征定义（已修正）

| 索引 | 特征名 | 计算方式 | 单位 | 说明 |
|------|--------|---------|------|------|
| 0 | mean_speed | mean(\|v\|) | m/s | 交互车平均速度 |
| 1 | max_speed | max(\|v\|) | m/s | 交互车最大速度 |
| 2 | speed_std | std(\|v\|) | m/s | 速度波动程度 |
| 3 | mean_accel | mean(\|a\|) | m/s² | 交互车平均加速度 |
| 4 | max_accel | max(\|a\|) | m/s² | 交互车最大加速度 |
| 5 | min_ttc_long | min(TTC_long) | s | 最小纵向TTC |
| 6 | mean_ttc_long | mean(TTC_long) | s | 平均纵向TTC |
| 7 | min_rel_dist | min(rel_dist) | m | 最小相对距离 |
| 8 | max_closing_speed | max(-v_long) | m/s | 最大接近速度 |
| 9 | trajectory_length | sum(\|ds\|) | m | 行驶距离 |
| 10 | max_curvature | max(\|dθ/ds\|) | 1/m | 最大曲率 |

#### 4.4.3 特征分类

**第一部分：交互车轨迹特征（0-4）**
- 基于交互车自身运动状态计算
- 反映交互车的驾驶行为特征

**第二部分：交互特征（5-10）**
- 基于主车-交互车相对运动计算
- 反映危险交互的严重程度

#### 4.4.4 数据结构

**TrajectoryFeatures** 包含以下11个字段，用于描述一条轨迹的核心特征：

| 分类 | 字段名 | 数据类型 | 单位 | 说明 |
|------|--------|----------|------|------|
| 速度统计 | mean_speed | float | m/s | 交互车速度均值 |
| | max_speed | float | m/s | 交互车速度最大值 |
| | speed_std | float | m/s | 交互车速度标准差 |
| 加速度统计 | mean_accel | float | m/s² | 交互车加速度均值 |
| | max_accel | float | m/s² | 交互车加速度最大值 |
| 交互特征 | min_ttc_long | float | s | 最小纵向TTC |
| | mean_ttc_long | float | s | 平均纵向TTC |
| | min_rel_dist | float | m | 最小相对距离 |
| | max_closing_speed | float | m/s | 最大纵向接近速度 |
| 几何特征 | trajectory_length | float | m | 交互车行驶距离 |
| | max_curvature | float | 1/m | 轨迹最大曲率 |

---

### 4.5 风险计算器 (core/risk_calculator.py)

#### 4.5.1 功能说明

基于TTC(Time-To-Collision)和车辆动力学计算场景风险分数，定位最危险帧。

#### 4.5.2 TTC计算公式

```
TTC = D / Vr  (当 Vr > 0 且 D > 0)
```

| 符号 | 含义 |
|------|------|
| D | 相对距离 (m) |
| Vr | 相对速度 (m/s)，正表示接近 |

#### 4.5.3 风险等级划分

| 风险等级 | TTC范围 | 说明 |
|---------|---------|------|
| 极高 | < 1.0s | 即将碰撞 |
| 高 | 1.0s ~ 2.0s | 紧急情况 |
| 中 | 2.0s ~ 3.0s | 需要注意 |
| 低 | > 3.0s | 安全范围 |

#### 4.5.4 输出数据

风险分析完成后，输出以下信息：

| 输出字段 | 数据类型 | 说明 |
|----------|----------|------|
| anchor_frame | int | 最危险帧在原始场景中的帧索引 |
| max_risk_score | float | 整条轨迹的最大风险分数 |
| risk_scores | ndarray | 每一帧的风险分数序列 (N, 7) |
| ttc_long | ndarray | 纵向TTC序列 |
| ttc_lat | ndarray | 横向TTC序列 |
| rel_dist | ndarray | 相对距离序列 |
| danger_type | str | **动态推断**，见下方说明 |
| danger_level | str | high/medium/low |

#### 4.5.5 危险类型判断（已修正）

> **⚠️ 已更新**：原设计假设简单场景（直道/交叉口），实际需要支持任意场景类型。

**原设计问题**：
- 原设计预设4种危险类型：rear_end、cut_in、head_on、side_swipe
- 假设场景为直道或简单交叉口

**解决方案**：
- **不预设场景类型**，计算所有可能的交互特征
- 基于实际特征值**动态推断**危险类型

**危险类型推断逻辑**：

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

**支持的场景类型**：
- 直道追尾/对向行驶
- 交叉口穿行
- 停车场/匝道等复杂场景

---

### 4.6 片段截取器 (core/fragment_extractor.py)

#### 4.6.1 功能说明

根据风险分析结果，截取包含危险交互的轨迹片段。

> **⚠️ 已更新**：截取参数由固定帧数改为可配置秒数，增加了边界处理。

#### 4.6.2 截取策略

| 参数 | 默认值 | 说明 |
|------|--------|------|
| n_before_sec | 2.0秒 | 锚点前截取时长 |
| n_after_sec | 3.0秒 | 锚点后截取时长 |
| 帧数计算 | n_before_frames = n_before_sec × 采样率(10Hz) | 自动计算 |
| 边界处理 | 自动调整 | 锚点靠近边界时自动扩展 |

**边界处理逻辑**：

```python
def extract_fragment_slice(anchor_frame, n_before_frames, n_after_frames, total_frames):
    start = anchor_frame - n_before_frames
    end = anchor_frame + n_after_frames + 1

    # 左边界处理
    if start < 0:
        end -= start
        start = 0

    # 右边界处理
    if end > total_frames:
        start -= (end - total_frames)
        start = max(0, start)
        end = total_frames

    return start, end
```

#### 4.6.3 同步输出

截取后的轨迹片段包含以下信息：

| 输出字段 | 数据类型 | 说明 |
|----------|----------|------|
| scenario_id | str | 原始场景的唯一标识 |
| anchor_frame | int | 最危险帧在原始场景中的索引 |
| ego_trajectory | List[TrajectoryPoint] | 主车（自车）在片段时间范围内的完整轨迹点序列 |
| target_trajectory | List[TrajectoryPoint] | 交互车（目标车）在片段时间范围内的完整轨迹点序列，与主车同步 |
| relative_features | RelativeFeatures | 主车与交互车之间的相对运动特征，包括相对距离、相对速度、相对角度等 |
| duration | float | 片段的时长，单位为秒 |
| frame_count | int | 片段包含的总帧数 |

---

### 4.7 CARLA适配器 (core/carla_adapter.py)

#### 4.7.1 功能说明

将处理后的轨迹转换为CARLA仿真可用的格式，支持JSON、OpenSCENARIO等格式。

#### 4.7.2 支持的输出格式

| 格式 | 文件扩展名 | 用途 |
|------|-----------|------|
| JSON | `.json` | 通用数据交换 |
| OpenSCENARIO | `.xosc` | 标准场景描述 |
| Python脚本 | `.py` | CARLA API直接使用 |

#### 4.7.3 坐标系转换

> **⚠️ 待验证问题**：Waymo → CARLA 坐标转换公式尚未验证，实现前需参考CARLA官方文档或实际测试确认。

Waymo → CARLA 坐标转换：

| 维度 | Waymo | CARLA |
|------|-------|-------|
| X | 右手系 | 左手系（取反） |
| Y | 与X、Z正交 | 与X、Z正交 |
| Z | 高度 | 高度 |
| Heading | 弧度 | 度（×180/π） |

#### 4.7.4 转换函数

坐标系转换需要处理以下要素的变换：

1. **位置转换**：Waymo使用右手坐标系（X向前，Y向左，Z向上），CARLA使用左手坐标系。需要将X坐标取反，Y和Z根据具体定义进行交换
2. **航向角转换**：Waymo中heading为弧度，CARLA中需要转换为度
3. **速度转换**：速度向量同样需要进行坐标系的变换

转换过程中需要注意timestamp的保持，确保时序信息一致。

#### 4.7.5 闭环测试轨迹设计

**测试场景配置原则**：

不同危险类型有不同的典型时间分配模式，具体值由用户选择轨迹长度时决定：

| 碰撞类型 | 典型时间分配 | 交互车行为 |
|----------|--------------|------------|
| 追尾 | 碰撞点前2/3，碰撞点后1/3 | 继续行驶或缓慢减速 |
| 切入 | 碰撞点前2/3，碰撞点后1/3 | 完成切入后继续行驶 |
| 对向 | 碰撞点前1/2，碰撞点后1/2 | 高速通过或让行 |
| 剐蹭 | 碰撞点前2/3，碰撞点后1/3 | 并行后继续行驶 |

**时间分配由用户选择**：
- 碰撞点位置由轨迹生成算法根据变异参数自动控制
- 用户选择总长度后，算法自动确定碰撞点时机
- 碰撞点后轨迹长度应足够展示避障效果

**评估指标**（需要碰撞点后轨迹才能计算）：
- **避障成功率**：是否避免碰撞
- **最小TTC**：避障过程中的最小碰撞时间
- **舒适性**：最大减速度、横向加速度
- **恢复时间**：回到正常行驶状态所需时间

**测试注意事项**：
- 主车接入自动驾驶算法，交互车使用生成的危险轨迹
- 碰撞点后轨迹应设计为"给空间"模式，而非"消失"模式
- 这样设计才能有效测试算法的避障能力和恢复能力

---

## 五、RAG知识库设计

### 5.1 知识库概念与数据库关系

**知识库是数据库架构的核心组成部分**，包含两层含义：

1. **狭义知识库**：存储在ChromaDB中的轨迹特征向量 + 相关元数据，用于RAG相似度检索
2. **广义知识库**：SQLite中 `knowledge_base` 表 + `trajectory_records` 表 + `evaluation_results` 表的联合视图

**数据库架构图**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                         应用层 (Python代码)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ 场景处理模块 │  │ 穷举生成模块 │  │ RAG评估模块 │  │ 人工审核模块 │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
└─────────┼───────────────┼───────────────┼───────────────┼─────────┘
          │               │               │               │
          ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    数据访问层 (DAO)                                   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   SQLiteManager (关系型数据)                   │   │
│  │                                                              │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                  │   │
│  │  │ trajectory_records │  │ evaluation_results│              │   │
│  │  │ (轨迹原始数据)    │  │ (评估结果)       │                  │   │
│  │  └─────────────────┘  └─────────────────┘                  │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                  │   │
│  │  │  knowledge_base  │  │  human_reviews   │                  │   │
│  │  │  (案例元数据)    │  │  (审核记录)       │                  │   │
│  │  └─────────────────┘  └─────────────────┘                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   ChromaDBManager (向量数据库)               │   │
│  │                                                              │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │         knowledge_base_vectors (向量集合)             │   │   │
│  │  │  - id: case_id                                       │   │   │
│  │  │  - embedding: 11维特征向量                             │   │   │
│  │  │  - metadata: {label, danger_type, evaluated_by}      │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 知识库完整数据结构

知识库中的每个案例（Case）包含以下完整信息：

| 分类 | 字段名 | 数据类型 | 说明 | 计算来源/存储方式 |
|------|--------|----------|------|-----------------|
| **标识** | `case_id` | string | 唯一标识 | 自动生成：`case_{timestamp}_{hash}` |
| **关联** | `trajectory_id` | string | 关联轨迹ID | 外键关联trajectory_records |
| **向量特征** | `embedding` | vector[11] | 11维特征向量 | TrajectoryFeatureExtractor计算，存ChromaDB |
| **轨迹数据** | `trajectory_data` | JSON | 完整轨迹点序列 | trajectory_records存储，关联查询获取 |
| **物理验证** | `physics_validation` | JSON | 物理指标验证结果 | PhysicsValidator输出 |
| | - `max_long_accel` | float | 最大纵向加速度 | 计算 |
| | - `max_lat_accel` | float | 最大横向加速度 | 计算 |
| | - `max_speed` | float | 最大速度 | 计算 |
| | - `collision_detected` | bool | 是否检测到碰撞 | 碰撞检测 |
| | - `failed_rules` | list | 失败的规则ID列表 | 验证规则 |
| **LLM评估** | `llm_evaluation` | JSON | LLM评估结果 | RAGEvaluator输出 |
| | - `is_reasonable` | bool | 是否合理 | LLM判断 |
| | - `confidence` | float | 置信度 | LLM返回 |
| | - `reasoning` | string | 评估理由 | LLM生成 |
| **标签** | `label` | enum | 合理/不合理 | reasonable/unreasonable |
| | `danger_type` | enum | 危险类型 | rear_end/cut_in/head_on/side_swipe |
| | `danger_level` | enum | 危险等级 | high/medium/low |
| **元数据** | `evaluated_by` | enum | 评估来源 | human/llm_auto |
| | `confidence` | float | 置信度 | LLM返回 |
| | `creation_context` | JSON | 生成参数 | 记录accel_offset等参数 |
| **统计** | `similar_cases` | list | 入库时检索到的相似案例 | RAG检索结果 |
| | `query_count` | int | 被查询次数 | 统计 |

**数据库存储位置**：

| 数据类型 | 存储位置 | 存储格式 |
|---------|---------|---------|
| 11维特征向量 | ChromaDB | float32[11] |
| 轨迹原始数据 | SQLite trajectory_records | BLOB (pickle) |
| 物理验证结果 | SQLite evaluation_results | JSON |
| LLM评估结果 | SQLite evaluation_results | JSON |
| 案例元数据 | SQLite knowledge_base | 各字段 |
| 标签信息 | SQLite knowledge_base + ChromaDB metadata | enum + string |

### 5.3 知识库构建完整流程

知识库的构建发生在每条轨迹通过评估之后：

```
候选轨迹生成
    │
    ▼
[Step 1] 物理验证 (PhysicsValidator)
    │ 计算物理指标：max_accel, max_speed, jerk, collision
    │ 输出：ValidationResult
    ▼
[Step 2] 特征提取 (TrajectoryFeatureExtractor)
    │ 提取11维特征向量
    │ 输出：embedding[12]
    ▼
[Step 3] RAG检索 (RAGEvaluator.query_similar_cases)
    │ ChromaDB相似度检索 top_k=5
    │ 获取相似案例的完整信息
    ▼
[Step 4] LLM评估 (RAGEvaluator.llm_reasonableness_eval)
    │ 结合相似案例进行评估
    │ 输出：is_reasonable, confidence, reasoning
    ▼
[Step 5] 人工审核（可选）
    │ 人工判断是否入库
    ▼
【知识库构建】
    │
    ├── 存储11维向量到ChromaDB
    │   └── collection.add(ids=[case_id], embeddings=[embedding],
    │                        metadatas=[{label, danger_type, evaluated_by}])
    │
    ├── 存储轨迹数据到SQLite trajectory_records
    │
    ├── 存储评估结果到SQLite evaluation_results
    │
    └── 存储案例元数据到SQLite knowledge_base
```

**入库判定规则**：

| 评估结果 | 置信度 | 人工审核 | 入库 | evaluated_by |
|---------|--------|---------|------|-------------|
| is_reasonable=true | ≥0.8 | 否 | 是 | llm_auto |
| is_reasonable=true | <0.8 | 是 | 审核后 | human/llm_auto |
| is_reasonable=false | any | 否 | 否 | llm_auto |

### 5.4 向量化编码器 (rag/embedding_encoder.py)

#### 5.4.1 功能说明

将轨迹数据编码为11维特征向量，用于ChromaDB存储和相似度检索。

#### 5.4.2 编码流程

```
原始轨迹数据
    ↓
提取运动学特征（速度、加速度）
    ↓
计算安全指标（TTC、碰撞风险）
    ↓
空间特征（横向偏移、曲率）
    ↓
归一化处理（L2归一化）
    ↓
11维特征向量
```

#### 5.4.3 编码器接口

TrajectoryEncoder 负责将轨迹数据编码为11维特征向量，编码流程包括：

1. **运动学特征提取**：从轨迹中计算速度、加速度等运动学量，提取统计特征（均值、最大值、标准差）
2. **空间特征提取**：计算横向偏移的相关特征
3. **安全指标提取**：计算TTC相关指标
4. **几何特征提取**：计算轨迹长度、曲率等几何属性
5. **归一化处理**：对特征向量进行L2归一化，使得不同轨迹的特征具有可比性

编码后的特征向量用于后续的RAG相似度检索。

### 5.5 RAG检索逻辑 (rag/rag_evaluator.py)

#### 5.5.1 检索流程

```
新候选轨迹（待评估）
    ↓
【Step 1】特征提取
    └── TrajectoryFeatureExtractor.encode() → 11维向量
    ↓
【Step 2】向量检索
    └── ChromaDB.query_similar(query_vector, top_k=5)
    ↓
【Step 3】获取相似案例完整信息
    ├── 根据case_id查询SQLite knowledge_base
    ├── 根据trajectory_id查询SQLite trajectory_records
    └── 根据evaluation_id查询SQLite evaluation_results
    ↓
【Step 4】构建RAG评估上下文
    ├── 候选轨迹：特征向量 + 物理验证结果
    ├── 相似案例：轨迹特征 + 评估结果 + 标签
    └── 危险类型 + 场景描述
    ↓
【Step 5】LLM合理性评估
    └── LLM综合判断
```

#### 5.5.2 RAG检索接口

RAGEvaluator 的检索流程如下：

1. **向量检索**：将待评估轨迹的11维特征向量作为查询向量，在ChromaDB中进行相似度检索，返回top_k个最相似的案例

2. **信息补全**：根据检索返回的case_id，在SQLite数据库中查询对应的完整案例信息，包括：
   - 轨迹原始数据（用于LLM参考）
   - 物理验证结果
   - 历史评估结果（是否合理、置信度、评估理由）

3. **结果组装**：将相似度得分与案例完整信息组装后返回，供后续LLM评估使用

#### 5.5.3 分支判断逻辑

| 条件 | 处理策略 | 说明 |
|------|---------|------|
| max_similarity > 0.75 | 标准RAG评估 | 基于相似案例判断，可信度高 |
| 0.5 < max_similarity ≤ 0.75 | LLM辅助评估 | 结合相似案例 + LLM分析 |
| max_similarity ≤ 0.5 | LLM深度分析 | 无足够相似案例，依赖LLM |

#### 5.5.4 相似度统计指标

| 指标 | 计算方式 | 用途 |
|------|---------|------|
| max_similarity | max(案例相似度列表) | 判断是否有高度相似案例 |
| avg_similarity | mean(案例相似度列表) | 整体相似程度 |
| min_similarity | min(案例相似度列表) | 排除异常案例 |
| reasonable_ratio | 相似案例中reasonable比例 | 辅助判断 |

### 5.6 LLM评估Prompt设计（含RAG上下文）

当使用RAG检索到的相似案例时，LLM评估Prompt设计如下：

```markdown
## 轨迹合理性评估任务

### 背景
你正在评估自动驾驶测试场景中的危险轨迹。该轨迹是通过对真实驾驶场景进行参数变异生成的。

### 输入信息
1. **待评估轨迹特征**:
   - 最大加速度: {max_accel} m/s²
   - 最大速度: {max_speed} m/s
   - 最小TTC: {min_ttc} s
   - 横向偏移范围: [{min_lat}, {max_lat}] m
   - 危险类型: {danger_type}

2. **物理验证结果**:
   - 通过状态: {is_physics_valid}
   - 失败规则: {failed_rules}
   - 警告信息: {warnings}

3. **RAG检索到的相似案例（共{num_similar_cases}个）**:

{% for case in similar_cases %}
#### 案例 {{ loop.index }}: {{ case.danger_type }}
- 相似度: {{ case.similarity }}
- 标签: {{ case.label }}
- 评估结果: {{ case.llm_evaluation.is_reasonable }}
- 置信度: {{ case.llm_evaluation.confidence }}
- 评估理由: {{ case.llm_evaluation.reasoning }}
- 物理指标: max_accel={{ case.physics_validation.max_long_accel }}, max_speed={{ case.physics_validation.max_speed }}
{% endfor %}

### 评估标准
1. **物理合理性**: 轨迹是否符合车辆动力学约束？
2. **场景合理性**: 这种驾驶行为在现实中是否可能发生？
3. **与相似案例一致性**: 与知识库中相似案例的评估结论是否一致？
4. **危险程度**: 是否构成有效的安全测试场景？

### 输出格式
```json
{
    "is_reasonable": true/false,
    "confidence": 0.85,
    "reasoning": "评估理由（需参考相似案例）",
    "danger_level": "high/medium/low",
    "consistency_with_cases": "与相似案例的一致性分析"
}
```
```

### 5.7 冷启动策略

```
系统启动 → 知识库为空
    ↓
穷举生成候选轨迹（基于参数空间计算的数量，如7000条）
    ↓
物理过滤 → 剩余约700条（过滤率约90%）
    ↓
人工审核50条（建立种子库）
    ↓
知识库初始化完成
    ↓
后续使用RAG辅助评估
```

### 5.8 LLM自补充机制 (rag/self_expansion.py)

#### 5.8.1 触发条件

当知识库中无足够相似案例时（max_similarity < 0.7），触发LLM自补充机制。

#### 5.8.2 自补充流程

```
LLM深度分析 → 置信度判断
    │
    ├── 置信度 ≥ 0.8 → 自动入库（标记llm_auto）
    │       ↓
    │       存储到SQLite + ChromaDB
    │
    └── 置信度 < 0.8 → 标记need_human_review
            ↓
        进入人工审核流程
```

#### 5.8.3 自动入库案例管理

| 属性 | 说明 |
|------|------|
| `evaluated_by` | 固定为 `llm_auto` |
| `confidence` | LLM返回的置信度 |
| `need_human_review` | `false`（自动入库） |

**定期抽检机制**：每周随机抽取5%的llm_auto案例进行人工复核。

### 5.9 知识库维护

#### 5.9.1 案例生命周期

```
创建 → 评估 → 入库 → 查询 → 更新/删除
```

#### 5.9.2 知识库统计

| 统计项 | 说明 |
|-------|------|
| total_cases | 总案例数 |
| reasonable_count | 合理案例数 |
| unreasonable_count | 不合理案例数 |
| llm_auto_count | LLM自动入库数 |
| pending_review_count | 待审核数 |

#### 5.9.3 知识库优化

| 操作 | 触发条件 | 动作 |
|------|---------|------|
| 去重检查 | 入库前 | 基于trajectory_hash查重 |
| 相似度阈值调整 | 查询命中率<50% | 降低相似度阈值 |
| 案例清理 | unreasonable比例>60% | 分析原因，调整生成策略 |
| 向量索引优化 | 案例数>10000 | 重建ChromaDB索引 |

---

## 六、数据库设计

> 本章节描述系统的完整数据库架构，包括关系型数据库（SQLite）和向量数据库（ChromaDB）的选型、表结构设计和数据流转。

### 6.1 数据库选型策略

| 用途 | 技术选型 | 选型理由 |
|------|---------|---------|
| 关系型数据存储 | SQLite | 本地轻量级、零配置、Python原生支持、开发测试阶段首选 |
| 向量检索 | ChromaDB | 专为RAG设计、支持持久化、本地部署、无需额外服务 |
| 配置存储 | config.yaml | 结构化配置、人类可读、支持环境变量覆盖 |

**为什么不选择其他方案：**

| 备选方案 | 未选原因 |
|---------|---------|
| PostgreSQL | 需要部署数据库服务，本项目为个人学习项目，SQLite足够 |
| Milvus | 需要额外服务部署，学习成本高，ChromaDB更适合本阶段 |
| MongoDB | 文档型数据库不适合本项目的强Schema场景 |

### 6.2 数据库架构设计

#### 6.2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         应用层 (Python代码)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ 场景处理模块 │  │ 穷举生成模块 │  │ RAG评估模块 │  │ 人工审核模块 │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
└─────────┼───────────────┼───────────────┼───────────────┼─────────┘
          │               │               │               │
          ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    数据访问层 (DAO)                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   SQLiteManager (关系型数据)                  │   │
│  │  • scenario_metadata (场景元数据)                           │   │
│  │  • trajectory_records (轨迹记录)                             │   │
│  │  • evaluation_results (评估结果)                             │   │
│  │  • human_reviews (审核记录)                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   ChromaDBManager (向量数据)                │   │
│  │  • knowledge_base (知识库向量)                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
          │                                       │
          ▼                                       ▼
┌─────────────────────┐               ┌─────────────────────┐
│   SQLite 数据库文件   │               │   ChromaDB 持久化    │
│  ./data/traj.db     │               │   ./data/chroma_db/  │
└─────────────────────┘               └─────────────────────┘
```

#### 6.2.2 表关系ER图

```
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│ scenario_metadata │       │trajectory_records│       │ evaluation_results│
├──────────────────┤       ├──────────────────┤       ├──────────────────┤
│ scenario_id (PK)  │──────<│ scenario_id (FK) │       │ evaluation_id (PK)│
│ source_dataset    │       │ trajectory_id(PK)│──────<│ trajectory_id (FK)│
│ map_name          │       │ vehicle_id       │       │ is_physics_valid │
│ scene_length      │       │ parent_traj_id   │       │ llm_confidence   │
│ ...               │       │ danger_type      │       │ final_status     │
└──────────────────┘       │ generation_batch │       └──────────────────┘
                            │ created_at       │               │
                            └──────────────────┘               │
                                   │                             │
                                   │               ┌───────────────┘
                                   ▼               ▼
                            ┌──────────────────┐ ┌──────────────────┐
                            │  knowledge_base   │ │  human_reviews   │
                            ├──────────────────┤ ├──────────────────┤
                            │ case_id (PK)     │ │ review_id (PK)   │
                            │ trajectory_id(FK) │ │ trajectory_id(FK)│
                            │ embedding [11维] │ │ evaluation_id(FK)│
                            │ label            │ │ review_status    │
                            │ confidence        │ │ corrected_label  │
                            └──────────────────┘ └──────────────────┘
```

### 6.3 各表详细设计

#### 6.3.1 场景元数据表 (scenario_metadata)

存储Waymo原始场景的基本信息。

| 字段名 | 数据类型 | 约束 | 说明 | 示例 |
|--------|----------|------|------|------|
| `scenario_id` | TEXT | **PRIMARY KEY** | 场景唯一标识 | `10135f16cd538e19` |
| `source_dataset` | TEXT | NOT NULL | 数据来源 | `waymo` |
| `map_name` | TEXT | - | 地图名称 | `Town04` |
| `scene_length` | INTEGER | - | 帧数 | `199` |
| `sampling_rate` | REAL | - | 采样率(Hz) | `10.0` |
| `ego_vehicle_id` | TEXT | - | 主车ID | `312` |
| `num_vehicles` | INTEGER | - | 车辆数量 | `8` |
| `num_pedestrians` | INTEGER | - | 行人数量 | `3` |
| `num_cyclists` | INTEGER | - | 骑车人数量 | `1` |
| `is_processed` | INTEGER | DEFAULT 0 | 是否已处理 | `0` |
| `processing_stage` | TEXT | DEFAULT `pending` | 处理阶段 | `pending` |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 创建时间 | `2024-01-01 10:00:00` |
| `processed_at` | TIMESTAMP | - | 处理完成时间 | `2024-01-01 12:00:00` |

**索引设计：**
```sql
CREATE INDEX idx_scenario_processed ON scenario_metadata(is_processed);
CREATE INDEX idx_scenario_source ON scenario_metadata(source_dataset);
```

#### 6.3.2 轨迹记录表 (trajectory_records)

存储生成的候选轨迹及其特征。

| 字段名 | 数据类型 | 约束 | 说明 | 示例 |
|--------|----------|------|------|------|
| `trajectory_id` | TEXT | **PRIMARY KEY** | 轨迹唯一标识 | `traj_20240101_001` |
| `scenario_id` | TEXT | **NOT NULL**, FK | 关联场景ID | `10135f16cd538e19` |
| `vehicle_id` | TEXT | NOT NULL | 车辆ID | `312` |
| `parent_trajectory_id` | TEXT | FK (self) | 父轨迹ID（用于变异追踪） | `traj_20240101_000` |
| `trajectory_data` | BLOB | - | 序列化的轨迹点列表 | (二进制) |
| `feature_vector` | BLOB | - | 11维特征向量 | (二进制) |
| `danger_type` | TEXT | - | 危险类型枚举 | `cut_in` |
| `danger_level` | TEXT | - | 危险等级 | `high` |
| `generation_batch` | TEXT | - | 生成批次 | `batch_001` |
| `generation_params` | TEXT | - | 生成参数JSON | `{"accel_offset": 2.0}` |
| `trajectory_hash` | TEXT | UNIQUE | 轨迹数据指纹（去重） | `a1b2c3d4...` |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 创建时间 | `2024-01-01 10:00:00` |
| `updated_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 更新时间 | `2024-01-01 10:00:00` |

**索引设计：**
```sql
CREATE INDEX idx_traj_scenario ON trajectory_records(scenario_id);
CREATE INDEX idx_traj_danger_type ON trajectory_records(danger_type);
CREATE INDEX idx_traj_danger_level ON trajectory_records(danger_level);
CREATE INDEX idx_traj_generation_batch ON trajectory_records(generation_batch);
CREATE UNIQUE INDEX idx_traj_hash ON trajectory_records(trajectory_hash);
```

#### 6.3.3 评估结果表 (evaluation_results)

存储每条轨迹的物理验证和LLM评估结果。

| 字段名 | 数据类型 | 约束 | 说明 | 示例 |
|--------|----------|------|------|------|
| `evaluation_id` | TEXT | **PRIMARY KEY** | 评估记录唯一标识 | `eval_20240101_001` |
| `trajectory_id` | TEXT | **NOT NULL**, FK | 关联轨迹ID | `traj_20240101_001` |
| `physics_validation` | TEXT | - | 物理验证详情JSON | 详见下方 |
| `is_physics_valid` | INTEGER | DEFAULT 0 | 物理验证是否通过 | `1` |
| `max_similarity` | REAL | - | RAG最高相似度 | `0.85` |
| `avg_similarity` | REAL | - | RAG平均相似度 | `0.72` |
| `similar_cases` | TEXT | - | 相似案例列表JSON | `[{"case_id": "..."}]` |
| `llm_is_reasonable` | INTEGER | - | LLM评估是否合理 | `1` |
| `llm_confidence` | REAL | - | LLM置信度 | `0.88` |
| `llm_reasoning` | TEXT | - | LLM评估理由 | `轨迹符合物理规律` |
| `needs_human_review` | INTEGER | DEFAULT 1 | 是否需要人工审核 | `1` |
| `final_status` | TEXT | DEFAULT `pending` | 最终状态 | `approved` |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 创建时间 | `2024-01-01 10:00:00` |

**physics_validation字段结构：**
```json
{
    "max_long_accel": 5.2,
    "max_long_decel": 6.8,
    "max_lat_accel": 3.1,
    "max_speed": 28.5,
    "min_speed": 0.0,
    "collision_detected": false,
    "failed_rules": [],
    "warnings": ["jerk_high"]
}
```

**索引设计：**
```sql
CREATE INDEX idx_eval_trajectory ON evaluation_results(trajectory_id);
CREATE INDEX idx_eval_status ON evaluation_results(final_status);
CREATE INDEX idx_eval_physics_valid ON evaluation_results(is_physics_valid);
```

#### 6.3.4 知识库表 (knowledge_base)

存储用于RAG检索的轨迹案例，向量存储在ChromaDB中。

| 字段名 | 数据类型 | 约束 | 说明 | 示例 |
|--------|----------|------|------|------|
| `case_id` | TEXT | **PRIMARY KEY** | 案例唯一标识 | `case_20240101_001` |
| `trajectory_id` | TEXT | **NOT NULL**, FK | 关联轨迹ID | `traj_20240101_001` |
| `label` | TEXT | NOT NULL | 标签 | `reasonable` |
| `danger_type` | TEXT | - | 危险类型 | `cut_in` |
| `danger_level` | TEXT | - | 危险等级 | `high` |
| `evaluated_by` | TEXT | - | 评估来源 | `human` |
| `confidence` | REAL | - | 置信度 | `0.92` |
| `evaluation_reasoning` | TEXT | - | 评估理由 | `轨迹自然流畅` |
| `creation_context` | TEXT | - | 生成参数JSON | `{"accel_offset": 2.0, "speed_scale": 1.2}` |
| `similar_cases` | TEXT | - | 入库时相似案例ID列表JSON | `["case_001", "case_002"]` |
| `query_count` | INTEGER | DEFAULT 0 | 被查询次数 | `15` |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 创建时间 | `2024-01-01 10:00:00` |
| `updated_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 更新时间 | `2024-01-01 10:00:00` |

**ChromaDB向量存储：**

| 字段 | 类型 | 维度 | 说明 |
|------|------|------|------|
| `id` | string | - | 唯一ID，等于case_id |
| `embedding` | float32[] | 11 | 轨迹11维特征向量 |
| `metadata` | dict | - | label、danger_type等 |

**11维特征向量定义：**

| 索引 | 特征名 | 说明 |
|------|--------|------|
| 0 | mean_speed | 平均速度(米/秒) |
| 1 | max_speed | 最大速度(米/秒) |
| 2 | speed_std | 速度标准差 |
| 3 | mean_accel | 平均加速度(米/秒²) |
| 4 | max_accel | 最大加速度(米/秒²) |
| 5 | max_lateral_offset | 最大横向偏移(米) |
| 6 | lateral_std | 横向偏移标准差 |
| 7 | min_ttc | 最小TTC(秒) |
| 8 | mean_ttc | 平均TTC(秒) |
| 9 | trajectory_length | 轨迹长度(米) |
| 10 | max_curvature | 最大曲率 |

**索引设计：**
```sql
CREATE INDEX idx_kb_label ON knowledge_base(label);
CREATE INDEX idx_kb_danger_type ON knowledge_base(danger_type);
CREATE INDEX idx_kb_evaluated_by ON knowledge_base(evaluated_by);
```

#### 6.3.5 人工审核记录表 (human_reviews)

存储人工审核的详细记录。

| 字段名 | 数据类型 | 约束 | 说明 | 示例 |
|--------|----------|------|------|------|
| `review_id` | TEXT | **PRIMARY KEY** | 审核记录唯一标识 | `review_20240101_001` |
| `trajectory_id` | TEXT | **NOT NULL**, FK | 关联轨迹ID | `traj_20240101_001` |
| `evaluation_id` | TEXT | **NOT NULL**, FK | 关联评估ID | `eval_20240101_001` |
| `reviewer_id` | TEXT | - | 审核人ID | `user_001` |
| `review_status` | TEXT | - | 审核状态 | `approved` |
| `reviewer_notes` | TEXT | - | 审核备注 | `轨迹过于激进` |
| `corrected_label` | TEXT | - | 修正后的标签 | `unreasonable` |
| `reviewed_at` | TIMESTAMP | - | 审核时间 | `2024-01-01 12:00:00` |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 创建时间 | `2024-01-01 10:00:00` |

**索引设计：**
```sql
CREATE INDEX idx_review_trajectory ON human_reviews(trajectory_id);
CREATE INDEX idx_review_evaluation ON human_reviews(evaluation_id);
CREATE INDEX idx_review_status ON human_reviews(review_status);
```

### 6.4 数据流转设计

#### 6.4.1 模块与数据库交互

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              数据写入流程                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Waymo原始数据                                                                │
│       │                                                                       │
│       ▼                                                                       │
│  ┌─────────────┐                                                            │
│  │ 数据加载模块 │ ──场景元数据──> │ scenario_metadata │                     │
│  └─────────────┘                                                            │
│       │                                                                       │
│       ▼                                                                       │
│  ┌─────────────┐                                                            │
│  │ 穷举生成模块 │ ──候选轨迹──> │ trajectory_records │                      │
│  └─────────────┘                                                            │
│       │                                                                       │
│       ▼                                                                       │
│  ┌─────────────┐                                                            │
│  │ 物理验证模块 │                                                            │
│  └─────────────┘                                                            │
│       │                                                                       │
│       ├─ 验证通过 ──轨迹ID + 验证结果 ──> │ evaluation_results │             │
│       │                                      │                              │
│       │                                      ▼                              │
│       │                              ┌─────────────┐                        │
│       │                              │ 知识库表    │ (经ChromaDB存储向量)    │
│       │                              └─────────────┘                        │
│       │                                                                       │
│       ▼                                                                       │
│  ┌─────────────┐                                                            │
│  │ LLM评估模块 │ ──评估结果──> │ evaluation_results │                       │
│  └─────────────┘                                                            │
│       │                                                                       │
│       ▼                                                                       │
│  ┌─────────────┐                                                            │
│  │ 人工审核模块 │ ──审核记录──> │ human_reviews │                           │
│  └─────────────┘                                                            │
│       │                                                                       │
│       └─ 审核通过 ──> │ knowledge_base │ (更新标签)                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### 6.4.2 RAG检索数据流

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              RAG检索流程                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  新候选轨迹                                                                │
│       │                                                                       │
│       ▼                                                                       │
│  ┌─────────────────┐                                                      │
│  │ 特征提取模块     │ ──11维向量──> │ ChromaDB │ query_similar(top_k=5)  │
│  └─────────────────┘              └─────┬─────┘                           │
│                                          │                                   │
│                                          ▼                                   │
│                                  相似案例列表                                  │
│                                          │                                   │
│                                          ▼                                   │
│                                  ┌─────────────────┐                        │
│                                  │ RAG评估模块      │                         │
│                                  │ • 计算相似度统计 │                        │
│                                  │ • 决定评估策略   │                        │
│                                  └─────────────────┘                        │
│                                          │                                   │
│                                          ▼                                   │
│                                  ┌─────────────────┐                        │
│                                  │ 评估结果写入    │                         │
│                                  │ evaluation_results │                      │
│                                  └─────────────────┘                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 6.5 接口设计

#### 6.5.1 SQLiteManager (SQLite操作封装)

| 方法 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `connect()` | 连接数据库 | - | `bool` |
| `disconnect()` | 断开连接 | - | - |
| `create_tables()` | 创建所有表 | - | - |
| `insert_scenario()` | 插入场景元数据 | `ScenarioMetadataRecord` | `bool` |
| `get_scenario()` | 获取场景 | `scenario_id` | `ScenarioMetadataRecord` |
| `insert_trajectory()` | 插入轨迹 | `TrajectoryRecord` | `bool` |
| `get_trajectory()` | 获取轨迹 | `trajectory_id` | `TrajectoryRecord` |
| `get_trajectories_by_scenario()` | 获取场景下所有轨迹 | `scenario_id` | `List[TrajectoryRecord]` |
| `insert_evaluation()` | 插入评估结果 | `EvaluationResultRecord` | `bool` |
| `get_evaluation()` | 获取评估结果 | `evaluation_id` | `EvaluationResultRecord` |
| `update_evaluation_status()` | 更新评估状态 | `evaluation_id`, `status` | `bool` |
| `insert_review()` | 插入审核记录 | `HumanReviewRecord` | `bool` |
| `get_reviews_by_trajectory()` | 获取轨迹的审核记录 | `trajectory_id` | `List[HumanReviewRecord]` |

#### 6.5.2 ChromaDBManager (向量数据库封装)

| 方法 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `connect()` | 连接数据库 | - | `bool` |
| `add_case()` | 添加案例向量 | `case_id`, `embedding`, `metadata` | `bool` |
| `add_batch()` | 批量添加 | `List[(case_id, embedding, metadata)]` | `bool` |
| `query_similar()` | 相似度检索 | `query_vector`, `top_k` | `List[Dict]` |
| `get_case()` | 获取案例 | `case_id` | `Dict` |
| `delete_case()` | 删除案例 | `case_id` | `bool` |
| `get_collection_stats()` | 获取集合统计 | - | `Dict` |

#### 6.5.3 数据访问层 (DAO) 便捷函数

| 函数 | 功能 | 位置 |
|------|------|------|
| `init_database()` | 初始化数据库连接和表 | `database_schema.py` |
| `save_trajectory_with_evaluation()` | 保存轨迹及评估结果 | `database_schema.py` |
| `get_scenarios_summary()` | 获取所有场景摘要 | `database_schema.py` |
| `export_to_knowledge_base()` | 导出评估通过的轨迹到知识库 | `database_schema.py` |

### 6.6 文件存储位置

```
data/
├── raw/                        # 原始数据（Waymo pkl）
├── processed/                  # 处理后数据
├── generated/                  # 生成的候选轨迹
│   └── {scenario_id}/
│       ├── candidates_001.json
│       └── candidates_002.json
├── evaluated/                  # LLM评估结果
├── reviewed/                   # 人工审核后轨迹
├── carla_ready/                # CARLA可导入格式
├── chroma_db/                  # ChromaDB持久化
│   └── trajectory_cases/
│       ├── 00000.variant
│       └── ...
└── trajectory_database.db       # SQLite数据库文件
```

### 6.7 备份与恢复策略

| 操作 | 策略 | 频率 |
|------|------|------|
| 数据库自动备份 | 每次启动前自动备份到 `backup/` | 每次启动 |
| 向量数据库备份 | 复制整个 `chroma_db/` 目录 | 每周或100条新增后 |
| 轨迹数据导出 | 导出为JSON格式存档 | 按批次 |
| 恢复流程 | 从最新备份恢复SQLite + ChromaDB | 按需 |

---

## 七、版本管理与迭代

### 7.1 版本规划

| 版本 | 周期 | 核心功能 | 里程碑 |
|------|------|---------|--------|
| v0.1 | Week 1 | 项目搭建，数据加载 | 能读取Waymo数据 |
| v0.2 | Week 2 | 风险计算，片段截取 | 能识别危险片段 |
| v0.3 | Week 3 | 穷举生成，物理验证 | 能生成并过滤轨迹 |
| v0.4 | Week 4-5 | RAG知识库，LLM评估 | 能智能评估轨迹 |
| v0.5 | Week 6 | 人工审核，Web界面 | 能交互式审核 |
| v0.6 | Week 7 | CARLA导出，仿真 | 能运行仿真测试 |
| v1.0 | Week 8 | 文档完善，优化 | 可演示的完整系统 |

### 7.2 Git工作流

```

main (稳定分支)
  │
  ├── develop (开发分支)
  │     │
  │     ├── feature/step1-data-loader
  │     ├── feature/step2-risk-calculation
  │     ├── feature/step3-feature-extraction
  │     ├── feature/step4-trajectory-generation
  │     └── ...
  │
  └── release/v0.x (发布分支)

```

---

## 八、测试策略

### 8.1 测试金字塔

```

    /
    /  \     E2E测试 (少量)
      /----
    /      \   集成测试 (中等)
    /--------
   /          \ 单元测试 (大量)
  /------------\

```

### 8.2 测试要求

| 类型 | 覆盖率 | 说明 |
|------|-------|------|
| 单元测试 | ≥80% | 每个函数都有测试用例 |
| 集成测试 | 核心流程 | 数据流完整流程测试 |
| E2E测试 | 关键路径 | 从加载到导出的全流程 |

### 8.3 测试文件命名

```

tests/
├── test_<module_name>.py      # 单元测试
├── test_integration_`<flow>`.py  # 集成测试
└── test_e2e_`<scenario>`.py      # E2E测试

```

---

## 九、文档管理

### 9.1 文档结构

| 文档 | 位置 | 维护频率 | 说明 |
|-----|------|---------|------|
| README.md | /docs | 每版本更新 | 项目入口文档 |
| CHANGELOG.md | /docs | 每次PR | 变更日志 |
| architecture.md | /docs | 架构变更时 | 架构设计 |
| api_reference.md | /docs | API变更时 | API文档 |
| learning_notes.md | /docs | 持续更新 | 学习笔记（面试用） |

### 9.2 代码注释规范

所有函数应遵循以下注释规范：

- **函数文档字符串（docstring）**：说明函数功能、参数、返回值、异常和示例
- **行内注释**：解释复杂逻辑或关键计算
- **类型注解**：为函数参数和返回值提供类型信息
- **示例代码**：提供函数用法的示例

---

## 十、面试准备材料

### 10.1 技术亮点（简历用）

1. **算法设计**: 设计了穷举+剪枝的混合算法，将LLM调用成本降低90%
2. **RAG应用**: 构建了基于向量检索的轨迹评估系统，支持知识库自扩充
3. **全栈能力**: 独立完成数据处理、算法设计、API开发、Web界面全流程
4. **工程规范**: 遵循企业级开发规范，包括版本控制、测试驱动、文档管理

### 10.2 常见问题准备

| 问题                                  | 回答要点                        |
| ------------------------------------- | ------------------------------- |
| 为什么选择穷举+剪枝而不是端到端生成？ | 可控性、成本、物理一致性        |
| RAG如何设计？                         | 向量编码、相似度检索、案例匹配  |
| 如何评估生成质量？                    | 物理规则+语义评估+人工审核      |
| 遇到的最大挑战？                      | 参数空间设计、LLM评估Prompt优化 |

---

## 十一、实施检查清单

### 11.1 每个Step的交付标准

| Step   | 功能完成 | 单元测试 | 集成测试 | 文档 | 可演示 |
| ------ | -------- | -------- | -------- | ---- | ------ |
| Step 1 | ✓       | ✓       | -        | ✓   | ✓     |
| Step 2 | ✓       | ✓       | ✓       | ✓   | ✓     |
| Step 3 | ✓       | ✓       | ✓       | ✓   | ✓     |
| Step 4 | ✓       | ✓       | ✓       | ✓   | ✓     |
| Step 5 | ✓       | ✓       | ✓       | ✓   | ✓     |
| Step 6 | ✓       | ✓       | ✓       | ✓   | ✓     |
| Step 7 | ✓       | ✓       | ✓       | ✓   | ✓     |
| Step 8 | ✓       | -        | ✓       | ✓   | ✓     |
| Step 9 | ✓       | -        | ✓       | ✓   | ✓     |

### 11.2 代码质量检查

- [ ] 所有函数都有类型注解
- [ ] 核心函数都有文档字符串
- [ ] 单元测试覆盖率≥80%
- [ ] 无硬编码（配置外置）
- [ ] 日志记录完整
- [ ] 异常处理完善

---

## 附录：文档更新日志

### 更新记录

| 版本 | 日期 | 更新内容 | 更新人 |
|------|------|---------|-------|
| v2.1 | 2026-04-12 | Week 1执行后更新：<br>1. 目录结构：添加types.py，标注已实现模块<br>2. 11维特征定义：完全修正<br>3. 危险类型判断：改为动态推断<br>4. 片段截取策略：改为可配置秒数+边界处理<br>5. 数据目录：raw→waymo-open<br>6. 存储方案：第一阶段使用JSON | Gumekn |
| v2.2 | 2026-04-12 | Week 1优化更新：<br>1. 风险分数计算：与Data_Processor.py一致<br>2. 默认主车：改为sdc_id，列表中用*标记<br>3. 边界检查：添加should_skip_fragment函数<br>4. 输出文件名：改为{scenario_id}_{ego_id}.json<br>5. 无片段时不生成JSON，输出原因分析<br>6. 代码清理：删除tests/目录，删除未使用代码 | Gumekn |

### Week 1优化后实现状态

| 模块 | 文件 | 状态 | 备注 |
|------|------|------|------|
| 核心模块整合 | core/processor.py | ✓ 已实现 | 数据类型、数据加载、风险计算、片段截取 |
| 主程序 | Main.py | ✓ 已实现 | 含命令行参数 |
| 存储 | JSON | ✓ 已实现 | 无片段不生成 |

### 当前目录结构

```
TrajectoryAnalysis/
├── core/                           # 核心处理模块
│   ├── Main.py                     # 主程序入口（Step 1-4 调度）
│   ├── processor.py                # 整合模块（见下方模块清单）
│   └── __init__.py                 # 模块导出
│
├── rag/                            # RAG知识库模块（待实现）
│   └── __init__.py
│
├── ui/                             # 用户界面模块（待实现）
│   └── __init__.py
│
├── utils/                          # 工具函数模块（待实现）
│   └── __init__.py
│
├── docs/                           # 文档目录
│   ├── 项目框架（claude）.md      # 本文档
│   ├── 项目执行计划_详细版.md     # 详细执行计划
│   ├── Week1执行总结.md          # Week 1执行总结
│   └── 轨迹生成模块设计.md       # 轨迹生成模块设计
│
├── data/                           # 数据目录
│   ├── waymo-open/                # 原始Waymo数据（pkl）
│   └── processed/                  # 处理后数据（JSON格式）
│
├── 素材参考/                        # 归档素材文件
│
├── requirements.txt               # 依赖管理
└── .gitignore                     # Git忽略规则
```

### 模块组织原则

**重要设计原则**：模块不必独立成文件，可以是函数或类，整合在同一文件中。

每个函数/类必须明确标注：
1. **所在文件**：函数/类在哪个文件中定义
2. **调用位置**：哪个文件/函数调用它
3. **功能说明**：做什么

示例（core/processor.py）：
```python
class WaymoDataLoader:
    """Waymo数据加载器

    被使用于:
        - core/Main.py main() 创建实例
        - ScenarioProcessor.__init__() 创建数据加载器
    """

class ScenarioRiskAnalyzer:
    """场景风险分析器

    被使用于:
        - core/Main.py run_pipeline() 分析所有车辆风险
        - ScenarioProcessor.process_scenario() 分析关键车辆
    """
```

### 配置参数（core/Main.py顶部）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| DATA_DIR | data/waymo-open | Waymo原始数据目录 |
| OUTPUT_DIR | data/processed | 处理后数据输出目录 |
| N_BEFORE_SEC | 2.0 | 锚点前截取时长（秒） |
| N_AFTER_SEC | 3.0 | 锚点后截取时长（秒） |
| MIN_BEFORE_FRAMES | 10 | 锚点前最小帧数阈值 |
| MIN_AFTER_FRAMES | 15 | 锚点后最小帧数阈值 |
| RISK_THRESHOLD | 3 | 风险分数阈值 |
| SPEED_THRESHOLD | 0.5 | 速度阈值 (m/s) |
| SCENARIO_ID | 10135f16cd538e19 | 场景ID |

---

*文档版本: v2.3*
*技术方案: 穷举生成 + LLM剪枝*
*最后更新: 2026-04-12*
*Week 1完成状态: ✓ 已完成（优化版）*
