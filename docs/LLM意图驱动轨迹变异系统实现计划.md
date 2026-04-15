# 基于 LLM 意图的轨迹变异系统 - 实现计划

## 目标

用 LLM 作为"驾驶策略规划器"，根据轨迹上下文生成语义连贯的驾驶意图，再基于意图生成符合物理规律的轨迹变异。避免随机扰动导致的轨迹震荡、不连贯问题。

---

## 一、数据输入

**数据来源**：`core/processor.py` 生成的 JSON 文件
- 路径：`data/processed/{scenario_id}_{ego_id}.json`
- 内容：一个场景下一个主车对应的所有交互车轨迹片段

**关键数据字段**：
- 场景信息：scenario_id、总帧数、采样率
- 片段信息：危险类型、危险等级、最低风险分数、锚点帧
- 轨迹数据：每帧的 positions、headings、velocities、accelerations
- 交互特征：ttc_long、ttc_lat、rel_dist、lateral_offset、longitudinal_offset
- 统计特征：min_ttc_long、min_rel_dist、max_closing_speed 等

---

## 二、LLM 输入端：轨迹编码

### 2.1 编码策略

**策略 A - 结构化文本描述**（通用）：
```
场景ID: xxx | 片段时长: 5s (50帧@10Hz)
危险类型: rear_end | 危险等级: high | 最低风险分数: 1.2

主车: 位置(100.5, 45.2), 速度8.5m/s, 航向0.52rad
交互车: 位置(95.2, 44.8), 速度10.2m/s, 航向0.48rad

交互统计: 最小TTC=0.8s, 最小距离=3.2m, 最大接近速度=5.2m/s

速度序列(每5帧采样): 10.2, 9.8, 9.1, 8.5, 8.2, 7.9, ...
```

**策略 B - 数值数组**（适合长上下文模型如 Qwen-Max、GPT-4）：
```
每行: frame\tx\ty\theading\tspeed\taccel_x\taccel_y
0\t100.5\t45.2\t0.523\t8.5\t0.2\t-0.1
1\t101.3\t45.8\t0.528\t8.6\t0.3\t-0.1
...
```

### 2.2 Prompt 构造

Prompt 包含：
1. 场景上下文（编码后的结构化描述）
2. 意图类型定义（10种驾驶意图）
3. 输出格式要求（JSON Schema）
4. 约束条件（语义连贯、物理可行、风险感知）

---

## 三、LLM 输出端：意图格式

### 3.1 输出结构

```json
{
    "intention_sequence": [
        {
            "start_frame": 0,
            "end_frame": 20,
            "primary_intention": "decelerate_to_yield",
            "confidence": 0.85,
            "reasoning": "检测到追尾风险，选择减速让行"
        },
        {
            "start_frame": 20,
            "end_frame": 50,
            "primary_intention": "cruise_maintain",
            "confidence": 0.75,
            "reasoning": "风险解除后恢复匀速"
        }
    ],
    "overall_strategy": "防御性驾驶，优先保证安全"
}
```

### 3.2 支持的意图类型

| 意图 | 含义 | 典型参数曲线 |
|------|------|-------------|
| cruise_maintain | 保持速度直行 | 速度恒定，加速度≈0 |
| decelerate_to_yield | 减速让行 | 速度指数衰减，减速度1.5-3m/s² |
| accelerate_through | 加速通过 | 速度S形增长，加速度2-4m/s² |
| emergency_brake | 紧急制动 | 速度快速降至0，减速度5-8m/s² |
| lane_change_left | 向左变道 | 横向位移+2~3.5m，钟形横向加速度 |
| lane_change_right | 向右变道 | 横向位移-3.5~-2m，钟形横向加速度 |
| turn_left/right | 左/右转 | 航向角持续变化 |
| go_straight | 直行 | 无横向运动 |

---

## 四、意图驱动的轨迹变异流程（穷举版本）

### 4.1 核心思想：穷举所有可能的参数组合

每个意图段内的参数是**固定的**（不随时间变化），但每个参数有有限的离散值：
- 速度缩放因子 speed_scale: {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2}
- 纵向加速度 accel_long: {-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0} m/s²
- 横向加速度 accel_lat: {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0} m/s²
- 航向变化率 heading_rate: {-0.3, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3} rad/s

### 4.2 意图约束

每个意图类型根据其语义方向筛选有效参数组合：
- decelerate_to_yield: accel_long 必须是负的（减速）
- accelerate_through: accel_long 必须是正的（加速）
- lane_change_left: heading_rate > 0（向左转）
- lane_change_right: heading_rate < 0（向右转）

### 4.3 穷举流程

```
输入: TrajectoryFragment (JSON)
         │
         ▼
┌─────────────────────────────────────────┐
│ Step 1: 意图序列解析                     │
│         每个意图段 [start_frame, end_frame] │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Step 2: 穷举参数组合                     │
│                                         │
│ For each intention segment:             │
│   - speed_scale: 从离散集选1个           │
│   - accel_long: 从离散集选1个            │
│   - accel_lat: 从离散集选1个             │
│   - heading_rate: 从离散集选1个          │
│                                         │
│ 按意图约束过滤，得到有效组合列表          │
│                                         │
│ Example: DECELERATE_TO_YIELD            │
│   → 5(speed_scale) × 5(accel_long) ×    │
│     3(accel_lat) × 3(heading_rate)      │
│   = 225 种组合/段                       │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Step 3: 运动学积分生成轨迹               │
│                                         │
│ 对于每一组参数组合:                      │
│   对每一帧 t = 0 to N-1:                │
│     v = v0 × speed_scale               │
│     v_next = v + accel_long × dt       │
│     h_next = h + heading_rate × dt     │
│     x_next = x + v × cos(h) × dt       │
│     y_next = y + v × sin(h) × dt       │
│     z_next = 0 (固定)                   │
│                                         │
│ 输出: 一条变异轨迹                      │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Step 4: 输出完整变体库                   │
│                                         │
│ 所有段的组合数相乘 = 总变体数量          │
│ Example: 2段 × 225 × 320 = 144,000 个变体│
└─────────────────────────────────────────┘
```

### 4.4 物理积分公式

```python
dt = 0.1  # 10Hz，每帧时间间隔

# 速度更新（纵向加速度影响）
v[t+1] = v[t] × speed_scale + accel_long × dt
v[t+1] = max(v[t+1], 0)  # 速度非负

# 航向更新（航向角变化率影响）
h[t+1] = h[t] + heading_rate × dt

# 位置更新（运动学积分）
x[t+1] = x[t] + v[t] × cos(h[t]) × dt
y[t+1] = y[t] + v[t] × sin(h[t]) × dt
z[t+1] = 0  # z坐标固定为0
```

### 4.5 各意图类型的组合数

| 意图类型 | speed_scale可选 | accel_long可选 | accel_lat可选 | heading_rate可选 | 有效组合数 |
|---------|----------------|----------------|---------------|-----------------|-----------|
| CRUISE_MAINTAIN | 5 | 5 | 3 | 3 | 225 |
| DECELERATE_TO_YIELD | 5 | 5 | 3 | 3 | 225 |
| DECELERATE_TO_STOP | 5 | 4 | 3 | 3 | 180 |
| ACCELERATE_THROUGH | 5 | 6 | 3 | 3 | 270 |
| EMERGENCY_BRAKE | 4 | 3 | 3 | 3 | 108 |
| LANE_CHANGE_LEFT | 5 | 4 | 4 | 4 | 320 |
| LANE_CHANGE_RIGHT | 5 | 4 | 4 | 4 | 320 |
| TURN_LEFT | 4 | 4 | 4 | 5 | 320 |
| TURN_RIGHT | 4 | 4 | 4 | 5 | 320 |
| GO_STRAIGHT | 5 | 3 | 3 | 3 | 135 |

### 4.6 总变体数量计算

如果有 N 个意图段，总变体数 = ∏(每段有效组合数)

Example: 3段 (DECELERATE_TO_YIELD → CRUISE_MAINTAIN → ACCELERATE_THROUGH)
- 225 × 225 × 270 = 13,668,750 种变体

可以通过限制意图段数量或进一步筛选参数范围来控制数量。

---

## 五、多模型适配

### 5.1 支持的模型

| 模型 | 提供商 | 调用方式 |
|------|--------|----------|
| Qwen-Plus/Turbo/Max | 阿里千问 | OpenAI 兼容 API (DashScope) |
| GPT-4o / GPT-4o-mini | OpenAI | OpenAI API |
| Claude Sonnet/Opus | Anthropic | Anthropic API |
| Gemini Flash/Pro | Google | Google AI API |

### 5.2 统一接口

```python
class LLMClient:
    def chat(self, prompt: str, **kwargs) -> str:
        """返回文本响应"""

    def structured_output(self, prompt: str, schema: dict, **kwargs) -> dict:
        """返回结构化 JSON"""
```

### 5.3 模型工厂

```python
def get_llm_client(model_name: str) -> LLMClient:
    """根据模型名获取对应客户端"""
    # qwen-plus → QwenClient
    # gpt-4o → OpenAIClient
    # claude-sonnet → AnthropicClient
    # gemini-flash → GeminiClient
```

---

## 六、文件结构

```
core/
├── processor.py                    # 已完成：Waymo数据加载、风险分析、片段截取
├── Main.py                         # 主程序入口
├── llm/
│   ├── __init__.py                 # 统一导出
│   ├── config.py                   # 配置管理
│   ├── intention_models.py          # 仅 LLM 客户端 (UnifiedLLMClient)
│   └── prompt_builder.py            # 提示词构建：意图类型定义 + SYSTEM_PROMPT + TrajectoryPromptBuilder
├── llm_intention_generator.py       # 调用入口：轨迹提示词 + 意图生成 + 数据存储
└── trajectory_mutator.py           # 轨迹变异器：意图驱动轨迹生成
```

### 模块职责划分

| 文件 | 职责 |
|------|------|
| `prompt_builder.py` | **唯一的提示词构建位置**：意图类型定义(DrivingIntention)、系统提示词(SYSTEM_PROMPT)、轨迹提示词构造(TrajectoryPromptBuilder) |
| `intention_models.py` | 仅 LLM 客户端：UnifiedLLMClient 支持 qwen/openai/gemini |
| `llm_intention_generator.py` | 调用入口：组合 prompt_builder 和 intention_models，提供 build_trajectory_prompt()、save_fragment_with_intention() 等 |
| `trajectory_mutator.py` | 轨迹变异器：从意图序列生成变异轨迹 |

---

## 七、实现顺序

1. ✅ `core/llm/intention_models.py` - LLM 客户端（UnifiedLLMClient）
2. ✅ `core/llm/prompt_builder.py` - 提示词构建：意图类型 + SYSTEM_PROMPT + TrajectoryPromptBuilder
3. ✅ `core/llm_intention_generator.py` - 调用入口：轨迹提示词 + 数据存储
4. ✅ `core/trajectory_mutator.py` - 轨迹变异器
5. ⏳ Step 2：LLM 调用生成意图序列（已注释，待启用）
6. ⏳ 集成测试

### 提示词构建使用示例

```python
from core.llm_intention_generator import build_trajectory_prompt, get_system_prompt

# Step 1: 构建轨迹提示词
prompt = build_trajectory_prompt(fragment)

# Step 2: 获取系统提示词
system = get_system_prompt()

# 组合完整提示词
full_prompt = f"{system}\n\n{prompt}\n\n为交互车生成驾驶意图序列："
```
