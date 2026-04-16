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

## 四、意图驱动的轨迹变异流程（v2：宏动作+DFS+TopK剪枝）

### 4.1 核心思想：数量级削减

**问题**：旧方案逐帧穷举会产生 $3^{50} ≈ 10^{24}$ 级别的计算爆炸

**解决方案**：
1. **意图合并（游程编码）**：8~13个意图帧 → 3~5个Block
2. **钦定变异子集**：每种意图只保留3种最优组合
3. **宏动作递推**：Block内保持控制量恒定，避免高频抖动
4. **DFS + Top-K%剪枝**：只保留最危险的轨迹

### 4.2 意图合并（游程编码）

相邻且相同的意图合并为一个**意图区块 (IntentBlock)**：

```
输入: [(0.0s, 'cruise'), (0.5s, 'cruise'), (0.8s, 'turn_left'), (1.5s, 'turn_left'), ...]
输出: [Block('cruise', 0.0s, 0.8s), Block('turn_left', 0.8s, 1.5s), ...]
```

**变异触发时机**：只在每个Block的起始帧进行一次变异

### 4.3 意图钦定变异组合（最大3种/Block）

| 意图 | 变异组合 $(\Delta a, \Delta \omega)$ |
|------|--------------------------------------|
| cruise_maintain | (-0.1, 0), (0, 0), (0.1, 0) |
| accelerate_through | (0, 0.0087), (0.1, 0), (0.2, 0) |
| decelerate_to_yield | (-0.1, 0), (-0.2, 0), (-0.5, 0) |
| decelerate_to_stop | (-0.1, 0), (-0.2, 0), (-0.5, 0) |
| emergency_brake | (-0.5, 0), (-1.0, 0), (-2.0, 0) |
| lane_change_left | (0, 0.0087), (0, 0.0175), (0, 0.0349) |
| lane_change_right | (0, -0.0087), (0, -0.0175), (0, -0.0349) |
| turn_left | (-0.1, 0.0175), (-0.2, 0.0175), (0, 0.0349) |
| turn_right | (-0.1, -0.0175), (-0.2, -0.0175), (0, -0.0349) |
| go_straight | (-0.1, 0), (0, 0), (0.1, 0) |

### 4.4 穷举流程

```
输入: TrajectoryFragment (JSON)
         │
         ▼
┌─────────────────────────────────────────┐
│ Step 1: 意图合并（游程编码）              │
│         8~13个意图帧 → 3~5个IntentBlock  │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Step 2: 宏动作变异（DFS遍历）            │
│                                         │
│ For each IntentBlock:                   │
│   - 钦定3种(Delta_a, Delta_omega)组合   │
│   - Block内保持控制量恒定               │
│                                         │
│ Example: 3个Block → 3³ = 27 条轨迹      │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Step 3: Top-K% 危险剪枝                 │
│                                         │
│ - 计算每条轨迹的危险分数                │
│ - 按危险分数降序排序                    │
│ - 保留前K%（如K=10 → 约2~3条轨迹）     │
└─────────────────────────────────────────┘
```

### 4.5 宏动作递推公式

**区块恒定控制量**（Block起始帧计算一次）：
$$a_{block} = \text{clamp}(a_{t-1} + \Delta a, -8.0, 3.0)$$
$$\omega_{block} = \text{clamp}(\omega_{t-1} + \Delta \omega, -0.35, 0.35)$$

**微步长递推**（Block内每帧执行）：
$$v_t = \max(0, v_{t-1} + a_{block} \cdot 0.1)$$
$$\theta_t = \theta_{t-1} + \omega_{block} \cdot 0.1$$
$$x_t = x_{t-1} + v_t \cdot \cos(\theta_t) \cdot 0.1$$
$$y_t = y_{t-1} + v_t \cdot \sin(\theta_t) \cdot 0.1$$

### 4.6 各意图类型的组合数（v2）

| 意图类型 | 变异组合数 |
|---------|-----------|
| cruise_maintain | 3 |
| accelerate_through | 3 |
| decelerate_to_yield | 3 |
| decelerate_to_stop | 3 (+ 停车特判) |
| emergency_brake | 3 (锁死方向盘) |
| lane_change_left | 3 |
| lane_change_right | 3 |
| turn_left | 3 |
| turn_right | 3 |
| go_straight | 3 |

### 4.7 复杂度对比

| 方案 | Block数 | 每Block分支 | 总变体数 | Top-K%后 |
|------|---------|-------------|----------|----------|
| 旧方案（逐帧穷举） | 50 | 3 | $3^{50} ≈ 10^{24}$ | - |
| 新方案（意图合并） | 3~5 | 3 | $3^5 = 243$ | ~24 |
| 新方案+Top-10% | 3~5 | 3 | $3^5 = 243$ | ~24 |

### 4.8 DFS + Top-K% 剪枝算法

```python
def dfs(block_idx, current_state, trajectory_states):
    if block_idx == len(blocks):
        # 到达叶子节点，计算危险分数
        risk = compute_risk(trajectory_states)
        results.append((trajectory_states, risk))
        return

    block = blocks[block_idx]
    for (delta_a, delta_omega) in block.mutations:
        # 计算区块恒定控制量
        a_block = clamp(current_state.a + delta_a, A_MIN, A_MAX)
        omega_block = clamp(current_state.omega + delta_omega, OMEGA_MIN, OMEGA_MAX)

        # 递推该Block内所有帧
        for _ in range(block.frame_count):
            state = kinematically_integrate(state, a_block, omega_block)
            trajectory_states.append(state)

        # 递归处理下一Block
        dfs(block_idx + 1, state, trajectory_states)

# 主流程
all_trajectories = dfs(0, initial_state, [])
top_k = sorted(all_trajectories, key=lambda x: x.risk, reverse=True)[:int(0.1 * len(all_trajectories))]
```

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
