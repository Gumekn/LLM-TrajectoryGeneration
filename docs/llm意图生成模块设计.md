# LLM 意图驱动轨迹变异系统 - 工作状态记录

## 概述

本文档记录轨迹变异生成系统当前各模块的实现状态，用于确保后续工作衔接顺畅。

---

## 一、已完成模块

### 1. 场景轨迹提示词构建 ✅

**文件位置**: `core/llm/prompt_builder.py` - `TrajectoryPromptBuilder` 类

**实现内容**:
- 7部分结构的轨迹信息提示词
  - Scene Context & Danger Profile（场景信息）
  - Spatial Relationship（空间关系）
  - Key Frame Analysis（关键帧分析）
  - Vehicle Trajectory Evolution（轨迹演变）
  - Interaction Context Summary（交互统计）
  - Trajectory Profile（轨迹剖面，每N帧采样）
- 关键帧识别：anchor、min_ttc、min_dist、max_closing
- 采样间隔控制（默认2帧）
- 相对距离、TTC、空间关系等计算

**调用方式**:
```python
from core.llm import TrajectoryPromptBuilder
builder = TrajectoryPromptBuilder(sample_interval=2)
prompt = builder.build_prompt(fragment)
```

---

### 2. 系统提示词构建 ✅

**文件位置**: `core/llm/prompt_builder.py` - `SYSTEM_PROMPT` 常量

**实现内容**:
- 角色定义：自动驾驶驾驶意图预测专家
- 分析原则：全局分析、因果推理、科学判断
- 10种意图类型定义：
  - cruise_maintain, accelerate_through, decelerate_to_yield
  - decelerate_to_stop, emergency_brake
  - lane_change_left, lane_change_right
  - turn_left, turn_right, go_straight
- 意图碰撞影响说明
- 输出格式规范（JSON）

**调用方式**:
```python
from core.llm import SYSTEM_PROMPT
```

---

### 3. 用户提示词构建 ✅

**文件位置**: `core/llm/prompt_builder.py` - `build_intention_query_prompt()` 函数

**实现内容**:
- 关键帧列表格式化
- 分析要求说明（全局信息结合关键帧）
- 意图分析指导原则

**调用方式**:
```python
from core.llm import build_intention_query_prompt
full_prompt = build_intention_query_prompt(trajectory_prompt, key_frames, SYSTEM_PROMPT)
```

---

### 4. LLM API 配置 ✅

**文件位置**: `core/llm/intention_models.py` - `UnifiedLLMClient` 类

**实现内容**:
- 支持的提供商：qwen、openai、gemini
- 统一 API 调用接口 `.chat(prompt)`
- 环境变量配置：
  - qwen: `DASHSCOPE_API_KEY`
  - openai: `OPENAI_API_KEY`
  - gemini: `GEMINI_API_KEY`
- Base URL 配置
- 温度和 max_tokens 参数控制

**调用方式**:
```python
from core.llm import UnifiedLLMClient
client = UnifiedLLMClient(provider="qwen", model="qwen3.6-plus")
response = client.chat("你的问题")
```

---

### 5. 场景轨迹生成代码 ✅

**文件位置**: `core/llm_intention_generator.py` - `LLMIntentionGenerator` 类

**实现内容**:
- 完整流程封装（Step 1-5）
  - Step 1: 轨迹提示词构建
  - Step 2: 关键帧识别
  - Step 3: 完整询问提示词构建
  - Step 4: LLM API 调用
  - Step 5: 响应解析
- `generate()` - 完整意图生成
- `generate_with_fallback()` - 带异常处理的版本
- `save_fragment_with_intention()` - 结果保存函数

**调用方式**:
```python
from core.llm_intention_generator import LLMIntentionGenerator
generator = LLMIntentionGenerator(provider="qwen", model="qwen3.6-plus")
result = generator.generate(fragment)
```

---

### 6. 意图识别代码 ✅

**文件位置**: `core/llm/intention_models.py`

**实现内容**:

#### 6.1 关键帧识别 `identify_key_frames()`
- 关键帧类型：anchor、min_ttc、min_dist、max_closing、start、end、pre_mid、post_mid
- 基于 interaction_features 和 metadata 自动识别
- 按帧索引排序返回

#### 6.2 LLM 响应解析 `parse_intention_response()`
- JSON 提取：`extract_json_from_response()`
- 结构化解析为 `IntentionFrame` 对象列表
- 验证必要字段（frame、intention）

#### 6.3 意图生成 `generate_intention()`
- 调用 UnifiedLLMClient
- 低温度（0.3）确保确定性输出

---

## 二、数据结构

### IntentionFrame
```python
@dataclass
class IntentionFrame:
    frame: int           # 帧索引
    frame_type: str      # 帧类型
    intention: str       # 意图字符串
    reasoning: str       # 推理原因（≤10字）
```

### DrivingIntention 枚举
```python
class DrivingIntention(Enum):
    CRUISE_MAINTAIN = "cruise_maintain"
    ACCELERATE_THROUGH = "accelerate_through"
    DECELERATE_TO_YIELD = "decelerate_to_yield"
    DECELERATE_TO_STOP = "decelerate_to_stop"
    EMERGENCY_BRAKE = "emergency_brake"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    GO_STRAIGHT = "go_straight"
    UNKNOWN = "unknown"
```

---

## 三、模块导出

**文件位置**: `core/llm/__init__.py`

统一导出所有公共接口：
```python
from core.llm import (
    UnifiedLLMClient, list_providers, get_provider_config,
    TrajectoryPromptBuilder, SYSTEM_PROMPT, build_intention_query_prompt,
    identify_key_frames, generate_intention, parse_intention_response,
    DrivingIntention, IntentionPhase, IntentionSequence, IntentionFrame
)
```

---

## 四、待办事项

| 序号 | 事项 | 状态 | 说明 |
|------|------|------|------|
| 1 | 轨迹变异器完善 | 进行中 | `trajectory_mutator.py` 已实现穷举版本，需与意图生成结果对接 |
| 2 | LLM 调用测试 | 待办 | 需要真实 API Key 测试完整流程 |
| 3 | 意图序列生成 | 待办 | 当前 `IntentionSequence` 数据结构已定义，LLM 实际输出解析需验证 |

---

## 五、文件清单

| 文件路径 | 说明 |
|----------|------|
| `core/llm/__init__.py` | 模块导出 |
| `core/llm/prompt_builder.py` | 提示词构造（系统、用户、轨迹） |
| `core/llm/intention_models.py` | LLM 客户端、意图识别、响应解析 |
| `core/llm_intention_generator.py` | 意图生成器封装 |
| `core/trajectory_mutator.py` | 轨迹变异器（穷举版） |

---

## 六、后续工作建议

1. **验证意图生成流程**：使用真实片段数据调用 `LLMIntentionGenerator.generate()` 验证完整流程
2. **对接轨迹变异**：将意图生成结果（IntentionSequence）接入 `IntentionDrivenTrajectoryMutator`
3. **测试不同场景**：验证 qwen/openai/gemini 三个提供商的兼容性

---

*最后更新: 2026-04-15*
