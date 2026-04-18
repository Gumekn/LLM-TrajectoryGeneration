# Trajectory Analysis Project

基于大语言模型的车辆轨迹分析与风险评估系统。0418

## 项目简介

本项目采用**穷举参数变异 + LLM智能剪枝**的技术路线，通过系统性地遍历参数空间生成大量候选轨迹，再利用LLM的语义理解能力进行合理性评估和筛选，从而高效生成符合物理约束的危险驾驶场景。

**核心思想**：让LLM做它最擅长的事（评估/判断），让计算机做它最擅长的事（批量生成/计算）。

## 项目结构

```
TrajectoryAnalysis/
├── core/                           # 核心处理模块
│   ├── Main.py                     # 主程序入口（Step 1-4 调度）
│   ├── processor.py                # 整合模块：
│   │                                #   - 数据类型定义 (VehicleTrajectory, RiskAnalysisResult等)
│   │                                #   - Waymo数据加载 (WaymoDataLoader)
│   │                                #   - 风险计算 (RiskCalculator, ScenarioRiskAnalyzer)
│   │                                #   - 片段截取与特征提取 (ScenarioProcessor)
│   │                                #   - 工具函数 (slice_trajectory, extract_fragment_slice等)
│   └── __init__.py                 # 模块导出
│
├── rag/                            # RAG知识库模块（待实现）
├── ui/                             # 用户界面模块（待实现）
├── utils/                          # 工具函数模块（待实现）
│
├── docs/                           # 项目文档
│   ├── 项目框架（claude）.md      # 系统架构设计
│   ├── 项目执行计划_详细版.md     # 详细执行计划
│   ├── Week1执行总结.md          # Week 1执行总结
│   └── 轨迹生成模块设计.md       # 轨迹生成模块设计
│
├── data/                           # 数据目录
│   ├── waymo-open/                # 原始Waymo数据（pkl）
│   └── processed/                  # 处理后数据（JSON）
│
├── 素材参考/                        # 归档素材文件
│
├── requirements.txt                # Python依赖
└── README.md                      # 本文件
```

## 模块组织原则

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
        - Main.py main() 创建实例
        - ScenarioProcessor.__init__() 创建数据加载器
    """
    ...

class ScenarioRiskAnalyzer:
    """场景风险分析器

    被使用于:
        - Main.py run_pipeline() 分析所有车辆风险
        - ScenarioProcessor.process_scenario() 分析关键车辆
    """
    ...
```

## 当前实现状态

| 阶段     | 模块      | 文件                                                     | 状态      |
| -------- | --------- | -------------------------------------------------------- | --------- |
| Week 1   | 数据加载  | core/processor.py (WaymoDataLoader)                      | ✓ 已实现 |
| Week 1   | 风险计算  | core/processor.py (RiskCalculator, ScenarioRiskAnalyzer) | ✓ 已实现 |
| Week 1   | 片段截取  | core/processor.py (ScenarioProcessor)                    | ✓ 已实现 |
| Week 1   | 特征提取  | core/processor.py (extract_11d_features)                 | ✓ 已实现 |
| Week 2-3 | RAG评估   | rag/                                                     | 待实现    |
| Week 4   | CARLA导出 | core/carla_adapter.py                                    | 待实现    |
| Week 5   | 界面开发  | ui/                                                      | 待实现    |

## 快速开始

1. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```
2. **准备数据**

   - 将Waymo数据放入 `data/waymo-open/`
3. **运行主程序**

   ```bash
   python core/Main.py
   ```
4. **查看结果**

   - 处理后的JSON在 `data/processed/`

## 核心处理流程

```
Waymo原始pkl文件
      ↓
[Step 1] WaymoDataLoader - 数据加载
         - 加载pkl文件
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
      ↓
[Step 4] FragmentExtractor - 片段截取
         - 截取锚点前后指定时长
         - 提取11维特征
         - JSON持久化
```

## 文档

详见 `docs/` 目录：

- **项目框架（claude）.md** - 系统架构设计
- **项目执行计划_详细版.md** - 详细执行计划
- **轨迹生成模块设计.md** - Week 1模块设计
- **Week1执行总结.md** - Week 1执行总结

## 作者

Gumekn (2796792623@qq.com)
