---
name: project_trajectory_analysis
description: LLM轨迹分析项目，stage1处理轨迹片段，stage2做意图生成和轨迹变异
type: project
---

## 项目结构
- stage1: `core/stage1/` - 轨迹片段处理，输出到 `data/processed/`
- stage2: `core/stage2/` - 意图生成 + 轨迹变异
  - `intention_generator.py` - 意图生成逻辑（含 LLMIntentionGenerator）
  - `storage.py` - 数据持久化
  - `mutator.py` - 轨迹变异算法（DFS + Top-K% 剪枝）
  - `run_intention.py` - 意图生成入口
  - `run_mutation.py` - 轨迹变异入口
  - `llm/config.py` - LLM 统一客户端

## 关键设计决策
- run 入口文件配置所有可调参数（路径、LLM模型、TOP_K 等）
- 模块内部不含默认值常量，强制由入口传入
- run_mutation 不需要 LLM 配置（纯算法）
- storage.py 中 `generation_info` 用 `algorithm` 而非 `provider/model`

## 当前分支
- 分支名: `意图生成260415`
- 状态: ahead origin/... by 6 commits
