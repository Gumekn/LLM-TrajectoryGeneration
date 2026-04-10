# Trajectory Analysis Project

基于大语言模型的车辆轨迹分析与风险评估系统。

## 项目简介

本项目利用大语言模型(LLM)分析车辆轨迹数据，进行驾驶场景的风险评估和关键事件识别。主要处理 Waymo Open Dataset 的车辆轨迹数据。

## 项目结构

```
TrajectoryAnalysis/
├── src/                    # 源代码
│   ├── Main.py                    # 主程序入口
│   ├── Data_INFO.py               # 数据信息模块
│   ├── Data_Processor.py          # 数据处理器
│   ├── Data_Processor_Enhanced.py # 增强版数据处理器
│   ├── scenario_visualization.py  # 场景可视化
│   ├── database_schema.py         # 数据库架构定义
│   └── utils.py                   # 工具函数
├── data/                   # 数据目录
│   ├── raw/                # 原始数据
│   ├── processed/          # 处理后的数据
│   ├── visualizations/     # 可视化输出
│   └── README.md          # 数据说明
├── docs/                   # 项目文档
├── notebooks/              # Jupyter笔记本
├── configs/                # 配置文件
├── .gitignore             # Git忽略配置
├── requirements.txt       # Python依赖
└── README.md             # 本文件
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

1. **准备数据**
   - 将轨迹数据放入 `data/raw/waymo-open/`
   - 参考 `data/README.md` 了解数据结构

2. **运行主程序**
   ```bash
   python src/Main.py
   ```

3. **查看结果**
   - 处理后的数据在 `data/processed/`
   - 可视化结果在 `data/visualizations/`

## 功能模块

- **数据加载与预处理**: `Data_Processor.py`, `Data_Processor_Enhanced.py`
- **场景分析**: 风险计算、关键车辆识别
- **LLM提示生成**: 生成自然语言描述
- **可视化**: 轨迹动画、风险热力图

## 文档

详见 `docs/` 目录：
- 学术版框架
- 工程版框架（穷举剪枝版）
- 项目执行计划（详细版）

## 作者

Gumekn (2796792623@qq.com)
