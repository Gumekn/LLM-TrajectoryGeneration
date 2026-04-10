# 数据目录

## 目录结构

```
data/
├── raw/                # 原始数据
│   └── waymo-open/     # Waymo Open Dataset 轨迹数据
├── processed/          # 处理后的数据
├── visualizations/     # 可视化输出
│   └── waymo-open/     # 场景视频文件
└── README.md          # 本文件
```

## 数据说明

### 原始数据 (raw/waymo-open/)
- 来源：`LLM-HYNdatasets/waymo-open/`
- 内容：Waymo Open Dataset 的车辆轨迹数据 (.pkl文件)
- 大小：约 XX GB
- **注意**：大文件不纳入Git管理，需单独存放

### 处理数据 (processed/)
- 存放数据预处理后的中间结果
- 包括：风险评分、关键车辆识别、文本提示等

### 可视化 (visualizations/)
- 场景可视化视频 (.mp4)
- 轨迹图、分析图表等

## 数据使用流程

1. 将原始数据复制到 `data/raw/waymo-open/`
2. 运行 `src/Main.py` 进行处理
3. 处理结果保存到 `data/processed/`
4. 可视化结果保存到 `data/visualizations/`

## 数据备份建议

由于数据文件较大，建议：
- 使用外部硬盘或云存储备份原始数据
- Git只跟踪代码，不跟踪数据文件
