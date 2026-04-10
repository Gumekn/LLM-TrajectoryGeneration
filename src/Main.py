import os
import pickle
from Data_Processor import (
    ScenarioDataLoader,
    RiskCalculator,
    KeyVehicleFilter,
    KeyVehicleTrajectoryExtractor,
    KeyVehicleTrajectoryClipper,
    TrajectoryTextualizer
)

# ---------------------------------------------------------
# 地址配置
# ---------------------------------------------------------
DATA_DIR = r"D:\PythonProjects\Python_projects_anaconda_zpb\LLM-trajctory\LLM-HYNdatasets\waymo-open"
SCENARIO_ID = '10135f16cd538e19'

def save_checkpoint(data, scenario_id, step_name):
    save_path = os.path.join(DATA_DIR, f"{scenario_id}_{step_name}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"[备份成功] -> {save_path}")
    return save_path

def main():
    try:
        # --- [Step 1: 加载] ---
        loader = ScenarioDataLoader(DATA_DIR, SCENARIO_ID)
        data = loader.load_data()
        obs_id = loader.select_observer()

        # --- [Step 2: 风险计算] ---
        risk_calc = RiskCalculator(data)
        data = risk_calc.run(obs_id)
        save_checkpoint(data, SCENARIO_ID, "Step2_Risk")

        # --- [Step 3: 关键车筛选] ---
        threshold_val = float(input("\n>>> 请输入筛选阈值 (0-5): ").strip() or 2.0)
        key_filter = KeyVehicleFilter(data)
        data = key_filter.run(threshold_val)
        save_checkpoint(data, SCENARIO_ID, "Step3_KeyCar")

        # --- [Step 4: 轨迹提取与交互特征计算] ---
        extractor = KeyVehicleTrajectoryExtractor(data)
        data = extractor.run()
        save_checkpoint(data, SCENARIO_ID, "Step4_Trajectory")

        # --- [Step 5: 指定关键车危险轨迹获取] ---
        # 设置截取参数：时间间隔(s)、向前帧数、向后帧数
        CLIP_INTERVAL = 0.3
        BACK_FRAMES = 15
        FORWARD_FRAMES = 5

        clipper = KeyVehicleTrajectoryClipper(data)
        data = clipper.run(CLIP_INTERVAL, BACK_FRAMES, FORWARD_FRAMES)
        save_checkpoint(data, SCENARIO_ID, "Step5_RiskClip")

        # --- [Step 6: 轨迹文本化与提示词生成] ---
        textualizer = TrajectoryTextualizer(data)
        data, prompt_text = textualizer.run()
        final_path = save_checkpoint(data, SCENARIO_ID, "Step6_TextPrompt")

        # 打印最终生成的文本
        print("\n" + "=" * 30 + " 生成的提示词内容 " + "=" * 30)
        print(prompt_text)
        print("=" * 78)

        print("\n任务全部执行成功！")
        print(f"最终数据集文件位于:\n   {final_path}")

    except Exception as e:
        print(f"\n运行出错: {e}")

if __name__ == "__main__":
    main()