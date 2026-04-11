import argparse
import os
import pandas as pd
import pickle

from utils import scenario_visualization


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default="D:\PythonProjects\Python_projects_anaconda_zpb\LLM-trajctory\LLM-HYNdatasets")
parser.add_argument('--subset_name', type=str, default='waymo-open')
parser.add_argument('--scenario_ids', type=str, nargs='+', default=['1008875645c20966'])    # default=['XXX', 'XXX', 'XXX', ...]
parser.add_argument('--output_path', type=str, default="D:\PythonProjects\Python_projects_anaconda_zpb\LLM-trajctory\LLM-HYNdatasets\Visualizations")
args = parser.parse_args()

# choose one or more scenario to visualize
scenario_ids = args.scenario_ids
for scenario_id in scenario_ids:
    with open(os.path.join(args.input_path, args.subset_name, scenario_id)+'.pkl', 'rb') as file:
        data = pickle.load(file)

    traffic_events = pd.DataFrame([])
    scenario_visualization(data, traffic_events, scenario_id, args.output_path, args.subset_name)
