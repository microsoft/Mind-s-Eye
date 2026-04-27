# Print columns of a modelname_phi3_judged_all_tasks.csv file
import pandas as pd
import os

# Iterate over all model and step folders, aggregate judge_scores for accuracy and std dev
import numpy as np
model_eval_dir = "model_evaluation"
for model_name in os.listdir(model_eval_dir):
	model_path = os.path.join(model_eval_dir, model_name)
	if os.path.isdir(model_path):
		all_task_scores = {}
		for step_folder in os.listdir(model_path):
			step_path = os.path.join(model_path, step_folder)
			if os.path.isdir(step_path):
				csv_name = f"{model_name}_phi3_judged_all_tasks.csv"
				csv_path = os.path.join(step_path, csv_name)
				if os.path.exists(csv_path):
					df = pd.read_csv(csv_path)
					if 'task_name' in df.columns and 'judge_score' in df.columns:
						for task, group in df.groupby('task_name'):
							scores = group['judge_score'].values
							if task not in all_task_scores:
								all_task_scores[task] = []
							all_task_scores[task].extend(scores)
		# Calculate accuracy (mean) and std dev for each task
		print(f"\nModel: {model_name}")
		for task, scores in all_task_scores.items():
			scores = np.array(scores, dtype=float)
			accuracy = np.mean(scores)
			stddev = np.std(scores)
			print(f"Task: {task} | Accuracy: {accuracy:.4f} | Std Dev: {stddev:.4f}")
