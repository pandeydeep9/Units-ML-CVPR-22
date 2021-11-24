import sys
import os

dataset = ["omniglot", "mini-imagenet", "cifarfs"]
dataset = dataset[0]
to_skip_json = ["train_seed", "val_seed", "kl_scaling_factor",
				"total_epochs", "keep_logs", "total_epochs_before_pause",
				"num_models_to_ensemble" ] #If want to change the values directly without changing the json file

for way in [5]:
	for shot in [1]:
		for seed in [2]:
			exp_name = f"the_name_to_save_model_in2"
			os.system(f"python3 train_vac_inc_system.py "
					  f"--json_file config_ts/{dataset}_{shot}_{way}.json "
					  f"--use_bash True  --use_bash_all True "
					  f"--to_skip_json {to_skip_json}  "
					  f"--experiment_name {exp_name} "
					  f"--vac_inc_balance 0.99 " # Start from 0.99 dynamically adjusted as training progresses
					  f"--keep_logs False " # If debugging 
					  f"--kl_scaling_factor 2.0 " #Eta value
					  f"--total_epochs 100 " 
					  f"--total_epochs_before_pause 81 "
					  f"--use_kl_error False " #Keep this False
					  f"--train_seed {seed} "
					  f"--val_seed {seed} "
					  f"--task_sel_with_inc True "

					  )

