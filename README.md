# Units-ML
This is the repository for the Units-ML code

## This code requires
* Python3
* Pytorch


Datasets: <br />
Omniglot Dataset is included with the code <br />
For MiniImageNet and CifarFS, Download the datasets, extract the datasets and place them in the Folder: datasets/ <br />
Dataset structure for the three datasets<br/>

Cifarfs
```
datasets/cifarfs
              ├── test
              ├── train
              └── val
```            
mini-ImageNet 
```
datasets/mini_imagenet_full_size
                ├── test
                ├── train
                └── val
```
Omniglot
```
datasets/omniglot_dataset
├── images_background
└── images_evaluation
```


# Running the code
To run the experiments, follow ```python3 train_vac_inc_system.py --json_file ABC``` where ```ABC``` denotes the json file for experiment (for_eg. ```config_units/cifarfs_1_5_0.json)``` 
The hyperparameter details are present in config file and can be set accordingly

By default, task selection is done on the basis of vacuous belief and conflicting belief.
To do task selection using uncertainty score (lambda * vacuous_belief + (1 - lambda) * conflicting_belief)
In the json config file, set ```"task_sel_with_inc"``` to ```True``` 

Alternatively, this can be done directly by
```python3 train_vac_inc_system.py --json_file json_file --vac_inc_balance vac_bal_value --use_bash "True" --experiment_name name_of_experiment --task_sel_with_inc "True"```
where ```json_file``` is ```config_units_ts/...```


# Units-ML Experiments

### To run omniglot N way K shot experiment (N=5/20, K=1/5)
```python3 train_vac_inc_system.py --json_file config_basic_units/omniglot_K_N.json```

### For eg, omniglot 5 way 1 shot experiment can be run as:
```python3 train_vac_inc_system.py --json_file config_basic_units/omniglot_1_5.json```
The seed values can be specified from the json file.

For other custom experiments (say 7way 3 shot), the json file can be edited 

### For mini-ImageNet experiments (N=5, K=1/5)
```python3 train_vac_inc_system.py --json_file config_basic_units/mini-imagenet_K_N.json```

### For CifarFS experiments (N=5, K=1/5)
run ```python3 train_vac_inc_system.py --json_file config_basic_units/cifarfs_K_N.json```<br />

# Task Selection Units-ML Experiments

### To run task selection after ```e``` epochs, set ```start_task_selection``` to ```e``` in the json file. Use ```select_from_tasks``` to specify the number of query sets in a task. 
```python3 train_vac_inc_system.py --json_file config_ts_units/omniglot_K_N.json```<br />
 
The json files can be edited and customized to run Units-ML for any specific setting. For e.g., ```num_samples_per_class``` in the json files can be changed to change the number of shots of the task.<br />


### Note <br />
This code is built on top of open source MAML code provided at https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch
