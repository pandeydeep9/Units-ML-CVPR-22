from torch import cuda


def get_args():
    import argparse
    import os
    import torch
    import json
    parser = argparse.ArgumentParser(description='Units-ML')

    parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='Batch_size for experiment')
    parser.add_argument('--image_height', nargs="?", type=int, default=28)
    parser.add_argument('--image_width', nargs="?", type=int, default=28)
    parser.add_argument('--image_channels', nargs="?", type=int, default=1)
    parser.add_argument('--reset_stored_filepaths', type=str, default="False")
    parser.add_argument('--reverse_channels', type=str, default="False")
    parser.add_argument('--num_of_gpus', type=int, default=1)
    parser.add_argument('--indexes_of_folders_indicating_class', nargs='+', default=[-2, -3])
    parser.add_argument('--train_val_test_split', nargs='+', default=[0.73982737361, 0.26, 0.13008631319])
    parser.add_argument('--samples_per_iter', nargs="?", type=int, default=1)
    parser.add_argument('--labels_as_int', type=str, default="False")
    parser.add_argument('--seed', type=int, default=104)

    parser.add_argument('--gpu_to_use', type=int)
    parser.add_argument('--num_dataprovider_workers', nargs="?", type=int, default=4)
    parser.add_argument('--max_models_to_save', nargs="?", type=int, default=5)
    parser.add_argument('--dataset_name', type=str, default="omniglot_dataset")
    parser.add_argument('--dataset_path', type=str, default="datasets/omniglot_dataset")
    parser.add_argument('--reset_stored_paths', type=str, default="False")
    parser.add_argument('--experiment_name', nargs="?", type=str, )
    parser.add_argument('--architecture_name', nargs="?", type=str)
    parser.add_argument('--continue_from_epoch', nargs="?", type=str, default='latest',
                        help='Continue from checkpoint of epoch')
    parser.add_argument('--dropout_rate_value', type=float, default=0.3, help='Dropout_rate_value')
    parser.add_argument('--num_target_samples', type=int, default=15, help='Dropout_rate_value')
    parser.add_argument('--second_order', type=str, default="False", help='Dropout_rate_value')
    parser.add_argument('--total_epochs', type=int, default=200, help='Number of epochs per experiment')
    parser.add_argument('--total_iter_per_epoch', type=int, default=500, help='Number of iters per epoch')
    parser.add_argument('--min_learning_rate', type=float, default=0.00001, help='Min learning rate')
    parser.add_argument('--meta_learning_rate', type=float, default=0.001, help='Learning rate of overall MAML system')
    parser.add_argument('--meta_opt_bn', type=str, default="False")
    parser.add_argument('--task_learning_rate', type=float, default=0.1, help='Learning rate per task gradient step')

    parser.add_argument('--norm_layer', type=str, default="batch_norm")
    parser.add_argument('--max_pooling', type=str, default="False")
    parser.add_argument('--per_step_bn_statistics', type=str, default="False")
    parser.add_argument('--num_classes_per_set', type=int, default=20, help='Number of classes to sample per set')
    parser.add_argument('--cnn_num_blocks', type=int, default=4, help='Number of classes to sample per set')
    parser.add_argument('--number_of_training_steps_per_iter', type=int, default=1,
                        help='Number of classes to sample per set')
    parser.add_argument('--number_of_evaluation_steps_per_iter', type=int, default=1,
                        help='Number of classes to sample per set')
    parser.add_argument('--cnn_num_filters', type=int, default=64, help='Number of classes to sample per set')
    parser.add_argument('--cnn_blocks_per_stage', type=int, default=1,
                        help='Number of classes to sample per set')
    parser.add_argument('--num_samples_per_class', type=int, default=1, help='Number of samples per set to sample')
    parser.add_argument('--json_file', type=str, default="None")

    parser.add_argument('--kl_scaling_factor', type=float, default=1.0, help='whether to scale the kl term in loss')
    parser.add_argument('--select_num_tasks', type=int, default=-1, help="how many tasks to select (no TS if <0)")
    parser.add_argument('--keep_logs', type=str, default="False", help="Whether to keep training log or not")
    parser.add_argument('--start_task_selection', type=int, default=-1, help="Start task selection frm specified epoch")

    parser.add_argument('--num_models_to_ensemble', type=int, default=3, help="number of models to ensemble")

    parser.add_argument('--imp_sampling', type=str, default="False", help="Whether or not change train distribution")
    parser.add_argument('--imp_sampling_class_level', type=str, default="False", help="Change based on class vacuity")
    parser.add_argument('--class_imp_tracking', type=str, default="False", help="Track Class Importance?")
    parser.add_argument('--imp_sampling_after', type=int, default=5, help="Change task distribution after ? epoch")

    parser.add_argument('--task_sel_with_inc', type=str, default="True", help="Use Incorrect/Conflicting belief for task sel.")
    parser.add_argument('--vac_inc_balance', type=float, default="1.0", help="Control amount of vacuity/inc belief 0-1")
    parser.add_argument('--use_bash', type=str, default="False", help="Whether control from bash")
    parser.add_argument('--use_bash_all', type=str, default="False", help="Whether to specify everything from bash")
    parser.add_argument('--to_skip_json', type=str, default=["nothing_to_skip"], nargs="+", help="Things to skip")

    parser.add_argument('--total_epochs_before_pause', type=int, default=101, help = 'when to stop')

    #The seed values
    parser.add_argument('--train_seed', type=int, default=0, help="The train seed for averaging")
    parser.add_argument('--val_seed', type=int, default=0, help="The validation seed for averaging")

    #KL or incorrect belief error
    parser.add_argument('--use_kl_error',type=str, default="True", help="Use KL reg. or incorrect bel. regularization")


    #OOD Appendix
    parser.add_argument('--query_rotate', type=float, default=0.0, help="Rotate query images (only for omniglot)")
    parser.add_argument('--query_scale', type=float, default=-1, help="Scale query images (only for omniglot)")

    #Fix annealing rate fix_annealing_rate
    parser.add_argument('--fix_annealing_rate', type=str, default="False", help="Whether or not to fix the annealing rate")
    parser.add_argument('--active_direct_test', type=str, default="False", help="Whether to directly look at test set results or not")

    parser.add_argument('--use_iter_for_ts', type=str, default="False", help= "Task selection based on iteration")
    parser.add_argument('--start_ts_after_iter', type=int, default=100, help="TS After 100 iterations")


    args = parser.parse_args()

    args.fix_annealing_rate = args.fix_annealing_rate.lower() == "true"
    args.active_direct_test = args.active_direct_test.lower() == "true"
    args_dict = vars(args)
    if args.json_file is not "None":
        args_dict = extract_args_from_json(args.json_file, args_dict)

    for key in list(args_dict.keys()):

        if str(args_dict[key]).lower() == "true":
            args_dict[key] = True
        elif str(args_dict[key]).lower() == "false":
            args_dict[key] = False
        if key == "dataset_path":
            # Keep data in dataset_path
            args_dict[key] = os.path.join("datasets/", args_dict[key])
            print(key, os.path.join("datasets/", args_dict[key]))

        print(key, args_dict[key], type(args_dict[key]))

    args = Bunch(args_dict)

    args.use_cuda = torch.cuda.is_available()

    if args.gpu_to_use == -1:
        args.use_cuda = False

    if args.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_to_use)
        device = cuda.current_device()
    else:
        device = torch.device('cpu')

    return args, device


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(json_file_path, args_dict):
    import json
    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)

    to_skip_json = args_dict["to_skip_json"]
    for index, a in enumerate(to_skip_json):
        a = a.replace('[','')
        a = a.replace(']','')
        a = a.replace(',','')
        print(a)
        to_skip_json[index] = a
    print("Passed from cmd: ", to_skip_json)

    for key in summary_dict.keys():
        if "gpu_to_use" in key or "to_skip_json" in key:
            continue

        if "experiment_name" in key and args_dict["use_bash"] == "True":
            # experiment name specified from bash
            continue

        if args_dict["use_bash_all"] == "True":
            if key in to_skip_json:
                print(key, " and value (From CMD not JSON) ", args_dict[key])
                continue

        args_dict[key] = summary_dict[key]
    return args_dict
