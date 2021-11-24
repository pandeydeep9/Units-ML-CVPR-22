import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from meta_neural_network_architectures import VGGReLUNormNetwork
from inner_loop_optimizers import LSLRGradientDescentLearningRule
import csv

from utils.storage import build_experiment_folder, save_statistics, save_to_json

def set_torch_seed(seed):
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)
    return rng

class ModelLoadSee(object):
    def __init__(self, args, data, model, device):

        self.args, self.device = args, device

        self.model = model

        self.saved_models_filepath, self.logs_filepath, self.samples_filepath = build_experiment_folder(
            experiment_name=self.args.experiment_name
        )

        once = True
        for i in range(1, 101):


            checkpoint = os.path.join(self.saved_models_filepath, f"train_model_{i}")

            if os.path.exists(checkpoint):

                #print("WHY")
                self.state = \
                self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="train_model",
                                      model_idx=i)
                self.start_epoch = int(self.state['current_iter'] / self.args.total_iter_per_epoch)

                #print("ITer: ", self.state['current_iter'])


                if once:
                    self.data = data(args=args, current_iter=self.state['current_iter'])
                total_losses = dict()
                with tqdm.tqdm(total=int(self.args.num_evaluation_tasks / self.args.batch_size)) as pbar_val:
                    for _, val_sample in enumerate(
                        self.data.get_val_batches(
                            total_batches=int(self.args.num_evaluation_tasks / self.args.batch_size),
                            augment_images=False)):

                        val_losses, total_losses = self.evaluation_iteration(val_sample=val_sample,
                                                                             total_losses=total_losses,
                                                                             pbar_val=pbar_val, phase='val')
                    print(val_losses)
                    val_losses['epoch'] = self.start_epoch
                    if once:
                        once = False
                        self.save_to_file(val_losses, "re_sum_val.csv", True)
                    else:
                        self.save_to_file(val_losses, "re_sum_val.csv")

    def evaluation_iteration(self, val_sample, total_losses,pbar_val, phase):
        x_support_set, x_target_set, y_support_set, y_target_set, seed, classes_sel = val_sample

        data_batch = (
            x_support_set, x_target_set, y_support_set, y_target_set)

        logging_dict = dict()
        epoch = self.start_epoch
        logging_dict['epoch'] = epoch
        logging_dict['sampled classes'] = classes_sel

        losses, _ = self.model.run_validation_iter(data_batch=data_batch, logging_dict=logging_dict, epoch = epoch-1)
        #print("OK ev it losses: ", losses)

        for key, value in zip(list(losses.keys()), list(losses.values())):
            if key not in total_losses:
                total_losses[key] = [float(value)]
            else:
                total_losses[key].append(float(value))

        val_losses = self.build_summary_dict(total_losses=total_losses)
        val_output_update = self.build_string_for_pbar(losses)


        pbar_val.update(1)
        pbar_val.set_description(
            "val_phase {} --> {}".format(epoch, val_output_update))

        return val_losses, total_losses

    def build_summary_dict(self, total_losses):
        summary_losses = dict()

        for key in total_losses:
            summary_losses['{}_{}_mean'.format("val", key)] = np.mean(total_losses[key])
            summary_losses['{}_{}_std'.format("val", key)] = np.std(total_losses[key])

        return summary_losses

    def build_string_for_pbar(self, losses):
        output_string=""
        for key, value in zip(list(losses.keys()),list(losses.values())):
            if "loss" in key or "accuracy" in key:
                value = float(value)
                output_string += "{}: {:.4f} ".format(key, value)

        return output_string

    def save_to_file(self, dict_to_save, file_to_save="re_sum_val.csv", create_new_csv=False):
        if create_new_csv:
            with open(file_to_save, 'w') as f:
                writer = csv.writer(f)
                writer.writerow( list(dict_to_save.keys()) )
                writer.writerow( list(dict_to_save.values()) )
        else:
            with open(file_to_save, 'a') as f:
                writer = csv.writer(f)
                writer.writerow( list(dict_to_save.values()) )





