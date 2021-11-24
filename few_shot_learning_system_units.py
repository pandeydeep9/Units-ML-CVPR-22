import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.distributions as dist

from meta_neural_network_architectures import VGGReLUNormNetwork
from inner_loop_optimizers import LSLRGradientDescentLearningRule


# From Source
def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng


# From Source (Edited)
class MAMLFewShotClassifier(nn.Module):
    def __init__(self, im_shape, device, args):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLFewShotClassifier, self).__init__()
        self.args = args
        self.device = device
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.im_shape = im_shape
        self.current_epoch = 0

        self.rng = set_torch_seed(seed=args.seed)
        self.classifier = VGGReLUNormNetwork(im_shape=self.im_shape, num_output_classes=self.args.
                                             num_classes_per_set,
                                             args=args, device=device, meta_classifier=True).to(device=self.device)
        self.task_learning_rate = args.task_learning_rate

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=device,
                                                                    init_learning_rate=self.task_learning_rate,
                                                                    total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                    use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate)
        self.inner_loop_optimizer.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(params=self.classifier.named_parameters()))

        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)

        self.use_cuda = args.use_cuda
        self.device = device
        self.args = args
        self.to(device)
        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)

        self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.total_epochs,
                                                              eta_min=self.args.min_learning_rate)

        # For logging purposes
        self.logging_dict = dict()
        self.create_new_train_log = True
        self.create_new_test_log = True
        self.create_new_val_log = True
        self.phase = "val"

    # Save logs to file (For debugging during train, results during test)
    def save_log_to_csv_file(self, logging_dict):
        line_to_add = logging_dict.keys()
        value_to_add = logging_dict.values()
        import csv
        file_to_save = self.args.experiment_name + self.phase + "log.csv"
        # print("saving at: ", file_to_save)
        if int(self.args.query_rotate)<0 or int(self.args.query_rotate)>0:
            file_to_save = self.args.experiment_name + self.phase + str(self.args.query_rotate)+ "log.csv"
        elif float(self.args.query_scale)>=.1:
            file_to_save = self.args.experiment_name + self.phase + str(self.args.query_scale) + "Scalelog.csv"
        if (self.current_epoch == 0 and self.create_new_train_log) or \
                (self.current_epoch == 0 and self.phase == "val" and self.create_new_val_log) \
                or (self.phase == "test" and self.create_new_test_log):
            if self.phase == "train": self.create_new_train_log = False
            if self.phase == "val": self.create_new_val_log = False
            if self.phase == "test": self.create_new_test_log = False
            with open(file_to_save, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(line_to_add)
                writer.writerow(value_to_add)
        else:
            with open(file_to_save, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(value_to_add)

    # From Source
    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if self.args.enable_inner_loop_optimizable_bn_params:
                    param_dict[name] = param.to(device=self.device)
                else:
                    if "norm_layer" not in name:
                        param_dict[name] = param.to(device=self.device)

        return param_dict

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        self.classifier.zero_grad(names_weights_copy)

        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order)
        names_grads_wrt_params = dict(zip(names_weights_copy.keys(), grads))

        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_wrt_params,
                                                                     num_step=current_step_idx)

        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = dict()

        losses['loss'] = torch.mean(torch.stack(total_losses))
        losses['accuracy'] = np.mean(total_accuracies)

        return losses

    # Edited
    def forward(self, data_batch, epoch, use_second_order, num_steps, training_phase, task_selection_call=False,iteration_cur = -1):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :param task_selection_call: Whether we are in forward for task selection ( or not
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        [b, ncs, spc] = y_support_set.shape

        self.num_classes_per_set = ncs

        total_losses = []
        total_accuracies = []
        per_task_target_preds = [[] for i in range(len(x_target_set))]
        self.classifier.zero_grad()

        average_query_vac = []
        average_query_wrong_bel = []
        average_query_cor_bel = []
        average_query_dis = []

        average_class_level_vacuity = []

        if (self.args.keep_logs or self.args.keep_val_logs) and self.phase != "task_selection":
            try:
                all_classes = self.logging_dict['sampled classes']
            except:
                pass

        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in \
                enumerate(zip(x_support_set,
                              y_support_set,
                              x_target_set,
                              y_target_set)):

            task_losses = []
            support_losses = []
            support_accuracies = []

            if (self.args.keep_logs or self.args.keep_val_logs) and self.phase != "task_selection":
                try:
                    self.logging_dict['sampled classes'] = [cl[task_id] for cl in all_classes]
                except:
                    pass

            names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

            n, s, c, h, w = x_support_set_task.shape # was x target set task

            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1)


            # vac - vacuity, bel - belief
            for num_step in range(num_steps):
                support_loss, support_preds, support_vac, sup_wrong_bel, sup_cor_bel,sup_dis = self.net_forward(
                    x=x_support_set_task,
                    y=y_support_set_task,
                    weights=names_weights_copy,
                    backup_running_statistics=
                    True if (num_step == 0) else False,
                    training=training_phase, num_step=num_step, all_see=True)

                support_losses.append(support_loss.detach())
                _, support_predicted = torch.max(support_preds.data, 1)
                sup_accuracy = torch.mean(support_predicted.float().eq(y_support_set_task.data.float()).cpu().float())
                support_accuracies.append(sup_accuracy)

                names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step)

                if num_step == (self.args.number_of_training_steps_per_iter - 1):
                    how_many_tasks = self.args.select_from_tasks

                    task_selection = False
                    if self.args.use_iter_for_ts and iteration_cur>0:
                        if iteration_cur >= self.args.start_ts_after_iter:
                            task_selection = True
                    elif int(epoch) >= int(self.args.start_task_selection):
                        task_selection = True

                    # print("task selection: ", int(epoch) <= int(self.args.start_task_selection ), iteration_cur,training_phase)
                    if int(self.args.select_from_tasks)<=1 or not task_selection \
                            or not training_phase \
                            or int(self.args.vac_inc_balance)>1.01:
                        # print("ok", iteration_cur, epoch)
                        how_many_tasks = 0
                    else:
                        query_scores = np.zeros((self.args.select_from_tasks))
                    for i in range(how_many_tasks):
                        # print("i: ", i, "shape : ", x_target_set_task.shape,  " and ", x_support_set_task.shape)
                        x_target_set_task_i = x_target_set_task[:,  self.args.num_target_samples * i: self.args.num_target_samples * (i+1)]
                        y_target_set_task_i = y_target_set_task[:,  self.args.num_target_samples * i: self.args.num_target_samples * (i+1)]


                        x_target_set_task_i = x_target_set_task_i.reshape(-1, c, h, w)
                        y_target_set_task_i = y_target_set_task_i.reshape(-1)
                        with torch.no_grad():
                            target_loss, target_preds, target_vac, tar_wrong_bel, tar_cor_bel,target_diss = self.net_forward(
                                x=x_target_set_task_i,
                                y=y_target_set_task_i, weights=names_weights_copy,
                                backup_running_statistics=False, training=False,
                                num_step=num_step, all_see=True, query_set=True)

                            av_q_vac = torch.mean(target_vac)
                            av_q_wr_bel = torch.mean(tar_wrong_bel)
                            av_q_diss = torch.mean(target_diss)
                            #Use Dissonance and vacuity
                            # print("Task id: , i ", i , av_q_vac, av_q_diss)

                            #Dynamically adjust the two
                            start_vl = self.args.vac_inc_balance
                            end_vl = 0.5
                            num_epochs_vl = 50
                            diff_vl = start_vl - end_vl
                            lam_vac_diss_bal = self.args.vac_inc_balance - diff_vl * min(1, epoch / num_epochs_vl)
                            # print('s: ','ep:',epoch, start_vl, 'e: ', end_vl, 'n: ', num_epochs_vl, 'l: ', lam_vac_diss_bal)

                            query_scores[i] = av_q_vac * lam_vac_diss_bal + \
                                              av_q_diss * (1.0 - lam_vac_diss_bal)


                    if how_many_tasks == 0:
                        x_target_set_task_i = x_target_set_task
                        y_target_set_task_i = y_target_set_task
                        if int(self.args.vac_inc_balance)>1.01:
                            x_target_set_task_i = x_target_set_task[:, :self.args.num_target_samples]
                            y_target_set_task_i = y_target_set_task[:, :self.args.num_target_samples]

                    else:
                        best_query_index = np.argmax(query_scores)
                        x_target_set_task_i = x_target_set_task[:,  self.args.num_target_samples * best_query_index: \
                                                                    self.args.num_target_samples * (best_query_index+1)]
                        y_target_set_task_i = y_target_set_task[:,  self.args.num_target_samples * best_query_index: \
                                                                    self.args.num_target_samples * (best_query_index+1)]

                    x_target_set_task_i = x_target_set_task_i.reshape(-1, c, h, w)
                    y_target_set_task_i = y_target_set_task_i.reshape(-1)

                    target_loss, target_preds, target_vac, tar_wrong_bel, tar_cor_bel, target_diss = self.net_forward(
                        x=x_target_set_task_i,
                        y=y_target_set_task_i, weights=names_weights_copy,
                        backup_running_statistics=False, training=training_phase,
                        num_step=num_step, all_see=True, query_set=True)

                    task_losses.append(target_loss)

                    average_query_vac.append(target_vac)

                    class_level_vac = np.mean(
                        target_vac.detach().cpu().numpy().reshape(-1, self.args.num_target_samples), axis=1)
                    # print("target class level: , ", class_level_vac)
                    average_class_level_vacuity.append(class_level_vac)
                    average_query_wrong_bel.append(tar_wrong_bel)
                    average_query_cor_bel.append(tar_cor_bel)
                    average_query_dis.append(target_diss)

            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            _, predicted = torch.max(target_preds.data, 1)

            accuracy = predicted.float().eq(y_target_set_task_i.data.float()).cpu().float()
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            total_accuracies.extend(accuracy)

            support_losses = torch.stack(support_losses)
            support_accuracies = torch.stack(support_accuracies)

            if (self.args.keep_logs or self.args.keep_val_logs) and self.phase != "task_selection":
                self.logging_dict['epoch'] = epoch
                self.logging_dict['task_id'] = task_id
                self.logging_dict['support losses'] = support_losses.detach().cpu().numpy()
                self.logging_dict['support accuracies'] = support_accuracies.numpy()
                self.logging_dict['query loss'] = task_losses.detach().cpu().numpy()
                self.logging_dict['query accuracy'] = torch.mean(accuracy).numpy()
                self.logging_dict['query vacuity'] = torch.mean(average_query_vac[-1]).detach().cpu().numpy()
                self.logging_dict['all query vacs'] = average_query_vac[-1].detach().cpu().numpy()
                self.logging_dict['all query accuracies'] = accuracy.numpy()
                self.logging_dict['all query cor bel'] = average_query_cor_bel[-1].detach().cpu().numpy()
                self.logging_dict['average query cor bel'] = torch.mean(
                    average_query_cor_bel[-1]).detach().cpu().numpy()
                self.logging_dict['all query inc bel'] = average_query_wrong_bel[-1].detach().cpu().numpy()
                self.logging_dict['average query inc bel'] = torch.mean(
                    average_query_wrong_bel[-1]).detach().cpu().numpy()

                self.save_log_to_csv_file(self.logging_dict)

            if not training_phase:
                self.classifier.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(total_losses=total_losses, total_accuracies=total_accuracies)

        #avg_q_unc is average query set vacuity (vacuity and uncertainty are used interchangebly)
        losses['avg_q_vac'] = torch.mean(torch.stack(average_query_vac))
        losses['av_q_vac_per_task'] = torch.mean(torch.stack(average_query_vac), dim=1).detach().cpu()
        losses['av_q_vac_per_class'] = average_class_level_vacuity  ################
        losses['avg_q_wrong_bel'] = torch.mean(torch.stack(average_query_wrong_bel))
        losses['avg_q_cor_bel'] = torch.mean(torch.stack(average_query_cor_bel))
        losses['avg_q_cor_bel_per_class'] = average_query_cor_bel
        losses['avg_q_inc_bel_per_class'] = average_query_wrong_bel
        losses['avg_q_dis'] = torch.mean(torch.stack(average_query_dis))

        if not task_selection_call:
            return losses, per_task_target_preds

        # Return task scores ( For task selection)
        av_q_vac = torch.stack(average_query_vac)  # query set vacuity
        av_q_wr_bel = torch.stack(average_query_wrong_bel)  # query set wrong belief
        av_q_cor_bel = torch.stack(average_query_cor_bel)  # query set correct belief
        av_q_dis = torch.stack(average_query_dis)

        av_q_vac = torch.mean(av_q_vac, dim=1)  # average query set vacuity
        av_q_wr_bel = torch.mean(av_q_wr_bel, dim=1)  # average query set wrong belief
        av_q_cor_bel = torch.mean(av_q_cor_bel, dim=1)  # average query set correct belief
        av_q_dis = torch.mean(av_q_dis, dim = 1)

        #Previous task selection idea (Units-ST). Not for Units-ML
        if self.args.task_sel_with_inc:
            # print("epoch: ", epoch)
            start = self.args.vac_inc_balance
            end = 0.5
            num_epochs = 50
            diff = start - end
            lambda_val = self.args.vac_inc_balance - diff * min(1,epoch/num_epochs)
            # print("da: ", av_q_vac, av_q_dis)
            return av_q_vac * lambda_val + av_q_dis * (1.0 - lambda_val)

            # return av_q_vac * self.args.vac_inc_balance + av_q_wr_bel * (1.0 - self.args.vac_inc_balance)
        return av_q_vac  # + av_q_wr_bel

    def KL_flat_dirichlet(self, alpha):
        """
        Calculate Kl divergence between a flat/uniform dirichlet distribution and a passed dirichlet distribution
        i.e. KL(dist1||dist2)
        distribution is a flat dirichlet distribution
        :param alpha: The parameters of dist2 (2nd distribution)
        :return: KL divergence
        """
        num_classes = alpha.shape[1]
        beta = torch.ones(alpha.shape, dtype=torch.float32, device=self.device)

        dist1 = dist.Dirichlet(beta)
        dist2 = dist.Dirichlet(alpha)

        kl = dist.kl_divergence(dist1, dist2).reshape(-1, 1)
        return kl

    # A function to calculate the loss based on eqn. 5 of the paper

    def dir_prior_mult_likelihood_loss(self, gt, alpha, current_epoch):
        """
        Calculate the loss based on the dirichlet prior and multinomial likelihoood
        :param gt: The ground truth (one hot vector)
        :param alpha: The prior parameters
        :param current_epoch: For the regularization parameter
        :return: loss
        """
        gt = gt.to(self.device)
        alpha = alpha.to(self.device)

        S = torch.sum(alpha, dim=1, keepdim=True)

        first_part_error = torch.sum(gt * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)
        annealing_rate = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(current_epoch / 10, dtype=torch.float32)
        )

        if self.args.fix_annealing_rate:
            annealing_rate = 1
            # print(annealing_rate)

        alpha_new = (alpha - 1) * (1 - gt) + 1
        kl_err = self.args.kl_scaling_factor * annealing_rate * self.KL_flat_dirichlet(alpha_new)


        #return loss

        dirichlet_strength = torch.sum(alpha, dim=1)
        dirichlet_strength = dirichlet_strength.reshape((-1, 1))

        # Belief
        belief = (alpha - 1) / dirichlet_strength

        inc_belief = belief * (1 - gt)
        inc_belief_error = self.args.kl_scaling_factor * annealing_rate * torch.mean(inc_belief, dim = 1, keepdim=True)

        if self.args.use_kl_error:
            loss = first_part_error + kl_err
            # print("kl using")
        else:
            loss = first_part_error + inc_belief_error

        return loss

    def calculate_dissonance_from_belief(self, belief):
        num_classes = len(belief)
        Bal_mat = torch.zeros((num_classes, num_classes)).to(self.device)
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                if belief[i] == 0 or belief[j] == 0:
                    Bal_mat[i, j] = 0
                else:
                    Bal_mat[i, j] = 1 - torch.abs(belief[i] - belief[j]) / (belief[i] + belief[j])
                Bal_mat[j, i] = Bal_mat[i, j]
        sum_belief = torch.sum(belief).to(self.device)
        dissonance = 0
        for i in range(num_classes):
            if torch.sum(belief * Bal_mat[i, :]) == 0: continue
            dissonance += belief[i] * torch.sum(belief * Bal_mat[i, :]) / (sum_belief - belief[i])
        return dissonance

    def calculate_dissonance(self, belief):

        dissonance = torch.zeros(belief.shape[0])
        for i in range(len(belief)):
            dissonance[i] = self.calculate_dissonance_from_belief(belief[i])
        return dissonance

    def calculate_dissonance_from_belief_vectorized(self, belief):
        # print("belief shape: ", belief.shape)
        sum_bel_mat = torch.transpose(belief, -2, -1) + belief
        sum_bel_mat[sum_bel_mat == 0] = -500
        # print("sum: ", sum_bel_mat)
        diff_bel_mat = torch.abs(torch.transpose(belief, -2, -1) - belief)
        # print("diff bel mat: ", diff_bel_mat)
        div_diff_sum = torch.div(diff_bel_mat, sum_bel_mat)
        # print("div diff sum up: ", div_diff_sum)

        div_diff_sum[div_diff_sum < -0] = 1
        # print("div diff sum: ", div_diff_sum)
        Bal_mat = 1 - div_diff_sum
        # print("Bal Mat vec: ", Bal_mat)
        # import sys
        # sys.exit()
        num_classes = belief.shape[1]
        Bal_mat[torch.eye(num_classes).byte().bool()] = 0  # torch.zeros((num_classes, num_classes))
        # print("BAL mat: ", Bal_mat)

        sum_belief = torch.sum(belief)

        bel_bal_prod = belief * Bal_mat
        # print("Prod: ", bel_bal_prod)
        sum_bel_bal_prod = torch.sum(bel_bal_prod, dim=1, keepdim=True)
        divisor_belief = sum_belief - belief
        scale_belief = belief / divisor_belief
        scale_belief[divisor_belief == 0] = 1
        each_dis = torch.matmul(scale_belief, sum_bel_bal_prod)

        return torch.squeeze(each_dis)

    def calculate_dissonance_from_belief_vectorized_again(self,belief):
        belief = torch.unsqueeze(belief, dim=1)

        sum_bel_mat = torch.transpose(belief, -2, -1) + belief  # a + b for all a,b in the belief
        diff_bel_mat = torch.abs(torch.transpose(belief, -2, -1) - belief)

        div_diff_sum = torch.div(diff_bel_mat, sum_bel_mat)  # |a-b|/(a+b)

        Bal_mat = 1 - div_diff_sum
        zero_matrix = torch.zeros(sum_bel_mat.shape, dtype=sum_bel_mat.dtype).to(sum_bel_mat.device)
        Bal_mat[sum_bel_mat == zero_matrix] = 0  # remove cases where a=b=0

        diagonal_matrix = torch.ones(Bal_mat.shape[1], Bal_mat.shape[2]).to(sum_bel_mat.device)
        diagonal_matrix.fill_diagonal_(0)  # remove j != k
        Bal_mat = Bal_mat * diagonal_matrix  # The balance matrix

        belief = torch.einsum('bij->bj', belief)
        sum_bel_bal_prod = torch.einsum('bi,bij->bj', belief, Bal_mat)
        sum_belief = torch.sum(belief, dim=1, keepdim=True)
        divisor_belief = sum_belief - belief
        scale_belief = belief / divisor_belief
        scale_belief[divisor_belief == 0] = 1

        each_dis = torch.einsum('bi,bi->b', scale_belief, sum_bel_bal_prod)

        return each_dis


    def calculate_dissonance2(self, belief):
        dissonance = torch.zeros(belief.shape[0])
        for i in range(len(belief)):
            dissonance[i] = self.calculate_dissonance_from_belief_vectorized(belief[i:i + 1, :])
            # break
        return dissonance

    def calculate_dissonance3(self, belief):
        # print("belief: ", belief.shape)
        dissonance = self.calculate_dissonance_from_belief_vectorized_again(belief)
            # break
        return dissonance

    def calc_loss_vac_bel(self, preds, y, query_set=False):
        """
        Calculate the loss, evidence, vacuity, correct belief, and wrong belief
        Prediction is done on the basis of evidence
        :param preds: the NN predictions
        :param y: the groud truth labels
        :param query_set: whether the query set or support set of ask
        :return: loss, vacuity, wrong_belief_vector and cor_belief_vector
        """

        # Make evidence non negative (use softplus)
        evidence = F.softplus(preds)

        # The prior parameters
        alpha = evidence + 1

        dirichlet_strength = torch.sum(alpha, dim=1)
        dirichlet_strength = dirichlet_strength.reshape((-1, 1))

        # Belief
        belief = evidence / dirichlet_strength

        # Total belief
        sum_belief = torch.sum(belief, dim=1)

        # Vacuity
        vacuity = 1 - sum_belief

        #Dissonance
        dissonance = self.calculate_dissonance3(belief)

        # one hot vector for ground truth
        gt = torch.eye(len(y))[y].to(self.device)
        gt = gt[:, :self.args.num_classes_per_set]

        wrong_belief_matrix = belief * (1 - gt)

        wrong_belief_vector = torch.sum(wrong_belief_matrix, dim=1)
        cor_belief_vector = torch.sum(belief * gt, dim=1)

        loss = self.dir_prior_mult_likelihood_loss(gt, alpha, self.current_epoch)

        loss = torch.mean(loss)

        return loss, vacuity, wrong_belief_vector, cor_belief_vector, dissonance

    # Modified
    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step, all_see=False, query_set=False):
        """
        A base model forward pass on some data points x. Using the parameters in the weights dictionary. Also requires
        boolean flags indicating whether to reset the running statistics at the end of the run (if at evaluation phase).
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param x: A data batch of shape b, c, h, w
        :param y: A data targets batch of shape b, n_classes
        :param weights: A dictionary containing the weights to pass to the network.
        :param backup_running_statistics: A flag indicating whether to reset the batch norm running statistics to their
         previous values after the run (only for evaluation)
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :return: the crossentropy losses with respect to the given y, the predictions of the base model.
        """
        preds = self.classifier.forward(x=x, params=weights,
                                        training=training,
                                        backup_running_statistics=backup_running_statistics, num_step=num_step)

        loss, vacuity, wrong_bel, cor_bel, dissonance = self.calc_loss_vac_bel(preds, y, query_set=query_set)

        if all_see:
            return loss, preds, vacuity, wrong_bel, cor_bel, dissonance
        else:
            return loss, preds

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch, iteration_cur = -1):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch,
                                                     use_second_order=self.args.second_order and
                                                                      epoch > self.args.first_order_to_second_order_epoch,
                                                     num_steps=self.args.number_of_training_steps_per_iter,
                                                     training_phase=True,
                                                     iteration_cur=iteration_cur)
        return losses, per_task_target_preds

    # Modified
    def evaluation_forward_prop(self, data_batch, epoch, task_selection_call=False):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :task_selection_call: Whether it is task selection. Then return the task uncertainty scores
        :return: A dictionary of losses for the current step.
        """
        if not task_selection_call:
            losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False,
                                                         num_steps=self.args.number_of_evaluation_steps_per_iter,
                                                         training_phase=False)

            return losses, per_task_target_preds

        target_uncs = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False,
                                   num_steps=self.args.number_of_evaluation_steps_per_iter,
                                   training_phase=False, task_selection_call=task_selection_call)
        return target_uncs

    def meta_update(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        if 'imagenet' in self.args.dataset_name:
            for name, param in self.classifier.named_parameters():
                if param.requires_grad:
                    param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed
        self.optimizer.step()

    def run_train_iter(self, data_batch, epoch, logging_dict=None,iteration_cur = -1):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param logging_dict: The details for logging
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """

        self.logging_dict = logging_dict
        self.phase = "train"

        epoch = int(epoch)
        self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        # print("current epoch: ",self.current_epoch)
        if not self.training:
            self.train()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)

        # print(y_target_set)
        # print(y_target_set.shape)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        # print(x_target_set.shape)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_target_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch, iteration_cur = iteration_cur)

        self.meta_update(loss=losses['loss'])
        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()

        return losses, per_task_target_preds

    def run_validation_iter(self, data_batch, epoch, logging_dict=None):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param logging_dict: ok
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """

        if self.current_epoch != epoch:
            self.current_epoch = epoch

        self.phase = "test"
        if self.current_epoch < self.args.total_epochs:
            self.phase = "val"

        if self.training:
            self.eval()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        if logging_dict is not None:
            self.logging_dict = logging_dict

        losses, per_task_target_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch)

        return losses, per_task_target_preds

    def save_model(self, model_save_dir, state, prob_dict={}):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        :param prob_dict: the probability of sampling from class (training only)
        """
        state['network'] = self.state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        state['scheduler'] = self.scheduler.state_dict()

        if self.args.imp_sampling:
            state['class_prb_dict'] = prob_dict

        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):
        """
        Load checkpoint and return the state dictionary containing the network state params and experiment state.
        :param model_save_dir: The directory from which to load the files.
        :param model_name: The model_name to be loaded from the direcotry.
        :param model_idx: The index of the model (i.e. epoch number or 'latest' for the latest saved model of the current
        experiment)
        :return: A dictionary containing the experiment state and the saved model parameters.
        """
        print("Model idx: ", model_idx)
        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath)
        state_dict_loaded = state['network']
        self.optimizer.load_state_dict(state['optimizer'])
        #        self.scheduler.load_state_dict(state['scheduler'])
        self.load_state_dict(state_dict=state_dict_loaded)
        return state

    def run_validation_iter_ts(self, data_batch, epoch):
        """
        Run meta-testing for task selection. Returns uncertainty score for each task.
        :param data_batch: All candidate tasks
        :param epoch: Which epoch is it?
        :return: Uncertainty score for all tasks
        """

        self.phase = "task_selection"
        self.current_epoch = epoch

        if self.training:
            self.eval()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        target_uncs = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch, \
                                                   task_selection_call=True)

        # losses['loss'].backward() # uncomment if you get the weird memory error
        # self.zero_grad()
        # self.optimizer.zero_grad()

        return target_uncs

    def run_uncertain_pred(self, x_support_set, x_target_set, y_support_set, y_target_set):
        """
        Get Uncertainty
        :param x_support_set: support set images
        :param x_target_set: support set labels
        :param y_support_set: query set images
        :param y_target_set: query set labels
        :return: uncertainty score
        """

        if self.training:
            self.eval()

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        target_uncs = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch, \
                                                   task_selection_call=True)

        return target_uncs
