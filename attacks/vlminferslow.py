
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from .score_black_box_attack import ScoreBlackBoxAttack
from .utils.compute import lp_step
from .utils.compute import norm
import torch.nn.functional as F




class VLMInferSlowAttack(ScoreBlackBoxAttack):
    def __init__(self, model, tokenizer, processor, device, logger, config):
        super(VLMInferSlowAttack, self).__init__(model, tokenizer, processor, device, logger, config)
        self.model.eval()
        self.fd_eta = self.config.fd_eta
        self.lr = self.config.lr
        self.q = self.config.q



    def out2loss(self, y):


        length = len(y['scores']) -1

        logits = torch.concatenate(list(y['scores'])) + 1e-9

        full_probs = torch.softmax(logits, dim=1)
        full_probs = full_probs[:-1]
        eos_probs= full_probs[:, self.sep_token_id]


        full_probs_new = get_topk_probs(full_probs, self.config.kl_k)
        uni_distribution_new = (torch.ones(full_probs_new.shape) / full_probs_new.shape[1]).to(self.device)

        loss_uncertainty = F.kl_div(torch.log(full_probs_new), uni_distribution_new, reduction='batchmean')

        loss_uncertainty = loss_uncertainty * self.config.kl_coef


        loss_eos = weighted_eos_loss(eos_probs, self.config.eos_factor)
        loss_eos = loss_eos * self.config.eos_coef

        loss = length - loss_eos - loss_uncertainty
        
        return loss





    def _perturb(self, sample): 
        # x: adv_sample
        sample1 = copy.deepcopy(sample)
        sample2 = copy.deepcopy(sample)

        x = sample['pixel_values']
        _shape = list(x.shape)
        num_axes = len(_shape[1:])
        gs_t = torch.zeros_like(x)
        gs_t = gs_t.to(self.device)
        for _ in range(self.q):
            exp_noise = torch.randn_like(x)
            fxs_t = x + self.fd_eta * exp_noise
            bxs_t = x - self.fd_eta * exp_noise
            sample1['pixel_values'] = fxs_t
            sample2['pixel_values'] = bxs_t
            output1 = self.get_output(sample1)
            output2 = self.get_output(sample2)
            est_deriv = (self.out2loss(output1) - self.out2loss(output2)) / (4. * self.fd_eta)
            gs_t += torch.tensor(est_deriv.reshape(-1, *[1] * num_axes) * exp_noise.to(self.device))
        
        # perform the step
        new_xs = lp_step(x, gs_t, self.lr, self.p)
        return new_xs 


def weighted_eos_loss(value, factor):
    length = value.shape[0]
    weights = torch.pow(factor, torch.arange(length, device=value.device))
    weights = weights.flip(dims=[0])
    weighted_values = value * weights
    weighted_sum = weighted_values.sum()
    return weighted_sum


def get_topk_probs(full_probs, k):
    topk_probs, _ = torch.topk(full_probs, k, dim=1)
    return topk_probs
