from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import torch 
from torch import Tensor as t
import os
import pynvml
import time
import copy
from .utils.compute import l2_proj_maker, linf_proj_maker
import sys
import matplotlib.pyplot as plt



class ScoreBlackBoxAttack(object):
    def __init__(self, model, tokenizer, processor, device, logger, config):

        self.model = model.to(device)
        self.device = device
        self.logger = logger
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.sep_token_id  = 50265
        self.p = self.config.p
        self.iter = self.config.iter
        self.epsilon = self.config.epsilon
        self._proj = None

        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.mean = torch.tensor(self.mean).view(1, -1, 1, 1)
        self.std = (0.26862954, 0.26130258, 0.27577711)
        self.std = torch.tensor(self.std).view(1, -1, 1, 1)
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)


    @torch.no_grad()
    def get_output(self, x):
        y = self.model.generate_logits(
            self.processor, pixel_values=x['pixel_values'], device=self.device,)

        return y



    @torch.no_grad()
    def measure_gpu(self, x, device_id):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        t1 = time.time()
        power_list = []
        avg_length = 0
        
        iter_num = 1
        for _ in range(iter_num):
            y = self.get_output(x)
            caption = self.processor.tokenizer.batch_decode(y['sequences'], skip_special_tokens=True)
            caption = [self.processor.remove_tags(t) for t in caption]
            length = len(y['scores']) -1
            avg_length += (length / iter_num)
            power = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_list.append(power)

        t2 = time.time()
        latency = (t2 - t1) / iter_num
        s_energy = sum(power_list) / len(power_list) * latency
        energy = s_energy / (10 ** 3) / iter_num
        pynvml.nvmlShutdown()
        return latency, energy, avg_length, caption, y


    def maybe_autocast(self, dtype=torch.float16):
        return torch.cuda.amp.autocast(dtype=dtype)


    def _perturb(self, sample):
        raise NotImplementedError

    
    def proj_replace(self, sugg_xs_t):
        sugg_xs_t = self._proj(sugg_xs_t)
        return sugg_xs_t


    def out2loss(self, x):
        return NotImplementedError



    @torch.no_grad()
    def run_attack(self, x):

        dim = len(x['pixel_values'].shape)
        ori_img = x['pixel_values'].clone().detach()
        adv_sample = copy.deepcopy(x)
        adv_image = ori_img.clone()

        metric_list = []
        ori_latency, ori_energy, ori_len, ori_caption, ori_output  = self.measure_gpu(x, self.config.device_id)
        ori_metric = self.out2loss(ori_output)
        metric_list.append(ori_metric.item())

        print(f"Original sequence: {ori_caption}, length: {ori_len}, latency: {ori_latency}, energy: {ori_energy}")
        self.logger.info(f"Original sequence: {ori_caption}, length: {ori_len}, latency: {ori_latency}, energy: {ori_energy}")
        
        best_adv, best_len, best_latency, best_energy, best_caption = ori_img, ori_len, ori_latency, ori_energy, ori_caption
        
    
        # make a projector into xs lp-ball and within valid pixel range
        if self.p == '2':
            _proj = l2_proj_maker(ori_img, self.epsilon)
            self._proj = lambda _: _proj(_)
        elif self.p == 'inf':
            _proj = linf_proj_maker(ori_img, self.epsilon)
            self._proj =  lambda _: _proj(_)
        else:
            raise Exception('Undefined l-p!')

        
        for its in range(self.iter):
            adv_image_new = self._perturb(adv_sample)

            if self.config.restrict == 'yes':
                adv_image = self.proj_replace(adv_image_new)

            else:
                adv_image = adv_image_new
            
            
            adv_sample['pixel_values'] = adv_image

            current_latency, current_energy, current_len, current_caption, current_output = self.measure_gpu(adv_sample, self.config.device_id)
            current_metric = self.out2loss(current_output)
            metric_list.append(current_metric.item())

            print(f"{its}-th: length: {current_len}, caption: {current_caption}")

            if current_len > best_len:
                best_len = current_len
                best_caption = current_caption
                best_adv = adv_image.clone()
                best_latency = current_latency
                best_energy = current_energy
        
        print(f"VLMInferSlow sequence length: {best_len}, caption: {best_caption}, latency: {best_latency}, energy: {best_energy}")
        self.logger.info(f"VLMInferSlow sequence length: {best_len}, caption: {best_caption}, latency: {best_latency}, energy: {best_energy}")
        best_adv = best_adv.detach()
        
        return True, [ori_img, best_adv], [ori_len, best_len], [ori_latency, best_latency], [ori_energy, best_energy]
    


