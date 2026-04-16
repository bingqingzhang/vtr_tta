from copy import deepcopy
import logging
import torch
import torch.nn as nn


import math
import numpy as np
from tta_model.utils import softmax_entropy
from tta_model.param import load_model_and_optimizer, copy_model_and_optimizer
from tta_model.loss_tracker import LossTracker
from tta_model.tta_base import TTABase

logger = logging.getLogger(__name__)

def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


class SAR(TTABase):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, tokenizer, optimizer, steps=1, episodic=False, margin_e0=0.4*math.log(1000), reset_constant_em=0.2, config=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.total_steps = steps
        assert steps > 0, "SAR requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.margin_e0 = margin_e0  # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = reset_constant_em  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria
        self.config = config

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        self.device = config.device
        self.con_ratio = config.con_ratio
        self.temperature = config.temperature
        self.t = config.t
        
        w = getattr(config, "loss_window_size", 6)
        e = getattr(config, "loss_log_every", 3)
        self.loss_tracker = LossTracker(adapt_method='sar', logger=logger, window_size=w, log_every=e)

    def forward(self, x, num_iter=None):
        if self.episodic:
            self.reset()

        for step in range(self.total_steps):
            sim_matrix, reset_flag = self.forward_and_adapt_sar(x, step, iter_idx=num_iter)
            # if reset_flag:
            #     self.reset()
        return sim_matrix

    def reset(self):
        super().reset()
        self.ema = None

    def forward_and_adapt_sar(self, modality_query, cur_step, iter_idx):
        with torch.set_grad_enabled(True):
            self.optimizer.zero_grad()
            outputs = self.forward_output(modality_query)
            entropys = softmax_entropy(outputs).sum(1)
            k = int(len(entropys) * self.margin_e0)
            top_k_values, _ = torch.topk(entropys, k, largest=False)
            threshold = top_k_values[-1]  # the k-th smallest entropy value

            # Filter out elements with entropy less than or equal to the threshold
            filter_ids_1 = torch.where(entropys <= threshold)
            
            if filter_ids_1[0].numel() > 0:  # Ensure there are valid elements after filtering
                entropys = entropys[filter_ids_1]
                loss = entropys.mean(0)
            else:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)  # Set loss to 0 if no valid samples
            loss.backward()
            self.optimizer.first_step(zero_grad=True)  # Compute \hat{\epsilon(\Theta)} for first-order approximation
            
            # second forward pass
            outputs2 = self.forward_output(modality_query)
            entropys2 = softmax_entropy(outputs2)
            
            if filter_ids_1[0].numel() > 0:  # Ensure there are valid elements after the first filtering
                entropys2 = entropys2[filter_ids_1]
                filter_ids_2 = torch.where(entropys2 < threshold)  # Re-filter reliable samples after model update
                
                if filter_ids_2[0].numel() > 0:  # Ensure there are valid elements after re-filtering
                    loss_second = entropys2[filter_ids_2].mean(0)
                else:
                    loss_second = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                loss_second = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            if not torch.isnan(loss_second):
                self.ema = update_ema(self.ema, loss_second.item())
            loss_second.backward()
            self.optimizer.second_step(zero_grad=True)

            reset_flag = False
            if self.ema is not None:
                if self.ema < 0.2:
                    # print("ema < 0.2, now reset the model")
                    reset_flag = True
            loss_dict = {
                'loss_first': loss.item(),
                'loss_second': loss_second.item() if not torch.isnan(loss_second) else 0.0,
                'total_loss': (loss.item() + loss_second.item()) / 2 if not torch.isnan(loss_second) else loss.item(),
            }
        if self.loss_tracker is not None:
            self.loss_tracker.add(
                iter_idx=iter_idx,
                step_idx=cur_step,
                steps_per_iter=self.total_steps,
                loss_dict=loss_dict
            )
        return outputs, reset_flag

"""
from https://github.com/davda54/sam
"""
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups