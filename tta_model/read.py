import torch
import math
import logging
import torch.nn as nn
import torch.nn.functional as F

from tta_model.param import load_model_and_optimizer, copy_model_and_optimizer
from tta_model.tta_base import TTABase
from tta_model.loss_tracker import LossTracker

logger = logging.getLogger(__name__)

class READ(TTABase):
    """READ adapts a model by entropy minimization during testing.

    Once READed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, tokenizer, optimizer, steps=1, episodic=False, config=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.total_steps = steps
        assert steps > 0, "READ requires >= 1 step(s) to forward and update"
        self.episodic = episodic
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
        self.loss_tracker = LossTracker(adapt_method='read', logger=logger, window_size=w, log_every=e)

    def forward(self, x, num_iter):
        if self.episodic:
            self.reset()
        for step in range(self.total_steps):
            sims_matrix = self.forward_and_adapt(x, step, iter_idx=num_iter)

        return sims_matrix
    
    def forward_and_adapt(self, modality_query, cur_step, iter_idx):
        with torch.set_grad_enabled(True):
            outputs = self.forward_output(modality_query)
            
            # pred = F.softmax(outputs / self.temperature, dim=-1) 
            # eps = 1e-12
            # loss_ra = -(pred * (pred.clamp_min(eps)).log()).sum(dim=-1).mean()
            # q = pred.mean(dim=0)   
            # loss_bal = -(q * (q.clamp_min(eps)).log()).sum()
            # loss_bal = loss_bal / math.log(q.numel() + 1e-12)
            
            p_sum = outputs.softmax(dim=-1).sum(dim=-2)
            loss_bal = - (p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum()
            # coef = getattr(self, "con_ratio", 0.1)
            # loss = loss_ra - coef * loss_bal
            
            pred = outputs.softmax(dim=-1)
            pred_max = pred.max(dim=-1)[0]
            gamma = math.exp(-1)
            t = torch.ones(outputs.shape[0], device=self.device) * gamma
            loss_ra = (pred_max * (1 - pred_max.log() + t.log())).mean()
            
            # loss = loss_ra - 0.01 * loss_bal
            loss = loss_ra
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        # loss_dict = {
        #     "loss": loss.item(),
        #     "loss_ra": loss_ra.item(),
        #     "loss_bal": loss_bal.item(),
        # }
        loss_dict = {
            "loss": loss.item(),
            "loss_ra": loss_ra.item(),
            "loss_bal": 0,
        }
        if self.loss_tracker is not None:
            self.loss_tracker.add(
                iter_idx=iter_idx,
                step_idx=cur_step,
                steps_per_iter=self.total_steps,
                loss_dict=loss_dict
            )
        return outputs
