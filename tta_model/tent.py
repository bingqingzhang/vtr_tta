import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from tta_model.param import load_model_and_optimizer, copy_model_and_optimizer
from tta_model.utils import softmax_entropy
from tta_model.tta_base import TTABase
from tta_model.loss_tracker import LossTracker

logger = logging.getLogger(__name__)

class Tent(TTABase):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, tokenizer, optimizer, steps=1, episodic=False, config=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.total_steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
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
        self.loss_tracker = LossTracker(adapt_method='tent', logger=logger, window_size=w, log_every=e)


    def forward(self, x, num_iter=None):
        if self.episodic:
            self.reset()

        for step in range(self.total_steps):
            sims_matrix = self.forward_and_adapt(x, step, iter_idx=num_iter)

        return sims_matrix
        
    def forward_and_adapt(self, modality_query, cur_step, iter_idx):
        with torch.set_grad_enabled(True):
            outputs = self.forward_output(modality_query)
            loss = softmax_entropy(outputs).sum(1).mean(0)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_dict = {
                'loss_total': loss.item(),
            }
        if self.loss_tracker is not None:
            self.loss_tracker.add(
                iter_idx=iter_idx,
                step_idx=cur_step,
                steps_per_iter=self.total_steps,
                loss_dict=loss_dict
            )
        return outputs




