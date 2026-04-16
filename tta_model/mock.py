import torch
import logging
import torch.jit
import torch.nn as nn
import torch.nn.functional as F

from tta_model.param import load_model_and_optimizer, copy_model_and_optimizer
from tta_model.utils import softmax_entropy
from tta_model.tta_base import TTABase

class Mock(TTABase):
    """
    Just for mock, do nothing
    """
    def __init__(self, model, tokenizer, optimizer, steps=1, episodic=False, config=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.steps = 1
        assert steps > 0, "mock requires >= 1 step(s) to forward and update"
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


    def forward(self, x, num_iter=None):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            loss_dict, sims_matrix = self.forward_and_adapt(x)

        return sims_matrix


    def forward_and_adapt(self, modality_query):
        outputs = self.forward_output(modality_query, do_temperature=False)
        loss_dict = {
                'loss_total': 0,
        }
        return loss_dict, outputs
