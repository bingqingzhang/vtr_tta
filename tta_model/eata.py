import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import math

from tta_model.param import load_model_and_optimizer, copy_model_and_optimizer
from tta_model.utils import softmax_entropy

from tta_model.tta_base import TTABase
from tta_model.loss_tracker import LossTracker

logger = logging.getLogger(__name__)

class EATA(TTABase):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, tokenizer, optimizer, fishers=None, fisher_alpha=2000.0, steps=1, episodic=False, e_margin=math.log(1000)/2-1, d_margin=0.05, config=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.total_steps = steps
        assert steps > 0, "EATA requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.config = config

        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = e_margin # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = d_margin # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5)

        self.current_model_probs = None # the moving average of probability vector (Eqn. 4)

        self.fishers = fishers # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
        self.fisher_alpha = fisher_alpha # trade-off \beta for two losses (Eqn. 8) 

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
        self.loss_tracker = LossTracker(adapt_method='eata', logger=logger, window_size=w, log_every=e)
    
    
    def forward(self, x, num_iter):
        if self.episodic:
            self.reset()

        for step in range(self.total_steps):
            outputs, num_counts_2, num_counts_1, updated_probs = self.forward_and_adapt_eata(x, step, iter_idx=num_iter)
            self.num_samples_update_2 += num_counts_2
            self.num_samples_update_1 += num_counts_1
            self.reset_model_probs(updated_probs)

        return outputs

    def reset_model_probs(self, probs):
        self.current_model_probs = probs
    
    def forward_and_adapt_eata(self, x, cur_step, iter_idx):        
        with torch.set_grad_enabled(True):
            outputs = self.forward_output(x)
            # adapt
            entropys = softmax_entropy(outputs).sum(1)
            # filter unreliable samples
            filter_ids_1 = torch.where(entropys < self.e_margin)
            
            if filter_ids_1[0].numel() == 0:
                n = entropys.numel()
                k = max(1, int(math.ceil(0.25 * n)))
                _, idx_small = torch.topk(entropys, k, largest=False)
                filter_ids_1 = (idx_small,)
            ids1 = filter_ids_1
            ids2 = torch.where(ids1[0]>-0.1)
            entropys = entropys[filter_ids_1]
            if self.current_model_probs is not None: 
                cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
                filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
                
                if filter_ids_2[0].numel() == 0:
                    n1 = entropys.numel()
                    k1 = max(1, int(math.ceil(0.1 * n1)))
                    _, idx_small_1 = torch.topk(entropys, k1, largest=False)
                    filter_ids_2 = (idx_small_1,)

                entropys = entropys[filter_ids_2]
                ids2 = filter_ids_2
                updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
            else:
                updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))
            
            coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))
            # implementation version 1, compute loss, all samples backward (some unselected are masked)
            entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
            loss = entropys.mean(0)
            if self.fishers is not None:
                ewc_loss = 0
                for name, param in self.model.named_parameters():
                    if name in self.fishers:
                        ewc_loss += self.fisher_alpha * (self.fishers[name][0] * (param - self.fishers[name][1])**2).sum()
                loss += ewc_loss
            loss_dict = {'total_loss': 0}
            # if x[ids1][ids2].size(0) != 0:
            loss.backward()
            self.optimizer.step()
            loss_dict['total_loss'] = loss.item()
            self.optimizer.zero_grad()
        if self.loss_tracker is not None:
            self.loss_tracker.add(
                iter_idx=iter_idx,
                step_idx=cur_step,
                steps_per_iter=self.total_steps,
                loss_dict=loss_dict
            )
        return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs

def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)