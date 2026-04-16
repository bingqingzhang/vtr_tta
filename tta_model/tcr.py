import torch
import logging
import torch.jit
import torch.nn as nn
import torch.nn.functional as F

from tta_model.param import load_model_and_optimizer, copy_model_and_optimizer
from tta_model.utils import softmax_entropy
from tta_model.tta_base import TTABase
from tta_model.loss_tracker import LossTracker


logger = logging.getLogger(__name__)

class TCR(TTABase):
    """TCR adapts a model by entropy minimization during testing.

    Once TCRed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, tokenizer, optimizer, steps=1, episodic=False, config=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.total_steps = steps # steps=3, tta for each step
        assert steps > 0, "TCR requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.config = config

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
            
        w = getattr(config, "loss_window_size", 6)
        e = getattr(config, "loss_log_every", 3)
        self.loss_tracker = LossTracker(adapt_method='tcr', logger=logger, window_size=w, log_every=e)
            
        self.queue_list = []
        self.max_queue_size = config.batch_size
        self.num_update_signal = 10
        self.update_signal = True
        self.device = config.device
        self.con_ratio = config.con_ratio # 0.3
        self.temperature = config.temperature # 0.02
        self.t = config.t # t=0.1
        
    def reset(self):
        super().reset()
        self.queue_list = []
        self.update_signal = True

    def forward(self, modality_query, num_iter):
        if self.episodic:
            self.reset()

        if num_iter >= self.num_update_signal:
            self.update_signal = False

        for step in range(self.total_steps):
            sims_matrix = self.forward_and_adapt(modality_query, step, iter_idx=num_iter)

        return sims_matrix

    def forward_and_adapt(self, modality_query, cur_step, iter_idx):
        with torch.set_grad_enabled(True):
            loss_REM, loss_UNI, loss_EMG, outputs = self.forward_tcr_tta(modality_query, cur_step)
            loss = loss_REM + loss_UNI + loss_EMG
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_dict = {
                'loss_REM': loss_REM.item(),
                'loss_UNI': loss_UNI.item(),
                'loss_EMG': loss_EMG.item(),
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

    def forward_tcr_tta(self, modality_query, cur_step):
        sim_matrix, modality_query_feat, modality_gallery_feat_all = self.forward_output(modality_query, do_temperature=False, return_features=True)
        nearest_neighbors_indices = (sim_matrix).argmax(dim=1)
        modality_gallery_feat = modality_gallery_feat_all[nearest_neighbors_indices]
        if (cur_step == 0 and self.update_signal):
            self.update_queue(modality_query_feat, modality_gallery_feat)
        
        margin, entropy_queue = self.get_current_value()
        outputs = (modality_query_feat @ modality_gallery_feat.t())
        sim_inter = outputs / self.temperature
        
        loss_REM=self.entropy_loss_against_noisy(sim_inter, entropy_queue)
        loss_UNI=self.center_uniform_loss(modality_query_feat, t=self.t)
        target_modality_gap=self.compute_modality_gap(modality_query_feat, modality_gallery_feat)
        loss_EMG=(target_modality_gap-margin)**2
        return loss_REM, loss_UNI, loss_EMG, sim_matrix
    
    def entropy_loss_against_noisy(self, outputs, entropy_queue, eps=1e-3):
        entropys = softmax_entropy(outputs).sum(1)
        entropy_threshold = entropy_queue.max()
        weight = torch.clamp(torch.tensor(1.0, device=self.device) - entropys.clone().detach() / (entropy_threshold + eps), min=0)
        loss = entropys.mul(weight)[entropys <= entropy_threshold]
        return loss.mean(0)

    def center_uniform_loss(self, x, t=0.1):
        center = x.mean(0)
        distances = torch.norm(x - center, dim=1) * t
        loss = (torch.exp(-distances)).mean()
        return loss

    def compute_modality_gap(self, all_image_embeds, all_text_embeds):
        all_image_embeds = F.normalize(all_image_embeds)
        all_text_embeds = F.normalize(all_text_embeds)
        image_embed = all_image_embeds.mean(dim=0)
        text_embed = all_text_embeds.mean(dim=0)
        modality_shift = image_embed - text_embed
        modality_gap = torch.norm(modality_shift, p=2)
        return modality_gap
    
    
    def update_queue(self, modality_1_feat, modality_2_feat):
        with torch.no_grad():
            num_to_select = int(self.con_ratio * modality_1_feat.size(0))
            sample_gap = torch.norm(modality_1_feat - modality_2_feat, p=2, dim=1)
            modality_1_center = modality_1_feat.mean(0)
            modality_2_center=modality_2_feat.mean(0)
            sample_1_diversity=torch.norm(modality_1_feat - modality_1_center, p=2, dim=1)
            sample_2_diversity=torch.norm(modality_2_feat - modality_2_center, p=2, dim=1)
            indictor=2*sample_gap-sample_1_diversity-sample_2_diversity

            entropys=softmax_entropy(modality_1_feat@modality_2_feat.t()/self.temperature).sum(1)

            sorted_indices = torch.argsort(indictor)[:num_to_select]

            for i in sorted_indices:
                indictor_item = indictor[i].detach().item()
                modality_1_feat_item = modality_1_feat[i].detach()
                modality_2_feat_item = modality_2_feat[i].detach()
                entropys_item=entropys[i].detach()
                self.queue_list.append((indictor_item,modality_1_feat_item, modality_2_feat_item,entropys_item))

            if len(self.queue_list) >= self.max_queue_size:
                self.queue_list = sorted(self.queue_list, key=lambda x: x[0], reverse=False)
                self.queue_list = self.queue_list[:self.max_queue_size]
    
    def get_current_value(self):
        with torch.no_grad():
            _, modality_1_embeds, modality_2_embeds, entropys = zip(*self.queue_list)
            feat_1_queue = torch.stack(modality_1_embeds)
            feat_2_queue = torch.stack(modality_2_embeds)
            current_margin=self.compute_modality_gap(feat_1_queue, feat_2_queue)
            entropys_queue=torch.stack(entropys)
            return current_margin, entropys_queue
        
