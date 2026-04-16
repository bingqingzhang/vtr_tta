import torch
import logging
import torch.jit
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tta_model.param import load_model_and_optimizer, copy_model_and_optimizer
from tta_model.utils import softmax_entropy
from tta_model.tta_base import TTABase
from tta_model.loss_tracker import LossTracker



logger = logging.getLogger(__name__)

class HATVTR(TTABase):

    def __init__(self, model, tokenizer, optimizer, steps=1, episodic=False, config=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.total_steps = steps
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
            
        self.reliable_memory = []
        self.max_queue_size = getattr(config, "max_queue_size", config.batch_size)
        self.device = config.device
        self.con_ratio = config.con_ratio
        self.temperature = config.temperature
        self.t = config.t
        self.hsm = []
        self.select_ratio = 0.1
        
    def reset(self):
        super().reset()
        self.reliable_memory = []
        self.hsm = []

    def forward(self, modality_query, num_iter):
        if self.episodic:
            self.reset()

        for step in range(self.total_steps):
            sims_matrix = self.forward_and_adapt(modality_query, step, iter_idx=num_iter)
        
        return self.adapt_sim_refine(sims_matrix)

    def adapt_sim_refine(self, sim_matrix):
        with torch.no_grad():
            sim_matrix = sim_matrix.cpu()
            batch_size, num_gallery = sim_matrix.shape
            if len(self.hsm) <= 0:
                self.hsm.append(sim_matrix)
                return sim_matrix
            else:
                cur_sims = torch.cat([*self.hsm, sim_matrix], dim=0)
                self.hsm.append(sim_matrix)
                cur_sims = bihub_supp(cur_sims)
                cur_selected_sims = cur_sims[-batch_size:]
                
                num_selected = max(1, int(self.select_ratio * num_gallery))
                if len(self.hsm) > num_selected:
                    self.hsm = self.hsm[-num_selected:]
                return cur_selected_sims
                

    def forward_and_adapt(self, modality_query, cur_step, iter_idx):
        with torch.set_grad_enabled(True):
            loss_NA, loss_MGUNI, loss_MGCM, outputs = self.forward_tcr_tta(modality_query, cur_step)
            loss = loss_NA + loss_MGUNI + loss_MGCM
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_dict = {
                'loss_NA': loss_NA.item(),
                'loss_MGUNI': loss_MGUNI.item(),
                'loss_MGCM': loss_MGCM.item(),
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
    
    @torch.no_grad()
    def hubness_neighbor_select(self, sim_matrix):
        sim_matrix = sim_matrix.cpu()
        if len(self.hsm) == 0:
            return sim_matrix.argmax(dim=1)
        b, g = sim_matrix.shape
        cur_sims = torch.cat([*self.hsm, sim_matrix], dim=0)
        cur_sims = bihub_supp(cur_sims)
        cur_selected_sims = cur_sims[-b:]
        return cur_selected_sims.argmax(dim=1).to(self.device)
    
    def forward_output(self,modality_query):
        modality_query_feat, modality_gallery_feat_all = self.base_forward(modality_query)
        video_features, text_features = self.get_multimodal_features(modality_query_feat, modality_gallery_feat_all)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        B, T, D = video_features.shape
        video_features = video_features.reshape(B*T, -1)
        video_features = video_features / video_features.norm(dim=1, keepdim=True)
        video_features = video_features.reshape(B, T, -1)
        if self.config.base_model == 'xpool':
            vid_pooled_embeds = self.model.pool_frames(text_features, video_features)
            vid_pooled_embeds = vid_pooled_embeds / vid_pooled_embeds.norm(dim=-1, keepdim=True)
            text_features = text_features.unsqueeze(1)
            vid_pooled_embeds = vid_pooled_embeds.permute(1, 2, 0)
            t2v_sims = torch.bmm(text_features, vid_pooled_embeds).squeeze(1)
            text_features = text_features.squeeze(1)
        else:
            video_embeds = video_features.mean(dim=1)
            video_embeds = video_embeds / video_embeds.norm(dim=-1, keepdim=True)
            video_embeds_transposed = video_embeds.permute(1, 0)
            t2v_sims = torch.mm(text_features, video_embeds_transposed)
        return self.combo_features_sims(video_features, text_features, t2v_sims)

    def forward_tcr_tta(self, modality_query, cur_step):
        modality_query_feat, modality_gallery_feat_all, sim_matrix = self.forward_output(modality_query) 
        nearest_neighbors_indices = self.hubness_neighbor_select(sim_matrix)
        modality_gallery_feat = modality_gallery_feat_all[nearest_neighbors_indices]
        if cur_step == 0:
            self.refine_reliable_mem(modality_query_feat, modality_gallery_feat)
        loss_NA = self.entropy_loss_against_noisy(modality_query_feat, modality_gallery_feat)
        loss_MGUNI=self.center_uniform_loss(modality_query_feat, t=self.t)
        loss_MGCM = self.multigrained_cross_modal_loss(modality_query_feat, modality_gallery_feat)
        return loss_NA, loss_MGUNI, loss_MGCM, sim_matrix

    def cross_modal_loss(self, modality_query_feat, modality_gallery_feat):
        queue_feat1, queue_feat2 = self.get_queue_features()
    
        queue_feat1 = F.normalize(queue_feat1)
        queue_feat2 = F.normalize(queue_feat2)
        modality_query_feat = F.normalize(modality_query_feat)
        modality_gallery_feat = F.normalize(modality_gallery_feat)

        queue_mean_gap = torch.norm(queue_feat1.mean(0) - queue_feat2.mean(0), p=2)
        batch_mean_gap = torch.norm(modality_query_feat.mean(0) - modality_gallery_feat.mean(0), p=2)
        loss = (batch_mean_gap - queue_mean_gap)**2
        
        return loss
    
    def multigrained_cross_modal_loss(self, modality_query_feat, modality_gallery_feat):
        queue_feat1, queue_feat2 = self.get_queue_features()
    
        if len(modality_query_feat.shape) == 3:
            video_features = modality_query_feat
            text_features = modality_gallery_feat
            queue_video_features = queue_feat1
            queue_text_features = queue_feat2
        else:
            video_features = modality_gallery_feat
            text_features = modality_query_feat
            queue_text_features = queue_feat1
            queue_video_features = queue_feat2
        B, T, D = video_features.shape
        
        video_global = video_features.mean(dim=1)
        
        video_global = F.normalize(video_global, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        queue_video_features = F.normalize(queue_video_features, dim=-1)
        queue_text_features = F.normalize(queue_text_features, dim=-1)
        
        global_queue_gap = torch.norm(queue_video_features.mean(0) - queue_text_features.mean(0), p=2)
        global_batch_gap = torch.norm(video_global.mean(0) - text_features.mean(0), p=2)
        global_loss = (global_batch_gap - global_queue_gap)**2
        
        video_frames_flat = video_features.view(-1, D)
        text_expanded_flat = text_features.unsqueeze(1).expand(-1, T, -1).reshape(-1, D)
        video_frames_flat = F.normalize(video_frames_flat, dim=-1)
        text_expanded_flat = F.normalize(text_expanded_flat, dim=-1)
        batch_frame_cross_cov = torch.mm(video_frames_flat.t(), text_expanded_flat) / video_frames_flat.size(0)
        
        queue_video_expanded = queue_video_features.unsqueeze(1).expand(-1, T, -1).reshape(-1, D)
        queue_text_expanded = queue_text_features.unsqueeze(1).expand(-1, T, -1).reshape(-1, D)
        queue_frame_cross_cov = torch.mm(queue_video_expanded.t(), queue_text_expanded) / queue_video_expanded.size(0)
        
        frame_cov_loss = F.mse_loss(batch_frame_cross_cov, queue_frame_cross_cov)
        
        return global_loss + frame_cov_loss
    
    def entropy_loss_against_noisy(self, modality_query_feat, modality_gallery_feat, eps=1e-3):
        entropy_queue = self.get_entropy_queue()
        if len(modality_query_feat.shape) == 3:
            modality_query_feat = modality_query_feat.mean(dim=1)
        elif len(modality_gallery_feat.shape) == 3:
            modality_gallery_feat = modality_gallery_feat.mean(dim=1)
        
        outputs = (modality_query_feat @ modality_gallery_feat.t())
        sim_inter = outputs / self.temperature
        entropys = softmax_entropy(sim_inter).sum(1)
        entropy_threshold = entropy_queue.max()
        weight = torch.clamp(torch.tensor(1.0, device=self.device) - entropys.clone().detach() / (entropy_threshold + eps), min=0)
        loss = entropys.mul(weight)[entropys <= entropy_threshold]
        return loss.mean(0)
    
    def cross_frame_uniform_loss(self, x, t=0.1, alpha=0.4, beta=0.3):
        B, T, D = x.shape
        x_pooled = x.mean(dim=1)
        center_global = x_pooled.mean(dim=0)
        distances_global = torch.norm(x_pooled - center_global, dim=1) * t
        loss_inter_video = torch.exp(-distances_global).mean()

        loss_intra_video = 0
        for b in range(B):
            video_frames = x[b]
            video_center = video_frames.mean(dim=0)
            frame_distances = torch.norm(video_frames - video_center, dim=1) * t
            loss_intra_video += torch.exp(-frame_distances).mean()
        loss_intra_video = loss_intra_video / B
        
        total_loss = loss_inter_video + loss_intra_video
        return total_loss    

    def center_uniform_loss(self, x, t=0.1):
        if len(x.shape) == 2:
            center = x.mean(0)
            distances = torch.norm(x - center, dim=1) * t
            loss = (torch.exp(-distances)).mean()
            return loss
        return self.cross_frame_uniform_loss(x, t)

    def compute_modality_gap(self, all_image_embeds, all_text_embeds):
        all_image_embeds = F.normalize(all_image_embeds)
        all_text_embeds = F.normalize(all_text_embeds)
        image_embed = all_image_embeds.mean(dim=0)
        text_embed = all_text_embeds.mean(dim=0)
        modality_shift = image_embed - text_embed
        modality_gap = torch.norm(modality_shift, p=2)
        return modality_gap
                
    def refine_reliable_mem(self, modality_1_feat, modality_2_feat):
        with torch.no_grad():
            if len(modality_1_feat.shape) == 3:
                modality_1_feat = modality_1_feat.mean(dim=1)
            if len(modality_2_feat.shape) == 3:
                modality_2_feat = modality_2_feat.mean(dim=1)
            num_to_select = int(self.con_ratio * modality_1_feat.size(0))      
            modality_1_center = modality_1_feat.mean(0)
            sample_1_diversity=torch.norm(modality_1_feat - modality_1_center, p=2, dim=1)
            cor_diversity = sample_1_diversity
            indicator =  cor_diversity
            entropys=softmax_entropy(modality_1_feat@modality_2_feat.t()/self.temperature).sum(1)
            sorted_indices = torch.argsort(indicator, descending=True)[:num_to_select]
            for i in sorted_indices:
                indictor_item = indicator[i].detach().item()
                modality_1_feat_item = modality_1_feat[i].detach()
                modality_2_feat_item = modality_2_feat[i].detach()
                entropys_item=entropys[i].detach()
                self.reliable_memory.append((indictor_item, modality_1_feat_item, modality_2_feat_item,entropys_item))
            if len(self.reliable_memory) >= self.max_queue_size:
                self.reliable_memory = sorted(self.reliable_memory, key=lambda x: x[0], reverse=True)
                self.reliable_memory = self.reliable_memory[:self.max_queue_size]
    
    def get_entropy_queue(self):
        with torch.no_grad():
            _, _, _, entropys = zip(*self.reliable_memory)
            entropys_queue=torch.stack(entropys)
            return entropys_queue
        
    def get_queue_features(self):
        with torch.no_grad():
            _, modality_1_embeds, modality_2_embeds, entropys = zip(*self.reliable_memory)
            feat_1_queue = torch.stack(modality_1_embeds)
            feat_2_queue = torch.stack(modality_2_embeds)
            return feat_1_queue, feat_2_queue
    
    def base_forward(self, modality_query):
        retrieval_type = self.config.retrieval_type
        if retrieval_type == 'v2t':
            text_features = self.text_features
            video_features = self.encode_videos(modality_query)
            modality_query_feat = video_features
            modality_gallery_feat_all = text_features
        elif retrieval_type == 't2v':
            video_features = self.video_features
            text_features = self.encode_texts(modality_query)
            modality_query_feat = text_features
            modality_gallery_feat_all = video_features
        return modality_query_feat, modality_gallery_feat_all
    
    def get_multimodal_features(self, modality_query_feat, modality_gallery_feat_all):
        retrieval_type = self.config.retrieval_type
        if retrieval_type == 'v2t':
            video_features = modality_query_feat
            text_features = modality_gallery_feat_all
        elif retrieval_type == 't2v':
            text_features = modality_query_feat
            video_features = modality_gallery_feat_all
        return video_features, text_features
    
    def combo_features_sims(self, video_features, text_features, sims):
        retrieval_type = self.config.retrieval_type
        if retrieval_type == 'v2t':
            sim_matrix = sims.t()
            modality_query_feat = video_features
            modality_gallery_feat_all = text_features
        elif retrieval_type == 't2v':
            sim_matrix = sims
            modality_query_feat = text_features
            modality_gallery_feat_all = video_features
        else:
            raise NotImplementedError("Only v2t and t2v retrieval types are implemented in TTA base class.")
        return modality_query_feat, modality_gallery_feat_all, sim_matrix
    
def calculate_cross_modal_weights(sims, amplified_weights=100.0, dim=0):
    y = sims * amplified_weights
    y = y - torch.max(y, dim=dim, keepdim=True).values
    y = torch.exp(y)
    y = y / torch.sum(y, dim=dim, keepdim=True)
    return y

def bihub_supp(all_sims, alpha=100.0, beta=10.0, m=0.5):
    col_weight = calculate_cross_modal_weights(all_sims, amplified_weights=alpha, dim=0)
    row_weight = calculate_cross_modal_weights(all_sims, amplified_weights=beta, dim=1)
    sim_col = all_sims * col_weight
    sim_row = all_sims * row_weight
    final_sim = m * sim_col + (1-m) * sim_row
    return final_sim
    
