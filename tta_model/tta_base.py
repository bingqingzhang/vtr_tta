import torch
import logging
import torch.jit
import torch.nn as nn
import torch.nn.functional as F

from tta_model.param import load_model_and_optimizer, copy_model_and_optimizer
from tta_model.utils import softmax_entropy

class TTABase(nn.Module):
    """
    Just for mock, do nothing
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x, num_iter=None):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.model.zero_grad(set_to_none=True)
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad = None
            
    def set_video_features(self, video_features=None):
        self.video_features = video_features
    
    def set_text_features(self, text_features=None):
        self.text_features = text_features
        
    def encode_videos(self, data):
        input_data = {}
        input_data['video'] = data['video'].to(self.device)
        video_features = self.model(input_data, return_video_only=True)
        return video_features
    
    def encode_texts(self, data):
        input_data = {}
        if self.config.base_model == 'languagebind':
            raise NotImplementedError("LanguageBind model is not supported for TTA yet.")
        elif self.config.base_model in ['clip4clip', 'xpool']:
            input_data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
        else:
            raise NotImplementedError(f"Base model {self.config.base_model} is not supported for TTA.")
        if isinstance(input_data['text'], torch.Tensor):
            input_data['text'] = input_data['text'].to(self.device)
        else:
            input_data['text'] = {key: val.to(self.device) for key, val in input_data['text'].items()}
        text_features = self.model(input_data, return_text_only=True)
        return text_features
    
    def forward_output(self, modality_query, do_temperature=True, return_features=False):
        retrieval_type = self.config.retrieval_type
        if retrieval_type == 'v2t':
            text_features = self.text_features
            video_features = self.encode_videos(modality_query)
        elif retrieval_type == 't2v':
            video_features = self.video_features
            text_features = self.encode_texts(modality_query)
        else:
            raise NotImplementedError("Only v2t and t2v retrieval types are implemented in TTA base class.")
        
        vid_pooled_embeds = self.model.pool_frames(text_features, video_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        vid_pooled_embeds = vid_pooled_embeds / vid_pooled_embeds.norm(dim=-1, keepdim=True)
        if self.config.base_model == 'xpool':
            text_features = text_features.unsqueeze(1)
            vid_pooled_embeds = vid_pooled_embeds.permute(1, 2, 0)
            t2v_sims = torch.bmm(text_features, vid_pooled_embeds).squeeze(1)
            text_features = text_features.squeeze(1)
        else:
            vid_pooled_embeds = vid_pooled_embeds.permute(1, 0)
            t2v_sims = torch.mm(text_features, vid_pooled_embeds)
            vid_pooled_embeds = vid_pooled_embeds.permute(1, 0)

        if retrieval_type == 'v2t':
            sim_matrix = t2v_sims.t()
            modality_gallery_feat_all = text_features
            modality_query_feat = video_features.mean(dim=1)
        else:
            sim_matrix = t2v_sims
            modality_gallery_feat_all = video_features.mean(dim=1)
            modality_query_feat = text_features
        if do_temperature:
            sim_matrix = sim_matrix/self.temperature
        if return_features:
            return sim_matrix, modality_query_feat, modality_gallery_feat_all
        return sim_matrix
    
    