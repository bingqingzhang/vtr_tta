import torch
import torch.nn as nn
from config.base_config import Config
from modules.transformer import Transformer

class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config
        
        if self.config.huggingface:
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            from model.clip_model import load_clip
            self.clip = load_clip(config.clip_arch)

        config.pooling_type = 'transformer'
        self.pool_frames = Transformer(config)


    def forward(self, data, return_all_frames=False, return_video_only=False, return_text_only=False):
        if return_video_only:
            return self.forward_video(data)
        if return_text_only:
            return self.forward_text(data)
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        
        if self.config.huggingface:
            text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data)
        else:
            text_features = self.clip.encode_text(text_data)
            video_features = self.clip.encode_image(video_data)
   
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        video_features_pooled = self.pool_frames(text_features, video_features)
            
        if return_all_frames:
            return text_features, video_features, video_features_pooled

        return text_features, video_features_pooled
    
    def forward_video(self, data):
        """
        Forward pass for video only, used in test time inference.
        """
        batch_size = data['video'].shape[0]
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        
        if self.config.huggingface:
            video_features = self.clip.get_image_features(video_data)
        else:
            video_features = self.clip.encode_image(video_data)

        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        return video_features

    def forward_text(self, data):
        """
        Forward pass for text only, used in test time inference.
        """
        if self.config.huggingface:
            text_features = self.clip.get_text_features(**data['text'])
        else:
            text_features = self.clip.encode_text(data['text'])

        return text_features