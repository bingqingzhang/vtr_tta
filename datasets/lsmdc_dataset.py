import os
import cv2
import sys
import logging
import torch
import random
import itertools
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture

logger = logging.getLogger(__name__)


class LSMDCDataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config: Config, split_type = 'train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        self.use_preprocessed = config.use_preprocessed
        
        # Set up preprocessed video directory
        if self.use_preprocessed:
            if config.preprocess_dir is None:
                logger.warning("use_preprocessed=True but preprocess_dir is not specified. Falling back to raw video loading.")
                self.use_preprocessed = False
            else:
                self.preprocess_dir = config.preprocess_dir
                assert os.path.exists(self.preprocess_dir), f"Preprocessed directory {self.preprocess_dir} not found."
        
        self.clip2caption = {}
        if split_type == 'train':
            train_file = 'data/LSMDC/LSMDC16_annos_training.csv'
            self._compute_clip2caption(train_file)
               
        else:
            test_file = 'data/LSMDC/LSMDC16_challenge_1000_publictect.csv'
            self._compute_clip2caption(test_file)
  

    def __getitem__(self, index):
        video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
        
        # Load video frames - either from preprocessed file or raw video
        if self.use_preprocessed:
            imgs = self._load_preprocessed_video(video_id)
        else:
            imgs, idxs = VideoCapture.load_frames_from_video(video_path, 
                                                             self.config.num_frames, 
                                                             self.config.video_sample_type)
            # process images of video
            if self.img_transforms is not None:
                imgs = self.img_transforms(imgs)

        return {
            'video_id': video_id,
            'video': imgs,
            'text': caption,
        }

    
    def __len__(self):
        return len(self.clip2caption)


    def _get_vidpath_and_caption_by_index(self, index):
        # returns video path and caption as string
        clip_id = list(self.clip2caption.keys())[index]
        caption = self.clip2caption[clip_id]
        clip_prefix = clip_id.split('.')[0][:-3]
        video_path = os.path.join(self.videos_dir, clip_prefix, clip_id + '.avi')

        return video_path, caption, clip_id

            
    def _compute_clip2caption(self, csv_file):
        with open(csv_file, 'r') as fp:
            for line in fp:
                line = line.strip()
                line_split = line.split("\t")
                assert len(line_split) == 6
                clip_id, _, _, _, _, caption = line_split
                if clip_id == '1012_Unbreakable_00.05.16.065-00.05.21.941':
                    continue
                self.clip2caption[clip_id] = caption
                
    def set_preprocess_dir(self, preprocess_dir):
        self.preprocess_dir = preprocess_dir

    def _load_preprocessed_video(self, video_id):
        """Load preprocessed video frames from .pt file"""
        # Create safe filename
        safe_video_id = video_id.replace('/', '_').replace('\\', '_')
        preprocessed_path = os.path.join(self.preprocess_dir, f"{safe_video_id}.pt")
        
        if os.path.exists(preprocessed_path):
            # Load preprocessed video data
            data = torch.load(preprocessed_path, map_location='cpu')
            return data['video']
        else:
            # Fallback to original loading if preprocessed file not found
            logger.error(f"Preprocessed file {preprocessed_path} not found. Loading from raw video.")
            clip_prefix = video_id.split('.')[0][:-3]
            video_path = os.path.join(self.videos_dir, clip_prefix, video_id + '.avi')
            imgs, idxs = VideoCapture.load_frames_from_video(video_path, 
                                                             self.config.num_frames, 
                                                             self.config.video_sample_type)
            if self.img_transforms is not None:
                imgs = self.img_transforms(imgs)
            return imgs

class LSMDCInferVideoDataset(LSMDCDataset):
    """
        Dataset for inference on LSMDC videos.
        Uses the same structure as LSMDCDataset but does not require captions.
    """
    def __init__(self, config: Config, split_type='test', img_transforms=None):
        assert split_type == 'test', "LSMDCInferVideoDataset only supports 'test' split"
        super().__init__(config, split_type, img_transforms)
        
    def _get_vidpath_by_index(self, index):
        clip_id = list(self.clip2caption.keys())[index]
        clip_prefix = clip_id.split('.')[0][:-3]
        video_path = os.path.join(self.videos_dir, clip_prefix, clip_id + '.avi')
        return video_path, clip_id
    
    def __getitem__(self, index):
        video_path, video_id = self._get_vidpath_by_index(index)
        
        if self.use_preprocessed:
            imgs = self._load_preprocessed_video(video_id)
        else:
            imgs, idxs = VideoCapture.load_frames_from_video(video_path, 
                                                             self.config.num_frames, 
                                                             self.config.video_sample_type)
            # process images of video
            if self.img_transforms is not None:
                imgs = self.img_transforms(imgs)
        
        return {
            'video_id': video_id,
            'video': imgs,
        }
    
    def __len__(self):
        return len(list(self.clip2caption.keys()))
    
class LSMDCInferTextDataset(LSMDCDataset):
    """
        Dataset for inference on LSMDC text.
        Uses the same structure as LSMDCDataset but does not require video loading.
    """
    def __init__(self, config: Config, split_type='test'):
        assert split_type == 'test', "LSMDCInferTextDataset only supports 'test' split"
        self.text_tta = False
        super().__init__(config, split_type, img_transforms=None)
    
    def set_new_text_anno(self, text_file_dir):
        assert self.split_type == 'test', "This method should only be called for test split."
        vid_text_pairs = load_json(text_file_dir)
        assert len(vid_text_pairs) == len(self.clip2caption), "Length of new text annotations must match existing captions."
        for i in range(len(vid_text_pairs)):
            ori_vid = list(self.clip2caption.keys())[i]
            new_vid, new_caption = vid_text_pairs[i]
            assert ori_vid == new_vid, f"Video ID mismatch at index {i}: {ori_vid} != {new_vid}"
        self.text_tta = True
        self.vid_text_pairs = vid_text_pairs

    def _get_caption_by_index(self, index):
        clip_id = list(self.clip2caption.keys())[index]
        caption = self.clip2caption[clip_id]
        return caption, clip_id
    
    def __getitem__(self, index):
        if self.text_tta:
            return {
                'video_id': self.vid_text_pairs[index][0],
                'text': self.vid_text_pairs[index][1],
            }
        caption, clip_id = self._get_caption_by_index(index)
        return {
            'video_id': clip_id,
            'text': caption
        }
    
    def __len__(self):
        return len(list(self.clip2caption.keys()))