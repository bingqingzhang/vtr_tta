import os
import cv2
import sys
import logging
import torch
import random
import itertools
import numpy as np
import ujson as json
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture

logger = logging.getLogger(__name__)

class ActivityNetDataset(Dataset):
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
        
        train_file = 'data/ActivityNet/train_anno.jsonl'
        val_file = 'data/ActivityNet/val1_anno.jsonl'
        self.anno = []
        if self.split_type == 'train':
            with open(train_file, 'r') as f:
                for line in f:
                    self.anno.append(json.loads(line))
        else:
            with open(val_file, 'r') as f:
                for line in f:
                    self.anno.append(json.loads(line))
                    
    
    def __getitem__(self, index):
        video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
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
        return len(self.anno)
        
    
    def _get_vidpath_and_caption_by_index(self, index):
        anno = self.anno[index]
        video_path = os.path.join(self.videos_dir, anno['clip_id'] + '.mp4')
        caption = anno['text']
        video_id = anno['clip_id']
        return video_path, caption, video_id

    def set_preprocess_dir(self, preprocess_dir):
        self.preprocess_dir = preprocess_dir

    def _load_preprocessed_video(self, video_id):
        preprocessed_path = os.path.join(self.preprocess_dir, f"{video_id}.pt")
        if os.path.exists(preprocessed_path):
            # Load preprocessed video data
            data = torch.load(preprocessed_path, map_location='cpu')
            return data['video']
        else:
            logger.warning(f"Preprocessed video {video_id} not found in {self.preprocess_dir}. Falling back to raw video loading.")
            video_path = os.path.join(self.videos_dir, video_id + '.mp4')
            imgs, idxs = VideoCapture.load_frames_from_video(video_path, 
                                                             self.config.num_frames, 
                                                             self.config.video_sample_type)
            if self.img_transforms is not None:
                imgs = self.img_transforms(imgs)
            return imgs

class ActivityNetInferVideoDataset(ActivityNetDataset):
    def __init__(self, config: Config, split_type='test', img_transforms=None):
        super().__init__(config, split_type, img_transforms)
        assert split_type not in ['train'], "ActivityNetInferVideoDataset should not be used for training."
        
    def _get_vidpath_by_index(self, index):
        anno = self.anno[index]
        video_path = os.path.join(self.videos_dir, anno['clip_id'] + '.mp4')
        video_id = anno['clip_id']
        return video_path, video_id
    
    def __getitem__(self, index):
        video_path, video_id = self._get_vidpath_by_index(index)
        
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
        }

class ActivityNetInferTextDataset(ActivityNetDataset):
    def __init__(self, config: Config, split_type='test'):
        super().__init__(config, split_type)
        self.text_tta = False
        assert split_type not in ['train'], "ActivityNetInferTextDataset should not be used for training."
        
    def _get_caption_by_index(self, index):
        anno = self.anno[index]
        caption = anno['text']
        video_id = anno['clip_id']
        return caption, video_id

    def set_new_text_anno(self, text_file_dir):
        assert self.split_type == 'test', "This method should only be called for test split."
        vid_text_pairs = load_json(text_file_dir)
        assert len(vid_text_pairs) == len(self.anno), "Number of video-text pairs must match the number of annotations."
        for i, cur_anno in enumerate(self.anno):
            ori_vid = cur_anno['clip_id']
            new_vid, new_caption = vid_text_pairs[i]
            assert ori_vid == new_vid, f"Video ID mismatch at index {i}: {ori_vid} != {new_vid}"
        self.text_tta = True
        self.vid_text_pairs = vid_text_pairs

    def __getitem__(self, index):
        if self.text_tta:
            return {
                'video_id': self.vid_text_pairs[index][0],
                'text': self.vid_text_pairs[index][1],
            }
        caption, video_id = self._get_caption_by_index(index)
        
        return {
            'video_id': video_id,
            'text': caption,
        }