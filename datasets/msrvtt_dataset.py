import os
import cv2
import sys
import logging
import torch
import random
import itertools
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture

logger = logging.getLogger(__name__)


class MSRVTTDataset(Dataset):
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
        
        db_file = 'data/MSRVTT/MSRVTT_data.json'
        test_csv = 'data/MSRVTT/MSRVTT_JSFUSION_test.csv'

        if config.msrvtt_train_file == '7k':
            train_csv = 'data/MSRVTT/MSRVTT_train.7k.csv'
        else:
            train_csv = 'data/MSRVTT/MSRVTT_train.9k.csv'

        self.db = load_json(db_file)
        if split_type == 'train':
            train_df = pd.read_csv(train_csv)
            self.train_vids = train_df['video_id'].unique()
            self._compute_vid2caption()
            self._construct_all_train_pairs()
        else:
            self.test_df = pd.read_csv(test_csv)

            
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
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.test_df)


    def _get_vidpath_and_caption_by_index(self, index):
        # returns video path and caption as string
        if self.split_type == 'train':
            vid, caption = self.all_train_pairs[index]
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
        else:
            vid = self.test_df.iloc[index].video_id
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            caption = self.test_df.iloc[index].sentence

        return video_path, caption, vid

    
    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        if self.split_type == 'train':
            for vid in self.train_vids:
                for caption in self.vid2caption[vid]:
                    self.all_train_pairs.append([vid, caption])

            
    def _compute_vid2caption(self):
        self.vid2caption = defaultdict(list)
        for annotation in self.db['sentences']:
            caption = annotation['caption']
            vid = annotation['video_id']
            self.vid2caption[vid].append(caption)

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
            video_path = os.path.join(self.videos_dir, video_id + '.mp4')
            imgs, idxs = VideoCapture.load_frames_from_video(video_path, 
                                                             self.config.num_frames, 
                                                             self.config.video_sample_type)
            if self.img_transforms is not None:
                imgs = self.img_transforms(imgs)
            return imgs
        
class MSRVTTInferVideoDataset(MSRVTTDataset):
    """
        For inference, we only need to load videos without captions.
        This class is used for
        inference purposes where we load videos without any text.
    """
    def __init__(self, config: Config, split_type='test', img_transforms=None):
        assert split_type not in ['train'], "MSRVTTInferVideoDataset should not be used for training."
        super().__init__(config, split_type, img_transforms)
        
    def _getvidpath_by_index(self, index):
        vid = self.test_df.iloc[index].video_id
        video_path = os.path.join(self.videos_dir, vid + '.mp4')
        return video_path, vid

    def __getitem__(self, index):
        video_path, video_id = self._getvidpath_by_index(index)
        
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

    def __len__(self):
        return len(self.test_df)

class MSRVTTInferTextDataset(MSRVTTDataset):
    """
        For inference, we only need to load text without videos.
        This class is used for inference purposes where we load text without any video.
    """
    def __init__(self, config: Config, split_type='test'):
        assert split_type not in ['train'], "MSRVTTInferTextDataset should not be used for training."
        self.text_tta = False
        super().__init__(config, split_type)
    
    def _get_caption_by_index(self, index):
        caption = self.test_df.iloc[index].sentence
        vid = self.test_df.iloc[index].video_id
        return caption, vid

    def set_new_text_anno(self, text_file_dir):
        assert self.split_type == 'test', "This method should only be called for test split."
        vid_text_pairs = load_json(text_file_dir)
        # verify the loaded annotations
        assert len(vid_text_pairs) == len(self.test_df), "Number of video-text pairs does not match the number of test samples."
        for i in range(len(self.test_df)):
            ori_caption = self.test_df.iloc[i].sentence
            ori_vid = self.test_df.iloc[i].video_id
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
        caption, vid = self._get_caption_by_index(index)
        return {
            'video_id': vid,
            'text': caption,
        }
    
    def __len__(self):
        return len(self.test_df)
