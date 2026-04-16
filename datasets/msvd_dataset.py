import os
import logging
import torch
from modules.basic_utils import load_json, read_lines
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture

logger = logging.getLogger(__name__)


class MSVDDataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
        Notes: for test split, we return one video, caption pair for each caption belonging to that video
               so when we run test inference for t2v task we simply average on all these pairs.
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
        
        db_file = 'data/MSVD/captions_msvd.json'
        test_file = 'data/MSVD/test_list.txt'
        train_file = 'data/MSVD/train_list.txt'
        self.vid2caption = load_json(db_file)

        if split_type == 'train':
            self.train_vids = read_lines(train_file) 
            self._construct_all_train_pairs()
        else:
            self.test_vids = read_lines(test_file)
            self._construct_all_test_pairs()


    def __getitem__(self, index):
        if self.split_type == 'train':
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index_train(index)
        else:
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index_test(index)

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

        ret = {
            'video_id': video_id,
            'video': imgs,
            'text': caption
        }

        return ret


    def _get_vidpath_and_caption_by_index_train(self, index):
        vid, caption = self.all_train_pairs[index]
        video_path = os.path.join(self.videos_dir, vid + '.avi')
        return video_path, caption, vid

    def _get_vidpath_and_caption_by_index_test(self, index):
        vid, caption = self.all_test_pairs[index]
        video_path = os.path.join(self.videos_dir, vid + '.avi')
        return video_path, caption, vid

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.all_test_pairs)


    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        for vid in self.train_vids:
            for caption in self.vid2caption[vid]:
                self.all_train_pairs.append([vid, caption])


    def _construct_all_test_pairs(self):
        self.all_test_pairs = []
        for vid in self.test_vids:
            for caption in self.vid2caption[vid]:
                self.all_test_pairs.append([vid, caption])
                
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
            video_path = os.path.join(self.videos_dir, video_id + '.avi')
            imgs, idxs = VideoCapture.load_frames_from_video(video_path, 
                                                             self.config.num_frames, 
                                                             self.config.video_sample_type)
            if self.img_transforms is not None:
                imgs = self.img_transforms(imgs)
            return imgs
        
class MSVDDatasetInferVideoDataset(MSVDDataset):
    """
    Dataset for inference on video-only tasks.
    """
    def __init__(self, config: Config, split_type='test', img_transforms=None):
        super().__init__(config, split_type, img_transforms)

    def _getvidpath_by_index(self, index):
        vid = self.test_vids[index]
        video_path = os.path.join(self.videos_dir, vid + '.avi')
        return video_path, vid
    
    def __len__(self):
        return len(self.test_vids)
    
    def __getitem__(self, index):
        video_path, video_id = self._getvidpath_by_index(index)
        
        if self.use_preprocessed:
            imgs = self._load_preprocessed_video(video_id)
        else:
            imgs, idxs = VideoCapture.load_frames_from_video(video_path, 
                                                             self.config.num_frames, 
                                                             self.config.video_sample_type)
            if self.img_transforms is not None:
                imgs = self.img_transforms(imgs)
        return {
            'video_id': video_id,
            'video': imgs,
        }
        
class MSVDDatasetInferTextDataset(MSVDDataset):
    """
    Dataset for inference on text-only tasks.
    """
    def __init__(self, config: Config, split_type='test'):
        self.text_tta = False
        super().__init__(config, split_type, img_transforms=None)
        self._construct_infer_test_pairs()
        
    def _get_caption_by_index(self, index):
        vid, caption = self.all_test_pairs[index]
        return caption, vid
    
    def _construct_infer_test_pairs(self):
        self.all_test_pairs = []
        for vid in self.test_vids:
            self.all_test_pairs.append([vid, self.vid2caption[vid][0]])
    
    def set_new_text_anno(self, text_file_dir):
        assert self.split_type == 'test', "This method should only be called for test split."
        vid_text_pairs = load_json(text_file_dir)
        assert len(vid_text_pairs) == len(self.all_test_pairs), "Number of video-text pairs must match number of test videos."
        for i in range(len(vid_text_pairs)):
            # Ensure video ID matches
            ori_vid = self.test_vids[i]
            new_vid, new_caption = vid_text_pairs[i]
            assert ori_vid == new_vid, f"Video ID mismatch at index {i}: {ori_vid} != {new_vid}"
        self.text_tta = True
        self.vid_text_pairs = vid_text_pairs
    
    def __len__(self):
        return len(self.test_vids)

    def __getitem__(self, index):
        if self.text_tta:
            return {
                'video_id': self.vid_text_pairs[index][0],
                'text': self.vid_text_pairs[index][1],
            }
        caption, video_id = self._get_caption_by_index(index)
        return {
            'video_id': video_id,
            'text': caption
        }
    
    
    
