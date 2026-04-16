import os
import torch
import argparse
from tqdm import tqdm
from config.all_config import AllConfig
from datasets.model_transforms import init_transform_dict
from datasets.video_capture import VideoCapture


class VideoPreprocessor:
    def __init__(self, config, output_dir=None):
        self.config = config
        self.img_transforms = init_transform_dict(config.input_res)
        
        # Use user specified output directory or default
        assert config.preprocess_dir is not None, "Please specify preprocessed_dir in config"
        self.preprocess_dir = config.preprocess_dir
        
        os.makedirs(self.preprocess_dir, exist_ok=True)
        
        # Create separate directories for train and test
        self.train_dir = self.preprocess_dir
        self.test_dir = self.preprocess_dir
        
    def get_unique_videos(self, split_type):
        """Extract unique video IDs from dataset"""
        print(f"Extracting unique videos for {split_type} split...")
        
        # Create dataset to get video IDs
        if self.config.dataset_name == "MSRVTT":
            from datasets.msrvtt_dataset import MSRVTTDataset
            dataset = MSRVTTDataset(self.config, split_type, None)
            
            if split_type == 'train':
                # Extract unique video IDs from all_train_pairs
                video_ids = set()
                for vid, caption in dataset.all_train_pairs:
                    video_ids.add(vid)
                return list(video_ids)
            else:
                # Extract unique video IDs from test_df
                return dataset.test_df['video_id'].unique().tolist()
                
        elif self.config.dataset_name == "MSVD":
            from datasets.msvd_dataset import MSVDDataset
            dataset = MSVDDataset(self.config, split_type, None)
            
            if split_type == 'train':
                return dataset.train_vids
            else:
                return dataset.test_vids
                
        elif self.config.dataset_name == "LSMDC":
            from datasets.lsmdc_dataset import LSMDCDataset
            dataset = LSMDCDataset(self.config, split_type, None)
            
            # Extract unique clip IDs
            return list(dataset.clip2caption.keys())

        elif self.config.dataset_name == "ActivityNet":
            from datasets.activitynet_dataset import ActivityNetDataset
            dataset = ActivityNetDataset(self.config, split_type, None)
            
            # Extract unique video IDs
            return [anno['clip_id'] for anno in dataset.anno]
            
        else:
            raise NotImplementedError(f"Dataset {self.config.dataset_name} not supported")
    
    def get_video_path(self, video_id):
        """Get video file path based on dataset and video ID"""
        if self.config.dataset_name == "MSRVTT":
            return os.path.join(self.config.videos_dir, video_id + '.mp4')
        elif self.config.dataset_name == "MSVD":
            return os.path.join(self.config.videos_dir, video_id + '.avi')
        elif self.config.dataset_name == "LSMDC":
            clip_prefix = video_id.split('.')[0][:-3]
            return os.path.join(self.config.videos_dir, clip_prefix, video_id + '.avi')
        elif self.config.dataset_name == "ActivityNet":
            return os.path.join(self.config.videos_dir, video_id + '.mp4')
        else:
            raise NotImplementedError(f"Dataset {self.config.dataset_name} not supported")
        
    def preprocess_split(self, split_type):
        """Preprocess videos for a specific split (train/test)"""
        print(f"Preprocessing {split_type} split for {self.config.dataset_name}...")
        
        # Get the appropriate transforms
        if split_type == 'train':
            transforms = self.img_transforms['clip_train']
            output_dir = self.train_dir
        else:
            transforms = self.img_transforms['clip_test']
            output_dir = self.test_dir
            
        # Get unique video IDs
        video_ids = self.get_unique_videos(split_type)
        print(f"Found {len(video_ids)} unique videos in {split_type} split")
        
        # Process each unique video
        processed_count = 0
        error_count = 0
        skipped_count = 0
        
        for video_id in tqdm(video_ids, desc=f"Processing {split_type} videos"):
            try:
                # Create safe filename
                safe_video_id = video_id.replace('/', '_').replace('\\', '_')
                output_path = os.path.join(output_dir, f"{safe_video_id}.pt")
                
                # Skip if already processed
                if os.path.exists(output_path):
                    skipped_count += 1
                    continue
                
                # Get video path
                video_path = self.get_video_path(video_id)
                
                # Check if video file exists
                if not os.path.exists(video_path):
                    print(f"Warning: Video file not found: {video_path}")
                    error_count += 1
                    continue
                
                # Load video frames
                imgs, idxs = VideoCapture.load_frames_from_video(
                    video_path, 
                    self.config.num_frames, 
                    self.config.video_sample_type
                )
                
                # Apply transforms
                if transforms is not None:
                    imgs = transforms(imgs)
                
                # Save preprocessed video tensor
                torch.save({
                    'video': imgs,
                    'video_id': video_id,
                    'frame_indices': idxs,
                    'original_path': video_path
                }, output_path)
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {video_id}: {e}")
                error_count += 1
                continue
        
        print(f"Completed {split_type}: {processed_count} processed, {skipped_count} skipped, {error_count} errors")
        
    def preprocess_all(self):
        """Preprocess both train and test splits"""
        self.preprocess_split('train')
        self.preprocess_split('test')


def main():
    # Parse arguments
    config = AllConfig()
    args = config.parse_args()
    
    # Override some settings for preprocessing
    parser = argparse.ArgumentParser(description='Video Preprocessing')
    parser.add_argument('--split', type=str, choices=['train', 'test', 'all'], default='all',
                       help='Which split to preprocess')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for preprocessed videos (default: use config.output_dir/preprocessed_videos)')
    preprocess_args, _ = parser.parse_known_args()
    
    # Create preprocessor
    preprocessor = VideoPreprocessor(args, preprocess_args.output_dir)
    
    # Run preprocessing
    if preprocess_args.split == 'all':
        preprocessor.preprocess_all()
    else:
        preprocessor.preprocess_split(preprocess_args.split)
    
    print("Preprocessing completed!")
    print(f"Preprocessed videos saved to: {preprocessor.preprocess_dir}")


if __name__ == "__main__":
    main()
