import os
import torch
import argparse
import math
import json
import cv2
import numpy as np
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen2_5_VLForConditionalGeneration, Owlv2ForObjectDetection
import spacy
from typing import List, Tuple, Dict, Any
from datasets.video_capture import get_frame_indices
from qwen_vl_utils import process_vision_info

def get_video_ids(config, split_type):
    """Extract unique video IDs from dataset"""
    print(f"Extracting unique videos for {split_type} split...")

    if config.dataset_name == "MSRVTT":
        from datasets.msrvtt_dataset import MSRVTTDataset
        dataset = MSRVTTDataset(config, split_type, None)
        
        if split_type == 'train':
            # Extract unique video IDs from all_train_pairs
            video_ids = set(vid for vid, _ in dataset.all_train_pairs)
            return list(video_ids)
        else:
            # Extract unique video IDs from test_df
            return dataset.test_df['video_id'].unique().tolist()
            
    elif config.dataset_name == "MSVD":
        from datasets.msvd_dataset import MSVDDataset
        dataset = MSVDDataset(config, split_type, None)
        
        if split_type == 'train':
            return dataset.train_vids
        else:
            return dataset.test_vids
            
    elif config.dataset_name == "LSMDC":
        from datasets.lsmdc_dataset import LSMDCDataset
        dataset = LSMDCDataset(config, split_type, None)
        
        # Extract unique clip IDs
        return list(dataset.clip2caption.keys())

    elif config.dataset_name == "ActivityNet":
        from datasets.activitynet_dataset import ActivityNetDataset
        dataset = ActivityNetDataset(config, split_type, None)
        vids = []
        for i in range(len(dataset.anno)):
            vids.append(dataset.anno[i]['clip_id'])
        vids = list(set(vids))
        return vids
    
    elif config.dataset_name == "Didemo":
        from datasets.didemo_dataset import DidemoDataset
        dataset = DidemoDataset(config, split_type, None)
        vids = []
        for i in range(len(dataset.anno)):
            vids.append(dataset.anno[i]['clip_id'])
        vids = list(set(vids))
        return vids

    else:
        raise ValueError(f"Unsupported dataset: {config.dataset_name}")
    
def get_video_path(dataset_name, video_id, videos_dir):
    if dataset_name == "MSRVTT":
        return os.path.join(videos_dir, video_id + '.mp4')
    elif dataset_name == "MSVD":
        return os.path.join(videos_dir, video_id + '.avi')
    elif dataset_name == "LSMDC":
        clip_prefix = video_id.split('.')[0][:-3]
        return os.path.join(videos_dir, clip_prefix, video_id + '.avi')
    elif dataset_name == "ActivityNet":
        return os.path.join(videos_dir, video_id + '.mp4')
    elif dataset_name == "Didemo":
        return os.path.join(videos_dir, video_id + '.mp4')
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
def read_video_by_frames(video_path, num_frames=12):
    frame_data = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    acc_frames = min(num_frames, total_frames)
    frame_indices = get_frame_indices(acc_frames, total_frames)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {idx} from video {video_path}.")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)
        frame_data.append({'frame_index': int(idx), 'image': pil_image})
    
    while len(frame_data) < num_frames:
        last_frame = frame_data[-1]
        frame_data.append({
            'frame_index': last_frame['frame_index'],
            'image': last_frame['image'].copy()
        })
    
    cap.release()
    return frame_data
    
class MainObjectIdentifier:
    def __init__(self, device='cuda:0'):
        """
        Initialize models and processors.
        """
        self.device = device
        print("Loading models...")
        # --- NEW: Load Qwen2.5-VL for captioning ---
        self.caption_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        self.caption_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        # Load open-set object detection model (OWLv2)
        self.detection_processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.detection_model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        ).to(self.device)
        
        # Load spaCy for keyword extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy 'en_core_web_sm' model not found. Downloading...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        print("Models loaded successfully.")
    
    def _generate_caption_and_keywords(self, image: Image.Image) -> Tuple[str, List[str]]:
        """
        Generate a caption for the image and extract nouns as keywords.
        """
        prompt = "Describe the main objects in this image in a short sentence."
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.caption_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.caption_model.device)

        with torch.no_grad():
            output_ids = self.caption_model.generate(**inputs, max_new_tokens=64)
        
        # Decode only the newly generated tokens
        input_token_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[:, input_token_len:]
        generated_text = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # --- Keyword extraction logic remains the same ---
        doc = self.nlp(generated_text)
        keywords = list(set([chunk.root.text.lower() for chunk in doc.noun_chunks]))
        
        if not keywords:
            keywords = list(set([token.lemma_.lower() for token in doc if token.pos_ == "NOUN"]))

        return generated_text, keywords
    
    def _generate_keywords(self, caption):        
        generated_text = caption

        doc = self.nlp(generated_text)
        keywords = list(set([chunk.root.text.lower() for chunk in doc.noun_chunks]))
        
        if not keywords:
            keywords = list(set([token.lemma_.lower() for token in doc if token.pos_ == "NOUN"]))

        return generated_text, keywords
    
    def _get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Extracts a visual embedding for a given image patch."""
        # Use the processor suitable for the vision model
        inputs = self.detection_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Get the features from the vision model component of OWL-ViT
            embedding = self.detection_model.owlv2.vision_model(**inputs).pooler_output
        # Normalize the embedding for cosine similarity
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy().squeeze()
    
    def _generate_video_caption(self, video_path: str) -> str:
        prompt = "Generate a concise and accurate one-sentence caption for this video, focusing on the main objects and their actions. The caption should be around 10-15 words."
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": 1, "max_frames": 16},
                {"type": "text", "text": prompt}
            ],
        }]
        
        tpl = self.caption_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        imgs, vids, vkw = process_vision_info(messages, return_video_kwargs=True)

        num_video_frames = 16
        inputs = self.caption_processor(
            text=[tpl],
            images=imgs,
            videos=vids,
            num_frames=num_video_frames,
            return_tensors="pt", 
            **vkw
        ).to(self.device)

        generation_kwargs = {
            "max_new_tokens": 32,
            "do_sample": False,
            "pad_token_id": self.caption_processor.tokenizer.eos_token_id,
            "temperature": None
        }
        with torch.no_grad():
            out_ids = self.caption_model.generate(**inputs, **generation_kwargs)
        
        generated_ids = out_ids[:, inputs.input_ids.shape[1]:]
        caption = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return caption
    
    def _detect_objects_in_frame(self, image: Image.Image, keywords: List[str], score_threshold=0.1) -> List[Dict[str, Any]]:
        """
        Detect objects for the given keywords in a single frame and extract their visual embeddings.
        """
        if not keywords:
            return []
            
        texts = [keywords]
        inputs = self.detection_processor(
            text=texts, 
            images=image, 
            return_tensors="pt", 
            padding="longest", 
            truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.detection_model(**inputs)
            
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.detection_processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=score_threshold
        )
        
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        # Create a list of all detected objects first
        all_detected_objects = []
        for box, score, label_idx in zip(boxes, scores, labels):
            all_detected_objects.append({
                'label': keywords[label_idx],
                'score': score.item(),
                'box_tensor': box, # Keep tensor for easy cropping
                'box': [round(i, 2) for i in box.tolist()]
            })

        # Sort by detection score in descending order
        all_detected_objects.sort(key=lambda x: x['score'], reverse=True)

        # Process only the top 3 objects
        top_objects = []
        for obj in all_detected_objects[:3]:
            img_width, img_height = image.size
            box_coords = obj['box']
            box_width = box_coords[2] - box_coords[0]
            box_height = box_coords[3] - box_coords[1]
            area_ratio = (box_width * box_height) / (img_width * img_height)
            
            xmin, ymin, xmax, ymax = [int(v) for v in obj['box_tensor'].tolist()]
            cropped_image = image.crop((xmin, ymin, xmax, ymax))
            
            if cropped_image.size[0] == 0 or cropped_image.size[1] == 0:
                continue
            
            embedding = self._get_image_embedding(cropped_image)

            top_objects.append({
                'label': obj['label'],
                'score': round(obj['score'], 3),
                'box': obj['box'],
                'area_ratio': round(area_ratio, 4),
                'embedding': embedding
            })            
        return top_objects
    
    def _associate_objects(self, all_objects: List[Dict[str, Any]], similarity_threshold=0.85) -> List[Dict[str, Any]]:
        """
        [TRACKING LOGIC]
        Groups a flat list of objects into tracks based on embedding similarity.
        """
        tracks = []
        # Create a copy to modify
        unassigned_objects = list(all_objects)
        
        while unassigned_objects:
            # Start a new track with the first available object as the seed
            seed_obj = unassigned_objects.pop(0)
            seed_frame_index = seed_obj['frame_index']
            new_track = {
                'track_id': len(tracks) + 1,
                'appearances': [seed_obj]
            }
            
            # Find all other objects that match this seed
            remaining_objects = []
            for other_obj in unassigned_objects:
                other_obj_frame_index = other_obj['frame_index']
                if seed_frame_index == other_obj_frame_index:
                    continue  # Skip objects from the same frame
                similarity = np.dot(seed_obj['embedding'], other_obj['embedding'])
                if similarity > similarity_threshold:
                    new_track['appearances'].append(other_obj)
                else:
                    remaining_objects.append(other_obj)
            
            unassigned_objects = remaining_objects
            tracks.append(new_track)
            
        return tracks

    def _calculate_and_rank_objects_for_each_frame(self, tracks: List[Dict[str, Any]], num_total_frames: int, all_frame_indexes: List[int]) -> Dict[int, List[Dict[str, Any]]]:
        """
        [RANKING & OUTPUT LOGIC]
        Calculates a final score for each object instance and formats the output per frame.
        """
        # First, create a map from an object's unique ID to its track's persistence score
        obj_id_to_persistence = {}
        for track in tracks:
            persistence_score = len(track['appearances']) / num_total_frames
            for appearance in track['appearances']:
                # Use object's id() as a unique key for this run
                obj_id_to_persistence[id(appearance)] = persistence_score

        # Prepare the final output structure, with keys for each sampled frame
        # final_ranked_frames = {i: [] for i in range(num_total_frames)}
        final_ranked_frames = {frame_index: [] for frame_index in all_frame_indexes}

        # Iterate through all tracks and appearances to calculate final scores
        for track in tracks:
            for obj in track['appearances']:
                persistence = obj_id_to_persistence[id(obj)]
                size_score = obj['area_ratio']
                confidence_score = obj['score']
                
                # Calculate the final rank score for this specific object instance
                rank_score = (0.5 * persistence) + (0.3 * size_score) + (0.2 * confidence_score)
                
                frame_index = obj['frame_index']
                
                # Add the final formatted object to the corresponding frame's list
                final_ranked_frames[frame_index].append({
                    'frame_index': frame_index,
                    'label': obj['label'],
                    'box': obj['box'],
                    'rank_score': round(rank_score, 4)
                })

        # Sort objects within each frame by their rank_score
        for frame_idx, objects in final_ranked_frames.items():
            objects.sort(key=lambda x: x['rank_score'], reverse=True)
            
        return final_ranked_frames
    
    def process_video(self, video_path: str, num_frames: int = 12, cur_video_caption=None) -> Dict[str, Any]:
        """
        The complete pipeline for processing a single video.
        """
        # 1. Read video frames
        sampled_frames_data = read_video_by_frames(video_path, num_frames)
        
        assert cur_video_caption is not None, "Current video caption must be provided."
        _, keywords = self._generate_keywords(cur_video_caption)
        
        # video_caption = self._generate_video_caption(video_path)
        
        # 2. Generate caption and detect objects for each frame
        all_detected_objects = []
        all_frame_indexes = []
        for i, frame_data in enumerate(sampled_frames_data):
            image = frame_data['image']
            # caption, keywords = self._generate_caption_and_keywords(image)
            objects = self._detect_objects_in_frame(image, keywords)
            frame_index = frame_data['frame_index']
            all_frame_indexes.append(frame_index)

            for obj in objects:
                # Add frame index for tracking
                obj['frame_index'] = frame_index
                all_detected_objects.append(obj)
            
        # 2. Associate all detected objects into tracks
        tracks = self._associate_objects(all_detected_objects)
        
        # 3. Calculate scores and structure the output per frame
        ranked_results_by_frame = self._calculate_and_rank_objects_for_each_frame(tracks, len(sampled_frames_data), all_frame_indexes)

        return {
            "video_path": video_path,
            "ranked_objects_per_frame": ranked_results_by_frame
        }
        
def main():    
    parser = argparse.ArgumentParser(description='Noisy Video Dataset Preprocessor')
    parser.add_argument('--dataset_name', type=str, default='MSRVTT', help="Dataset name")
    parser.add_argument('--videos_dir', type=str, default='data/MSRVTT/vids', help="Location of videos")
    parser.add_argument('--msrvtt_train_file', type=str, default='9k')
    parser.add_argument('--device', type=str, default='cuda:1')

    # irrelevant arguments
    parser.add_argument('--num_frames', type=int, default=12)
    parser.add_argument('--video_sample_type', default='uniform', help="'rand'/'uniform'")
    parser.add_argument('--input_res', type=int, default=224)
    parser.add_argument('--preprocess_dir', type=str, default=None, help="Directory containing preprocessed video files")
    parser.add_argument('--use_preprocessed', action='store_true', default=False, help="Use preprocessed videos if available")
    config = parser.parse_args()
    
    # Initialize the main object identifier
    identifier = MainObjectIdentifier(device=config.device)
    
    video_ids = get_video_ids(config, 'test')
    print(f"Found {len(video_ids)} unique videos in {config.dataset_name} test split.")
    
    caption_list = []
    for idx, cur_video_id in tqdm(list(enumerate(video_ids)), desc="Processing video captions", total=len(video_ids)):
        video_id = cur_video_id.replace('/', '_').replace('\\', '_')
        video_path = get_video_path(config.dataset_name, video_id, config.videos_dir)
        cur_video_caption = identifier._generate_video_caption(video_path)
        caption_list.append(cur_video_caption)

    all_video_results = {}
    for idx, video_id in tqdm(list(enumerate(video_ids)), desc="Processing videos", total=len(video_ids)):
        video_id = video_id.replace('/', '_').replace('\\', '_')
        video_path = get_video_path(config.dataset_name, video_id, config.videos_dir)
        cur_video_caption = caption_list[idx]
        video_result = identifier.process_video(video_path, num_frames=config.num_frames, cur_video_caption=cur_video_caption)
        video_result['video_caption'] = cur_video_caption
        all_video_results[video_id] = video_result

    output_path = os.path.join('cache_dir', f"{config.dataset_name}_main_objects.json")
    with open(output_path, 'w') as f:
        json.dump(all_video_results, f, indent=4)
    print(f"Video captions and object detections saved to {output_path}")

if __name__ == "__main__":
    main()