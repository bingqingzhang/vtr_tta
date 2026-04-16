import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import torch
import argparse
import av
import math
import json
import random
from types import SimpleNamespace
import numpy as np
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm
import subprocess
import tempfile
import cv2
import torch.nn as nn
import model.vgg_model as vgg_model
import torchvision.transforms
from scipy.ndimage import zoom as scizoom
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
from io import BytesIO
from scipy.ndimage.interpolation import map_coordinates
from skimage.filters import gaussian
from pathlib import Path
from PIL import ImageFile
from model.clip_baseline import CLIPBaseline


ImageFile.LOAD_TRUNCATED_IMAGES = True


from datasets.model_transforms import init_transform_dict
from datasets.video_capture import VideoCapture, get_frame_indices

def _load_video_from_path(video_path: str):
    """
    A utility function to load all frames from a video path and return video properties.
    
    Args:
        video_path (str): The path to the video file.

    Returns:
        A tuple of (height, width, fps, list_of_all_frames).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return height, width, fps, frames

def _extract_motion_vectors(frames, idxs):
    motion_vectors = []
    if len(idxs) == 0:
        return motion_vectors
    gray_frames_cache = {}

    for idx in idxs:
        if idx == 0:
            h, w = frames[0].shape[:2]
            zero_mv = np.zeros((h, w, 2), dtype=np.float32)
            motion_vectors.append(zero_mv)
            continue
        prev_idx = idx - 1
        current_idx = idx
        if prev_idx not in gray_frames_cache:
            gray_frames_cache[prev_idx] = cv2.cvtColor(frames[prev_idx], cv2.COLOR_BGR2GRAY)
        if current_idx not in gray_frames_cache:
            gray_frames_cache[current_idx] = cv2.cvtColor(frames[current_idx], cv2.COLOR_BGR2GRAY)
        
        prev_gray = gray_frames_cache[prev_idx]
        current_gray = gray_frames_cache[current_idx]
        flow = cv2.calcOpticalFlowFarneback(
            prev=prev_gray,
            next=current_gray,
            flow=None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        motion_vectors.append(flow)  
    return motion_vectors

def plasma_fractal(mapsize=1024, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()

def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    w = img.shape[1]
    #print("h:",h)
    #print("w:",w)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))
    cw = int(np.ceil(w / zoom_factor))
    #print("ch:",ch)
    #print("cw:",cw)

    top1 = (h - ch) // 2
    top2 = (w - cw) // 2
    img = scizoom(img[top1:top1 + ch, top2:top2 + cw], (zoom_factor, zoom_factor, 1), order=1)
    #print("img:", img.shape)
    # trim off any extra pixels
    trim_top1 = (img.shape[0] - h) // 2
    trim_top2 = (img.shape[1] - w) // 2
    
    temp = img[trim_top1:(trim_top1 + h), trim_top2:(trim_top2 + w)]
    #print("temp:", temp.shape)

    return img[trim_top1:(trim_top1 + h), trim_top2:(trim_top2 + w)]

class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

def _linear_motion_kernel(angle: float, length: int):
    """Generate a length×length PSF rotated by *angle* degrees."""
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= k.sum()
    return k

def _disk_kernel(radius: int, alias_blur: float = 0.1) -> np.ndarray:
    if radius <= 0:
        return np.array([[1]], dtype=np.float32)
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    k = ((X ** 2 + Y ** 2) <= radius ** 2).astype(np.float32)
    if k.sum() == 0:
        return np.array([[1]], dtype=np.float32)
    k /= k.sum()
    ksize = (3, 3) if radius <= 8 else (5, 5)
    return cv2.GaussianBlur(k, ksize, sigmaX=alias_blur)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.data.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.data.size()[:2] == style_feat.data.size()[:2])
    size = content_feat.data.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class NoisyVideoPreprocessor:
    """
    A class for adding noise to the test set of a video dataset and preprocessing it.
    It can load video frames, apply specified noises, and then save the results
    as .pt files or a sequence of .jpg images.
    """
    def __init__(self, noise_args):
        """
        Initializes the preprocessor.
        
        Args:
            noise_args (argparse.Namespace): An object containing specific arguments for noise processing.
        """
        self.noise_args = noise_args
        self.img_transforms = init_transform_dict(noise_args.input_res)
        
        assert len(self.noise_args.severity) == len(self.noise_args.noise_types), \
            "Each noise type must have a corresponding severity level."

        # Check and set the output directory
        assert self.noise_args.noised_video_dir is not None, "Please specify an output directory using --noised_video_dir"
        self.base_output_dir = self.noise_args.noised_video_dir
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        self.main_objects_data = None
        if ('main_object_occlusion' in self.noise_args.noise_types) or ('event_insertion' in self.noise_args.noise_types):
            json_path = os.path.join('cache_dir', f"{self.noise_args.dataset_name}_main_objects.json")
            print(f"Loading main object data for occlusion from: {json_path}")
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Main object data file not found at {json_path}. Please generate it first.")
            with open(json_path, 'r') as f:
                self.main_objects_data = json.load(f)
            print("Main object data loaded successfully.")
                                
        if 'style_transfer' in self.noise_args.noise_types:
            style_dir = Path('/path/to/adain/train/') 
            decoder_path = 'cache_dir/decoder.pth'
            vgg_path = 'cache_dir/vgg_normalised.pth'
            device = self.noise_args.model_device
            
            decoder = vgg_model.decoder
            vgg = vgg_model.vgg
            decoder.load_state_dict(torch.load(decoder_path, map_location=device))
            vgg.load_state_dict(torch.load(vgg_path, map_location=device))
            
            vgg = nn.Sequential(*list(vgg.children())[:31])
            vgg.to(device).eval()
            decoder.to(device).eval()

            exts = ['png', 'jpeg', 'jpg']
            style_paths = []
            for ext in exts + [e.upper() for e in exts]:
                style_paths.extend(style_dir.rglob(f'*.{ext}'))
            style_paths = [p for p in style_paths if p.is_file()]
            if not style_paths:
                print(f'[AdaIN] No style images found under: {style_dir}')
            
            self.adain_config = {
                'vgg': vgg,
                'decoder': decoder,
                'style_paths': style_paths,
                'device': device
            }
        
        if 'event_insertion' in self.noise_args.noise_types:
            self.base_video_dir = os.path.join('/path/to/', 'base_videodata')
            self.base_video = []
            video_extensions = ['.mp4', '.avi']
            for root, dirs, files in os.walk(self.base_video_dir):
                for file in files:
                    file_lower = file.lower()
                    if any(file_lower.endswith(ext) for ext in video_extensions):
                        video_path = os.path.abspath(os.path.join(root, file))
                        self.base_video.append(video_path)
            device = self.noise_args.model_device

            retrieval_config = {"huggingface":True, "pooling_type":"avg", "input_res":224, "num_frames":12}
            retrieval_config = SimpleNamespace(**retrieval_config)

            self.retrieval_model = CLIPBaseline(retrieval_config).to(device)
            from transformers import CLIPTokenizer
            self.retrieval_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
            self.retrieval_model.eval()

            if os.path.exists('cache_dir/base_video_embeddings.npy'):
                self._base_video_embeddings = np.load('cache_dir/base_video_embeddings.npy')
            else:
                feats = []
                self.retrieval_video_transform = self.img_transforms['clip_test']
                pbar = tqdm(total=len(self.base_video), desc="Processing base videos for retrieval embeddings")
                for i in range(0, len(self.base_video)):
                    video_path = self.base_video[i]
                    imgs, idxs = VideoCapture.load_frames_from_video(video_path, 12, 'uniform')
                    imgs = self.retrieval_video_transform(imgs).unsqueeze(0).to(device)
                    data = {'video': imgs}
                    with torch.no_grad():
                        emb = self.retrieval_model(data, return_video_only=True)
                        emb_pooled = self.retrieval_model.pool_frames(None, emb)
                    feats.append(emb_pooled.detach().cpu())
                    pbar.update(1)
                self._base_video_embeddings = torch.cat(feats, dim=0).numpy().astype('float32')  # [N, D]
                np.save('cache_dir/base_video_embeddings.npy', self._base_video_embeddings)
        
        print(f"Initializing noise preprocessor with the following settings:")
        print(f"  - Dataset: {self.noise_args.dataset_name}")
        print(f"  - Noise Types: {self.noise_args.noise_types}")
        print(f"  - Severity: {self.noise_args.severity}")
        print(f"  - Save Format: {self.noise_args.save_format}")
        print(f"  - Base Output Directory: {self.base_output_dir}")

    def get_unique_videos(self, split_type='test'):
        """
        Extracts unique video IDs for a specific split from the dataset.
        This method is based on the VideoPreprocessor code you provided.
        """
        print(f"Extracting unique video IDs for the {split_type} split...")
        
        # Dynamically load the corresponding dataset class to get the video list
        if self.noise_args.dataset_name == "MSRVTT":
            from datasets.msrvtt_dataset import MSRVTTDataset
            dataset = MSRVTTDataset(self.noise_args, split_type, None)
            if split_type=='train':
                return dataset.train_vids
            return dataset.test_df['video_id'].unique().tolist()
            
        elif self.noise_args.dataset_name == "MSVD":
            from datasets.msvd_dataset import MSVDDataset
            dataset = MSVDDataset(self.noise_args, split_type, None)
            if split_type == 'train':
                return dataset.train_vids
            return dataset.test_vids

        elif self.noise_args.dataset_name == "LSMDC":
            from datasets.lsmdc_dataset import LSMDCDataset
            dataset = LSMDCDataset(self.noise_args, split_type, None)
            return list(dataset.clip2caption.keys())

        elif self.noise_args.dataset_name == "ActivityNet":
            from datasets.activitynet_dataset import ActivityNetDataset
            dataset = ActivityNetDataset(self.noise_args, split_type, None)
            return [anno['clip_id'] for anno in dataset.anno]
        else:
            raise NotImplementedError(f"Dataset not supported: {self.noise_args.dataset_name}")

    def get_video_path(self, video_id):
        if self.noise_args.dataset_name == "MSRVTT":
            return os.path.join(self.noise_args.videos_dir, video_id + '.mp4')
        elif self.noise_args.dataset_name == "MSVD":
            return os.path.join(self.noise_args.videos_dir, video_id + '.avi')
        elif self.noise_args.dataset_name == "LSMDC":
            clip_prefix = video_id.split('.')[0][:-3]
            return os.path.join(self.noise_args.videos_dir, clip_prefix, video_id + '.avi')
        elif self.noise_args.dataset_name == "ActivityNet":
            return os.path.join(self.noise_args.videos_dir, video_id + '.mp4')
        else:
            raise NotImplementedError(f"Dataset not supported: {self.noise_args.dataset_name}")

    def _video_gaussian_noise(self, frames_np, severity):
        """
        Applies Gaussian noise to a list of frames in NumPy format.
        
        Args:
            frames_np (list[np.ndarray]): A list of RGB frames in (H, W, C) format with values in [0, 255].
            severity (int): The severity of the noise (1-5).
        
        Returns:
            list[np.ndarray]: The list of frames with added noise, in the same format and range.
        """
        stds = [0.08, 0.12, 0.18, 0.26, 0.38]
        if not (1 <= severity <= 5):
            raise ValueError("Severity must be between 1 and 5.")
        c = stds[severity - 1]

        H, W, C = frames_np[0].shape
        # Generate the same noise pattern for all frames to simulate sensor noise
        noise = np.random.normal(loc=0.0, scale=c, size=(H, W, C))
        
        noisy_frames = []
        for frame in frames_np:
            # Normalize to [0, 1] for noise application
            x = frame.astype(np.float32) / 255.0
            # Add noise and clip
            y = np.clip(x + noise, 0.0, 1.0)
            # Convert back to [0, 255]
            noisy = (y * 255.0).astype(np.uint8)
            noisy_frames.append(noisy)
        
        return noisy_frames
    
    def _video_impulse_noise(self, frames_np, severity):
        amounts = [0.03, 0.06, 0.09, 0.17, 0.27]
        if not (1 <= severity <= 5):
            raise ValueError("Severity must be between 1 and 5.")
        c = amounts[severity - 1]
        
        H, W, _ = frames_np[0].shape
        noise_mask = np.random.rand(H, W)
        salt_mask = noise_mask >= (1.0 - c / 2.0)
        pepper_mask = noise_mask < (c / 2.0)
        noisy_frames = []
        for frame in frames_np:
            noisy_frame_normalized = frame.astype(np.float32) / 255.0
            noisy_frame_normalized[salt_mask, :] = 1.0
            noisy_frame_normalized[pepper_mask, :] = 0.0
            noisy = (noisy_frame_normalized * 255.0).astype(np.uint8)
            noisy_frames.append(noisy)
        return noisy_frames
    
    def _video_fog(self, frames_np, severity):
        """
        Applies a temporally consistent fog effect to a list of video frames
        using the provided diamond-square plasma_fractal function.

        Args:
            frames_np (list[np.ndarray]): A list of RGB frames in (H, W, C) format
                                        with pixel values in [0, 255].
            severity (int): The severity of the fog effect (1-5).

        Returns:
            list[np.ndarray]: The list of frames with the fog effect added.
        """
        if not (1 <= severity <= 5):
            raise ValueError("Severity must be between 1 and 5.")

        params = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)]
        fog_intensity, wibbledecay = params[severity - 1]

        H, W, _ = frames_np[0].shape
        
        # 1. Determine the required mapsize for the fractal generator.
        max_dim = max(H, W)
        mapsize = 1 << (max_dim - 1).bit_length()
        # 2. Generate the fractal noise ONCE.
        fractal_noise = plasma_fractal(mapsize=mapsize, wibbledecay=wibbledecay)
        
        # 3. Crop the generated fractal to the exact frame dimensions.
        cropped_fractal = fractal_noise[:H, :W]

        # 4. Scale by intensity and add a new axis for broadcasting (to match H, W, 3 shape).
        fog_layer = fog_intensity * cropped_fractal[..., np.newaxis]
        foggy_frames = []
        for frame in frames_np:
            x = frame.astype(np.float32) / 255.0
            max_val = x.max()
            if max_val == 0: max_val = 1.0
            fogged_x = x + fog_layer
            rescaled_x = fogged_x * (max_val / (max_val + fog_intensity))
            clipped_x = np.clip(rescaled_x, 0.0, 1.0)
            foggy_frame = (clipped_x * 255.0).astype(np.uint8)

            foggy_frames.append(foggy_frame)
        return foggy_frames
    
    def _video_snow(self, frames_np, severity=1):
        """
        Applies a temporally consistent snow effect using the provided clipped_zoom.

        This works by creating a large "curtain" of snow and scrolling through it
        for each frame, creating a falling illusion.

        Args:
            frames_np (list[np.ndarray]): A list of RGB frames in (H, W, C) format.
            severity (int): The severity of the snow (1-5).

        Returns:
            list[np.ndarray]: The list of frames with the snow effect added.
        """
        if not (1 <= severity <= 5):
            raise ValueError("Severity must be between 1 and 5.")
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
            (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
            (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
            (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
            (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

        loc, scale, zoom, threshold, blur_r, blur_s, blend = c
        H, W, _ = frames_np[0].shape
        num_frames = len(frames_np)
        fall_speed = 3  # Pixels the snow falls per frame.
        # Calculate the total height needed for the curtain to scroll through all frames.
        canvas_H = H + (num_frames - 1) * fall_speed

        snow_canvas = np.random.normal(size=(canvas_H, W), loc=loc, scale=scale)
        snow_canvas = clipped_zoom(snow_canvas, zoom) 
        snow_canvas[snow_canvas < threshold] = 0

        blur_angle = np.random.uniform(-135, -45)
        snow_canvas_pil = Image.fromarray((np.clip(snow_canvas.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
        output = BytesIO()
        snow_canvas_pil.save(output, format='PNG')
        snow_canvas_motion = MotionImage(blob=output.getvalue())
        snow_canvas_motion.motion_blur(radius=blur_r, sigma=blur_s, angle=blur_angle)

        # Convert back to a NumPy array for processing.
        snow_curtain = cv2.imdecode(np.fromstring(snow_canvas_motion.make_blob(), np.uint8),
                                    cv2.IMREAD_UNCHANGED) / 255.
        snow_curtain = snow_curtain[..., np.newaxis]
        snowy_frames = []
        for i, frame in enumerate(frames_np):
            x = (frame[..., [2, 1, 0]].astype(np.float32)) / 255.
            brightened_x = blend * x + (1 - blend) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).reshape(H, W, 1) * 1.5 + 0.5)
            y_start = i * fall_speed
            y_end = y_start + H
            snow_layer_slice = snow_curtain[y_start:y_end, :]
            final_frame_bgr = np.clip(brightened_x + snow_layer_slice + np.rot90(snow_layer_slice, k=2), 0, 1)
            final_frame_rgb = (final_frame_bgr[..., [2, 1, 0]] * 255).astype(np.uint8)
            snowy_frames.append(final_frame_rgb)
        return snowy_frames
    
    def _video_elastic_transform(self, frames_np, severity=1):
        """
        Applies a temporally consistent elastic transform to a list of video frames.

        This works by generating a single, fixed distortion field (both affine
        and elastic) and applying it to every frame in the sequence.

        Args:
            frames_np (list[np.ndarray]): A list of RGB frames in (H, W, C) format.
            severity (int): The severity of the distortion (1-5).

        Returns:
            list[np.ndarray]: The list of frames with the distortion applied.
        """
        if not (1 <= severity <= 5):
            raise ValueError("Severity must be between 1 and 5.")

        c = [(244 * 2, 244 * 0.7, 244 * 0.1),
            (244 * 2, 244 * 0.08, 244 * 0.2),
            (244 * 0.05, 244 * 0.01, 244 * 0.02),
            (244 * 0.07, 244 * 0.01, 244 * 0.02),
            (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]
        
        alpha, sigma, affine_magnitude = c
        H, W, C = frames_np[0].shape
        shape = (H, W, C)
        shape_size = (H, W)
        
        # 1. Create a fixed Affine Transform
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size,
                        [center_square[0] + square_size, center_square[1] - square_size],
                        center_square - square_size])
        
        pts2 = pts1 + np.random.uniform(-affine_magnitude, affine_magnitude, size=pts1.shape).astype(np.float32)
        # Calculate the transformation matrix ONCE
        M = cv2.getAffineTransform(pts1, pts2)

        # 2. Create fixed Elastic Displacement Fields (dx, dy)
        # Generate random noise and blur it ONCE to get smooth displacement fields
        dx = (gaussian(np.random.uniform(-1, 1, size=shape_size),
                    sigma, mode='reflect', truncate=3) * alpha).astype(np.float32)
        dy = (gaussian(np.random.uniform(-1, 1, size=shape_size),
                    sigma, mode='reflect', truncate=3) * alpha).astype(np.float32)
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
            # 3. Create the final coordinate map ONCE
        x, y, z = np.meshgrid(np.arange(W), np.arange(H), np.arange(C))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        distorted_frames = []
        for frame in frames_np:
            # Normalize the frame
            frame_normalized = frame.astype(np.float32) / 255.0

            # Apply the pre-calculated affine transform
            frame_affine = cv2.warpAffine(frame_normalized, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)

            # Apply the pre-calculated elastic transform using the fixed coordinate map
            distorted_frame = map_coordinates(frame_affine, indices, order=1, mode='reflect').reshape(shape)
            
            # Denormalize and append
            final_frame = (np.clip(distorted_frame, 0, 1) * 255).astype(np.uint8)
            distorted_frames.append(final_frame)

        return distorted_frames
        
    def video_h264_compression(self, video_path: str, severity: int = 1, output_frames: int = 12):
        """
        Applies H.264 compression artifacts to a video and returns a sampled list of frames.

        This function simulates real-world video compression artifacts by encoding and
        decoding the video at a low bitrate using FFmpeg.

        Args:
            video_path (str): The path to the input video file (e.g., '.mp4', '.avi').
            severity (int): The severity of the compression (1-5). A higher level means
                            a lower bitrate and more artifacts.
            output_frames (int): The number of frames to uniformly sample from the
                                corrupted video.

        Returns:
            torch.Tensor: A tensor containing `output_frames` number of corrupted frames.
                        The tensor has a shape of (F, C, H, W), is in RGB order,
                        and has float values normalized to the [0, 1] range.
        """
        # Define bitrates for each severity level. Lower is worse.
        bitrates = ['500k', '250k', '100k', '50k', '25k']
        # bitrates = ['1000k', '500k', '250k', '150k', '100k']
        if not (1 <= severity <= 5):
            raise ValueError("Severity must be between 1 and 5.")
        bitrate = bitrates[severity - 1]

        # --- Step 1: Load original frames and video properties from path ---
        H, W, fps, original_frames = _load_video_from_path(video_path)
        if not original_frames:
            print("Warning: No frames found in the video.")
            return torch.empty(0)

        # --- Step 2: Encode the video with a low bitrate using FFmpeg ---
        temp_output_path = ''
        try:
            # Create a secure temporary file for the compressed video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output_file:
                temp_output_path = temp_output_file.name
            
            # FFmpeg command to read raw video from stdin and write a compressed mp4 file
            command = [
                'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{W}x{H}', '-pix_fmt', 'bgr24', '-r', str(fps),
                '-i', '-', '-an', '-vcodec', 'libx264', '-b:v', bitrate,
                '-preset', 'fast', temp_output_path
            ]

            # Start the FFmpeg process and hide its output
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Write all original frames to FFmpeg's stdin
            for frame in original_frames:
                proc.stdin.write(frame.tobytes())
            
            proc.stdin.close()
            proc.wait()

            # --- Step 3: Decode the compressed video to get the corrupted frames ---
            cap = cv2.VideoCapture(temp_output_path)
            corrupted_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                corrupted_frames.append(frame)
            cap.release()

        finally:
            # --- Step 4: Clean up the temporary file ---
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
        
        # --- Step 5: Uniformly sample and process the final frames ---
        if not corrupted_frames:
            print("Warning: Failed to generate any frames from the corrupted video.")
            return torch.empty(0)
            
        total_corrupted = len(corrupted_frames)
        indices = get_frame_indices(output_frames, total_corrupted)
        
        processed_frames_rgb = []
        for i in indices:
            frame_bgr = corrupted_frames[i]
            # Convert BGR -> RGB
            frame_rgb = frame_bgr[:, :, ::-1]
            processed_frames_rgb.append(frame_rgb)
        
        num_processed = len(processed_frames_rgb)
        if 0 < num_processed < output_frames:
            last_frame_rgb = processed_frames_rgb[-1]
            padding_needed = output_frames - num_processed
            processed_frames_rgb.extend([last_frame_rgb] * padding_needed)
        
        # Stack HWC frames into a single numpy array (F, H, W, C)
        stacked_frames_np = np.stack(processed_frames_rgb, axis=0)

        # Convert to a torch tensor, permute to (F, C, H, W), and normalize to [0, 1]
        final_tensor = torch.from_numpy(stacked_frames_np).permute(0, 3, 1, 2).float() / 255.0
            
        return final_tensor
    
    def video_event_insertion(self, video_path, severity=1, output_frames=12, video_id=None):
        """
        Event insertion with pre-extracted frames.
        Args:
            frames_np: List[np.ndarray], each frame is RGB (H, W, C), dtype=uint8.
            severity: 1..5 controlling the insertion ratio (10%..50%).
            output_frames: Total number of frames to be sampled from the original video.
            video_id: str, the video id of input video
        Returns:
            torch.Tensor: (F, C, H, W), float32 in [0, 1].
        """
        device = self.noise_args.model_device
        if not (1 <= severity <= 5):
            raise ValueError("Severity must be between 1 and 5.")
        assert video_id is not None, "video_id must be provided for event insertion."

        # cur_video_meta_info = self.video_meta_info[video_id]
        # caption = cur_video_meta_info['caption']
        # caption = self.video_caption
        caption = self.main_objects_data[video_id]['video_caption']

        tok = self.retrieval_tokenizer([caption],  return_tensors='pt', padding=True, truncation=True)
        # lang_inputs = {k: v.to(device) for k, v in tok.items()}
        data = {'text': tok.to(device)}
        with torch.no_grad():
            lang_emb = self.retrieval_model(data, return_text_only=True)
        # lang_emb = lang_emb / (lang_emb.norm(dim=-1, keepdim=True) + 1e-8)
        lang_np = lang_emb.detach().cpu().numpy().astype('float32')

        # ---- Retrieve the most similar base video ----
        sims = self._base_video_embeddings @ lang_np.T
        best_idx = int(np.argmax(sims))
        insert_video_path = self.base_video[best_idx]

        # ---- Decide how many frames to insert (30%..80%) ----
        insert_ratio = {1: 0.30, 2: 0.40, 3: 0.50, 4: 0.60, 5: 0.70}[severity]
        F_total = output_frames
        k_insert = max(1, min(F_total - 1, int(round(F_total * insert_ratio))))
        keep_n = F_total - k_insert
        
        orig_tensor, _ = VideoCapture.load_frames_from_video(
            video_path,
            num_frames=F_total,
            sample='uniform'  # Sample uniformly from the entire video
        )
        
        actual_orig_frames = orig_tensor.shape[0]
        _, _, H, W = orig_tensor.shape
        
        if actual_orig_frames < F_total:
            print(f"Warning: Video {video_id} has only {actual_orig_frames} frames, expected {F_total}")

            keep_n = min(keep_n, actual_orig_frames)
            k_insert = max(1, F_total - keep_n)
        
        insert_tensor, _ = VideoCapture.load_frames_from_video(
            insert_video_path,
            num_frames=k_insert,
            sample='uniform'
        )
        if insert_tensor.shape[-2:] != (H, W):
            from torch.nn import functional as F
            insert_tensor = F.interpolate(insert_tensor, size=(H, W), mode='bilinear', align_corners=False)
        fused = torch.cat([orig_tensor[:keep_n], insert_tensor[:k_insert]], dim=0)
        
        current_frames = fused.shape[0]
        if current_frames < F_total:
            frames_needed = F_total - current_frames
            print(f"Padding {frames_needed} frames using the last frame for video {video_id}")

            last_frame = fused[-1:].clone()
            padding_frames = last_frame.repeat(frames_needed, 1, 1, 1)
            fused = torch.cat([fused, padding_frames], dim=0)
        
        fused = fused[:F_total].clamp_(0.0, 1.0).contiguous()
        return fused
    
    def video_temporal_scrambling(self, video_path, severity=1, output_frames=12, video_id=None):
        """
        Applies a two-stage perturbation to a video:
        1. Temporal Trimming: Trims a percentage of frames from the start and end of the video.
        The total trim percentage is determined by 'severity', and is randomly distributed
        between the head and tail.
        2. Temporal Scrambling: Scrambles the temporal order of the *remaining middle* frames.

        Args:
            video_path (str): Path to the video file.
            severity (int): An integer from 1 to 5. Controls the percentage of the video
                            to trim (from 20% to 60%).
            output_frames (int): The number of frames in the final output tensor.
            video_id (any, optional): An identifier for the video. Not used.

        Returns:
            torch.Tensor: The processed video tensor of shape (output_frames, C, H, W).
        """
        if not (1 <= severity <= 5):
            raise ValueError("severity must be in 1…5")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if vlen == 0:
            cap.release()
            raise ValueError(f"Video file seems to be empty or corrupted: {video_path}")

        # --- STAGE 1: TEMPORAL TRIMMING (FROM START AND END) ---
        # Map severity to a trimming percentage (20% to 60%)
        trim_ratio = [0.60, 0.70, 0.80, 0.90, 0.95][severity - 1]
        num_to_trim = int(vlen * trim_ratio)
        
        # Randomly distribute the total number of trimmed frames between the start and end
        num_at_start = random.randint(0, num_to_trim)
        num_at_end = num_to_trim - num_at_start
        
        start_index = num_at_start
        end_index = vlen - num_at_end

        # Check if the trimming leaves any frames, if not, use the last frame
        if start_index >= end_index:
            print(f"Warning: Trimming removed all frames for video {video_path}. Using last frame.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, vlen - 1)
            ret, last_frame = cap.read()
            cap.release()
            
            if ret:
                last_frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
                last_frame_tensor = torch.from_numpy(last_frame_rgb).permute(2, 0, 1).float() / 255.0
                return last_frame_tensor.unsqueeze(0).repeat(output_frames, 1, 1, 1)
            else:
                cap.release()
                return torch.zeros(output_frames, 3, 224, 224)

        # The frames we will work with are in the middle of the video
        middle_indices = np.arange(start_index, end_index)
        vlen_middle = len(middle_indices)

        # --- STAGE 2: TEMPORAL SCRAMBLING (on the middle section) ---
        num_chunks = min(output_frames, vlen_middle)
        
        # Split the *middle* indices into chunks
        chunks = np.array_split(middle_indices, num_chunks)
        chunk_order = list(range(num_chunks))
        
        # Scrambling logic remains the same, but now acts on the middle chunks
        if severity == 1:
            if num_chunks > 1:
                idx = random.randint(0, num_chunks - 2)
                chunk_order[idx], chunk_order[idx + 1] = chunk_order[idx + 1], chunk_order[idx]
        elif 1 < severity < 5:
            if num_chunks > 1:
                swapped_indices = set()
                num_swaps = min(severity, num_chunks // 2)
                for _ in range(num_swaps):
                    available_indices = [i for i in range(num_chunks) if i not in swapped_indices]
                    if len(available_indices) < 2: break
                    idx1, idx2 = random.sample(available_indices, 2)
                    chunk_order[idx1], chunk_order[idx2] = chunk_order[idx2], chunk_order[idx1]
                    swapped_indices.update([idx1, idx2])
        elif severity == 5:
            random.shuffle(chunk_order)
            
        scrambled_middle_indices = np.concatenate([chunks[i] for i in chunk_order])

        # --- STAGE 3: SAMPLING & FRAME LOADING ---
        if vlen_middle < output_frames:
            final_frame_indices_to_load = np.random.choice(scrambled_middle_indices, size=output_frames, replace=True)
        else:
            sampling_points = np.linspace(0, vlen_middle - 1, num=output_frames, dtype=int)
            final_frame_indices_to_load = scrambled_middle_indices[sampling_points]

        # The efficient frame loading logic is reused without changes
        frames = []
        sorted_indices_map = sorted(enumerate(final_frame_indices_to_load), key=lambda x: x[1])
        
        loaded_frames = {}
        for _, frame_idx in sorted_indices_map:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                loaded_frames[int(frame_idx)] = frame_rgb
            else:
                if loaded_frames:
                    last_frame_key = max(loaded_frames.keys())
                    loaded_frames[int(frame_idx)] = loaded_frames[last_frame_key]
                else:
                    cap.release()
                    raise ValueError(f"Could not read frame {frame_idx} from {video_path} and no previous frame is available.")
        
        for original_pos, frame_idx in sorted(sorted_indices_map, key=lambda x: x[0]):
            frame_tensor = torch.from_numpy(loaded_frames[int(frame_idx)].copy())
            frames.append(frame_tensor.permute(2, 0, 1))

        cap.release()
        output_tensor = torch.stack(frames).float() / 255.0

        # Safeguard to ensure correct output size - use last frame for padding
        if len(output_tensor) != output_frames:
            if len(output_tensor) > output_frames:
                output_tensor = output_tensor[:output_frames]
            else:
                # Use the last frame to pad the tensor
                while len(output_tensor) < output_frames:
                    output_tensor = torch.cat([output_tensor, output_tensor[-1].unsqueeze(0)], dim=0)

        return output_tensor

    def _adaptive_motion_blur(self, frame: np.ndarray, mv: np.ndarray, severity: int):
        """Apply motion blur whose kernel length&angle depend on local MV magnitude.

        * High‑motion regions (top 25 % magnitude) get a **longer** blur kernel.
        * Low‑motion regions get a shorter kernel (still >0 for consistency).
        """
        assert mv.shape[:2] == frame.shape[:2]
        mag = np.linalg.norm(mv, axis=-1)
        thr = np.percentile(mag, 75)  # 75‑th percentile threshold

        # Base kernel sizes for severity 1‑5
        base_short = [5, 7, 9, 11, 13][severity - 1]
        base_long  = [9, 13, 17, 21, 25][severity - 1]

        # Global dominant direction (in degrees)
        mean_vec = mv.reshape(-1, 2).mean(axis=0)
        angle = math.degrees(math.atan2(mean_vec[1], mean_vec[0] + 1e-8))

        k_short = _linear_motion_kernel(angle, base_short)
        k_long  = _linear_motion_kernel(angle, base_long)

        blur_short = cv2.filter2D(frame, -1, k_short)
        blur_long  = cv2.filter2D(frame, -1, k_long)

        mask_long = (mag >= thr).astype(np.float32)[..., None]  # (H,W,1)
        blended = blur_short * (1.0 - mask_long) + blur_long * mask_long
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        return blended

    def _adaptive_defocus_blur(self, frame: np.ndarray, mv: np.ndarray, severity: int):
        """Apply defocus blur with radius amplified in high‑motion areas."""
        mag = np.linalg.norm(mv, axis=-1)
        thr = np.percentile(mag, 75)

        base_rad = [3, 4, 6, 8, 10][severity - 1]
        large_rad = base_rad * 2

        k_small = _disk_kernel(base_rad)
        k_large = _disk_kernel(large_rad)

        blur_small = cv2.filter2D(frame, -1, k_small)
        blur_large = cv2.filter2D(frame, -1, k_large)

        mask_large = (mag >= thr).astype(np.float32)[..., None]
        blended = blur_small * (1.0 - mask_large) + blur_large * mask_large
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        return blended
    
    def apply_video_blur(self, video_path: str, output_frames: int = 12, severity: int = 5, mode: str = "motion_blur"):
        """Return a tensor containing *output_frames* perturbed frames.

        The perturbation uses motion vectors whenever available; otherwise it
        computes dense optical flow.
        """
        if mode not in {"motion_blur", "video_defocus"}:
            raise ValueError("mode must be 'motion_blur' or 'video_defocus'")
        if not (1 <= severity <= 5):
            raise ValueError("severity must be in 1…5")
        
        h, w, fps, frames = _load_video_from_path(video_path)
        # frame sampling

        total_frames = len(frames)
        idxs = get_frame_indices(output_frames, total_frames)
        sampled_frames = [frames[i] for i in idxs]

        motion_vectors = _extract_motion_vectors(frames, idxs)

        assert len(sampled_frames) == len(motion_vectors), \
            "Number of sampled frames must match number of motion vectors"
        
        # Apply blur on sampled frames
        out_rgb: List[np.ndarray] = []
        for f, mv in zip(sampled_frames, motion_vectors):
            if mode=="motion_blur":
                bf = self._adaptive_motion_blur(f, mv, severity)
            elif mode == "video_defocus":
                bf = self._adaptive_defocus_blur(f, mv, severity)
            else:
                raise NotImplementedError(f"Blur mode not supported: {mode}")
            out_rgb.append(bf[:,:,::-1])  # BGR->RGB

        arr = np.stack(out_rgb, axis=0)
        tensor = torch.from_numpy(arr).permute(0,3,1,2).float()/255.0
        return tensor
    
    def _video_main_object_occlusion(self, frames_np: List[np.ndarray], severity: int, video_id: str, frame_indices: List[int]) -> List[np.ndarray]:
        """
        Applies an occlusion of a fixed relative size to the main object in every frame.

        Args:
            frames_np (List[np.ndarray]): A list of RGB frames in (H, W, C) format.
            severity (int): Controls the occlusion area's size relative to the object (1-5).
                            Level 1 is ~30% of the area, Level 5 is ~80%.
            video_id (str): The ID of the video to look up in the main objects JSON file.
            frame_indices (List[int]): The original indices of the frames in frames_np.

        Returns:
            List[np.ndarray]: The list of frames with occlusions applied.
        """
        if not (1 <= severity <= 5):
            raise ValueError("Severity must be between 1 and 5.")
        if self.main_objects_data is None:
            raise ValueError("Main objects data is not loaded. Please load the main objects JSON file first.")

        # New logic: Severity controls the occlusion area ratio (30% to 80%)
        occlusion_area_ratios = np.linspace(0.3, 0.8, 5)
        occ_area_ratio = occlusion_area_ratios[severity - 1]
        
        num_frames = len(frames_np)
        assert num_frames == len(frame_indices), f"Length mismatch: {num_frames} frames vs {len(frame_indices)} indices"
        
        # Get object info for this video from the main data source
        # video_object_data = self.main_objects_data.get(video_id, {})
        video_object_data = self.main_objects_data[video_id] 
        # ranked_objects_per_frame = video_object_data.get("ranked_objects_per_frame", {})
        ranked_objects_per_frame = video_object_data['ranked_objects_per_frame']

        occluded_frames = [f.copy() for f in frames_np]
        last_occlusion_box = None # For temporal consistency

        H, W, _ = frames_np[0].shape

        # New logic: Iterate through and occlude every frame
        for i in range(num_frames):
            original_frame_idx = frame_indices[i]
            frame_objects = ranked_objects_per_frame[str(original_frame_idx)]

            occlusion_box = None
            if frame_objects:
                # Get the main object (first in the ranked list)
                main_object = frame_objects[0]
                box = main_object['box']
                x_min, y_min, x_max, y_max = [int(v) for v in box]

                # Ensure bounding box coordinates are within frame boundaries
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(W, x_max), min(H, y_max)

                box_w, box_h = x_max - x_min, y_max - y_min

                if box_w > 0 and box_h > 0:
                    # Calculate the occlusion patch size based on severity
                    occ_h = int(box_h * np.sqrt(occ_area_ratio))
                    occ_w = int(box_w * np.sqrt(occ_area_ratio))
                    
                    # Randomly determine the top-left corner of the occlusion patch within the object
                    # Using max(x_min + 1, ...) prevents an invalid range for randint if occ_w is large
                    occ_x_start = np.random.randint(x_min, max(x_min + 1, x_max - occ_w))
                    occ_y_start = np.random.randint(y_min, max(y_min + 1, y_max - occ_h))
                    
                    occlusion_box = [occ_x_start, occ_y_start, occ_x_start + occ_w, occ_y_start + occ_h]
            
            else: # If no object is detected in the current frame
                if last_occlusion_box:
                    # Use the last occlusion box with a small random shift for continuity
                    last_x1, last_y1, last_x2, last_y2 = last_occlusion_box
                    occ_w, occ_h = last_x2 - last_x1, last_y2 - last_y1
                    
                    shift_x = np.random.randint(-5, 6)
                    shift_y = np.random.randint(-5, 6)
                    
                    new_x1 = last_x1 + shift_x
                    new_y1 = last_y1 + shift_y
                    occlusion_box = [new_x1, new_y1, new_x1 + occ_w, new_y1 + occ_h]
                else:
                    # If it's the first frame and has no object, create a random box in the center
                    occ_w = np.random.randint(W // 8, W // 4)
                    occ_h = np.random.randint(H // 8, H // 4)
                    occ_x_start = (W - occ_w) // 2
                    occ_y_start = (H - occ_h) // 2
                    occlusion_box = [occ_x_start, occ_y_start, occ_x_start + occ_w, occ_y_start + occ_h]

            if occlusion_box:
                x1, y1, x2, y2 = [int(v) for v in occlusion_box]
                # Clip coordinates again to ensure the occlusion box itself is within the frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                
                # Apply the black rectangle occlusion
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(occluded_frames[i], (x1, y1), (x2, y2), (0, 0, 0), -1)
                
                last_occlusion_box = [x1, y1, x2, y2]
        
        return occluded_frames

    
    def _video_style_transfer(self, frames_np, severity):
        """
        Applies a randomly chosen artistic style to a list of frames using AdaIN.
        Severity is controlled by the `alpha` parameter, which dictates the degree
        of stylization. Helper functions are nested for encapsulation.

        Args:
            frames_np (List[np.ndarray]): A list of RGB frames in (H, W, C) format.
            severity (int): The severity level (1-5).

        Returns:
            List[np.ndarray]: The list of frames with the style transfer applied.
        """
        def _input_transform(size=0):
            """Creates a transform to convert a PIL image to a PyTorch tensor."""
            transform_list = []
            if size:
                transform_list.append(torchvision.transforms.Resize(size))
            transform_list.append(torchvision.transforms.ToTensor())
            return torchvision.transforms.Compose(transform_list)

        def _style_transfer_core(vgg, decoder, content, style, alpha=1.0):
            """The core AdaIN style transfer logic."""
            assert 0.0 <= alpha <= 1.0
            with torch.no_grad():
                content_f = vgg(content)
                style_f = vgg(style)
                feat = adaptive_instance_normalization(content_f, style_f)
                feat = feat * alpha + content_f * (1 - alpha)
                return decoder(feat)
            
        if not self.adain_config or not self.adain_config['style_paths']:
            print("Skipping AdaIN style transfer: models not loaded or no style images found.")
            return frames_np

        if not (1 <= severity <= 5):
            raise ValueError("Severity must be between 1 and 5.")

        # Map severity to the alpha parameter
        alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
        alpha = alphas[severity - 1]
        
        device = self.adain_config['device']

        # Choose one style image for the entire video clip for temporal consistency
        style_path = random.choice(self.adain_config['style_paths'])
        
        try:
            img = Image.open(style_path)
            img.verify() 
            img = Image.open(style_path).convert('RGB')
            style_img_pil = img
        except Exception as e:
            print(f"Error loading style image {style_path}: {e}")
            default_style_path = self.adain_config['style_paths'][0]
            print(f"Using default style image {default_style_path} instead.")
            style_img_pil = Image.open(default_style_path).convert('RGB')

        # Define transforms
        content_tf = _input_transform() 
        style_tf = _input_transform(size=512)

        style_tensor = style_tf(style_img_pil).to(device).unsqueeze(0)

        stylized_frames = []
        for frame_np in frames_np:
            content_img_pil = Image.fromarray(frame_np)
            content_tensor = content_tf(content_img_pil).to(device).unsqueeze(0)

            # Perform the style transfer
            output_tensor = _style_transfer_core(
                self.adain_config['vgg'],
                self.adain_config['decoder'],
                content_tensor,
                style_tensor,
                alpha
            )
            
            # Post-process the output tensor back to a numpy array
            output_tensor = output_tensor.squeeze(0).cpu()
            stylized_np = output_tensor.permute(1, 2, 0).numpy()
            stylized_np = (np.clip(stylized_np, 0, 1) * 255).astype(np.uint8)
            
            stylized_frames.append(stylized_np)

        return stylized_frames

    def apply_noise(self, imgs_tensor, noise_type, severity, video_id, frame_indices, num_frames=12):
        """
        Applies specified noise to a video tensor. It converts the tensor to NumPy for the
        noise function and then converts it back.
        
        Args:
            imgs_tensor (torch.Tensor): A tensor of shape (F, C, H, W) with values in [0, 1].
            noise_type (str): The type of noise to apply (e.g., 'gaussian').
            severity (int): The severity level of the noise (1-5).

        Returns:
            torch.Tensor: A noisy tensor of the same shape and value range.
        """
        # 1. Convert tensor (F, C, H, W), [0, 1] to list of numpy frames (H, W, C), [0, 255]
        imgs_tensor_permuted = imgs_tensor.permute(0, 2, 3, 1)
        frames_np_all = (imgs_tensor_permuted * 255).byte().cpu().numpy()
        frames_np = [frame for frame in frames_np_all]

        # 2. Apply the actual noise function
        if noise_type == 'gaussian':
            noisy_frames_np = self._video_gaussian_noise(frames_np, severity)
        elif noise_type == 'impulse':
            noisy_frames_np = self._video_impulse_noise(frames_np, severity)
        elif noise_type == 'fog':
            noisy_frames_np = self._video_fog(frames_np, severity)
        elif noise_type == 'snow':
            noisy_frames_np = self._video_snow(frames_np, severity)
        elif noise_type == 'elastic_distortion':
            noisy_frames_np = self._video_elastic_transform(frames_np, severity)
        elif noise_type == 'main_object_occlusion':
            noisy_frames_np = self._video_main_object_occlusion(frames_np, severity, video_id, frame_indices)
        elif noise_type == 'style_transfer':
            noisy_frames_np = self._video_style_transfer(frames_np, severity)
        else:
            raise NotImplementedError(f"Noise type not supported: {noise_type}")
        
        num_processed_frames = len(noisy_frames_np)
        if num_processed_frames < num_frames:
            if num_processed_frames == 0:
                raise ValueError("Cannot pad an empty list of frames returned by the noise function.")
            
            last_frame = noisy_frames_np[-1]
            padding_needed = num_frames - num_processed_frames
            # Use extend with a list of copies of the last frame for efficiency
            noisy_frames_np.extend([last_frame] * padding_needed)

        # 3. Convert noisy numpy frames back to a tensor (F, C, H, W), [0, 1]
        noisy_frames_np_stacked = np.stack(noisy_frames_np, axis=0)
        noisy_tensor_permuted = torch.from_numpy(noisy_frames_np_stacked).to(imgs_tensor.device)
        noisy_tensor = noisy_tensor_permuted.permute(0, 3, 1, 2).float()
        noisy_tensor /= 255.0
        
        return noisy_tensor

    def apply_noise_from_video_path(self, video_path, noise_type, severity, video_id):
        if noise_type == 'h264_compression':
            return self.video_h264_compression(video_path, severity, self.noise_args.num_frames)
        elif noise_type == 'event_insertion':
            return self.video_event_insertion(video_path, severity, self.noise_args.num_frames, video_id)
        elif noise_type == 'temporal_scrambling':
            return self.video_temporal_scrambling(video_path, severity, self.noise_args.num_frames, video_id)
        elif noise_type in ['motion_blur', 'video_defocus']:
            return self.apply_video_blur(video_path, self.noise_args.num_frames, severity, noise_type)
        else:
            raise NotImplementedError(f"Noise type not supported: {noise_type}")
    
    def _save_frames_as_jpg(self, frames_np, output_folder, clip_id):
        """Saves a list of NumPy frames as a sequence of JPG images."""
        clip_folder = os.path.join(output_folder, clip_id)
        os.makedirs(clip_folder, exist_ok=True)
        for idx, frame_np in enumerate(frames_np):
            filename = f'frame_{idx:04d}.jpg'
            path = os.path.join(clip_folder, filename)
            # cv2.imwrite expects BGR format, while our frames are RGB, so a conversion is needed
            cv2.imwrite(path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))

    def preprocess_test_split(self):
        """
        Applies all specified noises to all videos in the test set and saves them
        according to the settings.
        """
        print("-" * 50)
        print("Starting noisy preprocessing for the test set...")
        
        video_ids = self.get_unique_videos('test')
        print(f"Found {len(video_ids)} unique videos in the test set.")

        # Process each specified noise type
        for idx, noise_type in enumerate(self.noise_args.noise_types):
            output_dir = os.path.join(self.base_output_dir, f"{noise_type}_{self.noise_args.severity[idx]}")
            os.makedirs(output_dir, exist_ok=True)

            print(f"\nProcessing noise type: '{noise_type}', Severity: {self.noise_args.severity[idx]}")
            print(f"Output will be saved to: {output_dir}")
            processed_count = 0
            skipped_count = 0
            for video_id in tqdm(video_ids, desc=f"Applying {noise_type} noise"):
 
                safe_video_id = video_id.replace('/', '_').replace('\\', '_')
                
                if self.noise_args.save_format == 'pt':
                    output_path = os.path.join(output_dir, f"{safe_video_id}.pt")
                else: # jpg
                    output_path = os.path.join(output_dir, safe_video_id)

                if os.path.exists(output_path):
                    skipped_count += 1
                    continue

                video_path = self.get_video_path(video_id)
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")

                if noise_type not in ['h264_compression','video_defocus', 'motion_blur','event_insertion', 'temporal_scrambling']:
                    imgs_tensor, frame_indices = VideoCapture.load_frames_from_video(
                        video_path,
                        self.noise_args.num_frames,
                        self.noise_args.video_sample_type
                    )

                    if imgs_tensor is None or len(imgs_tensor) == 0:
                        raise ValueError(f"No frames loaded from video: {video_path}")

                    # 2. Apply noise. Takes a tensor [0, 1] and returns a noisy tensor [0, 1].
                    noisy_imgs_tensor = self.apply_noise(imgs_tensor, noise_type, self.noise_args.severity[idx], video_id, frame_indices)

                else:
                    noisy_imgs_tensor = self.apply_noise_from_video_path(video_path, noise_type, self.noise_args.severity[idx], video_id)

                # 3. Save based on the specified format
                if self.noise_args.save_format == 'pt':
                    # Apply test transforms (Resize, Crop, Normalize) to the noisy tensor
                    transforms = self.img_transforms['clip_test']
                    if transforms:
                        transformed_tensor = transforms(noisy_imgs_tensor)
                    else:
                        transformed_tensor = noisy_imgs_tensor
                    
                    # Save the final transformed tensor as a .pt file
                    torch.save({
                        'video': transformed_tensor,
                        'video_id': video_id,
                    }, output_path)

                elif self.noise_args.save_format == 'jpg':
                    # Convert the noisy tensor [0, 1] to numpy frames [0, 255] for saving
                    noisy_frames_np = (noisy_imgs_tensor.permute(0, 2, 3, 1) * 255).byte().cpu().numpy()
                    self._save_frames_as_jpg(list(noisy_frames_np), output_dir, safe_video_id)

                processed_count += 1

            print(f"'{noise_type}' processing complete: {processed_count} processed, {skipped_count} skipped.")

    
def main():    
    parser = argparse.ArgumentParser(description='Noisy Video Dataset Preprocessor')
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file.')
    parser.add_argument('--model_device', type=str, default='cuda:0')

    args = parser.parse_args()

    # Load parameters from the JSON file
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    config = SimpleNamespace(**config_dict)
    config.model_device = args.model_device

    preprocessor = NoisyVideoPreprocessor(config)
    preprocessor.preprocess_test_split()
    
    print("\nAll noisy preprocessing tasks have been completed!")
    print(f"Results are saved to: {preprocessor.base_output_dir}")


if __name__ == "__main__":
    main()
