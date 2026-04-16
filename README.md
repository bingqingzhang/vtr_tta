# Robust Test-time Video-Text Retrieval: Benchmarking and Adapting for Query Shifts

[![Paper](https://img.shields.io/badge/ICLR%202026-Paper-blue)](https://openreview.net/forum?id=FRkJ3ehpNN)

> Modern video-text retrieval (VTR) models excel on in-distribution benchmarks but are highly vulnerable to real-world *query shifts*—distribution deviations between training and test queries that cause sharp performance degradation. Existing image-focused robustness solutions fall short for video, as they cannot handle the complex spatio-temporal dynamics inherent in such shifts.
>
> We address this with two contributions: **(1)** a comprehensive **MLVP Benchmark** featuring 12 video and 15 text perturbation types across five severity levels, and **(2)** **HAT-VTR** (Hubness Alleviation for Test-time Video-Text Retrieval), a test-time adaptation framework that directly counteracts the *hubness phenomenon*—where a few gallery items dominate retrieval—through a *Hubness Suppression Memory* and *multi-granular temporal losses*. Extensive experiments show that HAT-VTR consistently outperforms prior methods across diverse query shift scenarios.

## 🔥 Highlights

- 📊 **MLVP Benchmark** — 12 video perturbations + 15 text perturbations at 5 severity levels, evaluated on 4 major VTR datasets (MSRVTT, MSVD, ActivityNet, LSMDC).
- 🧠 **HAT-VTR Framework** — A novel TTA method combining Hubness Suppression Memory and multi-granular temporal losses to improve robustness without retraining.
- 🔁 **Comprehensive Baselines** — Implementations of 5 TTA baselines (Tent, SAR, EATA, TCR, READ) for comparison.
- 🏗️ **Multi-Model Support** — Supports CLIP4Clip and X-Pool as backbone VTR models for both video-to-text and text-to-video retrieval.

---

## 📜 Dataset Preparation

We provide dataset annotation files under `data/`. To download the raw videos, please refer to the [CLIP4Clip repository](https://github.com/ArrowLuo/CLIP4Clip).

---

## ⚙️ Environment Setup

This project requires **two separate Conda environments** for the benchmark construction and the TTA framework, respectively.

### MLVP Benchmark Environment

The MLVP benchmark runs on **Python 3.12** + **PyTorch 2.6.0**.

```bash
# Create and activate the environment
conda create -n mlvp python=3.12 -y
conda activate mlvp

# Install PyTorch 2.6.0 (visit pytorch.org for the correct command for your system)
# Example:
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r caption_requirements.txt
```

### TTA Framework Environment

The TTA framework runs on **Python 3.9** + **PyTorch 2.1.0**.

```bash
# Create and activate the environment
conda create -n tta python=3.9 -y
conda activate tta

# Install PyTorch 2.1.0 (visit pytorch.org for the correct command for your system)
# Example:
conda install pytorch==2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt
```

---

## 📊 Part I: MLVP Benchmark Construction

### Overview

The **Multi-Level Video Perturbation (MLVP)** benchmark is designed to systematically evaluate VTR model robustness under realistic query shifts. It applies controlled perturbations to either the video or text modality and measures the resulting retrieval performance degradation, revealing vulnerabilities that standard benchmarks overlook.

### Perturbation Types

#### Video Perturbations (12 types)

| Level | Perturbation | Severity Range |
|-------|-------------|----------------|
| **Low-Level** | Gaussian Noise | 1–5 |
| | Impulse Noise | 1–5 |
| | Fog | 1–5 |
| | Snow | 1–5 |
| | Elastic Distortion | 1–5 |
| | H.264 Compression | 1–5 |
| **Mid-Level** | Motion Blur | 1–5 |
| | Video Defocus | 1–5 |
| | Main Object Occlusion | 1–5 |
| **High-Level** | Style Transfer | 1–5 |
| | Event Insertion | 1–5 |
| | Temporal Scrambling | 1–5 |

#### Text Perturbations (15 types)

| Level | Perturbation | Severity Range |
|-------|-------------|----------------|
| **Character** | OCR Error | 1–7 |
| | Char Insert | 1–7 |
| | Char Replace | 1–7 |
| | Char Swap | 1–7 |
| | Char Delete | 1–7 |
| **Word** | Synonym Replace | 1–7 |
| | Word Insert | 1–7 |
| | Word Swap | 1–7 |
| | Word Delete | 1–7 |
| | Insert Punctuation | 1–7 |
| **Sentence** | Back Translation | 1 |
| | Formal Style | 1 |
| | Casual Style | 1 |
| | Passive Voice | 1 |
| | Active Voice | 1 |

### Prerequisites

Some perturbation types require additional resources:

- **Style Transfer** — Download the VGG and decoder models from [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) and place both files in the `cache_dir/` directory. The style images come from the [Kaggle Painter-by-Numbers](https://www.kaggle.com/c/painter-by-numbers/data) dataset (`train.zip`).
- **Event Insertion** — Requires a supplementary video pool. In our experiments, we randomly sample 2K videos from the MSRVTT training set. A pre-built pool is available for download at [base_videodata](https://drive.google.com/drive/folders/1Sf37yHYLmGbd6VTp-b3R2qW76jlVmqdc?usp=sharing).
- **Main Object Occlusion & Event Insertion** — Both require pre-computed video metadata. Generate it with:

```bash
PYTHONPATH=/path/to/tta_vtr/ python preprocess/get_main_objects.py \
  --dataset_name MSRVTT \
  --videos_dir /path/to/MSRVTT/MSRVTT_Videos
```

### Generate Video Perturbations

Configure perturbation types, severity levels, and paths in a JSON config file (see `config/msrvtt_video_pertrubation.json` for an example), then run:

```bash
python do_video_perturbation.py --config config/msrvtt_video_pertrubation.json
```

### Generate Text Perturbations

Similarly, configure and run text perturbations (see `config/msrvtt_text_pertrubation.json`):

```bash
python do_text_perturbation.py --config config/msrvtt_text_pertrubation.json
```

---

## 🎯 Part II: Test-Time Adaptation (HAT-VTR)

### VTR Model Preparation

You can either **train a model from scratch** or **download pre-trained checkpoints**.

**Train from scratch** (e.g., CLIP4Clip on MSRVTT):

```bash
python train.py \
  --exp_name=clip4clip_meanP \
  --videos_dir=/path/to/MSRVTT/MSRVTT_Videos \
  --arch=clip_baseline \
  --pooling_type=avg \
  --huggingface \
  --dataset_name=MSRVTT \
  --msrvtt_train_file=9k \
  --batch_size=32 \
  --noclip_lr=3e-5 \
  --transformer_dropout=0.3
```

**Download pre-trained checkpoints** from the official [X-Pool repository](https://github.com/layer6ai-labs/xpool.git).

### Video Preprocessing

Preprocess videos into `.pt` tensor files. This step is required for text perturbation-based TTA and also accelerates data loading for video perturbation experiments.

```bash
python preprocess_videos.py \
  --exp_name dataprocess \
  --dataset_name MSRVTT \
  --msrvtt_train_file=9k \
  --videos_dir=/path/to/MSRVTT/MSRVTT_Videos \
  --preprocess_dir=/path/to/MSRVTT/preprocess_videos
```

### Running HAT-VTR

Configure the TTA method, backbone model, and perturbation settings in a JSON config file (see `config_tta/` for examples), then run:

```bash
python tta_main.py \
  --config config_tta/msrvtt/v2t/clip4clip_vtr.json \
  --noise_types gaussian \
  --severity 5
```

The `--noise_types` flag specifies the perturbation type to evaluate, and `--severity` sets the perturbation intensity. Multiple perturbation types can be evaluated in a single run by listing them (e.g., `--noise_types gaussian impulse fog`).

---

## 📁 Project Structure

```
tta_vtr/
├── config/                      # Perturbation generation configs
├── config_tta/                  # TTA experiment configs (per dataset/model/method)
│   └── msrvtt/{v2t,t2v}/       #   e.g., clip4clip_vtr.json, xpool_sar.json
├── data/                        # Dataset annotation files
├── datasets/                    # Dataset loading & transforms
├── model/                       # VTR backbone models (CLIP4Clip, X-Pool)
├── modules/                     # Shared utilities (metrics, loss, optimization)
├── preprocess/                  # Video metadata extraction tools
├── tta_model/                   # TTA method implementations
│   ├── hat_vtr.py               #   HAT-VTR (proposed)
│   ├── tent.py                  #   Tent baseline
│   ├── sar.py                   #   SAR baseline
│   ├── eata.py                  #   EATA baseline
│   ├── tcr.py                   #   TCR baseline
│   └── read.py                  #   READ baseline
├── trainer/                     # Training loop
├── do_video_perturbation.py     # Video perturbation generation
├── do_text_perturbation.py      # Text perturbation generation
├── preprocess_videos.py         # Video-to-tensor preprocessing
├── train.py                     # VTR model training
├── test.py                      # VTR model evaluation
└── tta_main.py                  # TTA experiment entry point
```

---

## 📖 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{
  zhang2026robust,
  title={Robust Test-time Video-Text Retrieval: Benchmarking and Adapting for Query Shifts},
  author={Bingqing Zhang and Zhuo Cao and Heming Du and Yang Li and Xue Li and Jiajun Liu and Sen Wang},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=FRkJ3ehpNN}
}
```

## 🙏 Acknowledgement

This codebase builds upon the following projects:

- [X-Pool](https://github.com/layer6ai-labs/xpool) — Video-text retrieval framework
- [MM-Robustness](https://github.com/Jielin-Qiu/MM_Robustness) — Multimodal robustness benchmark
- [TCR-TTA](https://github.com/XLearning-SCU/2025-ICLR-TCR) — Test-time adaptation baseline