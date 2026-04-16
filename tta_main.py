import torch
import numpy as np
import argparse
import os
import time
import logging
import datetime
import json
import sys
import random
from pathlib import Path
from tqdm import tqdm
from types import SimpleNamespace

from datasets.data_factory import get_tta_dataset, get_tta_dataloader
from model.model_factory import get_tta_basemodel, load_checkpoint_for_tta_basemodel
from model.model_factory import freeze_parameters_using_ln, collect_parameters_with_ln
from tta_model.set_model import set_tta_optimizer, set_tta_model

all_v2t_perturbation_type = ["gaussian", "impulse", "fog", "snow", "elastic_distortion", "h264_compression",
                             "motion_blur", "video_defocus", "main_object_occlusion",
                             "style_transfer", "event_insertion", "temporal_scrambling", 
                             "zstta", "transdata"]

all_t2v_perturbation_type = ["ocr", "char_insert", "char_replace","char_swap","char_delete", 
                             "synonym_replace", "word_insert", "word_swap", "word_delete", "insert_punctuation", 
                             "back_translation","formal","casual","passive","active",
                             "zstta", "transdata"]

def setup_tta_logger(config, log_level=logging.INFO):
    output_dir = config.output_dir
    dataset_name = config.dataset_name.lower()
    base_model = config.base_model.lower()
    tta_method = config.tta_method.lower()
    retrieval_type = config.retrieval_type.lower()
    noise_type = config.noise_types[0]
    severity = config.severity_list[0]
    exp_dir = '_'.join([dataset_name, base_model, retrieval_type])
    log_dir = os.path.join(output_dir, exp_dir)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f'{tta_method}_' + f'{noise_type}_' + f'{str(severity)}_' + f'{time.strftime("%Y%m%d_%H%M%S")}.log')

    logger = logging.getLogger()
    logger.setLevel(log_level)

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(log_level)
    
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG) 
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    
    return logger

def check_noise_types(config):
    retrieval_type = config.retrieval_type
    noise_types = config.noise_types
    severity_list = config.severity_list
    assert len(noise_types) == len(severity_list), "Length of noise_types and severity must match."
    assert len(noise_types) == 1, "Only support one perturbation at a time for now."
    if noise_types[0] in ["zstta","transdata"]:
        assert severity_list[0] == 1, f"Severity for {noise_types[0]} must be 1."
    if retrieval_type == 'v2t':
        for cur_severity in severity_list:
            assert 1 <= cur_severity <= 5, "Severity must be between 1 and 5."
        for cur_noise_type in noise_types:
            assert cur_noise_type in all_v2t_perturbation_type, f"Noise type {cur_noise_type} is not supported for v2t retrieval."
    elif retrieval_type == 't2v':
        for i, cur_noise_type in enumerate(noise_types):
            assert cur_noise_type in all_t2v_perturbation_type, f"Noise type {cur_noise_type} is not supported for t2v retrieval."
            if cur_noise_type in ["back_translation","formal","casual","passive","active"]:
                assert severity_list[i] == 1, f"Severity for {cur_noise_type} must be 1."
            else:
                assert 1 <= severity_list[i] <= 7, f"Severity for {cur_noise_type} must be between 1 and 7."
    else:
        raise NotImplementedError("Only v2t and t2v retrieval is supported for now.")

def compute_gallery_embeds(model, gallery_dataloader, retrieval_type, tokenizer, device, config):
    gallery_embeded_arr = []
    vid_in_gallery = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(gallery_dataloader), total=len(gallery_dataloader), desc="Computing gallery embeds", leave=False):
            if retrieval_type == 'v2t':
                if config.base_model == 'languagebind':
                    raise NotImplementedError("LanguageBind model is not supported for TTA yet.")
                elif config.base_model in ['clip4clip', 'xpool']:
                    data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(device)
                else:
                    data['text'] = {key: val.to(device) for key, val in data['text'].items()}
                text_features = model(data, return_text_only=True)
                gallery_embeded_arr.append(text_features)
                vid_in_gallery.extend(data['video_id'])
            elif retrieval_type == 't2v':
                data['video'] = data['video'].to(device)
                video_features = model(data, return_video_only=True)
                gallery_embeded_arr.append(video_features)
                vid_in_gallery.extend(data['video_id'])
    gallery_embeds = torch.cat(gallery_embeded_arr, dim=0).detach()
    return gallery_embeds, vid_in_gallery

def set_gallery_embeds(tta_model, retrieval_type, gallery_embeds):
    if retrieval_type == 'v2t':
        tta_model.set_text_features(gallery_embeds)
    elif retrieval_type == 't2v':
        tta_model.set_video_features(gallery_embeds)
    else:
        raise NotImplementedError("Retrieval type must be either 'v2t' or 't2v'.")

def calculate_evaluation_metrics(all_sims, vid_in_gallery, vid_in_query, retrieval_type, logger):
    if retrieval_type == 'v2t':
        logger.info("Test-time adaptaion results for v2t retrieval...")
        sims_v2t = all_sims
        sorted_text_indices = torch.argsort(sims_v2t, dim=1, descending=True)
        all_vids = sorted(list(set(vid_in_gallery + vid_in_query)))
        vid_to_int = {vid: i for i, vid in enumerate(all_vids)}
        
        q_vids_int = torch.tensor([vid_to_int[v] for v in vid_in_query], device=all_sims.device)
        g_vids_int = torch.tensor([vid_to_int[v] for v in vid_in_gallery], device=all_sims.device)
        
        gt_mask = q_vids_int.unsqueeze(1) == g_vids_int.unsqueeze(0)
        
        gathered_gts = torch.gather(gt_mask, 1, sorted_text_indices)
        
        ranks = torch.argmax(gathered_gts.int(), dim=1) + 1
        ranks = ranks.float()
        num_queries = len(vid_in_query)
    elif retrieval_type == 't2v':
        logger.info("Test-time adaptaion results for t2v retrieval...")
        gallery_vids = np.array(vid_in_gallery)
        query_vids = np.array(vid_in_query)
        sorted_indices = torch.argsort(all_sims, dim=1, descending=True)
        gallery_vid_to_idx = {vid: i for i, vid in enumerate(gallery_vids)}
        gt_indices_list = [gallery_vid_to_idx[vid] for vid in query_vids]
        gt_indices = torch.tensor(gt_indices_list, device=all_sims.device)
        ranks = (sorted_indices == gt_indices.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
        ranks = ranks.float()
        num_queries = len(query_vids)
    else:
        raise NotImplementedError("Retrieval type must be either 'v2t' or 't2v'.")
    metrics = {}
    metrics["R1"] = (ranks <= 1).sum().item() / num_queries * 100
    metrics["R5"] = (ranks <= 5).sum().item() / num_queries * 100
    metrics["R10"] = (ranks <= 10).sum().item() / num_queries * 100
    metrics["R50"] = (ranks <= 50).sum().item() / num_queries * 100
    metrics["R100"] = (ranks <= 100).sum().item() / num_queries * 100
    metrics["MedR"] = ranks.median().item()
    metrics["MeanR"] = ranks.mean().item()
    logger.info("="*80)
    logger.info(f"{retrieval_type}")
    logger.info(f"R@1: {metrics['R1']:.2f}")
    logger.info(f"R@5: {metrics['R5']:.2f}")
    logger.info(f"R@10: {metrics['R10']:.2f}")
    logger.info(f"MedR: {metrics['MedR']:.2f}")
    logger.info(f"MeanR: {metrics['MeanR']:.2f}")
    return metrics

def print_all_res(all_res, logger):
    logger.info("="*80)
    logger.info("Final All Results")
    for cur_res in all_res:
        perturbation = cur_res['perturbation']
        severity = cur_res['severity']
        cur_metrics = cur_res['res']
        logger.info(f"Perturbation: {perturbation}, Severity: {severity}, TTA_Method: {cur_res['tta_method']}, RetrievalType: {cur_res['retrieval_type']}")
        logger.info(f"R@1: {cur_metrics['R1']:.2f}, R@5: {cur_metrics['R5']:.2f}, R@10: {cur_metrics['R10']:.2f}, MedR: {cur_metrics['MedR']:.2f}, MeanR: {cur_metrics['MeanR']:.2f}")

def test_time_tune(tta_model, query_dataloader, gallery_dataloader, tokenizer, retrieval_type, config, logger):
    device = config.device
    tta_method = config.tta_method
    
    tta_model.eval()
    gallery_embeds, vid_in_gallery = compute_gallery_embeds(tta_model.model, gallery_dataloader, retrieval_type, tokenizer, device, config)
    set_gallery_embeds(tta_model, retrieval_type, gallery_embeds)

    batch_size = config.batch_size
    vid_in_query = []
    all_sims = []
    with torch.no_grad():
        for i, query_data in enumerate(query_dataloader):
            vid_in_query.extend(query_data['video_id'])
            sims_matrix = tta_model(query_data, i)
            cur_sim = sims_matrix.cpu()
            all_sims.append(cur_sim)
    all_sims = torch.cat(all_sims, dim=0)
    metrics = calculate_evaluation_metrics(all_sims, vid_in_gallery, vid_in_query, retrieval_type, logger)
    return metrics

def do_qgs_tta(perturbation_type, severity, tta_model, tokenizer, retrieval_type, config, logger):
    assert perturbation_type in ["zstta","transdata"]
    logger.info("-"*50)
    logger.info("Starting QGS-based Test-time Adaptation with perturbations...")
    # build target dataset
    video_dataset, text_dataset = get_tta_dataset(config, split_type='test')
    if retrieval_type == 'v2t':
        gallery_dataset = text_dataset
        query_dataset = video_dataset
    else:
        gallery_dataset = video_dataset
        query_dataset = text_dataset
    query_dataloader, gallery_dataloader = get_tta_dataloader(config, query_dataset, gallery_dataset)
    total_inter = len(query_dataloader)
    tta_model.reset()
    if hasattr(tta_model, "loss_tracker") and tta_model.loss_tracker is not None:
        tta_model.loss_tracker.set_run_context(
            dataset_name=config.dataset_name,
            base_model=config.base_model,
            retrieval_type=config.retrieval_type,
            perturbation=perturbation_type,
            severity=severity,
            output_dir=config.output_dir,
            total_inter=total_inter
        )
    metrics = test_time_tune(tta_model, query_dataloader, gallery_dataloader, tokenizer, retrieval_type, config, logger)
    
    
def do_tta_task(perturbation_types, severity_list, tta_model, tokenizer, retrieval_type, config, logger):
    logger.info("Starting Test-time Adaptation with perturbations...")
    logger.info(f"number of perturbations: {len(perturbation_types)}")
    all_res = []
    for i, cur_perturbation in enumerate(perturbation_types):
        cur_severity = severity_list[i]
        logger.info("-"*50)
        logger.info(f"Applying perturbation {cur_perturbation} with severity {cur_severity}...")
        logger.info(f"in {i+1} of {len(perturbation_types)} ...")
        
        video_dataset, text_dataset = get_tta_dataset(config, split_type='test')
        if retrieval_type == 'v2t':
            video_dir = os.path.join(config.preprocess_dir, cur_perturbation + f"_{cur_severity}")
            video_dataset.set_preprocess_dir(video_dir)
            query_dataset = video_dataset
            gallery_dataset = text_dataset
        elif retrieval_type == 't2v':
            text_file_dir = os.path.join(config.noised_text_dir, cur_perturbation + f"_{cur_severity}.json")
            text_dataset.set_new_text_anno(text_file_dir)
            query_dataset = text_dataset
            gallery_dataset = video_dataset
            # raise NotImplementedError("T2V retrieval is not implemented yet for TTA.")
        query_dataloader, gallery_dataloader = get_tta_dataloader(config, query_dataset, gallery_dataset)
        total_inter = len(query_dataloader)
        tta_model.reset()
        if hasattr(tta_model, "loss_tracker") and tta_model.loss_tracker is not None:
            tta_model.loss_tracker.set_run_context(
                dataset_name=config.dataset_name,
                base_model=config.base_model,
                retrieval_type=config.retrieval_type,
                perturbation=cur_perturbation,
                severity=cur_severity,
                output_dir=config.output_dir,
                total_inter=total_inter
            )
        
        metrics = test_time_tune(tta_model, query_dataloader, gallery_dataloader, tokenizer, retrieval_type, config, logger)
        cur_res = {'perturbation': cur_perturbation, 'severity': cur_severity, 'retrieval_type': retrieval_type, 'tta_method': config.tta_method, 'res': metrics}
        all_res.append(cur_res)
        
    if getattr(config, "loss_dump", False) and hasattr(tta_model, "loss_tracker"):
        logger.info("="*80)
        csv_path = tta_model.loss_tracker.dump_csv(filename=f"{config.tta_method}_losses.csv")
        logger.info(f"Loss records saved to: {csv_path}")
    # print_all_res(all_res, logger)

def main(config):
    assert config.retrieval_type in ['v2t', 't2v'], "Retrieval type must be either 'v2t' or 't2v'."
    assert config.base_model in ['clip4clip', 'xpool', 'languagebind'], "Base model must be either 'clip4clip', 'xpool' or 'languagebind'."
    
    if config.retrieval_type == 'v2t' and config.noise_types[0] not in ["zstta","transdata"]:
        assert config.use_preprocessed==True, "Only support preprocessed data in v2t."
        config.preprocess_dir = config.videos_dir = config.noised_video_dir
    check_noise_types(config)

    logger = setup_tta_logger(config)
    dataset_name = config.dataset_name
    retrieval_type = config.retrieval_type
    base_model = config.base_model
    clip_arch = config.clip_arch
    tta_method = config.tta_method
    logger.info("="*80)
    logger.info("Starting Test-time Adaptation...")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Retrieval Type: {retrieval_type}")
    logger.info(f"Base Model: {base_model}")
    logger.info(f"CLIP Architecture: {clip_arch}")
    logger.info(f"TTA Method: {tta_method}")
    logger.info("-"*50)
    
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if config.base_model == "languagebind":
        raise NotImplementedError("LanguageBind model is not supported for TTA yet.")
    elif config.huggingface:
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
    else:
        raise NotImplementedError("Only HuggingFace models are supported.")

    logger.info(f"Loading base model, the base model is {base_model}")
    
    base_model = get_tta_basemodel(base_model, config)
    if config.noise_types[0] not in ["zstta"]:
        checkpoint_path = config.checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        base_model, msg = load_checkpoint_for_tta_basemodel(base_model, checkpoint_path)
        if msg:
            logger.info(f"Checkpoint loading message: \n{msg}")
    
    base_model = freeze_parameters_using_ln(base_model, retrieval_type=retrieval_type)
    # base_model = base_model.to(config.device)
    optimizing_params, _ = collect_parameters_with_ln(base_model, retrieval_type=retrieval_type)
    optimizer = set_tta_optimizer(optimizing_params, tta_method, config)
    logger.info(f"Optimizer set up with {len(optimizing_params)} parameters.")

    tta_model = set_tta_model(base_model, tokenizer, optimizer, tta_method, config)
    tta_model.model = tta_model.model.to(config.device)
    if config.noise_types[0] in ["zstta","transdata"]:
        do_qgs_tta(config.noise_types[0], config.severity_list[0], tta_model, tokenizer, retrieval_type, config, logger)
    else:
        do_tta_task(config.noise_types, config.severity_list, tta_model, tokenizer, retrieval_type, config, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test-time Adaptation for Video-Text Retrieval')
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file.')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='tta_outputs')
    parser.add_argument('--noise_types', type=str, required=True)
    parser.add_argument('--severity', type=int, required=True)
    parser.add_argument('--dataset_name', type=str,  default=None)
    parser.add_argument('--noised_video_dir', type=str,  default=None)
    parser.add_argument('--noised_text_dir', type=str,  default=None)
    parser.add_argument('--videos_dir', type=str,  default=None)
    parser.add_argument('--preprocess_dir', type=str,  default=None)
    parser.add_argument('--checkpoint_path', type=str,  default=None)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    config['device'] = args.device
    config['output_dir'] = args.output_dir
    config = SimpleNamespace(**config)
    
    if args.noise_types not in ['transdata', 'zstta']:
        if args.dataset_name is not None:
            assert args.checkpoint_path is not None, "Checkpoint path must be provided."
            config.checkpoint_path = args.checkpoint_path
            if config.retrieval_type == 'v2t':
                assert args.noised_video_dir is not None, "Noised video directory must be provided for v2t retrieval."
                config.noised_video_dir = args.noised_video_dir
            elif config.retrieval_type == 't2v':
                assert args.noised_text_dir is not None, "Noised text directory must be provided for t2v retrieval."
                assert args.videos_dir is not None, "Videos directory must be provided for t2v retrieval."
                config.noised_text_dir = args.noised_text_dir
                config.videos_dir = args.videos_dir
                if args.preprocess_dir is not None:
                    config.preprocess_dir = args.preprocess_dir
                    config.use_preprocessed = True
                else:
                    config.use_preprocessed = False
            config.dataset_name = args.dataset_name
    elif args.noise_types == 'transdata':
        assert args.checkpoint_path is not None, "Checkpoint path must be provided for transdata."
        assert args.dataset_name is not None, "Dataset name must be provided for transdata."
        assert args.preprocess_dir is not None, "Preprocess directory must be provided for transdata."
        config.checkpoint_path = args.checkpoint_path
        config.dataset_name = args.dataset_name
        config.preprocess_dir = args.preprocess_dir
        config.videos_dir = args.preprocess_dir
        config.use_preprocessed = True
    elif args.noise_types == 'zstta':
        config.use_preprocessed = True
        config.preprocess_dir = args.preprocess_dir
        config.videos_dir = args.preprocess_dir
        config.dataset_name = args.dataset_name
    config.noise_types = [args.noise_types]
    config.severity_list = [args.severity]
    main(config)
