import os
import re
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import argparse
import json
import warnings
from types import SimpleNamespace
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from styleformer import Styleformer
import nltk
from preprocess.eda import synonym_replacement, random_deletion, random_swap, random_insertion

warnings.filterwarnings("ignore")
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading wordnet data for NLTK...")
    nltk.download('wordnet')
    print("Download complete.")
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("Downloading averaged_perceptron_tagger for NLTK...")
    nltk.download('averaged_perceptron_tagger')
    print("Download complete.")
    


def extract_device_number(s: str) -> int:
    match = re.match(r"cuda:(\d+)", s.strip())
    if match:
        return int(match.group(1))
    return -1

PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
def insert_punctuation_marks(sentence, punc_ratio=0.1):
    words = sentence.split(' ')
    new_line = []
    q = max(1, int(punc_ratio * len(words)))
    if not words or len(words) == 0:
        return sentence
    qs = random.sample(range(0, len(words)), min(q, len(words)))

    for j, word in enumerate(words):
        if j in qs:
            new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
            new_line.append(word)
        else:
            new_line.append(word)
    return ' '.join(new_line)

class TextPerturbationProcessor:
    def __init__(self, config):
        print("Initializing TextPerturbationProcessor...")
        self.config = config
        self.device = config.model_device
        
        assert len(self.config.severity) == len(self.config.noise_types), \
            "Each noise type must have a corresponding severity level."

        assert self.config.noised_text_dir is not None, "Please specify the noised text directory."
        self.base_output_dir = self.config.noised_text_dir
        os.makedirs(self.base_output_dir, exist_ok=True)

        self.similarity_model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')
        
        print("Loading BackTranslation model...")
        self.back_translation_aug = naw.BackTranslationAug(
            from_model_name='facebook/wmt19-en-de', 
            to_model_name='facebook/wmt19-de-en',
            device=self.device
        )

        print("Loading Styleformer model...")
        self.style_former_formal = Styleformer(style=0)
        self.style_former_casual = Styleformer(style=1)
        self.style_former_passive = Styleformer(style=2)
        self.style_former_active = Styleformer(style=3)
        
        print("Initialization complete. Processor is ready.")
    
    def get_dataset_texts(self, split_type='test'):
        assert split_type == 'test', "Currently, only 'test' split is supported for TTA."
        dataset_name = self.config.dataset_name
        print(f"Getting dataset texts from {dataset_name}...")
        processed_text_list = []
        vid_test_pairs = []
        if dataset_name == 'MSRVTT':
            from datasets.msrvtt_dataset import MSRVTTDataset
            dataset = MSRVTTDataset(self.config, split_type, None)
            for i in range(len(dataset.test_df)):
                cur_vid = dataset.test_df.iloc[i].video_id
                cur_text = dataset.test_df.iloc[i].sentence
                processed_text_list.append(cur_text)
                vid_test_pairs.append((cur_vid, cur_text))
        elif dataset_name == 'MSVD':
            from datasets.msvd_dataset import MSVDDataset
            dataset = MSVDDataset(self.config, split_type, None)
            for i in range(len(dataset.test_vids)):
                cur_vid = dataset.test_vids[i]
                caption = dataset.vid2caption[cur_vid][0]
                processed_text_list.append(caption)
                vid_test_pairs.append((cur_vid, caption))
        elif dataset_name == 'LSMDC':
            from datasets.lsmdc_dataset import LSMDCDataset
            dataset = LSMDCDataset(self.config, split_type, None)
            for i in range(len(dataset.clip2caption)):
                cur_vid = list(dataset.clip2caption.keys())[i]
                cur_text = dataset.clip2caption[cur_vid]
                processed_text_list.append(cur_text)
                vid_test_pairs.append((cur_vid, cur_text))
        elif dataset_name == 'ActivityNet':
            from datasets.activitynet_dataset import ActivityNetDataset
            dataset = ActivityNetDataset(self.config, split_type, None)
            for i in range(len(dataset.anno)):
                cur_vid = dataset.anno[i]['clip_id']
                cur_text = dataset.anno[i]['text']
                processed_text_list.append(cur_text)
                vid_test_pairs.append((cur_vid, cur_text))
        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported for text perturbation.")
        self.vid_test_pairs = vid_test_pairs
        print(f"Found {len(processed_text_list)} unique texts in {dataset_name} {split_type} split.")

    def _apply_and_check_with_retry(self, original_sentence, perturbation_func):
        last_perturbed_sentence = original_sentence

        base_embedding = self.similarity_model.encode(original_sentence)

        for _ in range(100):
            perturbed_sentence = perturbation_func()

            if not perturbed_sentence or not isinstance(perturbed_sentence, str):
                perturbed_sentence = original_sentence
            
            last_perturbed_sentence = perturbed_sentence

            perturbed_embedding = self.similarity_model.encode(perturbed_sentence)
            score = float(util.cos_sim(base_embedding, perturbed_embedding))
            if score >= 0.9:
                return perturbed_sentence
        return last_perturbed_sentence
    
    def do_perturb(self, sentence, perturbation_type, severity=-1):
        if not (1 <= severity <= 7):
            raise ValueError("Severity must be between 1 and 7.")

        ratio = 0.05 * severity

        # --- character level ---
        if perturbation_type == 'keyboard':
            aug = nac.KeyboardAug(aug_word_p=ratio)
            return aug.augment(sentence)[0]
        
        elif perturbation_type == 'ocr':
            aug = nac.OcrAug(aug_word_p=ratio)
            return aug.augment(sentence)[0]

        elif perturbation_type == 'char_insert':
            aug = nac.RandomCharAug(action="insert", aug_word_p=ratio)
            return aug.augment(sentence)[0]

        elif perturbation_type == 'char_replace':
            aug = nac.RandomCharAug(action="substitute", aug_word_p=ratio)
            return aug.augment(sentence)[0]

        elif perturbation_type == 'char_swap':
            aug = nac.RandomCharAug(action="swap", aug_word_p=ratio)
            return aug.augment(sentence)[0]

        elif perturbation_type == 'char_delete':
            aug = nac.RandomCharAug(action="delete", aug_word_p=ratio)
            return aug.augment(sentence)[0]

        # word level ---
        elif perturbation_type in ['synonym_replace', 'word_insert', 'word_swap', 'word_delete']:
            words = sentence.split()
            num_words = len(words)
            
            if num_words == 0:
                return sentence

            if perturbation_type == 'synonym_replace':
                n_sr = max(1, int(ratio * num_words))
                perturb_func = lambda: ' '.join(synonym_replacement(words, n_sr))
            
            elif perturbation_type == 'word_insert':
                n_ri = max(1, int(ratio * num_words))
                perturb_func = lambda: ' '.join(random_insertion(words, n_ri))

            elif perturbation_type == 'word_swap':
                n_rs = max(1, int(ratio * num_words))
                perturb_func = lambda: ' '.join(random_swap(words, n_rs))

            elif perturbation_type == 'word_delete':
                perturb_func = lambda: ' '.join(random_deletion(words, ratio))

            return self._apply_and_check_with_retry(sentence, perturb_func)

        elif perturbation_type == 'insert_punctuation':
            perturb_func = lambda: insert_punctuation_marks(sentence, punc_ratio=ratio)
            return self._apply_and_check_with_retry(sentence, perturb_func)

        # sentence
        elif perturbation_type == 'back_translation':
            assert severity == 1, "Sentence level perturbation requires severity 1."
            return self.back_translation_aug.augment(sentence)[0]

        elif perturbation_type == 'formal':
            assert severity == 1, "Sentence level perturbation requires severity 1."
            device_id = extract_device_number(self.device)
            perturb_func = lambda: self.style_former_formal.transfer(sentence, inference_on=device_id)
            return self._apply_and_check_with_retry(sentence, perturb_func)

        elif perturbation_type == 'casual':
            assert severity == 1, "Sentence level perturbation requires severity 1."
            device_id = extract_device_number(self.device)
            perturb_func = lambda: self.style_former_casual.transfer(sentence, inference_on=device_id)
            return self._apply_and_check_with_retry(sentence, perturb_func)

        elif perturbation_type == 'passive':
            assert severity == 1, "Sentence level perturbation requires severity 1."
            device_id = extract_device_number(self.device)
            perturb_func = lambda: self.style_former_passive.transfer(sentence, inference_on=device_id)
            return self._apply_and_check_with_retry(sentence, perturb_func)

        elif perturbation_type == 'active':
            assert severity == 1, "Sentence level perturbation requires severity 1."
            device_id = extract_device_number(self.device)
            perturb_func = lambda: self.style_former_active.transfer(sentence, inference_on=device_id)
            return self._apply_and_check_with_retry(sentence, perturb_func)

        else:
            raise ValueError(f"Unknown perturbation_type: '{perturbation_type}'")

    def process(self):
        print("-" * 50)
        print("Starting noisy preprocessing for the test set...")
        self.get_dataset_texts()
        total_perturbation = len(self.config.noise_types)
        for idx, cur_perturbation_type in enumerate(self.config.noise_types):
            cur_severity = self.config.severity[idx]
            output_file_path = os.path.join(self.base_output_dir, f"{cur_perturbation_type}_{cur_severity}.json")
            print(f"\nIn {idx+1}/{total_perturbation}, perturbation type: {cur_perturbation_type}, severity: {cur_severity}")
            num_all_pairs = len(self.vid_test_pairs)
            new_vid_pairs = []
            for i, vid_text_pair in tqdm(enumerate(self.vid_test_pairs), total=num_all_pairs, desc=f"Processing {cur_perturbation_type}"):
                vid, text = vid_text_pair
                perturbated_text = self.do_perturb(text, cur_perturbation_type, severity=cur_severity)
                new_vid_pairs.append((vid, perturbated_text))
            with open(output_file_path, 'w') as f:
                json.dump(new_vid_pairs, f)
            print(f"Finished processing {cur_perturbation_type} with severity {cur_severity}. Output saved to {output_file_path}.")
        print("Noisy preprocessing completed for the test set.")

def main():
    parser = argparse.ArgumentParser(description='Text Perturbation Dataset Preprocessor')
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file.')
    parser.add_argument('--model_device', type=str, default='cuda:0')
    
    args = parser.parse_args()

    # Load parameters from the JSON file
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    config = SimpleNamespace(**config_dict)
    config.model_device = args.model_device
    processor = TextPerturbationProcessor(config)
    processor.process()

if __name__ == '__main__':
    main()