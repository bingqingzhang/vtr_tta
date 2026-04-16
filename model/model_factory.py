from config.base_config import Config
from model.clip_baseline import CLIPBaseline
from model.clip_transformer import CLIPTransformer
import torch
import torch.nn as nn

class ModelFactory:
    @staticmethod
    def get_model(config: Config):
        if config.arch == 'clip_baseline':
            return CLIPBaseline(config)
        elif config.arch == 'clip_transformer':
            return CLIPTransformer(config)
        else:
            raise NotImplementedError
        
def get_tta_basemodel(base_model, config):
    if base_model == 'clip4clip':
        return CLIPBaseline(config)
    elif base_model == 'xpool':
        return CLIPTransformer(config)
    else:
        raise NotImplementedError(f"Base model {base_model} is not implemented for TTA.")
   
def load_checkpoint_for_tta_basemodel(model, checkpoint_path):
    """
    Load from a specific checkpoint path
    :param checkpoint_path: Path to the checkpoint file
    """
    print("Loading checkpoint: {} ...".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    msg = model.load_state_dict(state_dict, strict=False)
    
    return model, msg

def freeze_parameters_using_ln(model, retrieval_type='v2t'):
    assert retrieval_type in ['v2t', 't2v'], "Retrieval type must be either 'v2t' or 't2v'."
    model.train()
    model.requires_grad_(False)
    if retrieval_type=='v2t':
        for name, param in model.clip.vision_model.named_parameters():
            if ('norm' in name.lower()) or ('ln' in name.lower()):
                param.requires_grad_(True)
    else:
        for name, param in model.clip.text_model.named_parameters():
            if ('norm' in name.lower()) or ('ln' in name.lower()):
                param.requires_grad_(True)
    return model

def collect_parameters_with_ln(model, retrieval_type='v2t'):
    assert retrieval_type in ['v2t', 't2v'], "Retrieval type must be either 'v2t' or 't2v'."
    params = []
    names = []
    if retrieval_type == 'v2t':
        for name, module in model.clip.vision_model.named_modules():
            if isinstance(module, (nn.LayerNorm)):
                for np, p in module.named_parameters():
                    if np in ['weight','bias']:
                        params.append(p)
                        names.append(f'{name}.{np}')
    else:
        for name, module in model.clip.text_model.named_modules():
            if isinstance(module, (nn.LayerNorm)):
                for np, p in module.named_parameters():
                    if np in ['weight','bias']:
                        params.append(p)
                        names.append(f'{name}.{np}')
    return params, names
