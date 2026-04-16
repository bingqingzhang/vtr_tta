import torch

from tta_model.mock import Mock
from tta_model.sar import SAM, SAR
from tta_model.tent import Tent
from tta_model.tcr import TCR
from tta_model.eata import EATA
from tta_model.read import READ
from tta_model.hat_vtr import HATVTR

def set_tta_optimizer(params, tta_method, config):
    if tta_method =='sar':
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(params=params, base_optimizer=base_optimizer, lr=config.init_lr, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.AdamW(params=params, lr=config.init_lr, weight_decay=config.weight_decay)
    return optimizer

def set_tta_model(base_model, tokenizer, optimizer, tta_method, config):
    if tta_method == 'tent':
        tta_model = Tent(model=base_model, tokenizer=tokenizer, optimizer=optimizer, steps=config.tta_steps, config=config)
    elif tta_method == 'sar':
        tta_model = SAR(model=base_model, tokenizer=tokenizer, optimizer=optimizer, steps=config.tta_steps, margin_e0=0.40, config=config)
    elif tta_method == 'read':
        tta_model = READ(model=base_model, tokenizer=tokenizer, optimizer=optimizer, steps=config.tta_steps, config=config)
    elif tta_method == 'tcr':
        tta_model = TCR(model=base_model, tokenizer=tokenizer, optimizer=optimizer, steps=config.tta_steps, config=config)
    elif tta_method == 'eata':
        tta_model = EATA(model=base_model, tokenizer=tokenizer, optimizer=optimizer, steps=config.tta_steps, config=config, fishers=None, e_margin=0.40)
    elif tta_method == 'mock':
        tta_model = Mock(model=base_model, tokenizer=tokenizer, optimizer=optimizer, steps=config.tta_steps, config=config)
    elif tta_method == 'hatvtr':
        tta_model = HATVTR(model=base_model, tokenizer=tokenizer, optimizer=optimizer, steps=config.tta_steps, config=config)
    else:
        raise NotImplementedError(f"TTA method {tta_method} is not implemented.")
    
    return tta_model