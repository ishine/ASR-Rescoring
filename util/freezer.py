from collections.abc import Iterable
from torch.nn import Module

def set_freeze_by_name(model: Module, layer_names, freeze=True):
    if isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze

def freeze_by_name(model: Module, layer_names):
    set_freeze_by_name(model, layer_names, True)

def unfreeze_by_name(model: Module, layer_names):
    set_freeze_by_name(model, layer_names, False)