import torch


def get_backbone(model):
    model_type = model.base_model_prefix
    backbone = getattr(model, model_type)
    return backbone


def register_mask_ffn(module, mask):
    hook = lambda _, inputs: (inputs[0] * mask, inputs[1])
    handle = module.register_forward_pre_hook(hook)
    return handle


def register_drop_layer(module):
    hook = lambda _, input, output: input[1]
    handle = module.register_forward_hook(hook)
    return handle


def register_drop_attention_layer(module):
    hook = lambda _, input, output: input
    handle = module.register_forward_hook(hook)
    return handle
