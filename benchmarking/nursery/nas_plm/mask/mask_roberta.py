from .mask_bert import mask_bert


def mask_roberta(model, neuron_mask, head_mask):
    return mask_bert(model, neuron_mask, head_mask)
