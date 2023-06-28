# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from .utils import get_backbone, register_drop_attention_layer


def get_ffn2(model, index):
    layer = get_layers(model)[index]
    ffn2 = layer.mlp.c_proj
    return ffn2


def register_drop_layer(module):
    hook = lambda _, inputs, output: (inputs[0], output[1:])
    handle = module.register_forward_hook(hook)
    return handle


def get_mlp(model, index):
    layer = get_layers(model)[index]
    return layer.mlp


def get_attention_output(model, index):
    layer = get_layers(model)[index]
    output = layer.attn
    return output


def get_layers(model):
    decoder = get_backbone(model)
    layers = decoder.h
    return layers


def register_mask(module, mask):
    hook = lambda _, inputs: (inputs[0] * mask)
    handle = module.register_forward_pre_hook(hook)
    return handle


def register_drop_mlp_layer(module):
    hook = lambda _, input, output: input[0]
    handle = module.register_forward_hook(hook)
    return handle


def mask_gpt(model, neuron_mask, head_mask):
    num_hidden_layers = neuron_mask.shape[0]

    assert head_mask.shape[0] == num_hidden_layers

    handles = []
    for layer_idx in range(num_hidden_layers):
        ffn2 = get_ffn2(model, layer_idx)
        handle = register_mask(ffn2, neuron_mask[layer_idx])
        handles.append(handle)

        if neuron_mask[layer_idx].sum() == 0 and head_mask[layer_idx].sum() == 0:
            layer = get_layers(model)[layer_idx]
            handle = register_drop_layer(layer)
            handles.append(handle)

        elif neuron_mask[layer_idx].sum() == 0:
            mlp = get_mlp(model, layer_idx)
            handle = register_drop_layer(mlp)
            handles.append(handle)

        elif head_mask[layer_idx].sum() == 0:
            attention = get_attention_output(model, layer_idx)
            handle = register_drop_attention_layer(attention)
            handles.append(handle)

    return handles
