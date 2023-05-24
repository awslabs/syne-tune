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
from collections import OrderedDict
import torch.nn as nn
from copy import deepcopy



class IdentityAttention(nn.Module):
    def __init__(self, model_type=None):
        self.model_type = model_type
        super(IdentityAttention, self).__init__()

    def forward(self, *input, **kwargs):
        if self.model_type == "distilbert-base-cased":
            return (kwargs["x"],)
        else:
            return input


class IdentityFFN(nn.Module):
    def __init__(self, model_type=None):
        self.model_type = model_type
        super(IdentityFFN, self).__init__()

    def forward(self, hidden_states):
        return hidden_states


class IdentityOutput(nn.Module):
    def __init__(self, model_type=None):
        self.model_type = model_type
        super(IdentityOutput, self).__init__()

    def forward(self, hidden_states, input_tensor):
        return hidden_states


def get_final_model(original_model, architecture_definition):

    new_model = deepcopy(original_model)

    for i in range(new_model.config.num_hidden_layers):
        if architecture_definition[f"layer_mha_{i}"] == 1:
            model.bert.encoder.layer[i].attention = IdentityAttention()
        if architecture_definition[f"layer_ffn_{i}"] == 1:
            model.bert.encoder.layer[i].intermediate = IdentityFFN()
            model.bert.encoder.layer[i].output = IdentityOutput()

    return new_model


def copy_linear_layer(new_layer, old_layer, weight_shape, bias_shape):
    old_state = old_layer.state_dict()
    new_state_dict = OrderedDict()
    print(weight_shape, old_state["weight"].shape)
    new_state_dict["weight"] = old_state["weight"][:weight_shape[0], :weight_shape[1]]
    new_state_dict["bias"] = old_state["bias"][: bias_shape]
    new_layer.load_state_dict(new_state_dict)

def get_final_bert_model(original_model, new_model_config):

    new_model = AutoModelForSequenceClassification.from_config(new_model_config)

    new_model.bert.embeddings = model.bert.embeddings
    new_model.bert.pooler = model.bert.pooler
    new_model.classifier = model.classifier

    num_attention_heads = config.num_attention_heads
    attention_head_size = int(config.hidden_size / original_model.config.num_attention_heads)
    all_head_size = num_attention_heads * attention_head_size
    for li, layer in enumerate(new_model.bert.encoder.layer):

        attention = layer.attention.self
        attention.query = nn.Linear(config.hidden_size, all_head_size)
        attention.key = nn.Linear(config.hidden_size, all_head_size)
        attention.value = nn.Linear(config.hidden_size, all_head_size)
        attention.all_head_size = all_head_size
        attention.attention_head_size = attention_head_size

        mha_original_model = model.bert.encoder.layer[li].attention.self
        copy_linear_layer(attention.query, mha_original_model.query,
                          (all_head_size, config.hidden_size), (all_head_size))

        copy_linear_layer(attention.key, mha_original_model.key,
                          (all_head_size, config.hidden_size), (all_head_size))

        copy_linear_layer(attention.value, mha_original_model.value,
                          (all_head_size, config.hidden_size), (all_head_size))

        ffn_layer = layer.intermediate.dense
        ffn_original_model = model.bert.encoder.layer[li].intermediate.dense
        copy_linear_layer(ffn_layer, ffn_original_model,
                          (config.intermediate_size, config.hidden_size), (config.intermediate_size))

    return new_model


if __name__ == "__main__":
    from transformers import AutoModelForSequenceClassification

    model_type = "bert-base-cased"
    model = AutoModelForSequenceClassification.from_pretrained(model_type)

    from transformers.models.bert.modeling_bert import BertConfig

    config = BertConfig(
        num_hidden_layers=6, num_attention_heads=6, intermediate_size=1024
    )
    new_model = get_final_bert_model(original_model=model, new_model_config=config)