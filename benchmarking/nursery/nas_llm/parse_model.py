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


def get_final_model(original_model, new_model_config):

    new_model = AutoModelForSequenceClassification.from_config(new_model_config)

    for name, module in new_model.bert.named_modules():
        if isinstance(module, nn.Linear):

            original_layer = original_model.bert.get_submodule(name).state_dict()

            weight_shape = module.state_dict()["weight"].shape
            bias_shape = module.state_dict()["bias"].shape

            new_state_dict = OrderedDict()
            new_state_dict["weight"] = original_layer["weight"][
                : weight_shape[0], : weight_shape[1]
            ]
            new_state_dict["bias"] = original_layer["bias"][: bias_shape[0]]
            module.load_state_dict(new_state_dict)

    new_model.bert.embeddings = model.bert.embeddings
    new_model.bert.pooler = model.bert.pooler
    new_model.classifier = model.classifier

    return new_model


if __name__ == "__main__":
    from transformers import AutoModelForSequenceClassification

    model_type = "bert-base-cased"
    model = AutoModelForSequenceClassification.from_pretrained(model_type)

    from transformers.models.bert.modeling_bert import BertConfig

    config = BertConfig(
        num_hidden_layers=6, num_attention_heads=6, intermediate_size=1024
    )
    new_model = get_final_model(original_model=model, new_model_config=config)
