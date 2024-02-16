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
import numpy as np
import torch

from syne_tune.config_space import randint, choice, Domain, ordinal


class SearchSpace(object):
    """
    Setting the mask to 1 means we keep the corresponding head / unit
    """

    def __init__(self, config, seed=None, **kwargs):
        self.config = config

        if config.model_type == "gpt2":
            self.num_heads = config.n_head
            self.num_layers = config.n_layer
            self.intermediate_size = (
                config.n_inner if config.n_inner is not None else 4 * config.hidden_size
            )

        else:
            self.num_heads = config.num_attention_heads
            self.num_layers = config.num_hidden_layers
            self.intermediate_size = config.intermediate_size

        if seed is None:
            self.rng = np.random.RandomState(np.random.randint(2**32 - 1))
        else:
            self.rng = np.random.RandomState(seed)
        self.config_space = self._define_config_space(**kwargs)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def get_smallest_sub_network(self):
        raise NotImplementedError

    def _define_config_space(self, **kwargs):
        return {}

    def get_syne_tune_config_space(self):
        return self.config_space

    def config_to_mask(self, config):
        raise NotImplementedError


class SmallSearchSpace(SearchSpace):
    def _define_config_space(self, power_of_2_encoding=False, **kwargs):
        config_space = {}
        config_space["num_layers"] = randint(0, self.num_layers)
        config_space["num_units"] = randint(1, self.intermediate_size)

        if power_of_2_encoding:
            values = [
                int(self.num_heads / 2**i)
                for i in range(int(np.log2(self.num_heads)) + 1)
            ]
            values.reverse()
            config_space["num_heads"] = ordinal(values, kind="nn-log")
        else:
            config_space["num_heads"] = randint(1, self.num_heads)

        return config_space

    def __call__(self, *args, **kwargs):
        # num_layers = self.rng.randint(self.num_layers)
        # num_heads = self.rng.randint(1, self.num_heads)
        # num_units = self.rng.randint(1, self.intermediate_size)

        config = {
            k: v.sample() if isinstance(v, Domain) else v
            for k, v in self.config_space.items()
        }
        return self.config_to_mask(config)

    def _create_mask(self, num_layers, num_heads, num_units):
        head_mask = torch.ones((self.num_layers, self.num_heads))
        ffn_mask = torch.ones((self.num_layers, self.intermediate_size))
        head_mask[num_layers:] = 0
        head_mask[:num_layers, num_heads:] = 0
        ffn_mask[num_layers:] = 0
        ffn_mask[:num_layers, num_units:] = 0
        return head_mask, ffn_mask

    def get_smallest_sub_network(self):
        num_layers = 1
        num_heads = 1
        num_units = 1

        return self._create_mask(num_layers, num_heads, num_units)

    def config_to_mask(self, config):
        num_layers = config["num_layers"]
        num_heads = config["num_heads"]
        num_units = config["num_units"]
        return self._create_mask(num_layers, num_heads, num_units)


class MediumSearchSpace(SearchSpace):
    def _define_config_space(self, **kwargs):
        config_space = {}
        for i in range(self.num_layers):
            # values = [
            #     int(self.num_heads / 2**i)
            #     for i in range(int(np.log2(self.num_heads)) + 1)
            # ]
            # values.reverse()
            # config_space[f"num_heads_{i}"] = ordinal(values, kind="nn-log")
            config_space[f"num_heads_{i}"] = randint(0, self.num_heads)
            config_space[f"num_units_{i}"] = randint(0, self.intermediate_size)
        return config_space

    def __call__(self, *args, **kwargs):
        # num_heads = self.rng.randint(0, self.num_heads, self.num_layers)
        # num_units = self.rng.randint(0, self.intermediate_size, self.num_layers)
        config = {
            k: v.sample() if isinstance(v, Domain) else v
            for k, v in self.config_space.items()
        }
        return self.config_to_mask(config)

    def config_to_mask(self, config):
        num_heads = [config[f"num_heads_{i}"] for i in range(self.num_layers)]
        num_units = [config[f"num_units_{i}"] for i in range(self.num_layers)]
        return self._create_mask(num_heads, num_units)

    def _create_mask(self, num_heads, num_units):
        head_mask = torch.ones((self.num_layers, self.num_heads))
        ffn_mask = torch.ones((self.num_layers, self.intermediate_size))

        for i, hi in enumerate(num_heads):
            head_mask[i, hi:] = 0
        for i, fi in enumerate(num_units):
            ffn_mask[i, fi:] = 0
        return head_mask, ffn_mask

    def get_smallest_sub_network(self):
        num_heads = [1] * self.num_layers
        num_units = [1] * self.num_layers

        return self._create_mask(num_heads, num_units)


class LayerSearchSpace(SearchSpace):
    def _define_config_space(self, **kwargs):
        config_space = {}
        for i in range(self.num_layers):
            config_space[f"layer_{i}"] = choice([0, 1])
        return config_space

    def config_to_mask(self, config):
        layers = []
        for i in range(self.num_layers):
            if config[f"layer_{i}"] == 1:
                layers.append(i)

        return self._create_mask(layers)

    def __call__(self, *args, **kwargs):
        n_layers = self.rng.randint(0, self.num_layers)
        layers = self.rng.choice(np.arange(self.num_layers), n_layers, replace=False)

        return self._create_mask(layers)

    def _create_mask(self, layers):
        head_mask = torch.zeros((self.num_layers, self.num_heads))
        ffn_mask = torch.zeros((self.num_layers, self.intermediate_size))
        for li in layers:
            head_mask[li, :] = 1
            ffn_mask[li, :] = 1
        return head_mask, ffn_mask

    def get_smallest_sub_network(self):
        return self._create_mask([])


class FullLayerSearchSpace(SearchSpace):
    def _define_config_space(self, **kwargs):
        config_space = {}
        for i in range(self.num_layers):
            config_space[f"layer_mha_{i}"] = choice([0, 1])
            config_space[f"layer_ffn_{i}"] = choice([0, 1])
        return config_space

    def config_to_mask(self, config):
        layers_mha = []
        layers_ffn = []
        for i in range(self.num_layers):
            if config[f"layer_mha_{i}"] == 1:
                layers_mha.append(i)
            if config[f"layer_ffn_{i}"] == 1:
                layers_ffn.append(i + self.num_layers)

        return self._create_mask(layers_mha + layers_ffn)

    def __call__(self, *args, **kwargs):
        n_layers = self.rng.randint(0, self.num_layers * 2)
        layers = self.rng.choice(
            np.arange(self.num_layers * 2), n_layers, replace=False
        )

        return self._create_mask(layers)

    def _create_mask(self, layers):
        head_mask = torch.zeros((self.num_layers, self.num_heads))
        ffn_mask = torch.zeros((self.num_layers, self.intermediate_size))
        for li in layers:
            if li < self.num_layers:
                head_mask[li, :] = 1
            else:
                ffn_mask[li - self.num_layers, :] = 1
        return head_mask, ffn_mask

    def get_smallest_sub_network(self):
        return self._create_mask([])


class FullSearchSpace(SearchSpace):
    def __call__(self, *args, **kwargs):
        num_layers = self.num_layers
        num_units = self.intermediate_size
        num_attention_heads = self.num_heads

        K = num_layers * num_units + num_attention_heads * num_layers
        k = np.random.randint(1, K)

        idx = np.random.choice(np.arange(K), k, replace=False)

        mask = torch.ones((K))
        mask[idx] = 0
        h = self.num_layers * self.num_heads
        head_mask = mask[:h].resize(self.num_layers, self.num_heads)
        ffn_mask = mask[h:].resize(self.num_layers, self.intermediate_size)
        return head_mask, ffn_mask

    def get_smallest_sub_network(self):
        head_mask = torch.zeros((self.num_layers, self.num_heads))
        ffn_mask = torch.zeros((self.num_layers, self.intermediate_size))

        head_mask[0, 0] = 1
        ffn_mask[0, 0] = 1
        return head_mask, ffn_mask

    def _define_config_space(self, **kwargs):
        config_space = {}
        for i in range(self.num_layers):
            for j in range(self.num_heads):
                config_space[f"layer_mha_{i}_{j}"] = choice([0, 1])
            for j in range(self.intermediate_size):
                config_space[f"layer_ffn_{i}_{j}"] = choice([0, 1])
        return config_space

    def config_to_mask(self, config):
        head_mask = torch.zeros((self.num_layers, self.num_heads))
        ffn_mask = torch.zeros((self.num_layers, self.intermediate_size))
        for i in range(self.num_layers):
            for j in range(self.num_heads):
                if config[f"layer_mha_{i}_{j}"] == 1:
                    head_mask[i, j] = 1
            for j in range(self.intermediate_size):
                if config[f"layer_ffn_{i}_{j}"] == 1:
                    ffn_mask[i, j] = 1

        return head_mask, ffn_mask
