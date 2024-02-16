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
import torch
import scipy
import json
import numpy as np
from sampling import SmallSearchSpace
from train_meta_model import VAE
import statsmodels.api as sm


class MetaSamplerSmallSearchSpace(SmallSearchSpace):
    def __init__(self, config, model_path, seed=None, **kwargs):
        super().__init__(config, seed, **kwargs)
        checkpoint = torch.load(model_path)
        model_config = checkpoint["config"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = VAE(
            input_dim=3,
            hidden_dim=model_config["hidden_dim"],
            latent_dim=model_config["latent_dim"],
            device=device,
        )
        self.model.load_state_dict(checkpoint["state"])
        self.lower_bound = np.empty((3))
        self.upper_bound = np.empty((3))
        for i, key in enumerate(["num_layers", "num_units", "num_heads"]):
            self.lower_bound[i] = self.config_space[key].lower
            self.upper_bound[i] = self.config_space[key].upper

    def sample(self):
        # num_layers = self.rng.randint(self.num_layers)
        # num_heads = self.rng.randint(1, self.num_heads)
        # num_units = self.rng.randint(1, self.intermediate_size)
        z = torch.randn(self.model.latent_dim)
        x_tilde = self.model.decode(z).detach().numpy()
        x = x_tilde * (self.upper_bound - self.lower_bound) + self.lower_bound
        # # num_layers, num_units, num_heads
        # x_hat = torch.rand(3)
        # x_tilde,_, _ = self.model(x_hat)
        # x = x_tilde.detach().numpy() * (self.upper_bound - self.lower_bound) + self.lower_bound
        config = {
            "num_layers": int(x[0]),
            "num_units": int(x[1]),
            "num_heads": int(x[2]),
        }
        return config

    def __call__(self, *args, **kwargs):
        config = self.sample()
        return self.config_to_mask(config)


class MetaSamplerDKESmallSearchSpace(SmallSearchSpace):
    def __init__(
        self, config, data_path, dataset_name, num_tasks=1, seed=None, **kwargs
    ):
        super().__init__(config, seed, **kwargs)
        self.lower_bound = np.empty((3))
        self.upper_bound = np.empty((3))
        for i, key in enumerate(["num_layers", "num_units", "num_heads"]):
            self.lower_bound[i] = self.config_space[key].lower
            self.upper_bound[i] = self.config_space[key].upper

        meta_data = json.load(open(data_path, "r"))

        meta_data.pop(dataset_name)
        data = []

        assert num_tasks <= len(meta_data.keys())
        for i in range(num_tasks):
            idx = np.random.randint(len(meta_data.keys()))
            source_data = list(meta_data.keys())[idx]
            run = np.random.randint(len(meta_data[source_data]))
            data.extend(meta_data[source_data][run])

            meta_data.pop(source_data)

        self.data = (data - self.lower_bound) / (self.upper_bound - self.lower_bound)

        dens_u = sm.nonparametric.KDEMultivariate(
            data=self.data, var_type="ccc", bw="normal_reference"
        )
        self.bw = dens_u.bw

    def sample(self):
        idx = np.random.randint(self.data.shape[0])
        x_hat = self.data[idx]
        x_tilde = scipy.stats.truncnorm.rvs(0, 1, loc=x_hat, scale=self.bw)
        x_tilde = np.clip(x_tilde, 0, 1)

        x = x_tilde * (self.upper_bound - self.lower_bound) + self.lower_bound
        config = {
            "num_layers": int(x[0]),
            "num_units": int(x[1]),
            "num_heads": int(x[2]),
        }
        return config

    def __call__(self, *args, **kwargs):
        config = self.sample()
        return self.config_to_mask(config)
