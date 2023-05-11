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
from dataclasses import dataclass
from typing import Dict

from syne_tune.config_space import Domain, Categorical
from syne_tune.optimizer.baselines import RandomSearch, MOREA, NSGA2
from modified_random_search import MORS
from local_search import LS


def get_default(config_space):
    config = {}
    for k, v in config_space.items():
        if isinstance(v, Domain):

            if isinstance(v, Categorical):
                config[k] = v.categories[0]
            else:
                config[k] = v.upper
    return config


@dataclass
class MethodArguments:
    config_space: Dict
    metrics: list
    mode: list
    random_seed: int
    # max_t: int
    # resource_attr: str


class Methods:
    RS = "random_search"
    MOREA = "morea"
    MORS = "modified_random_search"
    LS = "local_search"
    NSGA2 = "nsga2"


methods = {
    Methods.RS: lambda method_arguments: RandomSearch(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics[0],
        mode=method_arguments.mode[0],
        random_seed=method_arguments.random_seed,
        # points_to_evaluate=[get_default(config_space=method_arguments.config_space)],
    ),
    Methods.MOREA: lambda method_arguments: MOREA(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        sample_size=5,
        population_size=10,
        # points_to_evaluate=[get_default(config_space=method_arguments.config_space)],
    ),
    Methods.MORS: lambda method_arguments: MORS(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        # points_to_evaluate=[get_default(config_space=method_arguments.config_space)],
    ),
    Methods.LS: lambda method_arguments: LS(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        start_point=[get_default(config_space=method_arguments.config_space)],
    ),
    Methods.NSGA2: lambda method_arguments: NSGA2(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        population_size=10,
        # points_to_evaluate=[get_default(config_space=method_arguments.config_space)],
    ),
}
