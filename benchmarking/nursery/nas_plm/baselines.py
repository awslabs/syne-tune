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
from syne_tune.optimizer.baselines import (
    RandomSearch,
    GridSearch,
    MOREA,
    NSGA2,
    MORandomScalarizationBayesOpt,
    MOASHA,
)

# from ehvi import EHVI
from syne_tune.optimizer.schedulers.multiobjective.linear_scalarizer import (
    LinearScalarizedScheduler,
)

from local_search import LS


def get_random(config_space):
    config = {}
    for k, v in config_space.items():
        if isinstance(v, Domain):
            config[k] = v.sample()
    return config


def get_lower_bound(config_space):
    config = {}
    for k, v in config_space.items():
        if isinstance(v, Domain):
            if isinstance(v, Categorical):
                config[k] = v.categories[0]
            else:
                config[k] = v.lower
    return config


def get_upper_bound(config_space):
    config = {}
    for k, v in config_space.items():
        if isinstance(v, Domain):
            if isinstance(v, Categorical):
                config[k] = v.categories[-1]
            else:
                config[k] = v.upper
    return config


def get_mid_point(config_space):
    config = {}
    for i, (k, v) in enumerate(config_space.items()):
        if isinstance(v, Domain):
            if isinstance(v, Categorical):
                if i < len(config_space.keys()) // 2:
                    config[k] = v.categories[0]
                else:
                    config[k] = v.categories[-1]
            else:
                config[k] = (v.upper - v.lower) // 2
    return config


@dataclass
class MethodArguments:
    config_space: Dict
    metrics: list
    mode: list
    random_seed: int


def initial_design(config_space):
    points_to_evaluate = []
    upper_bound = get_upper_bound(config_space)
    points_to_evaluate.append(upper_bound)
    lower_bound = get_lower_bound(config_space)
    points_to_evaluate.append(lower_bound)
    mid_point = get_mid_point(config_space)
    points_to_evaluate.append(mid_point)
    return points_to_evaluate


class Methods:
    RS = "random_search"
    MOREA = "morea"
    LS = "local_search"
    LS_UPPER_BOUND = "local_search_upper_bound"
    LS_LOWER_BOUND = "local_search_lower_bound"
    LS_RANDOM = "local_search_random"
    NSGA2 = "nsga2"
    LSBO = "lsbo"
    RSBO = "rsbo"
    MOBORE = "mobore"
    EHVI = "ehvi"
    MOASHA = "moasha"
    GRIDSEARCH = "grid_search"


methods = {
    Methods.RS: lambda method_arguments: RandomSearch(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics[0],
        mode=method_arguments.mode[0],
        random_seed=method_arguments.random_seed,
        points_to_evaluate=initial_design(method_arguments.config_space),
    ),
    Methods.GRIDSEARCH: lambda method_arguments: GridSearch(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics[0],
        mode=method_arguments.mode[0],
        random_seed=method_arguments.random_seed,
        # points_to_evaluate=[{'num_layers': 12,'num_heads': 12, 'num_units': 3072},
        #                     {'num_layers': 10, 'num_heads': 10, 'num_units': 2560},
        #                     {'num_layers': 8, 'num_heads': 8, 'num_units': 2048},
        #                     {'num_layers': 6, 'num_heads': 6, 'num_units': 1536},
        #                     {'num_layers': 4, 'num_heads': 4, 'num_units': 1024},
        #                     {'num_layers': 2, 'num_heads': 2, 'num_units': 512}],
        points_to_evaluate=[
            {"num_layers": 12, "num_heads": 12, "num_units": 3072},
            {"num_layers": 11, "num_heads": 12, "num_units": 3072},
            {"num_layers": 10, "num_heads": 12, "num_units": 3072},
            {"num_layers": 9, "num_heads": 12, "num_units": 3072},
            {"num_layers": 8, "num_heads": 12, "num_units": 3072},
            {"num_layers": 7, "num_heads": 12, "num_units": 3072},
            {"num_layers": 6, "num_heads": 12, "num_units": 3072},
            {"num_layers": 5, "num_heads": 12, "num_units": 3072},
            {"num_layers": 4, "num_heads": 12, "num_units": 3072},
            {"num_layers": 3, "num_heads": 12, "num_units": 3072},
            {"num_layers": 2, "num_heads": 12, "num_units": 3072},
            {"num_layers": 1, "num_heads": 12, "num_units": 3072},
        ],
    ),
    Methods.MOREA: lambda method_arguments: MOREA(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        sample_size=5,
        population_size=10,
        points_to_evaluate=initial_design(method_arguments.config_space),
    ),
    Methods.LS_UPPER_BOUND: lambda method_arguments: LS(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        start_point=get_upper_bound(config_space=method_arguments.config_space),
        points_to_evaluate=[],
    ),
    Methods.LS: lambda method_arguments: LS(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        start_point=None,
        points_to_evaluate=initial_design(method_arguments.config_space),
    ),
    Methods.LS_LOWER_BOUND: lambda method_arguments: LS(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        start_point=get_lower_bound(config_space=method_arguments.config_space),
        points_to_evaluate=[],
    ),
    Methods.LS_RANDOM: lambda method_arguments: LS(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        start_point=get_random(config_space=method_arguments.config_space),
        points_to_evaluate=[],
    ),
    Methods.NSGA2: lambda method_arguments: NSGA2(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        population_size=10,
        points_to_evaluate=initial_design(method_arguments.config_space),
    ),
    Methods.LSBO: lambda method_arguments: LinearScalarizedScheduler(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        searcher="bayesopt",
        points_to_evaluate=initial_design(method_arguments.config_space),
    ),
    Methods.RSBO: lambda method_arguments: MORandomScalarizationBayesOpt(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=initial_design(method_arguments.config_space),
    ),
    Methods.MOASHA: lambda method_arguments: MOASHA(
        config_space=method_arguments.config_space,
        metrics=method_arguments.metrics,
        mode=method_arguments.mode,
        time_attr="epoch",
        max_t=method_arguments.config_space["num_train_epochs"],
        grace_period=1,
        reduction_factor=3,
        brackets=1,
        # random_seed=method_arguments.random_seed,
        points_to_evaluate=initial_design(method_arguments.config_space),
    ),
    # Methods.MOBORE: lambda method_arguments: MultiObjectiveBore(
    #     config_space=method_arguments.config_space,
    #     metric=method_arguments.metrics,
    #     mode=method_arguments.mode,
    #     random_seed=method_arguments.random_seed,
    #     points_to_evaluate=[]
    # ),
    Methods.EHVI: lambda method_arguments: EHVI(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=initial_design(method_arguments.config_space),
    ),
}


if __name__ == "__main__":
    from transformers.models.bert.modeling_bert import BertConfig
    from sampling import (
        SmallSearchSpace,
        LayerSearchSpace,
        MediumSearchSpace,
        FullSearchSpace,
    )

    config = BertConfig()
    spaces = [
        SmallSearchSpace(config),
        LayerSearchSpace(config),
        MediumSearchSpace(config),
        FullSearchSpace(config),
    ][:-1]

    for space in spaces:
        print(get_lower_bound(space.config_space))
        print(get_mid_point(space.config_space))
        print(get_upper_bound(space.config_space))
