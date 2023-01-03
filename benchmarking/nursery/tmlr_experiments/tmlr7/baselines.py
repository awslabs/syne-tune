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
from benchmarking.commons.baselines import (
    search_options,
)
from syne_tune.optimizer.baselines import (
    ASHA,
    MOBSTER,
    HyperTune,
    SyncHyperband,
    SyncMOBSTER,
    SyncBOHB,
    DEHB,
)
from syne_tune.optimizer.schedulers import HyperbandScheduler
from syne_tune.optimizer.schedulers.searchers.bore import MultiFidelityBore


class Methods:
    ASHA_4BR = "ASHA-4BR"
    MOBSTER_4BR = "MOBSTER-4BR"
    HYPERTUNE_4BR = "HyperTune-4BR"


methods = {
    Methods.ASHA_4BR: lambda method_arguments: ASHA(
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
        type="promotion",
        brackets=4,
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        **(
            method_arguments.scheduler_kwargs
            if method_arguments.scheduler_kwargs is not None
            else dict()
        ),
    ),
    Methods.MOBSTER_4BR: lambda method_arguments: MOBSTER(
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
        type="promotion",
        brackets=4,
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        **(
            method_arguments.scheduler_kwargs
            if method_arguments.scheduler_kwargs is not None
            else dict()
        ),
    ),
    Methods.HYPERTUNE_4BR: lambda method_arguments: HyperTune(
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
        type="promotion",
        brackets=4,
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        **(
            method_arguments.scheduler_kwargs
            if method_arguments.scheduler_kwargs is not None
            else dict()
        ),
    ),
}
