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

from typing import Dict, Any
from benchmarking.commons.hpo_main_simulator import main
from benchmarking.nursery.benchmark_multiobjective.baselines import methods
from benchmarking.nursery.benchmark_multiobjective.benchmark_definitions import (
    benchmark_definitions,
)
from syne_tune.util import recursive_merge

extra_args = [
    # dict(
    #     name="num_brackets",
    #     type=int,
    #     help="Number of brackets",
    # ),
    # dict(
    #     name="num_samples",
    #     type=int,
    #     default=50,
    #     help="Number of samples for Hyper-Tune distribution",
    # ),
]


def map_method_args(args, method: str, method_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return method_kwargs


if __name__ == "__main__":
    main(methods, benchmark_definitions, extra_args, map_method_args)
