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
import itertools
import logging
from typing import Optional, Callable, Dict, Any

import numpy as np
from tqdm import tqdm

from benchmarking.commons.baselines import MethodArguments, MethodDefinitions
from benchmarking.commons.benchmark_definitions.common import RealBenchmarkDefinition
from benchmarking.commons.hpo_main_common import (
    set_logging_level,
    get_metadata,
    ExtraArgsType,
    MapMethodArgsType,
    PostProcessingType,
    ConfigDict,
    DictStrKey,
    extra_metadata,
    str2bool,
    config_from_argparse,
)
from benchmarking.commons.utils import get_master_random_seed, effective_random_seed
from syne_tune.backend import LocalBackend
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner

logger = logging.getLogger(__name__)


RealBenchmarkDefinitions = Callable[..., Dict[str, RealBenchmarkDefinition]]
LOCAL_BACKEND_EXTRA_PARAMETERS = [
    dict(
        name="benchmark",
        type=str,
        default="resnet_cifar10",
        help="Benchmark to run",
    ),
    dict(
        name="verbose",
        type=str2bool,
        default=False,
        help="Verbose log output?",
    ),
    dict(
        name="instance_type",
        type=str,
        default=None,
        help="AWS SageMaker instance type",
    ),
]


def get_benchmark(
    configuration: ConfigDict,
    benchmark_definitions: RealBenchmarkDefinitions,
    **benchmark_kwargs,
) -> RealBenchmarkDefinition:
    do_scale = (
        configuration.scale_max_wallclock_time
        and configuration.n_workers is not None
        and configuration.max_wallclock_time is None
    )
    if do_scale:
        benchmark_default = benchmark_definitions(**benchmark_kwargs)[
            configuration.benchmark
        ]
        default_n_workers = benchmark_default.n_workers
    else:
        default_n_workers = None
    if configuration.n_workers is not None:
        benchmark_kwargs["n_workers"] = configuration.n_workers
    if configuration.max_wallclock_time is not None:
        benchmark_kwargs["max_wallclock_time"] = configuration.max_wallclock_time
    if configuration.instance_type is not None:
        benchmark_kwargs["instance_type"] = configuration.instance_type
    benchmark = benchmark_definitions(**benchmark_kwargs)[configuration.benchmark]
    if do_scale and configuration.n_workers < default_n_workers:
        # Scale ``max_wallclock_time``
        factor = default_n_workers / configuration.n_workers
        bm_mwt = benchmark.max_wallclock_time
        benchmark.max_wallclock_time = int(bm_mwt * factor)
        print(
            f"Scaling max_wallclock_time: {benchmark.max_wallclock_time} (from {bm_mwt})"
        )
    return benchmark


def create_objects_for_tuner(configuration: ConfigDict, methods: MethodDefinitions, method: str,
                             benchmark: RealBenchmarkDefinition, master_random_seed: int, seed: int, verbose: bool,
                             extra_tuning_job_metadata: Optional[DictStrKey] = None,
                             map_method_args: Optional[MapMethodArgsType] = None) -> Dict[str, Any]:

    method_kwargs = {"max_resource_attr": benchmark.max_resource_attr}
    if configuration.max_size_data_for_model is not None:
        method_kwargs["scheduler_kwargs"] = {
            "search_options": {
                "max_size_data_for_model": configuration.max_size_data_for_model
            },
        }

    if map_method_args is not None:
        method_kwargs = map_method_args(configuration, method, method_kwargs)

    method_kwargs.update(
        dict(
            config_space=benchmark.config_space,
            metric=benchmark.metric,
            mode=benchmark.mode,
            random_seed=effective_random_seed(master_random_seed, seed),
            resource_attr=benchmark.resource_attr,
            verbose=verbose,
        )
    )
    scheduler = methods[method](MethodArguments(**method_kwargs))

    stop_criterion = StoppingCriterion(
        max_wallclock_time=benchmark.max_wallclock_time,
        max_num_evaluations=benchmark.max_num_evaluations,
    )
    metadata = get_metadata(
        seed=seed,
        method=method,
        experiment_tag=configuration.experiment_tag,
        benchmark_name=configuration.benchmark,
        random_seed=master_random_seed,
        max_size_data_for_model=configuration.max_size_data_for_model,
        benchmark=benchmark,
        extra_metadata=extra_tuning_job_metadata,
    )
    return dict(
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=benchmark.n_workers,
        tuner_name=configuration.experiment_tag,
        metadata=metadata,
        save_tuner=configuration.save_tuner,
    )


def start_benchmark_local_backend(
    configuration: ConfigDict,
    methods: MethodDefinitions,
    benchmark_definitions: RealBenchmarkDefinitions,
    post_processing: Optional[PostProcessingType] = None,
    map_method_args: Optional[MapMethodArgsType] = None,
    extra_tuning_job_metadata: Optional[DictStrKey] = None,
):
    """
    Runs sequence of experiments with local backend sequentially.
    The loop runs over methods selected from ``methods`` and repetitions,

    ``map_method_args`` can be used to modify ``method_kwargs`` for constructing
    :class:`~benchmarking.commons.baselines.MethodArguments`, depending on
    ``configuration`` and the method. This allows for extra flexibility to specify specific arguments for chosen methods
    Its signature is :code:`method_kwargs = map_method_args(configuration, method, method_kwargs)`,
    where ``method`` is the name of the baseline.

    :param configuration: ConfigDict with parameters of the benchmark.
        Must contain all parameters from LOCAL_BACKEND_EXTRA_PARAMETERS
    :param methods: Dictionary with method constructors.
    :param benchmark_definitions: Definitions of benchmarks; one is selected from
        command line arguments
    :param post_processing: Called after tuning has finished, passing the tuner
        as argument. Can be used for postprocessing, such as output or storage
        of extra information
    :param map_method_args: See above, optional
    :param extra_tuning_job_metadata: Metadata added to the tuner, can be used to manage results
    """
    configuration.check_if_all_paremeters_present(LOCAL_BACKEND_EXTRA_PARAMETERS)
    configuration.expand_base_arguments(LOCAL_BACKEND_EXTRA_PARAMETERS)

    experiment_tag = configuration.experiment_tag
    benchmark_name = configuration.benchmark
    master_random_seed = get_master_random_seed(configuration.random_seed)
    set_logging_level(configuration)
    benchmark = get_benchmark(configuration, benchmark_definitions)

    combinations = list(itertools.product(list(methods.keys()), configuration.seeds))
    print(combinations)
    for method, seed in tqdm(combinations):
        random_seed = effective_random_seed(master_random_seed, seed)
        np.random.seed(random_seed)
        print(
            f"Starting experiment ({method}/{benchmark_name}/{seed}) of {experiment_tag}"
        )
        trial_backend = LocalBackend(entry_point=str(benchmark.script))

        tuner_kwargs = create_objects_for_tuner(configuration, methods=methods, method=method, benchmark=benchmark,
                                                master_random_seed=master_random_seed, seed=seed,
                                                verbose=configuration.verbose,
                                                extra_tuning_job_metadata=extra_tuning_job_metadata,
                                                map_method_args=map_method_args)
        tuner = Tuner(
            trial_backend=trial_backend,
            **tuner_kwargs,
        )
        tuner.run()
        if post_processing is not None:
            post_processing(tuner)


def main(
    methods: MethodDefinitions,
    benchmark_definitions: RealBenchmarkDefinitions,
    extra_args: Optional[ExtraArgsType] = None,
    map_extra_args: Optional[MapMethodArgsType] = None,
    post_processing: Optional[PostProcessingType] = None,
):
    """
    Runs sequence of experiments with local backend sequentially. The loop runs
    over methods selected from ``methods`` and repetitions, both controlled by
    command line arguments.

    ``map_extra_args`` can be used to modify ``method_kwargs`` for constructing
    :class:`~benchmarking.commons.baselines.MethodArguments`, depending on
    ``configuration`` returned by :func:`parse_args` and the method. Its signature is
    :code:`method_kwargs = map_extra_args(configuration, method, method_kwargs)`, where
    ``method`` is the name of the baseline.

    :param methods: Dictionary with method constructors
    :param benchmark_definitions: Definitions of benchmarks; one is selected from
        command line arguments
    :param extra_args: Extra arguments for command line parser. Optional
    :param map_extra_args: See above, optional
    :param post_processing: Called after tuning has finished, passing the tuner
        as argument. Can be used for postprocessing, such as output or storage
        of extra information
    """
    configuration = config_from_argparse(extra_args, LOCAL_BACKEND_EXTRA_PARAMETERS)
    method_names = (
        [configuration.method]
        if configuration.method is not None
        else list(methods.keys())
    )
    methods = {mname: methods[mname] for mname in method_names}
    if extra_args is not None:
        assert (
            map_extra_args is not None
        ), "map_extra_args must be specified if extra_args is used"

    start_benchmark_local_backend(
        configuration,
        methods=methods,
        benchmark_definitions=benchmark_definitions,
        map_method_args=map_extra_args,
        post_processing=post_processing,
        extra_tuning_job_metadata=None
        if extra_args is None
        else extra_metadata(configuration, extra_args),
    )
