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
import argparse
import logging
from pathlib import Path

from sagemaker_tune.backend.local_backend import LocalBackend
from sagemaker_tune.remote.remote_launcher import RemoteLauncher
from sagemaker_tune.tuner import Tuner
from sagemaker_tune.stopping_criterion import StoppingCriterion

from scheduler_factory import short_name_scheduler_factory, \
    supported_short_name_schedulers
from benchmark_factory import benchmark_factory

logging.getLogger().setLevel(logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, required=False)
    parser.add_argument('--scheduler', type=str, required=False)
    parser.add_argument('--n_seeds', type=int, required=False, default=3)
    parser.add_argument('--timeout', type=int, required=False, default=4 * 3600)
    args, _ = parser.parse_known_args()

    if args.benchmark is None:
        benchmark_selected = [
            "nasbench201_cifar100",
            "nasbench201_cifar10-valid",
            "nasbench201_ImageNet16-120",
            "mlp_fashionmnist",
            "resnet_cifar10",
            # "lstm_wikitext2",
        ]
    else:
        benchmark_selected = [args.benchmark]
    seeds = range(args.n_seeds)

    if args.scheduler is None:
        schedulers_selected = supported_short_name_schedulers
    else:
        schedulers_selected = [args.scheduler]

    n_exp = len(schedulers_selected) * len(benchmark_selected) * len(seeds)
    logging.info(f"Evaluating {n_exp} evaluations of {schedulers_selected} on {benchmark_selected} with {args.n_seeds} seeds.")
    params_common = {
        'backend': 'local',
        'dataset_path': './',
    }
    stop_criterion = StoppingCriterion(max_wallclock_time=args.timeout)
    for seed in seeds:
        for benchmark_name in benchmark_selected:
            for short_name_scheduler in schedulers_selected:
                params = dict(
                    params_common, benchmark_name=benchmark_name, run_id=seed)
                benchmark, default_params = benchmark_factory(params)
                endpoint_script = benchmark['script']
                myscheduler, params = short_name_scheduler_factory(
                    short_name_scheduler, params, benchmark, default_params)
                tuner_name = "-".join(
                    [Path(endpoint_script).stem, short_name_scheduler,
                     str(seed)])
                logging.info(f"scheduling {tuner_name}")

                # scheduler, benchmark_name, run_id (seed) are fields in params
                metadata = {
                    k: v for k, v in params.items() if v is not None}
                tuner = Tuner(
                    scheduler=myscheduler,
                    tuner_name=tuner_name,
                    n_workers=params['num_workers'],
                    backend=LocalBackend(endpoint_script),
                    stop_criterion=stop_criterion,
                    metadata=metadata,
                )
                tuner = RemoteLauncher(
                    tuner=tuner,
                    dependencies=[str(Path(__file__).parent.parent / "benchmarks/")],
                    # Extra arguments describing the ressource of the remote tuning instance and whether we want to wait
                    # the tuning to finish. The instance-type where the tuning job runs can be different than the
                    # instance-type used for evaluating the training jobs in case of a Sagemaker Backend.
                    instance_type=params['instance_type'],
                )
                # todo option to wait in case ResourceLimitExceeded is reached
                tuner.run(wait=False)
