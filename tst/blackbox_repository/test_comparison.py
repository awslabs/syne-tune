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
import pytest
import numpy as np

from yahpo_gym import BenchmarkSet

from syne_tune.blackbox_repository import load_blackbox
from syne_tune.blackbox_repository.utils import metrics_for_configuration
from benchmarking.commons.benchmark_definitions.lcbench import (
    lcbench_selected_datasets,
)
from benchmarking.commons.benchmark_definitions import (
    yahpo_lcbench_benchmark_definitions,
    lcbench_benchmark_definitions,
)
from syne_tune.config_space import cast_config_values


benchmark_pairs_lcbench = [
    (
        yahpo_lcbench_benchmark_definitions["yahpo-lcbench-" + dataset],
        lcbench_benchmark_definitions["lcbench-" + dataset],
    )
    for dataset in lcbench_selected_datasets
]


benchmark_pairs = benchmark_pairs_lcbench


# Issues so far:
# - lcbench: Metric values are quite different as well, in particular
#   "elapsed_time". They have 52 fidelities, while we have 50.
#
#   We found that in the original lcbench data, the first fidelity is for
#   the model with initial weights, before any training. Our import code
#   removes first and last fidelity, and corrects "elapsed_time" by time
#   recorded for first fidelity (setup time?).
#   Differences could be due to YAHPO not doing this, so they train on
#   data with values for first fidelity being nonsense. In particular, the
#   "elapsed_time" data for the first fidelity is very small.
@pytest.mark.skip(
    "Needs blackbox data files locally or on S3. Also, we are waiting for YAHPO authors to refit their surrogate to corrected lcbench data"
)
@pytest.mark.parametrize("benchmark_yahpo, benchmark_ours", benchmark_pairs)
def test_comparison_yahpo_ours(benchmark_yahpo, benchmark_ours):
    random_seed = 31415927
    np.random.seed(random_seed)
    num_configs = 50

    benchmark = {"yahpo": benchmark_yahpo, "ours": benchmark_ours}
    # Activates input checks for YAHPO BenchmarkSet.objective_function. We want
    # calls to fail if configurations are invalid. By default, these checks
    # are switched off, as this may be a little faster
    yahpo_kwargs = dict(check=True)
    blackbox = {
        name: load_blackbox(bm.blackbox_name, yahpo_kwargs=yahpo_kwargs)[
            bm.dataset_name
        ]
        for name, bm in benchmark.items()
    }
    # Sample `num_configs` configurations which are exactly supported by data
    hyperparameters = blackbox["ours"].hyperparameters
    random_inds = np.random.permutation(len(hyperparameters))[:num_configs]
    num_seeds = blackbox["ours"].num_seeds
    if num_seeds > 1:
        seeds = list(np.random.randint(num_seeds, size=num_configs))
    else:
        seeds = [0] * num_configs
    configs = [
        cast_config_values(
            hyperparameters.iloc[ind].to_dict(),
            config_space=blackbox["ours"].configuration_space,
        )
        for ind in random_inds
    ]
    resource_attr = "epoch"
    metric_attr = {name: bm.metric for name, bm in benchmark.items()}
    elapsed_time_attr = {name: bm.elapsed_time_attr for name, bm in benchmark.items()}
    metric_attr["yahpo_direct"] = metric_attr["yahpo"]
    elapsed_time_attr["yahpo_direct"] = elapsed_time_attr["yahpo"]
    # Loop over configs: Compare metrics returned
    for config, seed, ind in zip(configs, seeds, random_inds):
        config = {"ours": config}

        def map_name(name: str) -> str:
            prefix = "hp_"
            if name.startswith(prefix):
                return name[len(prefix) :]
            else:
                return name

        config["yahpo"] = {map_name(k): v for k, v in config["ours"].items()}
        task_attr = blackbox["yahpo"].benchmark.config.instance_names
        config["yahpo"][task_attr] = benchmark["yahpo"].dataset_name
        results = {
            name: metrics_for_configuration(
                blackbox=bb,
                config=config[name],
                resource_attr=resource_attr,
                seed=seed,
            )
            for name, bb in blackbox.items()
        }

        # Use YAHPO directly to fetch values
        scenario = benchmark["yahpo"].blackbox_name[len("yahpo-") :]
        instance = benchmark["yahpo"].dataset_name
        bb = BenchmarkSet(scenario)
        bb.set_instance(instance)
        num_fidelities = 100 if scenario == "fcnet" else 52
        config_yahpo = config["yahpo"]
        if scenario == "fcnet":
            config_yahpo = {**config_yahpo, "replication": seed}
        results["yahpo_direct"] = bb.objective_function(
            [
                {**config_yahpo, "epoch": fidelity}
                for fidelity in range(1, num_fidelities + 1)
            ],
            seed=seed,
        )

        metrics = {
            name: [res[metric_attr[name]] for res in res_dict]
            for name, res_dict in results.items()
        }
        elapsed_times = {
            name: [res[elapsed_time_attr[name]] for res in res_dict]
            for name, res_dict in results.items()
        }

        name = f"{benchmark_ours.blackbox_name}-{benchmark_ours.dataset_name}"
        errmsg_parts = [
            f"benchmark = {name}",
            f"config = [{ind}] {config['ours']}",
        ]
        if num_seeds > 1:
            errmsg_parts.append(f"seed = {seed}")
        for name, metric_vals in metrics.items():
            et_vals = elapsed_times[name]
            errmsg_parts.append(f"metric[{name}] = [{len(metric_vals)}] {metric_vals}")
            errmsg_parts.append(f"elapsed_time[{name}] = [{len(et_vals)}] {et_vals}")
        errmsg = "\n".join(errmsg_parts)

        case = "yahpo_direct vs yahpo\n"
        assert len(metrics["yahpo_direct"]) == len(metrics["yahpo"]), (
            "len: " + case + errmsg
        )
        assert np.allclose(metrics["yahpo_direct"], metrics["yahpo"]), (
            "metrics: " + case + errmsg
        )
        assert np.allclose(elapsed_times["yahpo_direct"], elapsed_times["yahpo"]), (
            "elapsed_times: " + case + errmsg
        )

        case = "ours vs yahpo\n"
        assert len(metrics["ours"]) == len(metrics["yahpo"]), "len: " + case + errmsg
        assert np.allclose(metrics["ours"], metrics["yahpo"]), (
            "metrics: " + case + errmsg
        )
        assert np.allclose(elapsed_times["ours"], elapsed_times["yahpo"]), (
            "elapsed_times: " + case + errmsg
        )
