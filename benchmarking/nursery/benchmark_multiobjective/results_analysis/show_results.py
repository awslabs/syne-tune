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
# %%
import logging
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas
import pandas as pd
import scipy

from syne_tune.constants import ST_TUNER_TIME

from benchmarking.nursery.benchmark_automl.results_analysis.utils import load_and_cache

from benchmarking.nursery.benchmark_multiobjective.baselines import Methods

from benchmarking.nursery.benchmark_multiobjective.results_analysis.utils import (
    METHOD_STYLES,
    plot_results,
)


def select_single_fidelity_for_benchmark(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the best result for any trial_id making sure only one number for each evaluation is stored.
    As a result, we will be able to have single-fidelity measurement.
    """
    measurements_by_trial= benchmark_df.groupby([
        'tuner_name',
        'seed',
        'algorithm',
        'tag',
        'benchmark',
        'random_seed',
        "scheduler_name",
        "trial_id"
    ])
    max_time_in_trial = measurements_by_trial[ST_TUNER_TIME].transform(max)
    max_time_in_trial_mask = max_time_in_trial == benchmark_df[ST_TUNER_TIME]
    return benchmark_df[max_time_in_trial_mask]




def rank_normalize_benchmark(benchmark_df: pandas.DataFrame):
    metrics_columns = [name for name in benchmark_df.columns if name.startswith("metric_names")]
    metric_names = benchmark_df.loc[:, metrics_columns].values[0].tolist()

    for metric_name in metric_names:
        benchmark_df[f"{metric_name}-base"] = benchmark_df[metric_name].copy()
        benchmark_df[metric_name] = scipy.stats.rankdata(benchmark_df[metric_name])
        benchmark_df[metric_name] = benchmark_df[metric_name] / benchmark_df[metric_name].max()

    return benchmark_df


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag",
        type=str,
        required=True,
        help="the experiment tag that was displayed when running the experiment",
    )
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    logging.getLogger().setLevel(logging.INFO)

    load_cache_if_exists = True
    select_single_fidelity = True
    normalize = True

    # benchmarks_to_df = {bench: df[] for bench, df in benchmarks_to_df.items()}
    methods_to_show = list(METHOD_STYLES.keys())
    benchmarks_to_df = load_and_cache(
        load_cache_if_exists=load_cache_if_exists,
        experiment_tag=experiment_tag,
        methods_to_show=methods_to_show,
    )

    for bench, df_ in benchmarks_to_df.items():
        df_methods = df_.algorithm.unique()
        for x in methods_to_show:
            if x not in df_methods:
                logging.warning(f"method {x} not found in {bench}")

    methods_to_show = [
        Methods.RS,
        Methods.LSOBO,
        Methods.MOREA,
        Methods.MSMOS,
    ]

    params = {
        "legend.fontsize": 18,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    }
    plt.rcParams.update(params)
    if select_single_fidelity:
        for name, benchmark_df in benchmarks_to_df.items():
            benchmarks_to_df[name] = select_single_fidelity_for_benchmark(benchmark_df)

    if normalize:
        for name, benchmark_df in benchmarks_to_df.items():
            benchmarks_to_df[name] = rank_normalize_benchmark(benchmark_df.copy())

    plot_results(benchmarks_to_df, METHOD_STYLES, methods_to_show=methods_to_show)


if (__name__ == "__main__"):
    main()