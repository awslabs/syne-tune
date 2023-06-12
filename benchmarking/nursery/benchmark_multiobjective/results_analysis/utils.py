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
import copy
import os
from dataclasses import dataclass
from typing import Dict, Optional, Union, List

import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm

from benchmarking.nursery.benchmark_multiobjective.baselines import Methods
from syne_tune.constants import ST_TUNER_TIME
from syne_tune.optimizer.schedulers.multiobjective.utils import hypervolume_cumulative

FIFO_STYLE = "solid"

@dataclass
class MethodStyle:
    color: str
    linestyle: str
    marker: str = None


METHOD_STYLES = {
    Methods.MOREA: MethodStyle("red", FIFO_STYLE),
    Methods.LSOBO: MethodStyle("green", FIFO_STYLE),
    Methods.RS: MethodStyle("blue", FIFO_STYLE),
    Methods.MSMOS: MethodStyle("orange", FIFO_STYLE),
}


@dataclass
class PlotArgs:
    xmin: float = None
    xmax: float = None
    ymin: float = None
    ymax: float = None


PLOT_RANGE = {
    # "nas201-ImageNet16-120": PlotArgs(1000, 21000, None, 0.8),
    # "nas201-cifar10": PlotArgs(2000, 21000, 0.05, 0.15),
    # "nas201-cifar100": PlotArgs(3000, 21000, 0.26, 0.35),
}


def plot_result_benchmark(
    df_task,
    title: str,
    metric2plot: Union[str, List[str]],
    mode: str,
    method_styles: Optional[Dict] = None,
    ax=None,
    methods_to_show: list = None,
):
    if len(df_task) == 0:
        return ax, None, None

    agg_results = {}
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(metric2plot, List):
        metric_names = metric2plot
        metric2plot = "Hypervolume"
        reference_point = df_task[metric_names].max().tolist()

    for algorithm, method_style in method_styles.items():
        if methods_to_show is not None and algorithm not in methods_to_show:
            continue

        x_values = []
        y_values = []

        df_scheduler = df_task[df_task.algorithm == algorithm]
        if len(df_scheduler) == 0:
            continue

        for i, tuner_name in tqdm(enumerate(df_scheduler.tuner_name.unique()), desc=f"Plotting for {title}/{algorithm}"):
            sub_df = df_scheduler[df_scheduler.tuner_name == tuner_name]
            sub_df = sub_df.sort_values(ST_TUNER_TIME).reset_index()
            sub_df = sub_df[10: ].copy() #TODO maybe we should find a better way of making sure initial points are selected randomly

            if metric2plot == "Hypervolume":
                results_array = sub_df[list(metric_names)].values
                sub_df[metric2plot] = hypervolume_cumulative(
                    results_array, reference_point
                )

            index = sub_df.index.to_list()
            y_best = (
                sub_df.loc[:, metric2plot].cummax().values
                if mode == "max"
                else sub_df.loc[:, metric2plot].cummin().values
            )
            x_values.append(index)
            y_values.append(y_best)

        # compute the mean/std over time-series of different seeds at regular time-steps
        # start/stop at respectively first/last point available for all seeds
        x_min = max(xx[0] for xx in x_values)
        x_max = min(xx[-1] for xx in x_values)
        if x_min > x_max:
            continue
        x_range = np.linspace(x_min, x_max)

        # find the best value at each regularly spaced time-step from x_range
        y_ranges = []
        for x, y in zip(x_values, y_values):
            indices = np.searchsorted(x, x_range, side="left")
            y_range = y[indices]
            y_ranges.append(y_range)
        y_ranges = np.stack(y_ranges)

        mean = y_ranges.mean(axis=0)
        std = y_ranges.std(axis=0)

        ax.fill_between(
            x_range,
            mean - std,
            mean + std,
            color=method_style.color,
            alpha=0.1,
        )
        ax.plot(
            x_range,
            mean,
            color=method_style.color,
            linestyle=method_style.linestyle,
            marker=method_style.marker,
            label=algorithm,
        )
        agg_results[algorithm] = mean

    ax.set_xlabel("Iterations")
    ax.set_ylabel(metric2plot)
    ax.legend()
    # ax.set_title(title)
    return ax, x_range, agg_results


def plot_results(
    benchmarks_to_df,
    method_styles: Optional[Dict] = None,
    prefix: str = "",
    title: str = None,
    ax=None,
    methods_to_show: list = None,
):
    agg_results = {}

    for benchmark, df_task in benchmarks_to_df.items():
        df_task = df_task.copy()
        # Gather the names of the metrics to plot
        metrics_columns = [name for name in df_task.columns if name.startswith("metric_names")]
        metric_names = df_task.loc[:, metrics_columns].values[0].tolist()

        modes_columns = [name for name in df_task.columns if name.startswith("metric_mode")]
        metric_modes = df_task.loc[:, modes_columns].values[0].tolist()
        assert np.all(
            [metric_mode == "min" for metric_mode in metric_modes]
        ), f"All metrics must be maximized but the following modes were selected: {metric_modes}"

        metric_names.append(copy.deepcopy(metric_names))
        metric_modes.append("max")
        for metric2plot, mode in zip(metric_names, metric_modes):
            ax, x_ranges, agg_result = plot_result_benchmark(
                df_task=df_task,
                title=benchmark,
                metric2plot=metric2plot,
                mode=mode,
                method_styles=method_styles,
                ax=ax,
                methods_to_show=methods_to_show,
            )
            if title is not None:
                ax.set_title(title)
            agg_results[benchmark] = agg_result
            if benchmark in PLOT_RANGE:
                plotargs = PLOT_RANGE[benchmark]
                ax.set_ylim([plotargs.ymin, plotargs.ymax])
                ax.set_xlim([plotargs.xmin, plotargs.xmax])

            if ax is not None:
                plt.tight_layout()
                os.makedirs("figures/", exist_ok=True)
                plt.savefig(f"figures/{prefix}{benchmark}-{metric2plot}.pdf")
            ax = None