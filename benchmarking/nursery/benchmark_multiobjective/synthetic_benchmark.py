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
from typing import Optional, Dict, Union, Any, Callable, List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from syne_tune.constants import ST_TUNER_TIME

from benchmarking.nursery.benchmark_multiobjective.baselines import methods, Methods

from benchmarking.commons.baselines import MethodArguments
from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.blackbox_repository import UserBlackboxBackend
from syne_tune.blackbox_repository.blackbox import Blackbox
from syne_tune.config_space import uniform, randint, choice
from syne_tune.experiments import load_experiment, ExperimentResult
from syne_tune.optimizer.scheduler import TrialScheduler
from syne_tune.optimizer.schedulers import FIFOScheduler
from syne_tune.util import catchtime


def circle_location(theta: float, r: float) -> np.ndarray:
    return np.array([-r * np.cos(theta), -r * np.sin(theta)])


class MOOArtificialBlackbox(Blackbox):
    objectives_names: List[str] = ["x", "y"]

    def __init__(
            self,
            target_function: Callable[[Any], np.ndarray] = circle_location
    ):
        self.num_fidelities = 1
        self.target_function = target_function
        super(MOOArtificialBlackbox, self).__init__(
            configuration_space={
                "theta": uniform(0, np.pi / 2),
                "r": uniform(0, 1),
            },
            fidelity_space={"time": choice([None])},
            objectives_names=self.objectives_names,
        )

    def _objective_function(
            self,
            configuration: Dict[str, Any],
            fidelity: Optional[dict] = None,
            seed: Optional[int] = None,
    ) -> Union[Dict[str, float], np.ndarray]:
        """

        :param configuration: configuration to be evaluated, should belong to
            :attr:`configuration_space`
        :param fidelity: not passing a fidelity is possible if either the blackbox
            does not have a fidelity space or if it has a single fidelity in its
            fidelity space. In the latter case, all fidelities are returned in
            form of a tensor with shape ``(num_fidelities, num_objectives)``.
        :param seed: Only used if the blackbox defines multiple seeds
        :return: dictionary of objectives evaluated or tensor with shape
            ``(num_fidelities, num_objectives)`` if no fidelity was given.
        """
        if fidelity is None:
            result = np.empty((self.num_fidelities, len(self.objectives_names)))
            result[0][:] = self.target_function(**configuration)
            return result
        else:
            raise

    @property
    def fidelity_values(self) -> Optional[np.array]:
        return np.arange(1, self.num_fidelities + 1)


def benchmark_method(
        name: str,
        method: Callable[[MethodArguments], TrialScheduler],
        blackbox: Blackbox,
        max_num_evaluations: int = 200,
        n_workers: int = 1
) -> ExperimentResult:
    mode = ["min", "min"]
    metric = ["x", "y"]

    trial_backend = UserBlackboxBackend(
        blackbox=blackbox,
        elapsed_time_attr="time",
    )
    with catchtime(f"run simulated tuning for {name} - {method}"):
        scheduler = method(MethodArguments(
            config_space=blackbox.configuration_space,
            mode=mode,
            metric=metric,
            random_seed=31415927,
            resource_attr="time",
        ))

        stop_criterion = StoppingCriterion(max_num_evaluations=max_num_evaluations)

        tuner = Tuner(
            trial_backend=trial_backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=n_workers,
            sleep_time=0.0,
            callbacks=[SimulatorCallback()],
            metadata=dict(multi_objective_metric_names=metric)
        )
        tuner.run()

    return load_experiment(tuner_name=tuner.name)


def plot_sampled_values(experiment: ExperimentResult, metrics_to_plot: Tuple[str, str], figure_path: str = None):
    """
    Plot all the points sampled by the method
    """
    results_df = experiment.results.sort_values(ST_TUNER_TIME).reset_index()
    results_df = experiment.results.sort_values(ST_TUNER_TIME).reset_index()
    num_iterations = len(results_df)
    for idx in np.linspace(0, num_iterations, num=10):
        idx = int(idx)
        local_results = results_df[0: idx]

        x = local_results.index.to_list()
        ys = [
            local_results.loc[:, metric_name] for metric_name in metrics_to_plot
        ]
        fig, ax = plt.subplots()
        ax.scatter(ys[0], ys[1])
        ax.set_xlabel(metrics_to_plot[0])
        ax.set_ylabel(metrics_to_plot[1])
        ax.set_title(f"Sampled values for {experiment.name}")
        if figure_path is not None:
            assert figure_path.endswith(".jpeg"), "Only jpeg supported for now"
            fig.savefig(figure_path.replace(".jpeg", f"-{idx}.jpeg"))
        else:
            fig.show()

def plot_sampled_configs(experiment: ExperimentResult, config_to_plot: str, figure_path: str = None):
    """
    Plot all the points sampled by the method
    """
    results_df = experiment.results.sort_values(ST_TUNER_TIME).reset_index()
    num_iterations = len(results_df)
    for idx in np.linspace(0, num_iterations, num=10):
        idx = int(idx)
        local_results = results_df[0: idx]
        x = local_results[f"config_{config_to_plot}"]
        fig, ax = plt.subplots()
        ax.hist(x)
        ax.set_title(f"Sampled points for {experiment.name}")
        if figure_path is not None:
            assert figure_path.endswith(".jpeg"), "Only jpeg supported for now"
            fig.savefig(figure_path.replace(".jpeg", f"-{idx}.jpeg"))
        else:
            fig.show()


if __name__ == "__main__":
    for name in [
        Methods.MOREA,
        Methods.RS,
        Methods.LSOBO,

    ]:
        method = methods[name]
        # --- Benchmarking starts
        res = benchmark_method(
            name=name,
            method=method,
            blackbox=MOOArtificialBlackbox(target_function=circle_location)
        )
        plot_sampled_values(
            experiment=res,
            metrics_to_plot=("x", "y"),
        )
        plot_sampled_configs(
            experiment=res,
            config_to_plot="theta",
        )
        plot_sampled_configs(
            experiment=res,
            config_to_plot="r",
        )
