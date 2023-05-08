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
import numpy as np

from typing import Optional, List, Union, Dict, Any

from syne_tune.optimizer.schedulers.searchers import StochasticSearcher
from syne_tune.config_space import Domain, Float, Integer, Categorical, FiniteRange

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.evaluator import Evaluator
from pymoo.problems.static import StaticProblem
from pymoo.core.mixed import (
    MixedVariableMating,
    MixedVariableSampling,
    MixedVariableDuplicateElimination,
)
from pymoo.core.variable import Real, Choice
from pymoo.core.variable import Integer as PyMOOInteger


class MultiObjectiveMixedVariableProblem(Problem):
    def __init__(self, n_obj, config_space, **kwargs):
        vars = {}

        for hp_name, hp in config_space.items():
            if isinstance(hp, Categorical):
                vars[hp_name] = Choice(options=hp.categories)
            elif isinstance(hp, Integer):
                vars[hp_name] = PyMOOInteger(bounds=(hp.lower, hp.upper))
            elif isinstance(hp, FiniteRange):
                vars[hp_name] = PyMOOInteger(bounds=(0, hp.size - 1))
            elif isinstance(hp, Float):
                vars[hp_name] = Real(bounds=(hp.lower, hp.upper))

        super().__init__(vars=vars, n_obj=n_obj, n_ieq_constr=0, **kwargs)


class NSGA2Searcher(StochasticSearcher):
    """
    This is a wrapper around the NSGA-2 [1] implementation of pymoo [2].

    [1] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan.
    A fast and elitist multiobjective genetic algorithm: nsga-II.
    Trans. Evol. Comp, 6(2):182–197, April 2002.

    [2] J. Blank and K. Deb
    pymoo: Multi-Objective Optimization in Python
    IEEE Access, 2020

    :param config_space: Configuration space
    :param metric: Name of metric passed to :meth:`~update`. Can be obtained from
        scheduler in :meth:`~configure_scheduler`. In the case of multi-objective optimization,
         metric is a list of strings specifying all objectives to be optimized.
    :param points_to_evaluate: List of configurations to be evaluated
        initially (in that order). Each config in the list can be partially
        specified, or even be an empty dict. For each hyperparameter not
        specified, the default value is determined using a midpoint heuristic.
        If ``None`` (default), this is mapped to ``[dict()]``, a single default config
        determined by the midpoint heuristic. If ``[]`` (empty list), no initial
        configurations are specified.
    :param mode: Should metric be minimized ("min", default) or maximized
        ("max"). In the case of multi-objective optimization, mode can be a list defining for
        each metric if it is minimized or maximized
    :param population_size: Size of the population, defaults to 100
    """

    def __init__(
        self,
        config_space,
        metric: List[str],
        mode: Union[List[str], str],
        points_to_evaluate: Optional[List[dict]] = None,
        population_size: int = 100,
        **kwargs,
    ):
        super(NSGA2Searcher, self).__init__(
            config_space, metric, points_to_evaluate=points_to_evaluate, **kwargs
        )
        if isinstance(mode, str):
            self._mode = [mode] * len(metric)
        else:
            self._mode = mode

        self.hp_names = []
        for hp_name, hp in config_space.items():
            if isinstance(hp, Domain):
                self.hp_names.append(hp_name)
                if (
                    not isinstance(hp, Categorical)
                    and not isinstance(hp, Integer)
                    and not isinstance(hp, Float)
                    and not isinstance(hp, FiniteRange)
                ):
                    raise Exception(
                        f"Type {type(hp)} for hyperparameter {hp_name} "
                        f"is not support for NSGA-2."
                    )

        self.problem = MultiObjectiveMixedVariableProblem(
            config_space=config_space, n_var=len(self.hp_names), n_obj=len(metric)
        )
        self.algorithm = NSGA2(
            pop_size=population_size,
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(
                eliminate_duplicates=MixedVariableDuplicateElimination()
            ),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
        )
        self.algorithm.setup(
            problem=self.problem, termination=("n_eval", 2**32 - 1), verbose=False
        )

        self.current_population = self.algorithm.ask()
        self.current_individual = 0
        self.observed_values = []

    def _update(self, trial_id: str, config: dict, result: dict):
        observed_metrics = {}
        for mode, metric in zip(self._mode, self._metric):
            value = result[metric]
            if mode == "max":
                value *= -1
            observed_metrics[metric] = value

        self.observed_values.append(list(observed_metrics.values()))

        if len(self.observed_values) == len(self.current_population):
            static = StaticProblem(self.problem, F=np.array(self.observed_values))
            Evaluator().eval(static, self.current_population)
            # self.algorithm.evaluator.eval(self.problem, self.current_population)
            self.algorithm.tell(infills=self.current_population)

            self.current_population = self.algorithm.ask()
            self.observed_values = []
            self.current_individual = 0

    def get_config(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Suggest a new configuration.

        Note: Query :meth:`_next_initial_config` for initial configs to return
        first.

        :param kwargs: Extra information may be passed from scheduler to
            searcher
        :return: New configuration. The searcher may return None if a new
            configuration cannot be suggested. In this case, the tuning will
            stop. This happens if searchers never suggest the same config more
            than once, and all configs in the (finite) search space are
            exhausted.
        """

        if self.current_individual >= len(self.current_population):
            raise Exception(
                "It seems that some configurations are sill pending, while querying a new configuration."
                "Note that NSGA-2 does not support asynchronous scheduling. To avoid this behavious, "
                "make sure to set num_workers = 1."
            )
        else:
            individual = self.current_population[self.current_individual]

        self.current_individual += 1
        config = {}
        for hp_name, hp in self.config_space.items():
            if isinstance(hp, FiniteRange):
                config[hp_name] = hp.values[individual.x[hp_name]]
            else:
                config[hp_name] = individual.x[hp_name]
        return config


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from syne_tune.config_space import uniform, randint, choice

    config_space = {
        "x0": uniform(0, 1),
        "x1": uniform(0, 1),
        "x2": randint(1, 100),
        "x3": choice(["a", "b"]),
    }
    pop_size = 50
    method = NSGA2Searcher(
        config_space, metric=["f0", "f1"], mode=["min", "min"], pop_size=pop_size
    )
    f = plt.figure(dpi=200)
    color = 0
    for i in range(300):
        config = method.get_config()
        f0 = (0.5 - config["x0"]) ** 2
        f1 = (0.5 - config["x1"]) ** 2
        color = i // pop_size
        plt.scatter(f0, f1, color=f"C{color}")
        method.on_trial_result(
            trial_id=i, config=config, result={"f0": f0, "f1": f1}, update=True
        )
    plt.show()