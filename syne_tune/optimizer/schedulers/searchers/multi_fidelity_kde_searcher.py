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
from typing import Dict, Optional, List
import logging
import numpy as np

from syne_tune.optimizer.schedulers.searchers.kde_searcher import KernelDensityEstimator

__all__ = ['MultiFidelityKernelDensityEstimator']

logger = logging.getLogger(__name__)


class MultiFidelityKernelDensityEstimator(KernelDensityEstimator):
    """
    Adapts the KernelDensityEstimator to the multi-fidelity setting as proposed by Falkner et al such that we can use
    it with Hyperband. Following Falkner et al, we fit the KDE only on the highest resource level where we
    have at least num_min_data_points.

    BOHB: Robust and Efficient Hyperparameter Optimization at Scale
    S. Falkner and A. Klein and F. Hutter
    Proceedings of the 35th International Conference on Machine Learning


    Parameters
    ----------
    config_space: dict
        Configuration space for trial evaluation function
    metric : str
        Name of metric to optimize, key in result's obtained via
        `on_trial_result`
    mode : str
        Mode to use for the metric given, can be 'min' or 'max', default to 'min'.
    num_min_data_points: int
        Minimum number of data points that we use to fit the KDEs. If set to None than we set this to the number of
        hyperparameters.
    top_n_percent: int
        Determines how many datapoints we use use to fit the first KDE model for modeling the well
        performing configurations.
    min_bandwidth: float
        The minimum bandwidth for the KDE models
    num_candidates: int
        Number of candidates that are sampled to optimize the acquisition function
    bandwidth_factor: int
        We sample continuous hyperparameter from a truncated Normal. This factor is multiplied to the bandwidth to
        define the standard deviation of this trunacted Normal.
    random_fraction: float
        Defines the fraction of configurations that are drawn uniformly at random instead of sampling from the model
    points_to_evaluate: List[Dict] or None
        List of configurations to be evaluated initially (in that order).
        Each config in the list can be partially specified, or even be an
        empty dict. For each hyperparameter not specified, the default value
        is determined using a midpoint heuristic.
        If None (default), this is mapped to [dict()], a single default config
        determined by the midpoint heuristic. If [] (empty list), no initial
        configurations are specified.
    """

    def __init__(
            self,
            configspace: Dict,
            metric: str,
            mode: str = "min",
            num_min_data_points: int = None,
            top_n_percent: int = 15,
            min_bandwidth: float = 0.1,
            num_candidates: int = 64,
            bandwidth_factor: int = 3,
            random_fraction: float = .33,
            resource_attr: str = 'epochs',
            points_to_evaluate: Optional[List[Dict]] = None,
            **kwargs
    ):
        super().__init__(configspace, metric, mode, num_min_data_points,
                         top_n_percent, min_bandwidth, num_candidates, bandwidth_factor, random_fraction,
                         points_to_evaluate, **kwargs)

        self.resource_attr = resource_attr

        self.resource_levels = []

    def _fit_kde_on_highest_resource_level(self, config, result):
        resource_level = result[self.resource_attr]
        self.resource_levels.append(resource_level)

        self.X.append(self.to_feature(
            config=config,
            configspace=self.configspace,
            categorical_maps=self.categorical_maps,
        ))
        self.y.append(self.to_objective(result))

        unique_resource_levels, counts = np.unique(self.resource_levels, return_counts=True)
        idx = np.where(counts >= self.num_min_data_points)[0]
        if len(idx) == 0:
            return

        highest_resource_level = unique_resource_levels[idx[-1]]
        indices = np.where(self.resource_levels == highest_resource_level)[0]

        train_data = np.array([self.X[i] for i in indices])
        train_targets = np.array([self.y[i] for i in indices])

        self.train_kde(train_data, train_targets)

    def _update(self, trial_id: str, config: Dict, result: Dict):
        self._fit_kde_on_highest_resource_level(config, result)
