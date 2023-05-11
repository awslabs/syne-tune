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
import logging

from typing import Optional, List, Dict, Any, Union

from syne_tune.optimizer.schedulers import FIFOScheduler
from syne_tune.optimizer.schedulers.searchers import StochasticSearcher
from syne_tune.optimizer.schedulers.searchers.searcher_base import (
    sample_random_configuration,
)

logger = logging.getLogger(__name__)


class MORS(FIFOScheduler):
    """

    See :class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param population_size: See
        :class:`~syne_tune.optimizer.schedulers.searchers.RegularizedEvolution`.
        Defaults to 100
    :param sample_size: See
        :class:`~syne_tune.optimizer.schedulers.searchers.RegularizedEvolution`.
        Defaults to 10
    :param random_seed: Random seed, optional
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: List[str],
        mode: Union[List[str], str] = "min",
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        super(MORS, self).__init__(
            config_space=config_space,
            metric=metric,
            mode=mode,
            searcher=ModifiedRandomSearch(
                config_space=config_space,
                metric=metric,
                mode=mode,
                random_seed=random_seed,
            ),
            random_seed=random_seed,
            **kwargs,
        )


class ModifiedRandomSearch(StochasticSearcher):
    """ """

    def __init__(
        self,
        config_space,
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
        mode: str = "min",
        **kwargs,
    ):
        super(ModifiedRandomSearch, self).__init__(
            config_space, metric, points_to_evaluate=points_to_evaluate, **kwargs
        )

    def _sample_random_config(self) -> Dict[str, Any]:
        return sample_random_configuration(self._hp_ranges, self.random_state)

    def get_config(self, **kwargs) -> Optional[dict]:
        hp = list(self.config_space.keys())[0]
        if hp.startswith("layer_"):
            num_layers = 0
            for hp in self.config_space:
                if hp.startswith("layer_mha_"):
                    num_layers += 1
            n_layers = self.random_state.randint(0, num_layers * 2)
            layers = self.random_state.choice(
                np.arange(num_layers * 2), n_layers, replace=False
            )
            config = {}
            for hp in self.config_space:
                config[hp] = 0

            for li in layers:
                if li < num_layers:
                    config[f"layer_mha_{li}"] = 1
                else:
                    config[f"layer_ffn_{li - num_layers}"] = 1

        else:
            config = self._sample_random_config()

        return config

    def _update(self, trial_id: str, config: Dict[str, Any], result: Dict[str, Any]):
        pass

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.scheduler_searcher import (
            TrialSchedulerWithSearcher,
        )

        assert isinstance(
            scheduler, TrialSchedulerWithSearcher
        ), "This searcher requires TrialSchedulerWithSearcher scheduler"
        super().configure_scheduler(scheduler)

    def clone_from_state(self, state: Dict[str, Any]):
        raise NotImplementedError


if __name__ == "__main__":
    import torch

    from transformers import AutoConfig

    from nas_fine_tuning.sampling import LayerSearchSpace

    config = AutoConfig.from_pretrained("bert-base-cased")
    ss = LayerSearchSpace(config)

    rs = ModifiedRandomSearch(
        ss.get_syne_tune_config_space(), metric=["a", "b"], random_seed=412
    )
    print(rs.get_config())
