import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

from syne_tune.optimizer.scheduler import TrialScheduler


@dataclass
class TransferLearningTaskEvaluations:
    """Class that contains offline evaluations for a task that can be used for transfer learning.
    Args:
        configuration_space: Dict the configuration space that was used when sampling evaluations.
        hyperparameters: pd.DataFrame the hyperparameters values that were acquired, all keys of configuration-space
         should appear as columns.
        objectives_names: List[str] the name of the objectives that were acquired
        objectives_evaluations: np.array values of recorded objectives, must have shape
            (num_evals, num_seeds, num_fidelities, num_objectives)
    """
    configuration_space: Dict
    hyperparameters: pd.DataFrame
    objectives_names: List[str]
    objectives_evaluations: np.array

    def __post_init__(self):
        assert len(self.objectives_names) == self.objectives_evaluations.shape[-1]
        assert len(self.hyperparameters) == self.objectives_evaluations.shape[0]
        for col in self.hyperparameters.keys():
            assert col in self.configuration_space

    def objective_values(self, objective_name: str) -> np.array:
        return self.objectives_evaluations[..., self.objective_index(objective_name=objective_name)]

    def objective_index(self, objective_name: str) -> int:
        matches = [i for i, name in enumerate(self.objectives_names) if name == objective_name]
        assert len(matches) >= 1, f"could not find objective {objective_name} in recorded objectives " \
                                  f"{self.objectives_names}"
        return matches[0]


class TransferLearningScheduler(TrialScheduler):

    def __init__(
            self,
            config_space: Dict,
            transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
            metric_names: List[str],
    ):
        """
        A scheduler that can levarages offline evaluations of related tasks.
        :param config_space: configuration space to be sampled from
        :param transfer_learning_evaluations: dictionary from task name to offline evaluations.
        :param metric_names: name of the metric to be optimized.
        """
        super(TransferLearningScheduler, self).__init__(config_space=config_space)
        for task, evals in transfer_learning_evaluations.items():
            for key in config_space.keys():
                assert key in evals.hyperparameters.columns, \
                    f"the key {key} of the config space should appear in transfer learning evaluations " \
                    f"hyperparameters {evals.hyperparameters.columns}"
            assert all([m in evals.objectives_names for m in metric_names]), \
                f"all objectives used in the scheduler {self.metric_names()} should appear in transfer learning " \
                f"evaluations objectives {evals.objectives_names}"
        self._metric_names = metric_names

    def metric_names(self) -> List[str]:
        return self._metric_names
