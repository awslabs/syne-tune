from typing import Dict
import datetime

from syne_tune.backend.trial_status import Trial, Status, TrialResult
from syne_tune.optimizer.scheduler import TrialScheduler


class AskTellScheduler:
    bscheduler: TrialScheduler
    trial_counter: int
    completed_experiments: Dict[int, TrialResult]

    def __init__(self, base_scheduler: TrialScheduler):
        self.bscheduler = base_scheduler
        self.trial_counter = 0
        self.completed_experiments = {}

    def ask(self) -> Trial:
        """
        Ask the scheduler for new trial to run
        :return: Trial to run
        """
        trial_suggestion = self.bscheduler.suggest(self.trial_counter)
        trial = Trial(
            trial_id=self.trial_counter,
            config=trial_suggestion.config,
            creation_time=datetime.datetime.now(),
        )
        self.trial_counter += 1
        return trial

    def tell(self, trial: Trial, experiment_result: Dict[str, float]):
        """
        Feed experiment results back to the Scheduler

        :param trial: Trial that was run
        :param experiment_result: {metric: value} dictionary with experiment results
        """
        trial_result = trial.add_results(
            metrics=experiment_result,
            status=Status.completed,
            training_end_time=datetime.datetime.now(),
        )
        self.bscheduler.on_trial_complete(trial=trial, result=experiment_result)
        self.completed_experiments[trial_result.trial_id] = trial_result

    def best_trial(self, metris: str) -> TrialResult:
        """
        Return the best trial according to the provided metric
        """
        if self.bscheduler.mode == "max":
            sign = 1.0
        else:
            sign = -1.0

        return max(
            [value for key, value in self.completed_experiments.items()],
            key=lambda trial: sign * trial.metrics[metris],
        )
