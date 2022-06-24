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
import os
from time import perf_counter
from typing import Dict, List, Tuple
import copy

from syne_tune.backend.trial_status import Trial

import pandas as pd

from syne_tune.constants import ST_DECISION, ST_TRIAL_ID, ST_STATUS, ST_TUNER_TIME
from syne_tune.util import RegularCallback


class TunerCallback:
    def on_tuning_start(self, tuner):
        pass

    def on_tuning_end(self):
        pass

    def on_loop_start(self):
        pass

    def on_loop_end(self):
        pass

    def on_fetch_status_results(
        self,
        trial_status_dict: Tuple[Dict[int, Tuple[Trial, str]]],
        new_results: List[Tuple[int, Dict]],
    ):
        """
        Called with the results of `backend.fetch_status_results`.
        """
        pass

    def on_trial_complete(self, trial: Trial, result: Dict):
        """
        Called when a trial completes.
        :param trial: trial that just completed.
        :param result: last result obtained.
        :return:
        """
        pass

    def on_trial_result(self, trial: Trial, status: str, result: Dict, decision: str):
        """
        Called when a new result is observed.
        :param trial:
        :param status: a string representing the status that is one value of `trial_status.Status`, such as
        `trial_status.Status.completed`.
        :param result:
        :param decision: decision that was returned by the scheduler
        :return:
        """
        pass

    def on_tuning_sleep(self, sleep_time: float):
        """
        Called when the tuner is put to sleep when no worker is available.
        :param sleep_time:
        :return:
        """
        pass


class StoreResultsCallback(TunerCallback):
    def __init__(
        self,
        add_wallclock_time: bool = True,
    ):
        """
        Minimal callback that enables plotting results over time,
        additional callback functionalities will be added as well as example to plot results over time.

        :param add_wallclock_time: whether to add wallclock time to results.
        """
        self.results = []

        self.csv_file = None
        self.save_results_at_frequency = None
        self.add_wallclock_time = add_wallclock_time
        self._start_time_stamp = None

    def _set_time_fields(self, result: Dict):
        """
        Note that we only add wallclock time to the result if this has not
        already been done (by the back-end)
        """
        if self._start_time_stamp is not None and ST_TUNER_TIME not in result:
            result[ST_TUNER_TIME] = perf_counter() - self._start_time_stamp

    def on_trial_result(self, trial: Trial, status: str, result: Dict, decision: str):
        assert (
            self.save_results_at_frequency is not None
        ), "on_tuning_start must always be called before on_trial_result."
        result = copy.copy(result)
        result[ST_DECISION] = decision
        result[ST_STATUS] = status
        result[ST_TRIAL_ID] = trial.trial_id

        for key in trial.config:
            result[f"config_{key}"] = trial.config[key]

        self._set_time_fields(result)

        self.results.append(result)

        if self.csv_file is not None:
            self.save_results_at_frequency()

    def store_results(self):
        if self.csv_file is not None:
            self.dataframe().to_csv(self.csv_file, index=False)

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def on_tuning_start(self, tuner):
        # we set the path of the csv file once the tuner is created since the path may change when the tuner is stop
        # and resumed again on a different machine.
        self.csv_file = str(tuner.tuner_path / "results.csv.zip")

        # we only save results every `results_update_frequency` seconds as this operation
        # may be expensive on remote storage.
        self.save_results_at_frequency = RegularCallback(
            lambda: self.store_results(),
            call_seconds_frequency=tuner.results_update_interval,
        )
        if self.add_wallclock_time:
            self._start_time_stamp = perf_counter()

    def on_tuning_end(self):
        # store the results in case some results were not committed yet (since they are saved every
        # `results_update_interval` seconds)
        self.store_results()


class TensorboardCallback(TunerCallback):
    def __init__(
            self,

            metric_to_optimize: 'str',
            mode: 'str',
            output_metrics: List[str] = [],
    ):
        """
        Simple callback that logs different statistics such that we can visualize them with Tensorboard.

        :param metric_to_optimize: The metric that we aim to optimize. We report the cumulative optimum over time. This
        should be the same metric that we pass to the scheduler / searcher.
        :param mode: Defines whether we minimize ('min') or maximize ('max'). Should be the same what we
        pass to scheduler / searcher.
        :param output_metrics: All other metrics that we want to log. Make sure that these metrics
         are reported correctly in the train function to Syne Tune via the report(...) function.
        """
        self.results = []

        self.output_metrics = output_metrics
        self.mode = mode
        self.metric = metric_to_optimize

        self.curr_best_value = None
        self.curr_best_config = None

        self._start_time_stamp = None
        self.writer = None
        self._iter = None
        self.trial_ids = set()

    def _set_time_fields(self, result: Dict):
        """
        Note that we only add wallclock time to the result if this has not
        already been done (by the back-end)
        """
        if self._start_time_stamp is not None and ST_TUNER_TIME not in result:
            result[ST_TUNER_TIME] = perf_counter() - self._start_time_stamp

    def on_trial_result(self, trial: Trial, status: str, result: Dict, decision: str):

        self._set_time_fields(result)

        new_result = result[self.metric]

        if self.mode == 'max':
            new_result *= -1

        if self.curr_best_value is None or self.curr_best_value > new_result:
            self.curr_best_value = new_result
            self.curr_best_config = trial.config
            self.writer.add_scalar(self.metric, result[self.metric], self._iter)

        else:
            opt = self.curr_best_value
            if self.mode == 'max':
                opt *= -1
            self.writer.add_scalar(self.metric, opt, self._iter)

        for key, value in self.curr_best_config.items():
            self.writer.add_scalar(f'optimal_{key}', value, self._iter)

        for metric in self.output_metrics:
            self.writer.add_scalar(metric, result[metric], self._iter)

        self.writer.add_scalar('runtime', result[ST_TUNER_TIME], self._iter)

        self.trial_ids.add(trial.trial_id)
        self.writer.add_scalar('num_trial', len(self.trial_ids),
                               self._iter, display_name='total number of trials')

        self._iter += 1

    def on_tuning_start(self, tuner):

        output_path = os.path.join(tuner.tuner_path, 'tensorboard_output')
        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(output_path)
        self._iter = 0
        self._start_time_stamp = perf_counter()

    def on_tuning_end(self):
        self.writer.close()
        self.writer = None
