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
from typing import Optional, List, Dict, Any, Union
import logging

from syne_tune.optimizer.schedulers.random_seeds import RANDOM_SEED_UPPER_BOUND
from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher
from syne_tune.optimizer.schedulers.searchers.searcher_factory import searcher_factory
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    check_and_merge_defaults,
    String,
    assert_no_invalid_options,
    Integer,
)
from syne_tune.optimizer.scheduler import (
    SchedulerDecision,
    TrialSuggestion,
)
from syne_tune.optimizer.schedulers.scheduler_searcher import TrialSchedulerWithSearcher

from syne_tune.backend.time_keeper import TimeKeeper, RealTimeKeeper
from syne_tune.backend.trial_status import Trial
from syne_tune.config_space import cast_config_values

logger = logging.getLogger(__name__)


_ARGUMENT_KEYS = {
    "searcher",
    "search_options",
    "metric",
    "mode",
    "points_to_evaluate",
    "random_seed",
    "max_resource_attr",
    "max_t",
    "time_keeper",
}

_DEFAULT_OPTIONS = {
    "searcher": "random",
    "mode": "min",
}

_CONSTRAINTS = {
    "random_seed": Integer(0, RANDOM_SEED_UPPER_BOUND),
    "max_resource_attr": String(),
    "max_t": Integer(1, None),
}


MetricModeType = Union[str, List[str]]


def _to_list(x) -> list:
    return x if isinstance(x, list) else [x]


class FIFOScheduler(TrialSchedulerWithSearcher):
    """Scheduler which executes trials in submission order.

    This is the most basic scheduler template. It can be configured to many use
    cases by choosing ``searcher`` along with ``search_options``.

    :param config_space: Configuration space for evaluation function
    :type config_space: Dict[str, Any]
    :param searcher: Searcher for ``get_config`` decisions. String values
        are passed to
        :func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory` along
        with ``search_options`` and extra information. Supported values:
        :const:`~syne_tune.optimizer.schedulers.searchers.searcher_factory.SUPPORTED_SEARCHERS_FIFO`.
        Defaults to "random" (i.e., random search)
    :type searcher: str or
        :class:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher`
    :param search_options: If searcher is ``str``, these arguments are
        passed to
        :func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory`
    :type search_options: Dict[str, Any], optional
    :param metric: Name of metric to optimize, key in results obtained via
        ``on_trial_result``. For multi-objective schedulers, this can also be a
        list
    :type metric: str or List[str]
    :param mode: "min" if ``metric`` is minimized, "max" if ``metric`` is
        maximized, defaults to "min". This can also be a list if ``metric`` is
        a list
    :type mode: str or List[str], optional
    :param points_to_evaluate: List of configurations to be evaluated
        initially (in that order). Each config in the list
        can be partially specified, or even be an empty dict. For each
        hyperparameter not specified, the default value is determined using
        a midpoint heuristic.
        If not given, this is mapped to ``[dict()]``, a single default config
        determined by the midpoint heuristic. If ``[]`` (empty list), no initial
        configurations are specified.
        Note: If ``searcher`` is of type :class:`BaseSearcher`,
        ``points_to_evaluate`` must be set there.
    :type points_to_evaluate: ``List[dict]``, optional
    :param random_seed: Master random seed. Generators used in the
        scheduler or searcher are seeded using :class:`RandomSeedGenerator`.
        If not given, the master random seed is drawn at random here.
    :type random_seed: int, optional
    :param max_resource_attr: Key name in config for fixed attribute
        containing the maximum resource. If this is given, ``max_t`` is not
        needed. We recommend to use ``max_resource_attr`` over ``max_t``.
        If given, we use it to infer ``max_resource_level``. It is also
        used to limit trial executions in promotion-based multi-fidelity
        schedulers (see class:``HyperbandScheduler``, ``type="promotion"``).
    :type max_resource_attr: str, optional
    :param max_t: Value for ``max_resource_level``. Needed for
        schedulers which make use of intermediate reports via
        ``on_trial_result``. If this is not given, we try to infer its value
        from ``config_space`` (see
        :class:`~syne_tune.optimizer.schedulers.ResourceLevelsScheduler`).
        checking ``config_space["epochs"]``, ``config_space["max_t"]``, and
        ``config_space["max_epochs"]``. If ``max_resource_attr`` is given, we use
        the value ``config_space[max_resource_attr]``. But if ``max_t`` is given
        here, it takes precedence.
    :type max_t: int, optional
    :param time_keeper: This will be used for timing here (see
        ``_elapsed_time``). The time keeper has to be started at the beginning
        of the experiment. If not given, we use a local time keeper here,
        which is started with the first call to :meth:`_suggest`. Can also be set
        after construction, with :meth:`set_time_keeper`.
        Note: If you use
        :class:`~syne_tune.backend.simulator_backend.SimulatorBackend`, you need
        to pass its ``time_keeper`` here.
    :type time_keeper: :class:`~syne_tune.backend.time_keeper.TimeKeeper`,
        optional
    """

    def __init__(self, config_space: Dict[str, Any], **kwargs):
        super().__init__(config_space, **kwargs)
        # Check values and impute default values
        assert_no_invalid_options(kwargs, _ARGUMENT_KEYS, name="FIFOScheduler")
        kwargs = check_and_merge_defaults(
            kwargs, set(), _DEFAULT_OPTIONS, _CONSTRAINTS, dict_name="scheduler_options"
        )
        self.metric, self.mode = self._check_metric_mode(
            kwargs.get("metric"), kwargs["mode"]
        )
        self.max_resource_attr = kwargs.get("max_resource_attr")
        # Setting max_t (if not provided as argument -> self.max_t)
        # This value can often be obtained from config_space. We check these
        # attributes (in order): epochs, max_t, max_epochs.
        # In any case, the max_t argument takes precedence. If it is None, we use
        # the one inferred from config_space.
        self.max_t = self._infer_max_resource_level(
            kwargs.get("max_t"), self.max_resource_attr
        )
        # Generate searcher
        searcher = kwargs["searcher"]
        if isinstance(searcher, str):
            search_options = kwargs.get("search_options")
            if search_options is None:
                search_options = dict()
            else:
                search_options = search_options.copy()
            search_options.update(
                {
                    "config_space": self.config_space.copy(),
                    "metric": self.metric,
                    "points_to_evaluate": kwargs.get("points_to_evaluate"),
                    "mode": kwargs["mode"],
                    "random_seed_generator": self.random_seed_generator,
                }
            )
            if self.max_t is not None:
                search_options["max_epochs"] = self.max_t
            # Subclasses may extend ``search_options``
            search_options = self._extend_search_options(search_options)
            # Adjoin scheduler info to search_options, if not already done by
            # subclass (via ``_extend_search_options``)
            if "scheduler" not in search_options:
                search_options["scheduler"] = "fifo"
            self._searcher: BaseSearcher = searcher_factory(searcher, **search_options)
        else:
            assert isinstance(searcher, BaseSearcher)
            self._searcher: BaseSearcher = searcher

        self._start_time = None  # Will be set at first ``suggest``
        # Time keeper
        time_keeper = kwargs.get("time_keeper")
        if time_keeper is not None:
            self.set_time_keeper(time_keeper)
        else:
            self.time_keeper = None

    @property
    def searcher(self) -> Optional[BaseSearcher]:
        return self._searcher

    def set_time_keeper(self, time_keeper: TimeKeeper):
        """Assign time keeper after construction.

        This is possible only if the time keeper was not assigned at
        construction, and the experiment has not yet started.

        :param time_keeper: Time keeper to be used
        """
        assert self.time_keeper is None, "Time keeper has already been assigned"
        assert isinstance(
            time_keeper, TimeKeeper
        ), "Argument must be of type TimeKeeper"
        self.time_keeper = time_keeper

    @staticmethod
    def _check_metric_mode(
        metric: MetricModeType, mode: MetricModeType
    ) -> (MetricModeType, MetricModeType):
        assert metric is not None, (
            "Argument `metric` is mandatory. Pass the name of the metric "
            + "reported by your training script, which you'd like to "
            + "optimize, and use `mode` to specify whether it should "
            + "be minimized or maximized"
        )
        if isinstance(metric, list):
            num_objectives = len(metric)
        else:
            num_objectives = 1
            metric = [metric]
        assert all(
            isinstance(x, str) for x in metric
        ), "Argument `metric` must be string or list of strings"
        if isinstance(mode, list):
            len_mode = len(mode)
            if len_mode == 1:
                mode = mode[0]
            else:
                assert (
                    len_mode == num_objectives
                ), "If arguments `metric`, `mode` are lists, they must have the same length"
        else:
            len_mode = 1
        if len_mode == 1:
            mode = [mode] * num_objectives
        allowed_values = {"min", "max"}
        assert all(
            x in allowed_values for x in mode
        ), "Value(s) of `mode` must be 'min' or 'max'"
        if num_objectives == 1:
            metric = metric[0]
            mode = mode[0]
        return metric, mode

    def _extend_search_options(self, search_options: Dict[str, Any]) -> Dict[str, Any]:
        """Allows child classes to extend ``search_options``.

        :param search_options: Original dict of options
        :return: Extended dict, to use instead of ``search_options``
        """
        return search_options

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        """Implements ``suggest``, except for basic postprocessing of config.

        We first check whether a paused trial can be promoted (by calling
        ``_promote_trial``). If this is not possible, we ask the searcher to
        suggest a config (by ``get_config``) for the new trial ``trial_id``.

        :param trial_id: ID for new trial to be started (ignored if existing
            trial to be resumed)
        :return: Suggestion for a trial to be started or to be resumed, see
            above. If no suggestion can be made, None is returned
        """
        # If no time keeper was provided at construction, we use a local
        # one which is started here
        if self.time_keeper is None:
            self.time_keeper = RealTimeKeeper()
            self.time_keeper.start_of_time()
        # For pause/resume schedulers: Can a paused trial be promoted?
        promote_trial_id, extra_kwargs = self._promote_trial(new_trial_id=str(trial_id))
        if promote_trial_id is not None:
            return TrialSuggestion.resume_suggestion(
                trial_id=int(promote_trial_id), config=extra_kwargs
            )
        # Ask searcher for config of new trial to start
        extra_kwargs["elapsed_time"] = self._elapsed_time()
        trial_id = str(trial_id)
        config = self.searcher.get_config(**extra_kwargs, trial_id=trial_id)
        if config is not None:
            config = cast_config_values(config, self.config_space)
            config = self._on_config_suggest(config, trial_id, **extra_kwargs)
            config = TrialSuggestion.start_suggestion(config)
        return config

    def _on_config_suggest(
        self, config: Dict[str, Any], trial_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Called by ``suggest`` to allow scheduler to register a new config.

        We register the config here, not in ``on_trial_add``. While this risks
        registering a config which is not successfully started, this is the
        right thing to do for batch suggestions. There, ``suggest`` is called
        multiple times in a row, and the batch trials are started together.
        If we did not register pending configs after being suggested (but
        before getting started), fantasizing would not be used for them.

        :param config: New config suggested for ``trial_id``
        :param trial_id: Input to ``_suggest``
        :param kwargs: Optional. Additional args
        :return: Configuration, potentially modified
        """
        self.searcher.register_pending(trial_id=trial_id, config=config)
        if self.searcher.debug_log is not None:
            # For log outputs:
            config = dict(config, trial_id=trial_id)
        return config

    def _promote_trial(self, new_trial_id: str) -> (Optional[str], Optional[dict]):
        """Checks whether any paused trial can be promoted.

        Has to be implemented by pause/resume schedulers.

        The second return argument, ``extra_kwargs``, plays different roles
        depending on the first return argument:

        * If ``trial_id is None`` (no promotion): ``extra_kwargs`` are args to be
          passed to ``get_config`` call of searcher.
        * If ``trial_id not None`` (promotion): ``extra_kwargs`` may be None or a
          dict. If a dict, ``extra_kwargs`` is used to update the config of the
          trial to be promoted. In this case, ``suggest`` will return the
          tuple ``(trial_id, extra_kwargs)``.

        :param new_trial_id: ID for new trial to be started, as passed to
            :meth:`_suggest`
        :return: ``(trial_id, extra_kwargs)``
        """
        return None, dict()

    def _elapsed_time(self):
        """
        :return: Time elapsed since start of experiment, as measured by
            ``self.time_keeper``
        """
        assert self.time_keeper is not None, "Experiment has not been started yet"
        return self.time_keeper.time()

    @staticmethod
    def _check_keys_of_result(result: Dict[str, Any], keys: List[str]):
        assert all(key in result for key in keys), (
            "Your training evaluation function needs to report values for the "
            + f"keys {keys}:\n   report("
            + ", ".join([f"{key}=..." for key in keys])
            + ", ...)"
        )

    def _check_result(self, result: Dict[str, Any]):
        self._check_keys_of_result(result, self.metric_names())

    def on_trial_result(self, trial: Trial, result: Dict[str, Any]) -> str:
        """
        We simply relay ``result`` to the searcher. Other decisions are done
        in ``on_trial_complete``.
        """
        self._check_result(result)
        trial_id = str(trial.trial_id)
        trial_decision = SchedulerDecision.CONTINUE
        config = self._preprocess_config(trial.config)
        self.searcher.on_trial_result(trial_id, config, result=result, update=False)
        # Extra info in debug mode
        log_msg = f"trial_id {trial_id} ("
        if self.is_multiobjective_scheduler():
            metrics = {k: result[k] for k in self.metric}
        else:
            metrics = {"metric": result[self.metric]}
        log_msg += ", ".join([f"{k} = {v:.3f}" for k, v in metrics.items()])
        for k, is_float in (("epoch", False), ("elapsed_time", True)):
            if k in result:
                if is_float:
                    log_msg += f", {k} = {result[k]:.2f}"
                else:
                    log_msg += f", {k} = {result[k]}"
        log_msg += f"): decision = {trial_decision}"
        logger.debug(log_msg)
        return trial_decision

    def metric_names(self) -> List[str]:
        return _to_list(self.metric)

    def metric_mode(self) -> Union[str, List[str]]:
        return self.mode

    def is_multiobjective_scheduler(self) -> bool:
        return isinstance(self.metric, list)
