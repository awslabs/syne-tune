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
import itertools
import json
import logging
from pathlib import Path
import time
from time import struct_time
from typing import Callable, Dict, Any, Optional, Union, Tuple, List

import pandas as pd

from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    s3_download_files_recursively,
)
from syne_tune.constants import (
    ST_DATETIME_FORMAT,
    ST_METADATA_FILENAME,
    ST_RESULTS_DATAFRAME_FILENAME,
)
from syne_tune.util import experiment_path, s3_experiment_path

logger = logging.getLogger(__name__)


_MapMetadataToSetup = Callable[[Dict[str, Any]], Optional[str]]

MapMetadataToSetup = Union[_MapMetadataToSetup, Dict[str, _MapMetadataToSetup]]

MapMetadataToSubplot = Callable[[Dict[str, Any]], Optional[int]]


def _strip_common_prefix(tuner_path: str) -> str:
    prefix_path = str(experiment_path())
    assert tuner_path.startswith(
        prefix_path
    ), f"tuner_path = {tuner_path}, prefix_path = {prefix_path}"
    start_pos = len(prefix_path)
    if tuner_path[start_pos] in ["/", "\\"]:
        start_pos += 1
    return tuner_path[start_pos:]


def _insert_into_nested_dict(
    metadata_values: Dict[str, Any],
    benchmark_name: str,
    key: str,
    setup_name: str,
    value: Any,
):
    inner_dict = metadata_values
    for name in [benchmark_name, key]:
        if name not in inner_dict:
            inner_dict[name] = dict()
        inner_dict = inner_dict[name]
    if setup_name in inner_dict:
        inner_dict[setup_name].append(value)
    else:
        inner_dict[setup_name] = [value]


DateTimeInterval = Tuple[Optional[str], Optional[str]]

DateTimeBounds = Union[DateTimeInterval, Dict[str, DateTimeInterval]]


def _convert_datetime_bound(
    bound: DateTimeInterval,
) -> Tuple[Optional[struct_time], Optional[struct_time]]:
    result = ()
    assert len(bound) == 2
    for elem in bound:
        if elem is not None:
            elem = time.strptime(elem, ST_DATETIME_FORMAT)
        result = result + (elem,)
    lower, upper = result
    assert (
        lower is None or upper is None or lower < upper
    ), f"Invalid time bound {bound}: First must be before second"
    return result


def _convert_datetime_bounds(
    datetime_bounds: Optional[DateTimeBounds], experiment_names: Tuple[str, ...]
) -> Dict[str, Tuple[Optional[struct_time], Optional[struct_time]]]:
    if datetime_bounds is None:
        result = {name: (None, None) for name in experiment_names}
    elif isinstance(datetime_bounds, dict):
        result = dict()
        for name in experiment_names:
            bound = datetime_bounds.get(name)
            result[name] = None if bound is None else _convert_datetime_bound(bound)
    else:
        tbound = _convert_datetime_bound(datetime_bounds)
        result = {name: tbound for name in experiment_names}
    return result


def _extract_datetime(tuner_name: str) -> struct_time:
    assert (
        tuner_name[-4] == "-" and tuner_name[-24] == "-"
    ), f"tuner_name = {tuner_name} has invalid format. Postfix should be like {ST_DATETIME_FORMAT}-XYZ"
    return time.strptime(tuner_name[-23:-4], ST_DATETIME_FORMAT)


def _skip_based_on_datetime(
    tuner_name: str,
    datetime_lower: Optional[struct_time],
    datetime_upper: Optional[struct_time],
) -> bool:
    if datetime_lower is None and datetime_upper is None:
        return False
    datetime = _extract_datetime(tuner_name)
    return (datetime_lower is not None and datetime < datetime_lower) or (
        datetime_upper is not None and datetime > datetime_upper
    )


def _get_benchmark_name(
    benchmark_key: Optional[str], metadata: Dict[str, Any], tuner_path: Path
) -> str:
    if benchmark_key is not None:
        assert benchmark_key in metadata, (
            f"Metadata for tuner_path = {tuner_path} does not contain "
            f"key {benchmark_key}:\n{metadata}"
        )
        benchmark_name = metadata[benchmark_key]
    else:
        benchmark_name = "SINGLE_BENCHMARK"  # Key for single dict entry
    return benchmark_name


def create_index_for_result_files(
    experiment_names: Tuple[str, ...],
    metadata_to_setup: MapMetadataToSetup,
    metadata_to_subplot: Optional[MapMetadataToSubplot] = None,
    metadata_keys: Optional[List[str]] = None,
    benchmark_key: Optional[str] = "benchmark",
    with_subdirs: Optional[Union[str, List[str]]] = "*",
    datetime_bounds: Optional[DateTimeBounds] = None,
) -> Dict[str, Any]:
    """
    Helper function for :class:`ComparativeResults`.

    Runs over all result directories for experiments of a comparative study.
    For each experiment, we read the metadata file, extract the benchmark name
    (key ``benchmark_key``), and use ``metadata_to_setup``,
    ``metadata_to_subplot`` to map the metadata to setup name and subplot index.
    If any of the two return ``None``, the result is not used. Otherwise, we
    enter ``(result_path, setup_name, subplot_no)`` into the list for benchmark
    name.
    Here, ``result_path`` is the result path for the experiment, without the
    :meth:`~syne_tune.util.experiment_path` prefix. The index returned is the
    dictionary from benchmark names to these list. It allows loading results
    specifically for each benchmark, and we do not have to load and parse the
    metadata files again.

    If ``benchmark_key is None``, the returned index is a dictionary with a
    single element only, and the metadata files need not contain an entry for
    benchmark name.

    Result files have the path
    ``f"{experiment_path()}{ename}/{patt}/{ename}-*/"``, where ``path`` is from
    ``with_subdirs``, and ``ename`` from ``experiment_names``. The default is
    ``with_subdirs="*"``. If ``with_subdirs`` is ``None``, result files have
    the path ``f"{experiment_path()}{ename}-*/"``. This is an older convention,
    which makes it harder to sync files from S3, it is not recommended.

    If ``metadata_keys`` is given, it contains a list of keys into the
    metadata. In this case, a nested dictionary ``metadata_values`` is
    returned, where ``metadata_values[benchmark_name][key][setup_name]``
    contains a list of metadata values for this benchmark, key in
    ``metadata_keys``, and setup name.

    If ``datetime_bounds`` is given, it contains a tuple of strings
    ``(lower_time, upper_time)``, or a dictionary mapping experiment names (from
    ``experiment_names``) to such tuples. Both strings are time-stamps in the
    format :const:`~syne_tune.constants.ST_DATETIME_FORMAT` (example:
    "2023-03-19-22-01-57"), and each can be ``None`` as well. This serves to
    filter out any result whose time-stamp does not fall within the interval
    (both sides are inclusive), where ``None`` means the interval is open on
    that side. This feature is useful to filter out results of erroneous
    attempts.

    :param experiment_names: Tuple of experiment names (prefixes, without the
        timestamps)
    :param metadata_to_setup: See above
    :param metadata_to_subplot: See above. Optional
    :param metadata_keys: See above. Optional
    :param benchmark_key: Key for benchmark in metadata files. Defaults to
        "benchmark"
    :param with_subdirs: See above. Defaults to "*"
    :param datetime_bounds: See above
    :return: Dictionary; entry "index" for index (see above); entry
        "setup_names" for setup names encountered; entry "metadata_values" see
        ``metadata_keys``
    """
    reverse_index = dict()
    setup_names = set()
    metadata_values = dict()
    if metadata_keys is None:
        metadata_keys = []
    datetime_bounds = _convert_datetime_bounds(datetime_bounds, experiment_names)
    is_map_dict = isinstance(metadata_to_setup, dict)
    for experiment_name in experiment_names:
        datetime_lower, datetime_upper = datetime_bounds[experiment_name]
        patterns = [experiment_name + "-*/" + ST_METADATA_FILENAME]
        if with_subdirs is not None:
            if not isinstance(with_subdirs, list):
                with_subdirs = [with_subdirs]
            pattern = patterns[0]
            patterns = [experiment_name + f"/{x}/" + pattern for x in with_subdirs]
        logger.info(f"Patterns for result files: {patterns}")
        for meta_path in itertools.chain(
            *[experiment_path().glob(pattern) for pattern in patterns]
        ):
            tuner_path = meta_path.parent
            if _skip_based_on_datetime(tuner_path.name, datetime_lower, datetime_upper):
                continue  # Skip this result
            try:
                with open(str(meta_path), "r") as f:
                    metadata = json.load(f)
            except FileNotFoundError:
                metadata = None
            if metadata is None:
                continue
            benchmark_name = _get_benchmark_name(benchmark_key, metadata, tuner_path)
            try:
                # Extract ``setup_name``
                map = (
                    metadata_to_setup[benchmark_name]
                    if is_map_dict
                    else metadata_to_setup
                )
                setup_name = map(metadata)
            except BaseException as err:
                logger.error(f"Caught exception for {tuner_path}:\n" + str(err))
                raise
            if setup_name is None:
                continue
            if metadata_to_subplot is not None:
                # Extract ``subplot_no``
                subplot_no = metadata_to_subplot(metadata)
            else:
                subplot_no = 0
            if subplot_no is None:
                continue
            if benchmark_name not in reverse_index:
                reverse_index[benchmark_name] = []
            reverse_index[benchmark_name].append(
                (
                    _strip_common_prefix(str(tuner_path)),
                    setup_name,
                    subplot_no,
                )
            )
            setup_names.add(setup_name)
            for key in metadata_keys:
                if key in metadata:
                    _insert_into_nested_dict(
                        metadata_values,
                        benchmark_name,
                        key,
                        setup_name,
                        value=metadata[key],
                    )

    result = {
        "index": reverse_index,
        "setup_names": setup_names,
    }
    if metadata_keys:
        result["metadata_values"] = metadata_values
    return result


def load_results_dataframe_per_benchmark(
    experiment_list: List[Tuple[str, str, int]]
) -> Optional[pd.DataFrame]:
    """
    Helper function for :class:`ComparativeResults`.

    Loads time-stamped results for all experiments in ``experiments_list``
    and returns them in a single dataframe with additional columns
    "setup_name", "suplot_no", "tuner_name", whose values are constant
    across data for one experiment, allowing for later grouping.

    :param experiment_list: Information about experiments, see
        :func:`create_index_for_result_files`
    :return: Dataframe with all results combined
    """
    dfs = []
    for tuner_path, setup_name, subplot_no in experiment_list:
        tuner_path = experiment_path() / tuner_path
        df_filename = str(tuner_path / ST_RESULTS_DATAFRAME_FILENAME)
        try:
            df = pd.read_csv(df_filename)
        except FileNotFoundError:
            df = None
        except Exception as ex:
            logger.error(f"{df_filename}: Error in pd.read_csv\n{ex}")
            df = None
        if df is None:
            logger.warning(
                f"{tuner_path}: Meta-data matches filter, but "
                "results file not found. Skipping."
            )
        else:
            df["setup_name"] = setup_name
            df["subplot_no"] = subplot_no
            df["tuner_name"] = tuner_path.name
            dfs.append(df)

    if not dfs:
        res_df = None
    else:
        res_df = pd.concat(dfs, ignore_index=True)
    return res_df


# HIER: This is VERY slow! Try to run "aws s3 sync" as a subprocess, as in
# sagemaker_utils.download_sagemaker_results. Only if this fails ...
def download_result_files_from_s3(
    experiment_names: Tuple[str, ...],
    s3_bucket: Optional[str] = None,
):
    """
    Recursively downloads result files from S3. This works only if the result
    objects on S3 have prefixes ``f"{s3_experiment_path(s3_bucket)}{ename}/"``,
    where ``ename`` is in ``experiment_names``. Only files with names
    :const:` ST_METADATA_FILENAME` and :const:`ST_RESULTS_DATAFRAME_FILENAME`
    are downloaded.

    :param experiment_names: Tuple of experiment names (prefixes, without the
        timestamps)
    :param s3_bucket: If not given, the default bucket for the SageMaker session
        is used
    """
    for experiment_name in experiment_names:
        s3_source_path = s3_experiment_path(s3_bucket) + experiment_name + "/"
        target_path = str(experiment_path() / experiment_name)
        valid_postfixes = [ST_METADATA_FILENAME, ST_RESULTS_DATAFRAME_FILENAME]
        logger.info(f"Downloading result files from {s3_source_path}")
        result = s3_download_files_recursively(
            s3_source_path=s3_source_path,
            target_path=target_path,
            valid_postfixes=valid_postfixes,
        )
        num_action_calls = result["num_action_calls"]
        if num_action_calls == 0:
            logger.info(f"No result files found")
        else:
            num_successful_action_calls = result["num_successful_action_calls"]
            assert num_successful_action_calls == num_action_calls, (
                f"{num_successful_action_calls} files downloaded successfully, "
                + f"{num_action_calls - num_successful_action_calls} failures. "
                + "Error:\n"
                + result["first_error_message"]
            )
            logger.info(f"Downloaded {num_action_calls} result files")
