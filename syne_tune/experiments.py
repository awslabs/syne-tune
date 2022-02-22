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
import json
import logging
from datetime import datetime
from typing import List, Dict, Callable, Optional

import boto3
import pandas as pd

from dataclasses import dataclass

from botocore.exceptions import ClientError

from syne_tune.constants import ST_TUNER_TIME, ST_TUNER_CREATION_TIMESTAMP
from syne_tune import Tuner
from syne_tune.util import experiment_path, s3_experiment_path


@dataclass
class ExperimentResult:
    name: str
    results: pd.DataFrame
    metadata: Dict
    tuner: Tuner

    def __str__(self):
        res = f"Experiment {self.name}"
        if self.results is not None:
            res += f" contains {len(self.results)} evaluations from {len(self.results.trial_id.unique())} trials"
        return res

    def creation_date(self):
        return datetime.fromtimestamp(self.metadata[ST_TUNER_CREATION_TIMESTAMP])

    def plot(self, **plt_kwargs):
        import matplotlib.pyplot as plt

        metric = self.metric_names()[0]
        df = self.results
        df = df.sort_values(ST_TUNER_TIME)
        x = df.loc[:, ST_TUNER_TIME]
        y = df.loc[:, metric].cummax() if self.metric_mode() == "max" else df.loc[:, metric].cummin()
        plt.plot(x, y, **plt_kwargs)
        plt.xlabel("wallclock time")
        plt.ylabel(metric)
        plt.title(self.entrypoint_name() + " " + self.name)
        plt.legend()
        plt.show()

    def metric_mode(self) -> str:
        return self.metadata['metric_mode']

    def metric_names(self) -> List[str]:
        return self.metadata['metric_names']

    def entrypoint_name(self) -> str:
        return self.metadata['entrypoint']

    def best_config(self) -> Dict:
        """
        Return the best config found for the first metric defined in the scheduler.
        :param self:
        :return:
        """
        metric_names = self.metric_names()
        metric_mode = self.metric_mode()

        if len(metric_names) > 1:
            logging.warning("Several metrics exists so the best is not defined, this will return the best other the"
                            f"first metric {metric_names}.")
        metric_name = metric_names[0]

        # locate best result
        if metric_mode == 'min':
            best_index = self.results.loc[:, metric_name].argmin()
        else:
            best_index = self.results.loc[:, metric_name].argmax()
        res = dict(self.results.loc[best_index])

        # dont include internal fields
        return {k: v for k, v in res.items() if not k.startswith("st_")}

def download_single_experiment(
        tuner_name: str,
        s3_bucket: Optional[str] = None,
        experiment_name: Optional[str] = None,
):
    """
    Downloads results from s3 of a tuning experiment previously run with remote launcher.
    :param tuner_name: named of the tuner to be retrieved.
    :param s3_bucket: If not given, the default bucket for the SageMaker session is used
    :param experiment_name: If given, this is used as first directory.
    :return:
    """
    s3_path = s3_experiment_path(s3_bucket=s3_bucket, tuner_name=tuner_name, experiment_name=experiment_name)
    tgt_dir = experiment_path(tuner_name=tuner_name)
    tgt_dir.mkdir(exist_ok=True, parents=True)
    s3 = boto3.client('s3')
    s3_bucket = s3_path.replace("s3://", "").split("/")[0]
    s3_key = "/".join(s3_path.replace("s3://", "").split("/")[1:])
    for file in ["metadata.json", "results.csv.zip", "tuner.dill"]:
        try:
            logging.info(f"downloading {file} on {s3_path}")
            s3.download_file(s3_bucket, f"{s3_key}/{file}", str(tgt_dir / file))
        except ClientError as e:
            logging.info(f"could not find {file} on {s3_path}")


def load_experiment(
        tuner_name: str,
        download_if_not_found: bool = True,
        force_download: bool = False,
) -> ExperimentResult:
    """
    :param tuner_name: name of a tuning experiment previously run
    :param download_if_not_found: whether to fetch the experiment from s3 if not found locally
    :param force_download: whether to force fetching the experiment from s3 even if found locally
    :return:
    """
    if force_download and download_if_not_found:
        logging.info(f"force_download=True so trying to download experiment {tuner_name} from s3.")
        download_single_experiment(tuner_name=tuner_name)
    
    path = experiment_path(tuner_name)

    metadata_path = path / "metadata.json"
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        logging.info(f"experiment {tuner_name} not found locally, trying to get it from s3.")
        if download_if_not_found:
            download_single_experiment(tuner_name=tuner_name)
        metadata = None
    try:
        if (path / "results.csv.zip").exists():
            results = pd.read_csv(path / "results.csv.zip")
        else:
            results = pd.read_csv(path / "results.csv")
    except Exception:
        results = None
    try:
        tuner = Tuner.load(path)
    except FileNotFoundError:
        tuner = None
    except Exception:
        tuner = None

    return ExperimentResult(
        name=tuner.name if tuner is not None else path.stem,
        results=results,
        tuner=tuner,
        metadata=metadata,
    )


def list_experiments(
        name_filter: Callable[[str], bool] = None,
        experiment_filter: Callable[[ExperimentResult], bool] = None
) -> List[ExperimentResult]:
    res = []
    for path in experiment_path().rglob("*/results.csv*"):
        if name_filter is None or name_filter(str(path)):
            exp = load_experiment(path.parent.name)
            if experiment_filter is None or experiment_filter(exp):
                if exp.results is not None and exp.tuner is not None and exp.metadata is not None:
                    res.append(exp)
    return sorted(res, key=lambda exp: exp.metadata.get(ST_TUNER_CREATION_TIMESTAMP, 0), reverse=True)



# TODO: Use conditional imports, in order not to fail if dependencies are not
# installed
def scheduler_name(scheduler):
    from syne_tune.optimizer.schedulers.fifo import FIFOScheduler

    if isinstance(scheduler, FIFOScheduler):
        scheduler_name = f"ST-{scheduler.__class__.__name__}"
        searcher = scheduler.searcher.__class__.__name__
        return "-".join([scheduler_name, searcher])
    else:
        from syne_tune.optimizer.schedulers.ray_scheduler import RayTuneScheduler

        if isinstance(scheduler, RayTuneScheduler):
            scheduler_name = f"Ray-{scheduler.scheduler.__class__.__name__}"
            searcher = scheduler.searcher.__class__.__name__
            return "-".join([scheduler_name, searcher])
        else:
            return scheduler.__class__.__name__


def load_experiments_df(
        name_filter: Callable[[str], bool] = None,
        experiment_filter: Callable[[ExperimentResult], bool] = None
) -> pd.DataFrame:
    """
    :param: name_filter: if specified, only experiment whose name matches the filter will be kept.
    :param experiment_filter: only experiment where the filter is True are kept, default to None and returns everything.
    :return: a dataframe that contains all evaluations reported by tuners according to the filter given.
    The columns contains trial-id, hyperparameter evaluated, metrics observed by `report`:
     metrics collected automatically by syne-tune:
     `st_worker_time` (indicating time spent in the worker when report was seen)
     `time` (indicating wallclock time measured by the tuner)
     `decision` decision taken by the scheduler when observing the result
     `status` status of the trial that was shown to the tuner
     `config_{xx}` configuration value for the hyperparameter {xx}
     `tuner_name` named passed when instantiating the Tuner
     `entry_point_name`/`entry_point_path` name and path of the entry point that was tuned
    """
    dfs = []
    for experiment in list_experiments(name_filter=name_filter, experiment_filter=experiment_filter):
        assert experiment.tuner is not None
        assert experiment.results is not None
        assert experiment.metadata is not None

        df = experiment.results
        df["tuner_name"] = experiment.name
        df["scheduler"] = scheduler_name(experiment.tuner.scheduler)
        df["backend"] = experiment.tuner.trial_backend.__class__.__name__
        for k, v in experiment.metadata.items():
            if isinstance(v, List):
                if len(v) > 1:
                    for i, x in enumerate(v):
                        df[f"{k}-{i}"] = x
                else:
                    df[k] = v[0]
            else:
                df[k] = v
        metrics = experiment.tuner.scheduler.metric_names()
        if len(metrics) == 1:
            df["metric"] = experiment.tuner.scheduler.metric_names()[0]
        else:
            # assume error is always the first
            df["metric"] = experiment.tuner.scheduler.metric_names()[0]

        scheduler_mode = experiment.tuner.scheduler.metric_mode()
        if isinstance(scheduler_mode, str):
            df["mode"] = scheduler_mode
        else:
            df["mode"] = scheduler_mode[0]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


if __name__ == '__main__':
    for exp in list_experiments():
        if exp.results is not None:
            print(exp)

