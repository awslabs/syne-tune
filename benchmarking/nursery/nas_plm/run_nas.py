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
import os

logging.basicConfig(level=logging.INFO)

from pathlib import Path
from transformers import AutoModelForSequenceClassification

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint
from syne_tune.experiments import load_experiment
from baselines import MethodArguments, methods

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset_seed", type=int)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--runtime", type=int)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--iterations", type=int, default=-1)
    parser.add_argument("--method", type=str, default="random_search")

    args, _ = parser.parse_known_args()

    config_space = {
        "num_layers": randint(0, 12),
        "num_heads": randint(0, 12),
        "num_units": randint(0, 3072),
        "model_name_or_path": args.model_name,
        "output_dir": "./nas_output",
        "task_name": args.dataset,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": 2e-05,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 8,
        "seed": args.seed,
        "dataset_seed": args.dataset_seed,
    }

    if args.dataset == "stsb":
        config_space["is_regression"] = True

    base_scheduler = methods[args.method](
        MethodArguments(
            config_space=config_space,
            metrics=["valid", "params"],
            mode=["min", "min"],
            random_seed=args.seed,
        )
    )
    if args.iterations > -1:
        stop_criterion = (StoppingCriterion(max_num_trials_finished=args.iterations),)
    else:
        stop_criterion = StoppingCriterion(max_wallclock_time=args.runtime)

    tuner = Tuner(
        trial_backend=LocalBackend(
            entry_point=str(Path(__file__).parent / "run_from_scratch_nas.py")
        ),
        scheduler=base_scheduler,
        # scheduler=LS(
        #     config_space,
        #     metric=["valid", "params"],
        #     mode=["min", "min"],
        #     search_options={"debug_log": False},
        # ),
        stop_criterion=stop_criterion,
        n_workers=1,  # how many trials are evaluated in parallel
    )
    tuner.run()

    df = load_experiment(tuner.name).results

    runtime_traj = []
    params = []
    test_error = []
    valid_error = []
    configs = []

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for trial, trial_df in df.groupby("trial_id"):
        idx = trial_df.valid.argmin()
        params.append(float(trial_df.params.iloc[0]) * total_params)
        valid_error.append(float(trial_df.valid.iloc[idx]))
        runtime_traj.append(float(trial_df.st_tuner_time.iloc[-1]))
        test_error.append(float(trial_df.test.iloc[idx]))

        config = []
        for hyper in config_space.keys():
            c = trial_df[hyper].iloc[idx]
            config.append(c)
        configs.append(config)

    results = {
        "runtime_traj": runtime_traj,
        "params": params,
        "valid_error": valid_error,
        "test_error": test_error,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(results, open(args.output_dir + "/results.json", "w"))
