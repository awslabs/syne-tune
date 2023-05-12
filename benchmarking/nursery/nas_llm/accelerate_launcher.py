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
import sys
import os
import subprocess
import json
import sys
import logging
from argparse import ArgumentParser

logger = logging.getLogger(__name__)


# hyperparameters defined in estimator as passed to your training container
# and saved in /opt/ml/input/config/hyperparameters.json. SM reads these hyperparameters
# and parse_args() is used to extract them
def parse_args():
    parser = ArgumentParser(
        description=(
            "SageMaker DeepSpeed Launch helper utility that will spawn deepspeed training scripts"
        )
    )
    parser.add_argument(
        "--training_script",
        type=str,
        help="Path to the training program/script to be run in parallel, can be either absolute or relative",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the accelerate config file",
    )

    # rest from the training program
    parsed, nargs = parser.parse_known_args()

    return parsed.training_script, parsed.config_file, nargs


def accelerate_launch(command):
    try:
        proc = subprocess.run(command, shell=True)
    except Exception as e:
        logger.info(e)

    sys.exit(proc.returncode)


def main():
    train_script, config_file, args = parse_args()
    command = (
        f"accelerate launch --config_file {config_file} {train_script} {' '.join(args)}"
    )
    print(f"command = {command}")
    # launch deepspeed training
    accelerate_launch(command)


if __name__ == "__main__":
    main()
