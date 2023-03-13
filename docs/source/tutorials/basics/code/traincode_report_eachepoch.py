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
import argparse
import logging

from syne_tune import Reporter
from benchmarking.training_scripts.mlp_on_fashion_mnist.mlp_on_fashion_mnist import (
    download_data,
    split_data,
    model_and_optimizer,
    train_model,
    validate_model,
)


def objective(config):
    # Download data
    data_train = download_data(config)
    # Report results to Syne Tune
    report = Reporter()
    # Split into training and validation set
    train_loader, valid_loader = split_data(config, data_train)
    # Create model and optimizer
    state = model_and_optimizer(config)
    # Training loop
    for epoch in range(1, config["epochs"] + 1):
        train_model(config, state, train_loader)
        # Report validation accuracy to Syne Tune
        accuracy = validate_model(config, state, valid_loader)
        report(epoch=epoch, accuracy=accuracy)


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    # Hyperparameters
    parser.add_argument("--n_units_1", type=int, required=True)
    parser.add_argument("--n_units_2", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--dropout_1", type=float, required=True)
    parser.add_argument("--dropout_2", type=float, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)

    args, _ = parser.parse_known_args()
    # Evaluate objective and report results to Syne Tune
    objective(config=vars(args))
