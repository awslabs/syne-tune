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
import logging
import numpy as np

from datasets import load_dataset
from torch.utils.data import Subset, DataLoader

from data_wrapper import DataWrapper
from task_data import GLUE_TASK_INFO

logger = logging.getLogger(__name__)


class IMDB(DataWrapper):
    def _load_data(self):
        raw_datasets = load_dataset("imdb", cache_dir=self.model_args.cache_dir)

        def preprocess_function(examples):
            # Tokenize the texts
            result = self.tokenizer(
                examples["text"],
                padding=self.padding,
                max_length=self.max_seq_length,
                truncation=True,
            )

            return result

        with self.training_args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        train_dataset = raw_datasets["train"]
        test_dataset = raw_datasets["test"]

        # Split training dataset in training / validation
        split = train_dataset.train_test_split(
            train_size=0.7, seed=self.data_args.dataset_seed
        )  # fix seed, all trials have the same data split
        train_dataset = split["train"]
        valid_dataset = split["test"]

        return train_dataset, valid_dataset, test_dataset
