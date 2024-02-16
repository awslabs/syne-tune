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
import numpy as np

from datasets import load_dataset
from torch.utils.data import Subset

from task_data import GLUE_TASK_INFO
from data_wrapper import DataWrapper


class Glue(DataWrapper):
    def _load_data(self):
        raw_datasets = load_dataset(
            "glue", self.data_args.task_name, cache_dir=self.model_args.cache_dir
        )

        # Preprocessing the raw_datasets
        sentence1_key, sentence2_key = GLUE_TASK_INFO[self.data_args.task_name]["keys"]

        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],)
                if sentence2_key is None
                else (examples[sentence1_key], examples[sentence2_key])
            )
            result = self.tokenizer(
                *args,
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
        test_dataset = raw_datasets[
            "validation_matched" if self.data_args.task_name == "mnli" else "validation"
        ]

        train_dataset = train_dataset.remove_columns(["idx"])
        test_dataset = test_dataset.remove_columns(["idx"])

        # Split training dataset in training / validation
        split = train_dataset.train_test_split(
            train_size=0.7, seed=self.data_args.dataset_seed
        )  # fix seed, all trials have the same data split
        train_dataset = split["train"]
        valid_dataset = split["test"]

        if self.data_args.task_name in ["sst2", "qqp", "qnli", "mnli"]:
            valid_dataset = Subset(
                valid_dataset,
                np.random.choice(len(valid_dataset), 2048).tolist(),
            )

        return train_dataset, valid_dataset, test_dataset
