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

from datasets import load_dataset


logger = logging.getLogger(__name__)


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def load_glue_datasets(tokenizer, dataset_name):
    raw_datasets = load_dataset(
        "glue", dataset_name
    )

    # Preprocessing the raw_datasets
    sentence1_key, sentence2_key = task_to_keys[dataset_name]

    # Padding strategy
    padding = "max_length"
    print(tokenizer.model_max_length)
    max_seq_length = tokenizer.model_max_length
    # max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )

        # Map labels to IDs (not necessary for GLUE tasks)
        # if label_to_id is not None and "label" in examples:
        #     result["label"] = [
        #         (label_to_id[l] if l != -1 else -1) for l in examples["label"]
        #     ]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
    )

    train_dataset = raw_datasets["train"]
    test_dataset = raw_datasets[
        "validation_matched" if dataset_name == "mnli" else "validation"
    ]

    train_dataset = train_dataset.remove_columns(["idx"])
    test_dataset = test_dataset.remove_columns(["idx"])

    # Split training dataset in training / validation
    split = train_dataset.train_test_split(
        train_size=0.7, seed=0
    )  # fix seed, all trials have the same data split
    valid_dataset = split["test"]

    return train_dataset, valid_dataset, test_dataset
