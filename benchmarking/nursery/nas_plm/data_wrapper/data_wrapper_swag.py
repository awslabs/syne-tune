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
import torch
import numpy as np

from dataclasses import dataclass
from typing import Optional, Union

from datasets import load_dataset

from torch.utils.data import Subset
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy,
)
from data_wrapper import DataWrapper

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


class SWAG(DataWrapper):
    def _load_data(self):
        raw_datasets = load_dataset(
            "swag", "regular", cache_dir=self.model_args.cache_dir
        )

        def preprocess_function(examples):
            # Repeat each first sentence four times to go with the four possibilities of second sentences.
            first_sentences = [[context] * 4 for context in examples["sent1"]]
            # Grab all second sentences possible for each context.
            question_headers = examples["sent2"]
            ending_names = ["ending0", "ending1", "ending2", "ending3"]
            second_sentences = [
                [f"{header} {examples[end][i]}" for end in ending_names]
                for i, header in enumerate(question_headers)
            ]

            # Flatten everything
            first_sentences = sum(first_sentences, [])
            second_sentences = sum(second_sentences, [])

            # Tokenize
            tokenized_examples = self.tokenizer(
                first_sentences, second_sentences, truncation=True
            )
            # Un-flatten
            return {
                k: [v[i : i + 4] for i in range(0, len(v), 4)]
                for k, v in tokenized_examples.items()
            }

        with self.training_args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        train_dataset = raw_datasets["train"]
        # Split training dataset in training / validation
        split = train_dataset.train_test_split(
            train_size=0.7, seed=self.data_args.dataset_seed
        )  # fix seed, all trials have the same data split
        train_dataset = split["train"]
        valid_dataset = split["test"]

        valid_dataset = Subset(
            valid_dataset,
            np.random.choice(len(valid_dataset), 2048).tolist(),
        )

        test_dataset = raw_datasets["validation"]

        return train_dataset, valid_dataset, test_dataset

    def get_data_collator(self):
        from transformers.trainer_utils import RemoveColumnsCollator

        data_collator = RemoveColumnsCollator(
            DataCollatorForMultipleChoice(tokenizer=self.tokenizer),
            ["input_ids", "attention_mask", "label"],
        )
        return data_collator
