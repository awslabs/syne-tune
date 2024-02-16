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

from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)


logger = logging.getLogger(__name__)


class DataWrapper(object):
    def __init__(self, training_args, model_args, data_args):
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        self.model_type = model_args.model_name_or_path

        # Load tokenizer
        self.tokenizer = self.get_tokenizer()

        if (
            self.model_type.startswith("gpt2")
            or "pythia" in self.model_type
            or self.model_type.startswith("distilgpt2")
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Padding strategy
        if self.data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        # Determine max_seq_length
        if self.data_args.max_seq_length > self.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({self.data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
            )
        self.max_seq_length = min(
            self.data_args.max_seq_length, self.tokenizer.model_max_length
        )

        self.train_data, self.valid_data, self.test_data = self._load_data()

        data_collator = self.get_data_collator()

        self.train_dataloader = DataLoader(
            self.train_data,
            batch_size=self.training_args.per_device_eval_batch_size,
            collate_fn=data_collator,
        )

        self.eval_dataloader = DataLoader(
            self.valid_data,
            batch_size=self.training_args.per_device_eval_batch_size,
            collate_fn=data_collator,
        )

        self.test_dataloader = DataLoader(
            self.test_data,
            batch_size=self.training_args.per_device_eval_batch_size,
            collate_fn=data_collator,
        )

        self.num_labels = self.get_num_labels(self.data_args)

    def get_num_labels(self, data_args):
        if data_args.is_regression:
            num_labels = 1
        else:
            label_list = self.train_data.features["label"].names
            num_labels = len(label_list)
        return num_labels

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_type,
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )

    def get_data_loaders(self):
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_collator(self):
        if self.data_args.pad_to_max_length:
            data_collator = default_data_collator
        elif self.training_args.fp16:
            data_collator = DataCollatorWithPadding(
                self.tokenizer, pad_to_multiple_of=8
            )
        else:
            data_collator = None
        return data_collator

    def _load_data(self):
        pass
