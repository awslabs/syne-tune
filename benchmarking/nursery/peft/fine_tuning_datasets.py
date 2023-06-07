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
import enum
import numpy as np

from functools import partial
from dataclasses import dataclass

from datasets import Dataset

from data_loading.load_swag import load_swag, DataCollatorForMultipleChoice, accuracy
from data_loading.load_glue import load_glue_datasets
from transformers import (
    default_data_collator,
    EvalPrediction
)


from evaluate import load


class Tasks(str, enum.Enum):
    SEQ_CLS = "sequence_classification"
    MUL_QA = "multiple_choice_qa"


@dataclass
class FineTuningDataset:

    valid_dataset: Dataset
    train_dataset: Dataset
    collator: object
    metric: object
    type: str


def get_glue_dataset(tokenizer, dataset_name):
    train, valid, test = load_glue_datasets(
        tokenizer=tokenizer, dataset_name=dataset_name
    )

    metric = load("glue", dataset_name)
    is_regression = True if dataset_name == 'stsb' else False
    def compute_metrics(p: EvalPrediction, is_regression):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    return FineTuningDataset(
        valid_dataset=valid,
        train_dataset=train,
        collator=default_data_collator,
        metric=partial(compute_metrics, is_regression=is_regression),
        type=Tasks.SEQ_CLS,
    )


def get_swag(tokenizer):
    valid_dataset, train_dataset = load_swag()
    return FineTuningDataset(
        valid_dataset=valid_dataset,
        train_dataset=train_dataset,
        collator=DataCollatorForMultipleChoice(tokenizer),
        metric=accuracy,
        type=Tasks.MUL_QA,
    )


fine_tuning_datasets = {
    "rte": partial(get_glue_dataset, dataset_name="rte"),
    "mrpc": partial(get_glue_dataset, dataset_name="rte"),
    "cola": partial(get_glue_dataset, dataset_name="rte"),
    "stsb": partial(get_glue_dataset, dataset_name="stsb"),
    "sst2": partial(get_glue_dataset, dataset_name="sst2"),
    "qnli": partial(get_glue_dataset, dataset_name="qnli"),
    "qqp": partial(get_glue_dataset, dataset_name="qqp"),
    "mnli": partial(get_glue_dataset, dataset_name="mnli"),
    "swag": get_swag,
}
