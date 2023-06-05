import enum
from functools import partial
from dataclasses import dataclass

from datasets import Dataset

from data_loading.load_swag import load_swag, DataCollatorForMultipleChoice, accuracy
from data_loading.load_glue import load_glue_datasets
from transformers import (
    default_data_collator,
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
    train, valid, test = load_glue_datasets(tokenizer=tokenizer, dataset_name=dataset_name)
    return FineTuningDataset(
        valid_dataset=valid, train_dataset=train, collator=default_data_collator,
        metric=load("glue", dataset_name), type=Tasks.SEQ_CLS
    )


def get_swag(tokenizer):
    valid_dataset, train_dataset = load_swag()
    return FineTuningDataset(
        valid_dataset=valid_dataset,
        train_dataset=train_dataset,
        collator=DataCollatorForMultipleChoice(tokenizer),
        metric=accuracy,
        type=Tasks.MUL_QA
    )


fine_tuning_datasets = {
    "rte": partial(get_glue_dataset, dataset_name='rte'),
    "mrpc": partial(get_glue_dataset, dataset_name='rte'),
    "cola": partial(get_glue_dataset, dataset_name='rte'),
    "stsb": partial(get_glue_dataset, dataset_name='stsb'),
    "sst2": partial(get_glue_dataset, dataset_name='sst2'),
    "qnli": partial(get_glue_dataset, dataset_name='qnli'),
    "qqp":  partial(get_glue_dataset, dataset_name='qqp'),
    "mnli": partial(get_glue_dataset, dataset_name='mnli'),
    "swag": get_swag
}
