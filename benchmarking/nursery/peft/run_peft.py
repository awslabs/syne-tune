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

from transformers import (
    AutoModelForMultipleChoice,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    AutoTokenizer,
)

from peft import get_peft_model, LoraConfig, TaskType, PromptEncoderConfig

from transformers.trainer_callback import TrainerCallback

from syne_tune.report import Reporter

from fine_tuning_datasets import fine_tuning_datasets, Tasks
from hf_args import ModelArguments, DataArguments, SearchArguments


class ReportBackMetrics(TrainerCallback):
    """
    This callback is used in order to report metrics back to Syne Tune, using a
    ``Reporter`` object.

    If ``test_dataset`` is given, we also compute and report test set metrics here.
    These are just for final evaluations. HPO must use validation metrics (in
    ``metrics`` passed to ``on_evaluate``).

    If ``additional_info`` is given, it is a static dict reported with each call.
    """

    def __init__(self, trainer, additional_info=None):
        self.trainer = trainer
        self.additional_info = (
            additional_info if additional_info is not None else dict()
        )
        self.report = Reporter()

    def on_evaluate(self, args, state, control, **kwargs):
        results = kwargs["metrics"].copy()
        results["step"] = state.global_step
        results["epoch"] = int(state.epoch)
        for k, v in self.additional_info.items():
            results[k] = v
        # Report results back to Syne Tune
        self.report(**results)


if __name__ == "__main__":

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, SearchArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        search_args,
    ) = parser.parse_args_into_dataclasses()

    model_name = model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if 'gpt' in model_name:
        tokenizer.pad_token = tokenizer.eos_token

    if search_args.peft_method == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=search_args.lora_r,
            lora_alpha=search_args.lora_alpha,
            lora_dropout=search_args.lora_dropout,
        )
    if search_args.peft_method == "p_tuning":
        peft_config = PromptEncoderConfig(
            task_type="SEQ_CLS",
            num_virtual_tokens=search_args.p_tuning_num_virtual_tokens,
            encoder_hidden_size=search_args.p_tuning_encoder_hidden_size,
        )

    dataset = fine_tuning_datasets[data_args.task_name](tokenizer, data_args)

    if dataset.type == Tasks.MUL_QA:
        model = AutoModelForMultipleChoice.from_pretrained(model_name)
    elif dataset.type == Tasks.SEQ_CLS:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if 'gpt' in model_name:
        model.config.pad_token_id = model.config.eos_token_id

    if search_args.peft_method != "fine_tuning":
        model = get_peft_model(model, peft_config)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params}")
    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.valid_dataset,
        tokenizer=tokenizer,
        data_collator=dataset.collator,
        compute_metrics=dataset.metric,
    )

    trainer.add_callback(
        ReportBackMetrics(trainer=trainer, additional_info={"num_params": n_params})
    )

    trainer.train()
