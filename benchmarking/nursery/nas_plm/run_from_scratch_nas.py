#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import logging
import sys
import time

from dataclasses import dataclass, field

import torch
import datasets
import evaluate
import numpy as np
from syne_tune.report import Reporter
from mask import mask_bert, mask_gpt, mask_gpt_neox
from torch.optim import AdamW

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    get_scheduler,
    HfArgumentParser,
    TrainingArguments,
)
from hf_args import DataTrainingArguments, ModelArguments, parse_model_name
from data_wrapper import Glue, IMDB, SWAG
from task_data import GLUE_TASK_INFO
from estimate_efficency import compute_parameters
from model_data import get_model_data


logger = logging.getLogger(__name__)

report = Reporter()


@dataclass
class PruningArguments:
    prune_top_n_layers: int = field(
        default=2,
    )


@dataclass
class NASArguments:
    do_nas: bool = field(
        default=False,
    )

    num_layers: int = field(default=12)
    num_heads: int = field(default=12)
    num_units: int = field(default=3072)
    st_checkpoint_dir: str = field(default=".")


def main():
    start_time = time.time()
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            TrainingArguments,
            PruningArguments,
            NASArguments,
        )
    )

    (
        model_args,
        data_args,
        training_args,
        pruning_args,
        nas_args,
    ) = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Load data
    if data_args.task_name in GLUE_TASK_INFO:
        data = Glue(
            training_args=training_args, model_args=model_args, data_args=data_args
        )
        metric = evaluate.load("glue", data_args.task_name)
        metric_name = GLUE_TASK_INFO[data_args.task_name]["metric"]
    elif data_args.task_name == "imdb":
        data = IMDB(
            training_args=training_args, model_args=model_args, data_args=data_args
        )
        metric = evaluate.load("accuracy")
        metric_name = "accuracy"
    elif data_args.task_name == "swag":
        data = SWAG(
            training_args=training_args, model_args=model_args, data_args=data_args
        )
        metric = evaluate.load("accuracy")
        metric_name = "accuracy"
    # elif data_args.task_name == "imdb":
    #     data = Imdb(training_args=training_args, model_args=model_args, data_args=data_args)
    # elif data_args.task_name == "custom":
    #     data = Custom(training_args=training_args, model_args=model_args, data_args=data_args)
    train_dataloader, eval_dataloader, test_dataloader = data.get_data_loaders()
    num_labels = data.num_labels

    model_type = parse_model_name(model_args)

    config = AutoConfig.from_pretrained(
        model_type,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if data_args.task_name in ["swag"]:
        model_cls = AutoModelForMultipleChoice
    else:
        model_cls = AutoModelForSequenceClassification
    model = model_cls.from_pretrained(
        model_type,
        from_tf=bool(".ckpt" in model_type),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model_type in ["gpt2", "gpt2-medium", "distilgpt2"]:
        model.config.pad_token_id = model.config.eos_token_id
        # if tokenizer.pad_token is None:
        #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # model.resize_token_embeddings(len(tokenizer))

    model_data = get_model_data(model)

    mask = model_data["mask"]
    num_attention_heads = model_data["num_attention_heads"]
    attention_head_size = model_data["attention_head_size"]
    num_layers = model_data["num_layers"]
    intermediate_size = model_data["intermediate_size"]
    n_params_emb = model_data["n_params_emb"]
    n_params_classifier = model_data["n_params_classifier"]
    attention_size = model_data["attention_size"]

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    head_mask = torch.ones((num_layers, num_attention_heads))
    neuron_mask = torch.ones((num_layers, intermediate_size))

    head_mask[nas_args.num_layers :] = 0
    head_mask[: nas_args.num_layers, nas_args.num_heads :] = 0
    neuron_mask[nas_args.num_layers :] = 0
    neuron_mask[: nas_args.num_layers, nas_args.num_units :] = 0

    head_mask = head_mask.to(device=device)
    neuron_mask = neuron_mask.to(device=device)
    model.to(device)

    handles = mask(model, neuron_mask, head_mask)

    is_regression = True if data_args.task_name == "stsb" else False

    # compute number of parameters
    n_params_model = compute_parameters(
        dmodel=attention_size,
        dhead=attention_head_size,
        num_heads_per_layer=head_mask.sum(dim=1),
        num_neurons_per_layer=neuron_mask.sum(dim=1),
    )
    n_params = n_params_emb + n_params_model + n_params_classifier

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    num_training_steps = int(training_args.num_train_epochs * len(train_dataloader))
    warmup_steps = int(training_args.warmup_ratio * num_training_steps)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training
    for epoch in range(int(training_args.num_train_epochs)):
        model.train()

        train_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch, head_mask=head_mask)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss
        runtime = time.time() - start_time
        print(
            f"epoch {epoch}: training loss = {train_loss / len(train_dataloader)}, "
            # f"evaluation metrics = {eval_metric}, "
            f"runtime = {runtime}"
        )

        model.eval()

        results = {}
        for mode, dataloader in zip(
            ["valid", "test"], [eval_dataloader, test_dataloader]
        ):
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch, head_mask=head_mask)

                logits = outputs.logits
                # predictions = torch.argmax(logits, dim=-1)
                predictions = (
                    torch.squeeze(logits)
                    if is_regression
                    else torch.argmax(logits, dim=-1)
                )

                metric.add_batch(predictions=predictions, references=batch["labels"])

            error = 1 - metric.compute()[metric_name]
            if np.isnan(error) and is_regression:
                error = 1
            results[mode] = error
        report(**results, params=n_params / total_params, epoch=epoch)


if __name__ == "__main__":
    main()
