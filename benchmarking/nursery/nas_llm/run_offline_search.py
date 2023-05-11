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
import json

# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import os
import time
import logging
import sys

from dataclasses import dataclass, field

import numpy as np
import torch
import datasets
import transformers
import accelerate

from masking import apply_neuron_mask
from masking_gpt import apply_neuron_mask_gpt2
from multi_objective import get_pareto_optimal

from torch.utils.data import DataLoader, Subset

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    AutoConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2ForSequenceClassification

from tensorboardX import SummaryWriter

from evaluate import load

from estimate_efficency import compute_parameters
from task_data import TASKINFO
from sampling import (
    SmallSearchSpace,
    MediumSearchSpace,
    LayerSearchSpace,
    FullSearchSpace,
)
from baselines import MethodArguments, methods
from ask_tell_scheduler import AskTellScheduler
from hf_args import DataTrainingArguments, ModelArguments
from load_glue_datasets import load_glue_datasets


accelerator = accelerate.Accelerator()

SEARCHSPACES = {
    "small": SmallSearchSpace,
    "medium": MediumSearchSpace,
    "layer": LayerSearchSpace,
    "uniform": FullSearchSpace,
}

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

logger = logging.getLogger(__name__)


@dataclass
class SearchArguments:
    """
    Arguments to define the search
    """

    if "SM_CHANNEL_MODEL" in os.environ:
        checkpoint_dir_model: str = field(
            metadata={"help": ""}, default=os.environ["SM_CHANNEL_MODEL"]
        )
    else:
        checkpoint_dir_model: str = field(
            metadata={"help": ""}, default="/home/ubuntu/seed_42/"
        )

    search_strategy: str = field(metadata={"help": ""}, default="random")
    search_space: str = field(metadata={"help": ""}, default="small")
    use_accelerate: bool = field(metadata={"help": ""}, default=False)
    num_samples: int = field(default=500)
    log_dir: str = field(metadata={"help": ""}, default="./tensorboard_log_dir")


def main():
    start_time = time.time()
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, SearchArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        search_args,
    ) = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    writer = SummaryWriter(logdir=search_args.log_dir)

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.

    if int(training_args.seed) == -1:
        training_args.seed = np.random.randint(2**32 - 1)
    print(training_args.seed)
    set_seed(training_args.seed)

    # model_type = model_args.model_name_or_path
    if model_args.model_name_or_path in ["bert-small", "bert-medium", "bert-tiny"]:
        model_type = "prajjwal1/" + model_args.model_name_or_path
    elif model_args.model_name_or_path in ["electra-base"]:
        model_type = "google/electra-base-discriminator"
    elif model_args.model_name_or_path in ["electra-small"]:
        model_type = "google/electra-small-discriminator"
    else:
        model_type = model_args.model_name_or_path

    st = time.time()
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
        "glue", data_args.task_name, cache_dir=model_args.cache_dir
    )

    # Labels
    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    metric = load("glue", data_args.task_name)

    _, eval_dataloader, test_dataloader = load_glue_datasets(
        training_args=training_args, model_args=model_args, data_args=data_args
    )

    # Load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_type,
    #     cache_dir=model_args.cache_dir,
    #     use_fast=model_args.use_fast_tokenizer,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    #
    # if model_type.startswith('gpt2'):
    #     tokenizer.pad_token = tokenizer.eos_token
    #
    # # Preprocessing the raw_datasets
    # sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    #
    # # Padding strategy
    # if data_args.pad_to_max_length:
    #     padding = "max_length"
    # else:
    #     # We will pad later, dynamically at batch creation, to the max sequence length in each batch
    #     padding = False
    #
    # if data_args.max_seq_length > tokenizer.model_max_length:
    #     logger.warning(
    #         f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
    #         f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    #     )
    # max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    #
    # def preprocess_function(examples):
    #     # Tokenize the texts
    #     args = (
    #         (examples[sentence1_key],)
    #         if sentence2_key is None
    #         else (examples[sentence1_key], examples[sentence2_key])
    #     )
    #     result = tokenizer(
    #         *args, padding=padding, max_length=max_seq_length, truncation=True
    #     )
    #
    #     # Map labels to IDs (not necessary for GLUE tasks)
    #     # if label_to_id is not None and "label" in examples:
    #     #     result["label"] = [
    #     #         (label_to_id[l] if l != -1 else -1) for l in examples["label"]
    #     #     ]
    #     return result
    #
    # with training_args.main_process_first(desc="dataset map pre-processing"):
    #     raw_datasets = raw_datasets.map(
    #         preprocess_function,
    #         batched=True,
    #         load_from_cache_file=not data_args.overwrite_cache,
    #         desc="Running tokenizer on dataset",
    #     )
    #
    # train_dataset = raw_datasets["train"]
    # test_dataset = raw_datasets[
    #     "validation_matched" if data_args.task_name == "mnli" else "validation"
    # ]
    #
    # train_dataset = train_dataset.remove_columns(["idx"])
    # test_dataset = test_dataset.remove_columns(["idx"])
    #
    # # Split training dataset in training / validation
    # split = train_dataset.train_test_split(
    #     train_size=0.7, seed=0
    # )  # fix seed, all trials have the same data split
    # valid_dataset = split["test"]
    #
    # if data_args.task_name in ['sst2', 'qqp', 'qnli', 'mnli']:
    #     valid_dataset = Subset(
    #         valid_dataset,
    #         np.random.choice(len(valid_dataset), 2048).tolist(),
    #     )
    #
    # # Get the metric function
    # metric = load('glue', data_args.task_name)
    #
    # # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # # we already did the padding.
    # if data_args.pad_to_max_length:
    #     data_collator = default_data_collator
    # elif training_args.fp16:
    #     data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    # else:
    #     data_collator = None
    #
    # eval_dataloader = DataLoader(
    #     valid_dataset,
    #     batch_size=training_args.per_device_eval_batch_size,
    #     collate_fn=data_collator,
    # )
    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=training_args.per_device_eval_batch_size,
    #     collate_fn=data_collator,
    # )

    data_loading_time = time.time() - st

    st = time.time()
    teacher_config = AutoConfig.from_pretrained(
        model_type,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_type.startswith("bert"):
        model = BertForSequenceClassification(teacher_config)

        attention_size = teacher_config.hidden_size
        num_attention_heads = teacher_config.num_attention_heads
        attention_head_size = int(attention_size / num_attention_heads)

        n_params_emb = sum(
            p.numel() for p in model.bert.embeddings.parameters() if p.requires_grad
        )
        n_params_pooler = sum(
            p.numel() for p in model.bert.pooler.parameters() if p.requires_grad
        )
        n_params_classifier = sum(
            p.numel() for p in model.classifier.parameters() if p.requires_grad
        )
        n_params_classifier += n_params_pooler

    elif model_type.startswith("gpt2"):
        model = GPT2ForSequenceClassification(teacher_config)
        model.config.pad_token_id = model.config.eos_token_id

        num_attention_heads = teacher_config.n_head
        attention_size = teacher_config.hidden_size
        attention_head_size = int(attention_size / num_attention_heads)

        wte = sum(
            p.numel() for p in model.transformer.wte.parameters() if p.requires_grad
        )
        wpe = sum(
            p.numel() for p in model.transformer.wpe.parameters() if p.requires_grad
        )
        n_params_emb = wte + wpe
        n_params_classifier = sum(
            p.numel() for p in model.score.parameters() if p.requires_grad
        )

    if search_args.use_accelerate:
        model = accelerator.prepare(model)
        model = model.from_pretrained(search_args.checkpoint_dir_model)
    else:
        model.load_state_dict(
            torch.load(
                os.path.join(search_args.checkpoint_dir_model, "checkpoint.pt"),
                map_location="cuda:0",
            ),
        )
    model_loading_time = time.time() - st

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    metric_name = TASKINFO[data_args.task_name]["metric"]

    if model_type.startswith("gpt2"):
        neuron_mask = apply_neuron_mask_gpt2
    elif model_type.startswith("bert"):
        neuron_mask = apply_neuron_mask

    def evaluate_masks(head_mask, ffn_mask, dataloader):

        n_params_model = compute_parameters(
            dmodel=attention_size,
            dhead=attention_head_size,
            num_heads_per_layer=head_mask.sum(dim=1),
            num_neurons_per_layer=ffn_mask.sum(dim=1),
        )
        n_params = n_params_emb + n_params_model + n_params_classifier

        handles = neuron_mask(model, ffn_mask)

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(head_mask=head_mask, **batch)

            logits = outputs.logits
            predictions = (
                torch.squeeze(logits) if is_regression else torch.argmax(logits, dim=-1)
            )

            metric.add_batch(predictions=predictions, references=batch["labels"])

        eval_metric = metric.compute()
        for handle in handles:
            handle.remove()

        return 1 - eval_metric[metric_name], n_params

    search_space = SEARCHSPACES[search_args.search_space](model.config)

    base_scheduler = methods[search_args.search_strategy](
        MethodArguments(
            config_space=search_space.get_syne_tune_config_space(),
            metrics=["error", "params"],
            mode=["min", "min"],
            random_seed=training_args.seed,
        )
    )

    scheduler = AskTellScheduler(base_scheduler=base_scheduler)

    costs = np.empty((search_args.num_samples, 2))
    masks = []
    runtime = []
    configs = []
    for i in range(search_args.num_samples):
        trial_suggestion = scheduler.ask()
        head_mask, ffn_mask = search_space.config_to_mask(trial_suggestion.config)
        head_mask = head_mask.to(device)
        ffn_mask = ffn_mask.to(device)
        error, params = evaluate_masks(head_mask, ffn_mask, eval_dataloader)
        scheduler.tell(trial_suggestion, {"error": error, "params": params})
        costs[i][0] = error
        costs[i][1] = params
        masks.append((head_mask, ffn_mask))
        configs.append(trial_suggestion.config)
        print(trial_suggestion.config)
        print(f"iteration={i};")
        print(f"error={error};")
        print(f"params={params};")
        writer.add_scalar("error", float(error), i)
        writer.add_scalar("params", int(params), i)

        runtime.append(time.time() - start_time)
        writer.add_scalar("runtime", runtime[-1], i)
        logger.info(f"runtime = {runtime[-1]}")

    idx = get_pareto_optimal(costs)
    indices = np.arange(costs.shape[0])[idx]
    masks = [masks[i] for i in indices]

    test_pareto = []
    for i, (head_mask, ffn_mask) in enumerate(masks):
        error, n_params = evaluate_masks(
            head_mask, ffn_mask, dataloader=test_dataloader
        )
        test_pareto.append(error)

        torch.save(
            head_mask.cpu(), os.path.join(training_args.output_dir, f"head_mask_{i}.pt")
        )
        torch.save(
            ffn_mask.cpu(),
            os.path.join(training_args.output_dir, f"neuron_mask_{i}.pt"),
        )

    results = {}
    results["dataset"] = data_args.task_name
    # results['test_' + metric_name] = float(test_metric[metric_name])
    results[metric_name] = list(costs[:, 0])
    results["params"] = list(costs[:, 1])
    results["test_pareto"] = test_pareto
    results["config"] = configs
    results["eval_pareto"] = list(costs[idx, 0])
    results["params_pareto"] = list(costs[idx, 1])
    results["model_loading_time"] = model_loading_time
    results["data_loading_time"] = data_loading_time
    results["runtime"] = runtime

    print(results)
    fname = os.path.join(
        training_args.output_dir, f"results_{data_args.task_name}.json"
    )
    json.dump(results, open(fname, "w"))


if __name__ == "__main__":
    main()
