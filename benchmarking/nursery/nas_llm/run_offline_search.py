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

from mask import mask_bert, mask_gpt, mask_gpt_neox
from multi_objective import get_pareto_optimal

from transformers import (
    HfArgumentParser,
    AutoConfig,
    TrainingArguments,
    set_seed,
    AutoModelForSequenceClassification,
)

from transformers.models.bert.modeling_bert import (
    BertForSequenceClassification,
    BertConfig,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2ForSequenceClassification

from tensorboardX import SummaryWriter

from evaluate import load
from functools import partial

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
from hf_args import DataTrainingArguments, ModelArguments, parse_model_name
from load_glue_datasets import load_glue_datasets


accelerator = accelerate.Accelerator()

SEARCHSPACES = {
    "small": SmallSearchSpace,
    "medium": MediumSearchSpace,
    "layer": LayerSearchSpace,
    "uniform": FullSearchSpace,
    "smallpower2": partial(SmallSearchSpace, power_of_2_encoding=True),
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
    optimize_memory_footprint: bool = field(metadata={}, default=False)


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

    model_type = parse_model_name(model_args)

    st = time.time()
    # Downloading and loading a dataset from the hub.
    metric = load("glue", data_args.task_name)

    _, eval_dataloader, test_dataloader, tokenizer, num_labels = load_glue_datasets(
        training_args=training_args, model_args=model_args, data_args=data_args
    )

    is_regression = data_args.task_name == "stsb"

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

    model = AutoModelForSequenceClassification.from_config(teacher_config)
    if model_type.startswith("bert"):
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
    elif "pythia" in model_type:
        model.config.pad_token_id = model.config.eos_token_id

        num_attention_heads = teacher_config.num_attention_heads
        attention_size = teacher_config.hidden_size
        attention_head_size = int(attention_size / num_attention_heads)

        n_params_emb = sum(
            p.numel() for p in model.gpt_neox.embed_in.parameters() if p.requires_grad
        )
        final_ln = sum(
            p.numel()
            for p in model.gpt_neox.final_layer_norm.parameters()
            if p.requires_grad
        )
        n_params_classifier = sum(
            p.numel() for p in model.score.parameters() if p.requires_grad
        )

        n_params_classifier += final_ln
    n_params_super_net = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if search_args.use_accelerate:
        model = accelerator.prepare(model)
    model = model.from_pretrained(search_args.checkpoint_dir_model)
    memory_footprint_supernet = model.get_memory_footprint()
    model_loading_time = time.time() - st

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    metric_name = TASKINFO[data_args.task_name]["metric"]

    if model_type.startswith("gpt2"):
        mask = mask_gpt
    elif model_type.startswith("bert"):
        mask = mask_bert
    elif "pythia" in model_type:
        mask = mask_gpt_neox

    def evaluate_masks(head_mask, ffn_mask, dataloader):
        n_params_model = compute_parameters(
            dmodel=attention_size,
            dhead=attention_head_size,
            num_heads_per_layer=head_mask.sum(dim=1),
            num_neurons_per_layer=ffn_mask.sum(dim=1),
        )
        n_params = n_params_emb + n_params_model + n_params_classifier

        handles = mask(model, ffn_mask, head_mask)

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

        return 1 - eval_metric[metric_name], n_params / n_params_super_net

    search_space = SEARCHSPACES[search_args.search_space](model.config)

    if search_args.optimize_memory_footprint:
        metrics = ["error", "memory"]
    else:
        metrics = ["error", "params"]

    base_scheduler = methods[search_args.search_strategy](
        MethodArguments(
            config_space=search_space.get_syne_tune_config_space(),
            metrics=metrics,
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

        if np.isnan(error) and is_regression:
            error = 1

        if search_args.optimize_memory_footprint:
            hypers = trial_suggestion.config
            c = BertConfig(
                num_hidden_layers=hypers["num_layers"],
                num_attention_heads=hypers["num_heads"],
                intermediate_size=hypers["num_units"],
                attention_size=attention_head_size,
            )
            temp_model = AutoModelForSequenceClassification.from_config(c)
            memory = temp_model.get_memory_footprint() / memory_footprint_supernet
            scheduler.tell(trial_suggestion, {"error": error, "memory": memory})
            costs[i][0] = error
            costs[i][1] = memory
            print(memory)
        else:
            scheduler.tell(trial_suggestion, {"error": error, "params": params})
            costs[i][0] = error
            costs[i][1] = params * n_params_super_net
        masks.append((head_mask, ffn_mask))
        configs.append(trial_suggestion.config)
        print(trial_suggestion.config)
        writer.add_scalar("error", float(error), i)
        writer.add_scalar("params", int(params), i)

        runtime.append(time.time() - start_time)
        writer.add_scalar("runtime", runtime[-1], i)
        logger.info(
            f"iteration {i}: error={error} ; params={params}; runtime = {runtime[-1]}"
        )

    idx = get_pareto_optimal(costs)
    indices = np.arange(costs.shape[0])[idx]
    masks = [masks[i] for i in indices]

    os.makedirs(training_args.output_dir, exist_ok=True)
    test_pareto = []
    model.eval()
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
    results[metric_name] = list(costs[:, 0])
    if search_args.optimize_memory_footprint:
        results["memory"] = list(costs[:, 1])
        results["memory_pareto"] = list(costs[idx, 1])

    else:
        results["params"] = list(costs[:, 1])
        results["params_pareto"] = list(costs[idx, 1])

    results["test_pareto"] = test_pareto
    if search_args.search_space != "uniform":
        results["config"] = configs
    results["eval_pareto"] = list(costs[idx, 0])
    results["model_loading_time"] = model_loading_time
    results["data_loading_time"] = data_loading_time
    results["runtime"] = runtime
    results["indices"] = [int(i) for i in indices]
    print(results)

    fname = os.path.join(
        training_args.output_dir, f"results_{data_args.task_name}.json"
    )
    json.dump(results, open(fname, "w"))


if __name__ == "__main__":
    main()
