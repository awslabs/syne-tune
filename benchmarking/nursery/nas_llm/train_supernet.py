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
import os
import time
import json
import logging
import sys

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
import transformers
import evaluate

from accelerate import Accelerator
from tensorboardX import SummaryWriter
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from sampling import FullSearchSpace, SmallSearchSpace, LayerSearchSpace, MediumSearchSpace
from task_data import TASKINFO
from masking import apply_neuron_mask
from masking_gpt import apply_neuron_mask_gpt2
from hf_args import DataTrainingArguments, ModelArguments


accelerator = Accelerator()


def loss_KD_fn(
        student_logits,
        teacher_logits,
        targets,
        alpha=0.5,
        temperature=1,
        is_regression=False
):
    # alpha=0, ignore soft labels
    # alpha=1, ignore hard labels
    # for rand1, rand2, and smallest, they can ignore hard labels
    # for largest, it can ignore soft labels
    loss = 0
    if alpha != 0:
        if is_regression:
            kd_loss = F.mse_loss(student_logits, teacher_logits)
        else:
            kd_loss = F.cross_entropy(
                student_logits / temperature, F.softmax(teacher_logits / temperature, dim=1)
            )
        loss += alpha * temperature ** 2 * kd_loss
    if alpha != 1:
        if is_regression:
            predictive_loss = F.mse_loss(student_logits, targets)
        else:
            predictive_loss = F.cross_entropy(student_logits, targets)
        loss += (1 - alpha) * predictive_loss

    return loss


sampling = {
    'small': SmallSearchSpace,
    'medium': MediumSearchSpace,
    'layer': LayerSearchSpace,
    'uniform': FullSearchSpace,
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
class NASArguments:
    search_space: str = field(metadata={"help": ""}, default="small")
    use_accelerate: bool = field(metadata={"help": ""}, default=False)
    sampling_strategy: str = field(metadata={"help": ""}, default=None)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, NASArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        nas_args
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

    # Set seed before initializing model.

    if int(training_args.seed) == -1:
        training_args.seed = np.random.randint(2 ** 32 - 1)
    print(training_args.seed)
    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)

    if model_args.model_name_or_path in ["bert-small", "bert-medium", "bert-tiny"]:
        model_type = "prajjwal1/" + model_args.model_name_or_path
    elif model_args.model_name_or_path in ["electra-base"]:
        model_type = "google/electra-base-discriminator"
    elif model_args.model_name_or_path in ["electra-small"]:
        model_type = "google/electra-small-discriminator"
    else:
        model_type = model_args.model_name_or_path

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

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_type,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_type,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_type,
        from_tf=bool(".ckpt" in model_type),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model_type.startswith('gpt2'):
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        # if tokenizer.pad_token is None:
        #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the raw_datasets
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

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
        if label_to_id is not None and "label" in examples:
            result["label"] = [
                (label_to_id[l] if l != -1 else -1) for l in examples["label"]
            ]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = raw_datasets["train"]
    test_dataset = raw_datasets[
        "validation_matched" if data_args.task_name == "mnli" else "validation"
    ]

    train_dataset = train_dataset.remove_columns(["idx"])
    test_dataset = test_dataset.remove_columns(["idx"])

    # Split training dataset in training / validation
    split = train_dataset.train_test_split(
        train_size=0.7, seed=0
    )  # fix seed, all trials have the same data split
    train_dataset = split["train"]
    valid_dataset = split["test"]

    # Get the metric function
    metric = evaluate.load("glue", data_args.task_name)

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        valid_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )

    from tqdm.auto import tqdm

    from transformers import get_scheduler
    from torch.optim import AdamW

    LOG_DIR = "/opt/ml/output/tensorboard"
    writer = SummaryWriter(logdir=LOG_DIR)
    # writer = SummaryWriter(logdir=training_args.output_dir,
    #                        comment=f"{data_args.task_name}_{model_type}_{nas_args.sampling_strategy}")

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    num_training_steps = int(training_args.num_train_epochs * len(train_dataloader))
    warmup_steps = int(training_args.warmup_ratio * num_training_steps)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    start_time = time.time()
    dropout_rate = np.linspace(0, 1, num_training_steps)
    step = 0
    logger.info(f"Use {nas_args.sampling_strategy} to update super-network training")

    metric_name = TASKINFO[data_args.task_name]['metric']

    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    if model_type.startswith('gpt2'):
        neuron_mask = apply_neuron_mask_gpt2
    elif model_type.startswith('bert'):
        neuron_mask = apply_neuron_mask

    if nas_args.use_accelerate:
        train_dataloader, eval_dataloader, test_dataloader, model, optimizer = accelerator.prepare(
            train_dataloader, eval_dataloader, test_dataloader, model, optimizer)

    sampler = sampling[nas_args.search_space](config, rng=np.random.RandomState(seed=training_args.seed))

    for epoch in range(int(training_args.num_train_epochs)):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            if not nas_args.use_accelerate:
               batch = {k: v.to(device) for k, v in batch.items()}

            if nas_args.sampling_strategy == "one_shot":

                # update largest sub-network (i.e super-network)
                outputs = model(**batch)
                loss = outputs.loss
                y_teacher = outputs.logits.detach()
                writer.add_scalar('loss largest sub-network', loss, step)
                accelerator.backward(loss) if nas_args.use_accelerate else loss.backward()

                # update smallest sub-network
                head_mask, ffn_mask = sampler.get_smallest_sub_network()
                head_mask = head_mask.to(device='cuda', dtype=model.dtype)
                ffn_mask = ffn_mask.to(device='cuda', dtype=model.dtype)
                handles = neuron_mask(model, ffn_mask)
                outputs = model(head_mask=head_mask, **batch)

                for handle in handles:
                    handle.remove()
                # loss = loss_KD_fn(outputs.logits, y_teacher, batch['labels'], is_regression=is_regression)
                loss = kl_loss(F.log_softmax(outputs.logits, dim=-1), F.log_softmax(y_teacher, dim=-1))
                accelerator.backward(loss) if nas_args.use_accelerate else loss.backward()
                writer.add_scalar('loss smallest sub-network', loss, step)

                # update random sub-network
                head_mask, ffn_mask = sampler()
                head_mask = head_mask.to(device='cuda', dtype=model.dtype)
                ffn_mask = ffn_mask.to(device='cuda', dtype=model.dtype)

                handles = neuron_mask(model, ffn_mask)
                outputs = model(head_mask=head_mask, **batch)
                for handle in handles:
                    handle.remove()
                # loss = loss_KD_fn(outputs.logits, y_teacher, batch['labels'], is_regression=is_regression)
                loss = kl_loss(F.log_softmax(outputs.logits, dim=-1), F.log_softmax(y_teacher, dim=-1))
                writer.add_scalar('loss random sub-network', loss, step)
                accelerator.backward(loss) if nas_args.use_accelerate else loss.backward()

                # update random sub-network
                head_mask, ffn_mask = sampler()
                head_mask = head_mask.to(device='cuda', dtype=model.dtype)
                ffn_mask = ffn_mask.to(device='cuda', dtype=model.dtype)

                handles = neuron_mask(model, ffn_mask)
                outputs = model(head_mask=head_mask, **batch)

                for handle in handles:
                    handle.remove()
                loss = kl_loss(F.log_softmax(outputs.logits, dim=-1), F.log_softmax(y_teacher, dim=-1))
                writer.add_scalar('loss random sub-network', loss, step)
                accelerator.backward(loss) if nas_args.use_accelerate else loss.backward()

            elif nas_args.sampling_strategy == "sandwich":

                # update largest sub-network (i.e super-network)
                outputs = model(**batch)
                loss = outputs.loss
                writer.add_scalar('loss largest sub-network', loss, step)
                accelerator.backward(loss) if nas_args.use_accelerate else loss.backward()

                # update smallest sub-network
                head_mask, ffn_mask = sampler.get_smallest_sub_network()
                head_mask = head_mask.to(device='cuda', dtype=model.dtype)
                ffn_mask = ffn_mask.to(device='cuda', dtype=model.dtype)

                handles = neuron_mask(model, ffn_mask)
                outputs = model(head_mask=head_mask, **batch)

                for handle in handles:
                    handle.remove()

                loss = outputs.loss
                accelerator.backward(loss) if nas_args.use_accelerate else loss.backward()
                writer.add_scalar('loss smallest sub-network', loss, step)

                # update random sub-network
                head_mask, ffn_mask = sampler()
                head_mask = head_mask.to(device='cuda', dtype=model.dtype)
                ffn_mask = ffn_mask.to(device='cuda', dtype=model.dtype)

                handles = neuron_mask(model, ffn_mask)
                outputs = model(head_mask=head_mask, **batch)

                for handle in handles:
                    handle.remove()

                loss = outputs.loss
                writer.add_scalar('loss random sub-network', loss, step)
                accelerator.backward(loss) if nas_args.use_accelerate else loss.backward()

                # update random sub-network
                head_mask, ffn_mask = sampler()
                head_mask = head_mask.to(device='cuda', dtype=model.dtype)
                ffn_mask = ffn_mask.to(device='cuda', dtype=model.dtype)

                handles = neuron_mask(model, ffn_mask)
                outputs = model(head_mask=head_mask, **batch)

                for handle in handles:
                    handle.remove()
                loss = outputs.loss
                writer.add_scalar('loss random sub-network', loss, step)
                accelerator.backward(loss) if nas_args.use_accelerate else loss.backward()

            elif nas_args.sampling_strategy == "random":
                head_mask, ffn_mask = sampler()
                head_mask = head_mask.to(device='cuda', dtype=model.dtype)
                ffn_mask = ffn_mask.to(device='cuda', dtype=model.dtype)

                handles = neuron_mask(model, ffn_mask)
                outputs = model(head_mask=head_mask, **batch)

                for handle in handles:
                    handle.remove()

                loss = outputs.loss
                writer.add_scalar('train-loss', outputs.loss, step)
                accelerator.backward(loss) if nas_args.use_accelerate else loss.backward()

            elif nas_args.sampling_strategy == "linear_random":
                if np.random.rand() <= dropout_rate[step]:
                    head_mask, ffn_mask = sampler()
                    head_mask = head_mask.to(device='cuda', dtype=model.dtype)
                    ffn_mask = ffn_mask.to(device='cuda', dtype=model.dtype)

                    handles = neuron_mask(model, ffn_mask)
                    outputs = model(head_mask=head_mask, **batch)

                    for handle in handles:
                        handle.remove()
                else:
                    outputs = model(**batch)
                loss = outputs.loss
                writer.add_scalar('train-loss', outputs.loss, step)
                accelerator.backward(loss) if nas_args.use_accelerate else loss.backward()

            elif nas_args.sampling_strategy == "meta":

                config = sampler(data_args.task_name)[0]
                num_layers = config['num_layers']
                num_heads = config['num_heads']
                num_units = config['num_units']

                head_mask = torch.ones((model.config.num_hidden_layers,
                                        model.config.num_attention_heads)).cuda()
                ffn_mask = torch.ones((model.config.num_hidden_layers,
                                       model.config.intermediate_size)).cuda()
                head_mask[num_layers:] = 0
                head_mask[:num_layers, num_heads:] = 0
                ffn_mask[num_layers:] = 0
                ffn_mask[:num_layers, num_units:] = 0

                handles = neuron_mask(model, ffn_mask)
                outputs = model(head_mask=head_mask, **batch)

                for handle in handles:
                    handle.remove()

                loss = outputs.loss
                writer.add_scalar('train-loss', loss.item(), step)

                accelerator.backward(loss) if nas_args.use_accelerate else loss.backward()

            elif nas_args.sampling_strategy == "standard":
                outputs = model(**batch)
                writer.add_scalar('train-loss', outputs.loss, step)
                loss = outputs.loss
                accelerator.backward(loss) if nas_args.use_accelerate else loss.backward()

            step += 1

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            writer.add_scalar('lr', lr_scheduler.get_lr(), step)

            train_loss += loss

        model.eval()
        for batch in eval_dataloader:
            if not nas_args.use_accelerate:
                batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)

            logits = outputs.logits
            # predictions = torch.argmax(logits, dim=-1)
            predictions = torch.squeeze(logits) if is_regression else torch.argmax(logits, dim=-1)

            metric.add_batch(predictions=predictions, references=batch["labels"])

        eval_metric = metric.compute()
        runtime = time.time() - start_time
        logger.info(
            f"epoch {epoch}: training loss = {train_loss / len(train_dataloader)}, "
            f"evaluation metrics = {eval_metric}, "
            f"runtime = {runtime}"
        )
        logger.info(f'epoch={epoch};')
        logger.info(f'training loss={train_loss / len(train_dataloader)};')
        logger.info(f'evaluation metrics={eval_metric[metric_name]};')
        logger.info(f'runtime={runtime};')

        for k, v in eval_metric.items():
            writer.add_scalar(f'eval-{k}', v, epoch)
        writer.add_scalar('runtime', runtime, epoch)

        if training_args.save_strategy == "epoch":
            os.makedirs(training_args.output_dir, exist_ok=True)
            logger.info(f"Store checkpoint in: {training_args.output_dir}")
            if nas_args.use_accelerate:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    training_args.output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model)
                )
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(training_args.output_dir, "checkpoint.pt"),
                )

    if not nas_args.use_accelerate:

        model.eval()
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)

            logits = outputs.logits
            # predictions = torch.argmax(logits, dim=-1)
            predictions = torch.squeeze(logits) if is_regression else torch.argmax(logits, dim=-1)

            metric.add_batch(predictions=predictions, references=batch["labels"])

        test_metric = metric.compute()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results = {}
        results['dataset'] = data_args.task_name
        results['params'] = n_params
        results['search_space'] = nas_args.search_space

        results[metric_name] = float(eval_metric[metric_name])
        results['test_' + metric_name] = float(test_metric[metric_name])
        fname = os.path.join(training_args.output_dir, f'results_{data_args.task_name}.json')
        json.dump(results, open(fname, 'w'))


if __name__ == "__main__":
    main()
