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

import numpy as np
import torch
import torch.nn.functional as F
import datasets
import transformers
import evaluate

from tqdm.auto import tqdm
from accelerate import Accelerator
from tensorboardX import SummaryWriter
from functools import partial

from torch.optim import AdamW

from transformers import (
    AutoConfig,
    get_scheduler,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from sampling import (
    FullSearchSpace,
    SmallSearchSpace,
    LayerSearchSpace,
    MediumSearchSpace,
)
from task_data import TASKINFO
from mask import mask_bert, mask_gpt, mask_gpt_neox
from hf_args import DataTrainingArguments, ModelArguments, parse_model_name
from load_glue_datasets import load_glue_datasets


def kd_loss(
    student_logits,
    teacher_logits,
    targets,
    temperature=1,
    is_regression=False,
):
    if is_regression:
        return F.mse_loss(student_logits, teacher_logits)
    else:
        kd_loss = F.cross_entropy(
            student_logits / temperature,
            F.softmax(teacher_logits / temperature, dim=1),
        )
        predictive_loss = F.cross_entropy(student_logits, targets)
        return temperature**2 * kd_loss + predictive_loss


sampling = {
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
class NASArguments:
    search_space: str = field(metadata={"help": ""}, default="small")
    use_accelerate: bool = field(metadata={"help": ""}, default=False)
    sampling_strategy: str = field(metadata={"help": ""}, default=None)
    log_dir: str = field(metadata={"help": ""}, default="./tensorboard_log_dir")
    num_random_sub_nets: int = field(metadata={"help": ""}, default=1)
    temperature: float = field(metadata={"help": ""}, default=1)


def main():
    start_time = time.time()
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, NASArguments)
    )

    (
        model_args,
        data_args,
        training_args,
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

    # Set seed before initializing model.
    if int(training_args.seed) == -1:
        training_args.seed = np.random.randint(2**32 - 1)
    print(training_args.seed)
    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)

    model_type = parse_model_name(model_args)

    (
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        tokenizer,
        num_labels,
    ) = load_glue_datasets(
        training_args=training_args, model_args=model_args, data_args=data_args
    )

    accelerator = Accelerator()

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

    if model_type.startswith("gpt2") or "pythia" in model_type:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        # if tokenizer.pad_token is None:
        #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # model.resize_token_embeddings(len(tokenizer))

    # Get the metric function
    metric = evaluate.load("glue", data_args.task_name)

    writer = SummaryWriter(logdir=nas_args.log_dir)

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

    if not nas_args.use_accelerate:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model.to(device)

    dropout_rate = np.linspace(0, 1, num_training_steps)
    step = 0
    logger.info(f"Use {nas_args.sampling_strategy} to update super-network training")

    metric_name = TASKINFO[data_args.task_name]["metric"]
    is_regression = True if data_args.task_name == "stsb" else False
    distillation_loss = partial(
        kd_loss, is_regression=is_regression, temperature=nas_args.temperature
    )
    # if is_regression:
    #     distillation_loss = nn.MSELoss()
    # else:
    #     kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    #     distillation_loss = lambda x, y: kl_loss(
    #         F.log_softmax(x, dim=-1), F.log_softmax(y, dim=-1)
    #     )

    if model_type.startswith("gpt2"):
        mask = mask_gpt
    elif model_type.startswith("bert"):
        mask = mask_bert
    elif "pythia" in model_type:
        mask = mask_gpt_neox

    if nas_args.use_accelerate:
        (
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            optimizer,
            lr_scheduler,
        ) = accelerator.prepare(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            optimizer,
            lr_scheduler,
        )

    sampler = sampling[nas_args.search_space](
        config, rng=np.random.RandomState(seed=training_args.seed)
    )

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
                writer.add_scalar("loss largest sub-network", loss, step)
                accelerator.backward(
                    loss
                ) if nas_args.use_accelerate else loss.backward()

                # update smallest sub-network
                head_mask, ffn_mask = sampler.get_smallest_sub_network()
                if nas_args.use_accelerate:
                    head_mask = head_mask.to(device=accelerator.device)
                    ffn_mask = ffn_mask.to(device=accelerator.device)
                else:
                    head_mask = head_mask.to(device="cuda", dtype=model.dtype)
                    ffn_mask = ffn_mask.to(device="cuda", dtype=model.dtype)

                handles = mask(model, ffn_mask, head_mask)
                outputs = model(head_mask=head_mask, **batch)

                for handle in handles:
                    handle.remove()
                # loss = loss_KD_fn(outputs.logits, y_teacher, batch['labels'], is_regression=is_regression)
                # loss = distillation_loss(
                #     F.log_softmax(outputs.logits, dim=-1),
                #     F.log_softmax(y_teacher, dim=-1),
                # )
                loss = distillation_loss(outputs.logits, y_teacher, batch["labels"])
                accelerator.backward(
                    loss
                ) if nas_args.use_accelerate else loss.backward()
                writer.add_scalar("loss smallest sub-network", loss, step)

                # update random sub-network
                for k in range(nas_args.num_random_sub_nets):
                    head_mask, ffn_mask = sampler()
                    if nas_args.use_accelerate:
                        head_mask = head_mask.to(device=accelerator.device)
                        ffn_mask = ffn_mask.to(device=accelerator.device)
                    else:
                        head_mask = head_mask.to(device="cuda", dtype=model.dtype)
                        ffn_mask = ffn_mask.to(device="cuda", dtype=model.dtype)

                    handles = mask(model, ffn_mask, head_mask)

                    outputs = model(head_mask=head_mask, **batch)
                    for handle in handles:
                        handle.remove()

                    loss = distillation_loss(outputs.logits, y_teacher, batch["labels"])
                    writer.add_scalar("loss random sub-network", loss, step)
                    accelerator.backward(
                        loss
                    ) if nas_args.use_accelerate else loss.backward()

            elif nas_args.sampling_strategy == "sandwich":

                # update largest sub-network (i.e super-network)
                outputs = model(**batch)
                loss = outputs.loss
                writer.add_scalar("loss largest sub-network", loss, step)
                accelerator.backward(
                    loss
                ) if nas_args.use_accelerate else loss.backward()

                # update smallest sub-network
                head_mask, ffn_mask = sampler.get_smallest_sub_network()
                if nas_args.use_accelerate:
                    head_mask = head_mask.to(device=accelerator.device)
                    ffn_mask = ffn_mask.to(device=accelerator.device)
                else:
                    head_mask = head_mask.to(device="cuda", dtype=model.dtype)
                    ffn_mask = ffn_mask.to(device="cuda", dtype=model.dtype)

                handles = mask(model, ffn_mask, head_mask)
                outputs = model(head_mask=head_mask, **batch)

                for handle in handles:
                    handle.remove()

                loss = outputs.loss
                accelerator.backward(
                    loss
                ) if nas_args.use_accelerate else loss.backward()
                writer.add_scalar("loss smallest sub-network", loss, step)

                # update random sub-network
                for k in range(nas_args.num_random_sub_nets):

                    head_mask, ffn_mask = sampler()
                    if nas_args.use_accelerate:
                        head_mask = head_mask.to(device=accelerator.device)
                        ffn_mask = ffn_mask.to(device=accelerator.device)
                    else:
                        head_mask = head_mask.to(device="cuda", dtype=model.dtype)
                        ffn_mask = ffn_mask.to(device="cuda", dtype=model.dtype)

                    handles = mask(model, ffn_mask, head_mask)
                    outputs = model(head_mask=head_mask, **batch)

                    for handle in handles:
                        handle.remove()

                    loss = outputs.loss
                    writer.add_scalar("loss random sub-network", loss, step)
                    accelerator.backward(
                        loss
                    ) if nas_args.use_accelerate else loss.backward()

            elif nas_args.sampling_strategy == "random":

                for k in range(nas_args.num_random_sub_nets):

                    head_mask, ffn_mask = sampler()
                    if nas_args.use_accelerate:
                        head_mask = head_mask.to(device=accelerator.device)
                        ffn_mask = ffn_mask.to(device=accelerator.device)
                    else:
                        head_mask = head_mask.to(device="cuda", dtype=model.dtype)
                        ffn_mask = ffn_mask.to(device="cuda", dtype=model.dtype)

                    handles = mask(model, ffn_mask, head_mask)
                    outputs = model(head_mask=head_mask, **batch)

                    for handle in handles:
                        handle.remove()

                    loss = outputs.loss
                    writer.add_scalar("train-loss", outputs.loss, step)
                    accelerator.backward(
                        loss
                    ) if nas_args.use_accelerate else loss.backward()

            elif nas_args.sampling_strategy == "linear_random":
                if np.random.rand() <= dropout_rate[step]:
                    for k in range(nas_args.num_random_sub_nets):

                        head_mask, ffn_mask = sampler()
                        if nas_args.use_accelerate:
                            head_mask = head_mask.to(device=accelerator.device)
                            ffn_mask = ffn_mask.to(device=accelerator.device)
                        else:
                            head_mask = head_mask.to(device="cuda", dtype=model.dtype)
                            ffn_mask = ffn_mask.to(device="cuda", dtype=model.dtype)

                        handles = mask(model, ffn_mask, head_mask)
                        outputs = model(head_mask=head_mask, **batch)

                        for handle in handles:
                            handle.remove()
                        loss = outputs.loss
                        accelerator.backward(
                            loss
                        ) if nas_args.use_accelerate else loss.backward()
                else:
                    outputs = model(**batch)
                    loss = outputs.loss

                    accelerator.backward(
                        loss
                    ) if nas_args.use_accelerate else loss.backward()

            elif nas_args.sampling_strategy == "kd":
                y_teacher = model(**batch)
                if np.random.rand() <= dropout_rate[step]:
                    for k in range(nas_args.num_random_sub_nets):

                        head_mask, ffn_mask = sampler()
                        if nas_args.use_accelerate:
                            head_mask = head_mask.to(device=accelerator.device)
                            ffn_mask = ffn_mask.to(device=accelerator.device)
                        else:
                            head_mask = head_mask.to(device="cuda", dtype=model.dtype)
                            ffn_mask = ffn_mask.to(device="cuda", dtype=model.dtype)

                        handles = mask(model, ffn_mask, head_mask)
                        outputs = model(head_mask=head_mask, **batch)

                        for handle in handles:
                            handle.remove()
                        loss = distillation_loss(
                            outputs.logits, y_teacher.logits.detach(), batch["labels"]
                        )
                        accelerator.backward(
                            loss
                        ) if nas_args.use_accelerate else loss.backward()
                else:
                    loss = y_teacher.loss

                    accelerator.backward(
                        loss
                    ) if nas_args.use_accelerate else loss.backward()

            elif nas_args.sampling_strategy == "standard":
                outputs = model(**batch)
                writer.add_scalar("train-loss", outputs.loss, step)
                loss = outputs.loss
                accelerator.backward(
                    loss
                ) if nas_args.use_accelerate else loss.backward()

            step += 1

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            writer.add_scalar("lr", lr_scheduler.get_lr(), step)

            train_loss += loss

        model.eval()
        for batch in eval_dataloader:
            if not nas_args.use_accelerate:
                batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)

            logits = outputs.logits
            # predictions = torch.argmax(logits, dim=-1)
            predictions = (
                torch.squeeze(logits) if is_regression else torch.argmax(logits, dim=-1)
            )

            metric.add_batch(predictions=predictions, references=batch["labels"])

        eval_metric = metric.compute()
        runtime = time.time() - start_time
        logger.info(
            f"epoch {epoch}: training loss = {train_loss / len(train_dataloader)}, "
            f"evaluation metrics = {eval_metric}, "
            f"runtime = {runtime}"
        )
        logger.info(f"epoch={epoch};")
        logger.info(f"training loss={train_loss / len(train_dataloader)};")
        logger.info(f"evaluation metrics={eval_metric[metric_name]};")
        logger.info(f"runtime={runtime};")

        for k, v in eval_metric.items():
            writer.add_scalar(f"eval-{k}", v, epoch)
        writer.add_scalar("runtime", runtime, epoch)

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
                    state_dict=accelerator.get_state_dict(model),
                )
            else:
                # torch.save(
                #     model.state_dict(),
                #     os.path.join(training_args.output_dir, "checkpoint.pt"),
                # )
                model.save_pretrained(training_args.output_dir)

    if not nas_args.use_accelerate:

        model.eval()
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)

            logits = outputs.logits
            # predictions = torch.argmax(logits, dim=-1)
            predictions = (
                torch.squeeze(logits) if is_regression else torch.argmax(logits, dim=-1)
            )

            metric.add_batch(predictions=predictions, references=batch["labels"])

        test_metric = metric.compute()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results = {}
        results["dataset"] = data_args.task_name
        results["params"] = n_params
        results["search_space"] = nas_args.search_space
        results["runtime"] = time.time() - start_time

        results[metric_name] = float(eval_metric[metric_name])
        results["test_" + metric_name] = float(test_metric[metric_name])
        fname = os.path.join(
            training_args.output_dir, f"results_{data_args.task_name}.json"
        )
        json.dump(results, open(fname, "w"))


if __name__ == "__main__":
    main()
