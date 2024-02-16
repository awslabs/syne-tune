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
from functools import partial

from torch.optim import AdamW

from transformers import (
    AutoConfig,
    get_scheduler,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    AutoModelForCausalLM,
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
from task_data import GLUE_TASK_INFO
from mask import mask_bert, mask_roberta, mask_gpt, mask_gpt_neox
from hf_args import DataTrainingArguments, ModelArguments, parse_model_name
from data_wrapper import Glue, IMDB, SWAG, AlpacaDataset
from sampler import MetaSamplerDKESmallSearchSpace


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
    "meta_small_kde": MetaSamplerDKESmallSearchSpace,
    "medium": MediumSearchSpace,
    "layer": LayerSearchSpace,
    "uniform": FullSearchSpace,
    "smallpower2": partial(SmallSearchSpace, power_of_2_encoding=True),
}

logging.basicConfig(level=logging.INFO)


@dataclass
class NASArguments:
    search_space: str = field(metadata={"help": ""}, default="small")
    use_accelerate: bool = field(metadata={"help": ""}, default=False)
    sampling_strategy: str = field(metadata={"help": ""}, default=None)
    log_dir: str = field(metadata={"help": ""}, default="./tensorboard_log_dir")
    num_random_sub_nets: int = field(metadata={"help": ""}, default=1)
    temperature: float = field(metadata={"help": ""}, default=1)
    do_hpo: bool = field(metadata={"help": ""}, default=False)
    store_debug_info: bool = field(metadata={"help": ""}, default=False)

    # TODO: not used
    st_checkpoint_dir: str = field(metadata={"help": ""}, default=None)


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

    if nas_args.do_hpo:
        from syne_tune.report import Reporter

        report = Reporter()
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
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
    elif data_args.task_name == "alpaca":
        data = AlpacaDataset(
            training_args=training_args, model_args=model_args, data_args=data_args
        )
        metric = evaluate.load("./cross_entropy.py")
        metric_name = "cross_entropy"

    # elif data_args.task_name == "custom":
    #     data = Custom(training_args=training_args, model_args=model_args, data_args=data_args)
    train_dataloader, eval_dataloader, test_dataloader = data.get_data_loaders()
    num_labels = data.num_labels

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

    if data_args.task_name in ["swag"]:
        model_cls = AutoModelForMultipleChoice
    elif data_args.task_name in ['alpaca']:
        model_cls = AutoModelForCausalLM
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

    if model_type.startswith("gpt2") or "pythia" in model_type:
        model.config.pad_token_id = model.config.eos_token_id
        # if tokenizer.pad_token is None:
        #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # model.resize_token_embeddings(len(tokenizer))

    if data_args.task_name == 'alpaca':

        model.resize_token_embeddings(len(data.tokenizer))

        if data.num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-data.num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings[:-data.num_new_tokens].mean(
                dim=0, keepdim=True
            )

            input_embeddings[-data.num_new_tokens:] = input_embeddings_avg
            output_embeddings[-data.num_new_tokens:] = output_embeddings_avg

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    num_training_steps = int(training_args.num_train_epochs * len(train_dataloader))
    warmup_steps = int(training_args.warmup_ratio * num_training_steps)

    if training_args.lr_scheduler_type == "linear":
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif training_args.lr_scheduler_type == "cosine_with_restarts":
        lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=training_args.num_train_epochs,
        )

    progress_bar = tqdm(range(num_training_steps))

    if not nas_args.use_accelerate:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model.to(device)

    dropout_rate = np.linspace(0, 1, num_training_steps)
    step = 0
    logging.info(f"Use {nas_args.sampling_strategy} to update super-network training")

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
    elif model_type.startswith("roberta"):
        mask = mask_roberta
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

    kwargs = {"rng": np.random.RandomState(seed=training_args.seed)}
    if nas_args.search_space == "meta_small_kde":
        kwargs["dataset_name"] = data_args.task_name
        kwargs["num_tasks"] = 1
        kwargs["data_path"] = f"meta_model/meta_data_{model_type}.json"
        print(kwargs)

    sampler = sampling[nas_args.search_space](config, **kwargs)

    if nas_args.store_debug_info:
        from collections import defaultdict

        debug_info = defaultdict(list)

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
                    accelerator.backward(
                        loss
                    ) if nas_args.use_accelerate else loss.backward()

            elif nas_args.sampling_strategy == "one_shot_upper":
                # update largest sub-network (i.e super-network)
                outputs = model(**batch)
                loss = outputs.loss
                y_teacher = outputs.logits.detach()
                accelerator.backward(
                    loss
                ) if nas_args.use_accelerate else loss.backward()

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
                    accelerator.backward(
                        loss
                    ) if nas_args.use_accelerate else loss.backward()

            elif nas_args.sampling_strategy == "sandwich":
                # update largest sub-network (i.e super-network)
                outputs = model(**batch)
                loss = outputs.loss
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
                loss = outputs.loss
                accelerator.backward(
                    loss
                ) if nas_args.use_accelerate else loss.backward()

            elif nas_args.sampling_strategy == "fix":
                configs = [
                    # {"num_layers": 12, "num_heads": 12, "num_units": 3072},
                    {"num_layers": 11, "num_heads": 12, "num_units": 3072},
                    {"num_layers": 10, "num_heads": 12, "num_units": 3072},
                    {"num_layers": 9, "num_heads": 12, "num_units": 3072},
                    {"num_layers": 8, "num_heads": 12, "num_units": 3072},
                    {"num_layers": 7, "num_heads": 12, "num_units": 3072},
                    {"num_layers": 6, "num_heads": 12, "num_units": 3072},
                    {"num_layers": 5, "num_heads": 12, "num_units": 3072},
                    {"num_layers": 4, "num_heads": 12, "num_units": 3072},
                    {"num_layers": 3, "num_heads": 12, "num_units": 3072},
                    {"num_layers": 2, "num_heads": 12, "num_units": 3072},
                    # {'num_layers':12, 'num_heads': 12, 'num_units': 3072},
                    # {'num_layers':10, 'num_heads': 10, 'num_units': 2560},
                    # {'num_layers':8, 'num_heads': 8, 'num_units': 2048},
                    # {'num_layers':6, 'num_heads': 6, 'num_units': 1536},
                    # {'num_layers':4, 'num_heads': 4, 'num_units': 1024},
                    # {'num_layers':2, 'num_heads': 2, 'num_units': 512}
                ]

                y_teacher = model(**batch)
                loss = y_teacher.loss
                accelerator.backward(
                    loss
                ) if nas_args.use_accelerate else loss.backward()
                for k in range(nas_args.num_random_sub_nets):
                    idx = np.random.randint(len(configs))
                    config = configs[idx]
                    # config = np.random.sample(configs, 1)
                    head_mask, ffn_mask = sampler.config_to_mask(config)
                    if nas_args.use_accelerate:
                        head_mask = head_mask.to(device=accelerator.device)
                        ffn_mask = ffn_mask.to(device=accelerator.device)
                    else:
                        head_mask = head_mask.to(device="cuda", dtype=model.dtype)
                        ffn_mask = ffn_mask.to(device="cuda", dtype=model.dtype)

                    handles = mask(model, ffn_mask, head_mask)

                    outputs = model(head_mask=head_mask, **batch)
                    # loss = outputs.loss
                    loss = distillation_loss(
                        outputs.logits, y_teacher.logits.detach(), batch["labels"]
                    )
                    if nas_args.store_debug_info:
                        # debug_info[f'index'].append(idx)
                        debug_info[f"distillation_loss_{idx}"].append(loss.item())
                        debug_info[f"nll_{idx}"].append(outputs.loss.item())
                        debug_info[f"loss_largest"].append(y_teacher.loss.item())

                    accelerator.backward(
                        loss
                    ) if nas_args.use_accelerate else loss.backward()
                    for handle in handles:
                        handle.remove()

            if nas_args.store_debug_info:
                # debug_info[f'index'].append(idx)
                debug_info[f"loss"].append(loss.item())
                debug_info[f"step"].append(step)
                debug_info["lr"].append(lr_scheduler.get_lr())

            step += 1

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

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
        if nas_args.do_hpo:
            report(epoch=epoch, metric=eval_metric[metric_name])
        runtime = time.time() - start_time
        logging.info(
            f"epoch {epoch}: training loss = {train_loss / len(train_dataloader)}, "
            f"evaluation metrics = {eval_metric}, "
            f"runtime = {runtime}"
        )
        logging.info(f"epoch={epoch};")
        logging.info(f"training loss={train_loss / len(train_dataloader)};")
        logging.info(f"evaluation metrics={eval_metric[metric_name]};")
        logging.info(f"runtime={runtime};")

        if training_args.save_strategy == "epoch":
            os.makedirs(training_args.output_dir, exist_ok=True)
            logging.info(f"Store checkpoint in: {training_args.output_dir}")
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

        if nas_args.store_debug_info:
            fname = os.path.join(
                training_args.output_dir, f"debug_info_{data_args.task_name}.json"
            )
            json.dump(debug_info, open(fname, "w"))


if __name__ == "__main__":
    main()
