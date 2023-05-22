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
import os
import torch
import accelerate
import numpy as np

from syne_tune.optimizer.baselines import RandomSearch

from estimate_efficency import compute_parameters
from ask_tell_scheduler import AskTellScheduler
from sampling import SmallSearchSpace
from multi_objective import get_pareto_optimal
from masking import apply_neuron_mask
from masking_gpt import apply_neuron_mask_gpt2


accelerator = accelerate.Accelerator()


def multi_objective_search(model, eval_dataloader, metric, metric_name, search_args):

    model_type = model.config._name_or_path
    if model_type.startswith("bert"):

        attention_size = model.config.hidden_size
        num_attention_heads = model.config.num_attention_heads
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

        num_attention_heads = model.config.n_head
        attention_size = model.config.hidden_size
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
        # model = model.from_pretrained(search_args.checkpoint_dir_model)
    # else:
    #     model.load_state_dict(
    #         torch.load(
    #             os.path.join(search_args.checkpoint_dir_model, "checkpoint.pt"),
    #             map_location="cuda:0",
    #         ),
    #     )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

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
                torch.squeeze(logits) if search_args.is_regression else torch.argmax(logits, dim=-1)
            )

            metric.add_batch(predictions=predictions, references=batch["labels"])

        eval_metric = metric.compute()
        for handle in handles:
            handle.remove()

        return 1 - eval_metric[metric_name], n_params

    search_space = SmallSearchSpace(model.config)

    base_scheduler = RandomSearch(
        config_space=search_space.get_syne_tune_config_space(),
        metric=["error", "params"],
        mode=["min", "min"],
    )

    scheduler = AskTellScheduler(base_scheduler=base_scheduler)

    costs = np.empty((search_args.num_samples, 2))
    masks = []
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

    idx = get_pareto_optimal(costs)
    indices = np.arange(costs.shape[0])[idx]
    masks = [masks[i] for i in indices]

    pareto_set = {'masks': masks}

    return pareto_set
