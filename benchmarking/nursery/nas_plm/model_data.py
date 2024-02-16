import transformers

from mask import mask_bert, mask_gpt, mask_gpt_neox, mask_roberta


def get_model_data(model):
    config = model.config
    model_type = model.config.model_type

    if model_type.startswith("bert"):
        attention_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        attention_head_size = int(attention_size / num_attention_heads)
        num_layers = config.num_hidden_layers
        intermediate_size = config.intermediate_size
        mask = mask_bert

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

    elif model_type.startswith("roberta"):
        attention_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        attention_head_size = int(attention_size / num_attention_heads)
        num_layers = config.num_hidden_layers
        intermediate_size = config.intermediate_size
        mask = mask_roberta

        n_params_emb = sum(
            p.numel() for p in model.roberta.embeddings.parameters() if p.requires_grad
        )
        n_params_classifier = sum(
            p.numel() for p in model.classifier.parameters() if p.requires_grad
        )

    elif model_type.startswith("distilbert"):
        attention_size = config.dim
        num_attention_heads = config.n_heads
        attention_head_size = int(attention_size / num_attention_heads)
        num_layers = config.n_layers
        intermediate_size = config.hidden_dim
        mask = mask_bert

        n_params_emb = sum(
            p.numel()
            for p in model.distilbert.embeddings.parameters()
            if p.requires_grad
        )
        n_params_pooler = sum(
            p.numel() for p in model.pre_classifier.parameters() if p.requires_grad
        )
        n_params_classifier = sum(
            p.numel() for p in model.classifier.parameters() if p.requires_grad
        )
        n_params_classifier += n_params_pooler

    elif model_type.startswith("gpt2"):
        model.config.pad_token_id = model.config.eos_token_id
        mask = mask_gpt

        num_attention_heads = config.n_head
        attention_size = config.hidden_size
        attention_head_size = int(attention_size / num_attention_heads)
        num_layers = config.n_layer
        intermediate_size = (
            config.n_inner if config.n_inner is not None else 4 * config.hidden_size
        )
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

    elif isinstance(model, transformers.models.gpt_neox.GPTNeoXForCausalLM):
        model.config.pad_token_id = model.config.eos_token_id
        mask = mask_gpt_neox

        num_attention_heads = config.num_attention_heads
        attention_size = config.hidden_size
        attention_head_size = int(attention_size / num_attention_heads)
        num_layers = config.num_hidden_layers
        intermediate_size = config.intermediate_size

        n_params_emb = sum(
            p.numel() for p in model.gpt_neox.embed_in.parameters() if p.requires_grad
        )
        final_ln = sum(
            p.numel()
            for p in model.gpt_neox.final_layer_norm.parameters()
            if p.requires_grad
        )
        n_params_classifier = sum(
            p.numel() for p in model.embed_out.parameters() if p.requires_grad
        )

        n_params_classifier += final_ln

    elif "pythia" in model_type:
        model.config.pad_token_id = model.config.eos_token_id
        mask = mask_gpt_neox

        num_attention_heads = config.num_attention_heads
        attention_size = config.hidden_size
        attention_head_size = int(attention_size / num_attention_heads)
        num_layers = config.num_hidden_layers
        intermediate_size = config.intermediate_size

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

    else:
        print(f"Model {model_type} is currently not supported!")
        return

    return {
        "mask": mask,
        "attention_size": attention_size,
        "num_attention_heads": num_attention_heads,
        "attention_head_size": attention_head_size,
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
        "n_params_emb": n_params_emb,
        "n_params_classifier": n_params_classifier,
    }
