from transformers.trainer import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names


def create_optimizer(opt_model, args):
    """
    Setup the optimizer with different learning rates for different parts of the model.
    """
    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS, ["bias", "layernorm", "rmsnorm"])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    # Group 0: vision encoder parameters
    vision_encoder_parameters = [
        name for name, _ in opt_model.named_parameters() 
        if any(x in name for x in [
            "paligemma_with_expert.paligemma.vision_tower",
            "paligemma_with_expert.paligemma.multi_modal_projector"
        ])
    ]

    # Group 1: paligemma parameters (excluding vision encoder)
    paligemma_parameters = [
        name for name, _ in opt_model.named_parameters() 
        if "paligemma_with_expert.paligemma" in name and name not in vision_encoder_parameters
    ]

    # Group 2: world model expert parameters and world model related parameters
    gen_expert_parameters = [name for name, _ in opt_model.named_parameters() if "paligemma_with_expert.gemma_wm_expert" in name]
    gen_parameters = [
        name for name, _ in opt_model.named_parameters() 
        if any(x in name for x in [
            "temporal_conv", "wm_embeddings", "wm_hist_pos_embs", 
            "wm_sos_token_embs", "wm_cond_pos_embs", "wm_output_position_embedding",
            "lvl_embed", "wm_out_layer_norm", "wm_out_proj"
        ])
    ]

    # Group 3: gemma expert parameters and action-related parameters
    act_expert_parameters = [name for name, _ in opt_model.named_parameters() if "paligemma_with_expert.gemma_expert" in name]
    action_parameters = [
        name for name, _ in opt_model.named_parameters() 
        if any(x in name for x in [
            "state_proj", "action_in_proj", "action_out_proj", 
            "action_time_mlp_in", "action_time_mlp_out"
        ])
    ]

    optimizer_grouped_parameters = []

    # Group 0: vision encoder parameters (if vision_encoder_lr is provided)
    if len(vision_encoder_parameters) > 0:
        optimizer_grouped_parameters.extend([
            {
                "params": [
                    p for n, p in opt_model.named_parameters() 
                    if n in decay_parameters and n in vision_encoder_parameters and p.requires_grad
                ],
                "weight_decay": args.weight_decay,
                "lr": args.vision_encoder_lr,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() 
                    if n not in decay_parameters and n in vision_encoder_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": args.vision_encoder_lr,
            },
        ])

    # Group 1: paligemma parameters (excluding vision encoder if separate lr is provided)
    optimizer_grouped_parameters.extend([
        {
            "params": [
                p for n, p in opt_model.named_parameters() 
                if n in decay_parameters and n in paligemma_parameters and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
            "lr": args.und_expert_lr,
        },
        {
            "params": [
                p for n, p in opt_model.named_parameters() 
                if n not in decay_parameters and n in paligemma_parameters and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": args.und_expert_lr,
        },
    ])

    # Group 2: world model expert and world model parameters
    if len(gen_parameters) > 0:
        optimizer_grouped_parameters.extend([
            {
                "params": [
                    p for n, p in opt_model.named_parameters() 
                    if n in decay_parameters and (n in gen_expert_parameters or n in gen_parameters) and p.requires_grad
                ],
                "weight_decay": args.weight_decay,
                "lr": args.gen_expert_lr,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() 
                    if n not in decay_parameters and (n in gen_expert_parameters or n in gen_parameters) and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": args.gen_expert_lr,
            },
        ])

    # Group 3: gemma expert and action parameters
    optimizer_grouped_parameters.extend([
        {
            "params": [
                p for n, p in opt_model.named_parameters() 
                if n in decay_parameters and (n in act_expert_parameters or n in action_parameters) and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
            "lr": args.act_expert_lr,
        },
        {
            "params": [
                p for n, p in opt_model.named_parameters() 
                if n not in decay_parameters and (n in act_expert_parameters or n in action_parameters) and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": args.act_expert_lr,
        },
    ])

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer
