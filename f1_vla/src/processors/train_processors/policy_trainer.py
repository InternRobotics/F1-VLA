import os
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Union, Tuple

import torch

from f1_vla.src.utils.utils import LargeScaleWeightedRandomSampler
from f1_vla.src.policies.f1_policy import F1_VLA

from lerobot.policies.pretrained import PreTrainedPolicy
from transformers import Trainer, __version__, PretrainedConfig
from transformers.trainer import (
    logger, 
    FSDP_MODEL_NAME, 
    TRAINING_ARGS_NAME, 
    is_peft_available, 
    _get_fsdp_ckpt_kwargs,
    _is_peft_model,
)
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    is_sagemaker_mp_enabled,
    is_accelerate_available,
)

if is_accelerate_available():
    from accelerate.utils import load_fsdp_model


@dataclass
class PolicyTrainingArguments(TrainingArguments):
    train_dir: str | None = None
    eval_dir: str | None = None
    num_eval_episodes: int = 50
    stage: str = "stage3_finetune_vla"
    language_tokenizer_path: str | None = None

    freeze_vision_encoder: bool = False
    freeze_gen_expert: bool = False
    train_act_expert_only: bool = False
    train_gen_expert_only: bool = False
    train_state_proj: bool = True

    gen_out_loss_ratio: float = 0.0

    resize_imgs_with_padding: Tuple[int, int] = (224, 224)

    image_transforms_enabled: bool = True
    image_transforms_max_num_transforms: int = 3
    image_transforms_random_order: bool = True
    image_transforms_type: List[str] = field(
        default_factory=lambda: ["brightness", "contrast", "saturation", "random_crop", "random_rotation"]
    )

    und_expert_lr: float = 0.0
    act_expert_lr: float = 0.0
    gen_expert_lr: float = 0.0
    vision_encoder_lr: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        random.seed(self.seed)


class PolicyTrainerCallback(TrainerCallback):
    policy: None
    image_transforms: None
    def __init__(self, policy, image_transforms):
        self.policy = policy
        self.image_transforms = image_transforms

    def on_train_begin(self, args, state, control, **kwargs):
        """ move the normalize_inputs and normalize_targets to the device """
        if self.image_transforms is not None:
            self.image_transforms.to(args.device)


class PolicyTrainer(Trainer):
    def __init__(
        self, 
        policy: Union[PreTrainedPolicy, F1_VLA], 
        image_transforms=None, 
        use_world_model=True,
        cur_n_obs_img_steps=None, 
        cur_n_pred_img_steps=None, 
        training_ds_sample_weights=None,
        *args, 
        **kwargs
    ):
        self.policy = policy
        self.image_transforms = image_transforms
        self.use_world_model = use_world_model
        # TODO: make this configurable
        self.pred_img_keys = ["observation.images.image0_history"]
        assert len(self.pred_img_keys) == 1, "Only one image key is supported for now"

        self.cur_n_obs_img_steps = cur_n_obs_img_steps
        self.cur_n_pred_img_steps = cur_n_pred_img_steps

        move_callbacks = [PolicyTrainerCallback(policy=policy, image_transforms=image_transforms)]

        self.training_ds_sample_weights = training_ds_sample_weights

        self.worker_idx = int(os.environ.get("MLP_ROLE_INDEX", 0))
        self.local_rank_idx = int(os.environ.get('LOCAL_RANK', -1))

        super().__init__(model=policy, callbacks=move_callbacks, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # apply image transforms to the inputs of understanding expert
        if self.image_transforms is not None:
            for key, value in inputs.items():
                if "history" in key or "mask" in key:
                    continue
                if key.startswith("observation.images"):
                    inputs[key] = self.image_transforms(value)

        outputs = self.policy.forward_with_world_model(
            inputs, 
            cur_n_obs_img_steps=self.cur_n_obs_img_steps, 
            cur_n_pred_img_steps=self.cur_n_pred_img_steps,
            train_gen_expert_only=self.args.train_gen_expert_only,
            gen_out_loss_ratio=self.args.gen_out_loss_ratio,
        )

        loss = outputs["loss"]

        if self.state.is_local_process_zero and self.state.is_world_process_zero:
            if self.state.global_step % self.state.logging_steps == 0 and self.state.global_step != 0:
                action_lr_log = {
                    "action_learning_rate": self.optimizer.param_groups[-1]["lr"],
                }
                action_log = {
                    "action_loss": outputs.get("action_loss", torch.tensor(0)).cpu().item(),
                }
                if self.policy.use_world_model:
                    wm_log = {
                        "wm_out_loss": outputs.get("wm_loss", torch.tensor(0)).cpu().item(),
                        "wm_acc_mean": outputs.get("wm_acc_mean", torch.tensor(0)).cpu().item(),
                        "wm_acc_tail": outputs.get("wm_acc_tail", torch.tensor(0)).cpu().item(),
                        "wm_learning_rate": self.optimizer.param_groups[4]["lr"],
                    }
                    vit_log = {
                        "vit_learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                    if self.policy.model.train_gen_expert_only:
                        loss_dict = {**wm_log, **vit_log}
                    else:
                        loss_dict = {**wm_log, **vit_log, **action_lr_log, **action_log}
                else:
                    loss_dict = {**action_lr_log, **action_log}

                self.log(loss_dict)

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.accelerator.unwrap_model(self.model)._save_pretrained(Path(output_dir))
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):

        if model is None:
            model = self.model

        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)
        is_fsdp_ckpt = os.path.isdir(resume_from_checkpoint) and (
            # this checks the FSDP state dict when `SHARDED_STATE_DICT` is used
            any(
                FSDP_MODEL_NAME in folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
            )
            # this checks the FSDP state dict when `FULL_STATE_DICT` is used
            or os.path.isfile(os.path.join(resume_from_checkpoint, f"{FSDP_MODEL_NAME}.bin"))
        )
        # if multiple adapters exist, they get saved in sub directories
        adapter_subdirs = (
            [
                folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
                and (
                    os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_WEIGHTS_NAME))
                    or os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_SAFE_WEIGHTS_NAME))
                )
            ]
            if os.path.isdir(resume_from_checkpoint)
            else []
        )

        if is_fsdp_ckpt and not self.is_fsdp_enabled:
            raise ValueError(f"Checkpoint found at {resume_from_checkpoint} is only supported when using PyTorch FSDP")

        if not (
            any(
                os.path.isfile(f)
                for f in [
                    weights_file,
                    safe_weights_file,
                    weights_index_file,
                    safe_weights_index_file,
                    adapter_weights_file,
                    adapter_safe_weights_file,
                ]
            )
            or is_fsdp_ckpt
            or adapter_subdirs
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(config_file):
            config = PretrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file) or is_fsdp_ckpt:
            weights_only_kwarg = {"weights_only": True}
            # If the model is on the GPU, it still works!
            if is_sagemaker_mp_enabled():
                if os.path.isfile(os.path.join(resume_from_checkpoint, "user_content.pt")):
                    # If the 'user_content.pt' file exists, load with the new smp api.
                    # Checkpoint must have been saved with the new smp api.
                    smp.resume_from_checkpoint(
                        path=resume_from_checkpoint, tag=WEIGHTS_NAME, partial=False, load_optimizer=False
                    )
                else:
                    # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                    # Checkpoint must have been saved with the old smp api.
                    if hasattr(self.args, "fp16") and self.args.fp16 is True:
                        logger.warning(
                            "Enabling FP16 and loading from smp < 1.10 checkpoint together is not suppported."
                        )
                    state_dict = torch.load(
                        weights_file,
                        map_location="cpu",
                        **weights_only_kwarg,
                    )
                    # Required for smp to not auto-translate state_dict from hf to smp (is already smp).
                    state_dict["_smp_is_partial"] = False
                    load_result = model.load_state_dict(state_dict, strict=True)
                    # release memory
                    del state_dict
            elif self.is_fsdp_enabled:
                load_fsdp_model(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    model,
                    resume_from_checkpoint,
                    **_get_fsdp_ckpt_kwargs(),
                )
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                    model = PreTrainedPolicy._load_as_safetensor(model, safe_weights_file, "cpu", False)
                    logger.info(f"\033[31mLoading model from {safe_weights_file} complete !!\033[0m")
                else:
                    raise NotImplementedError("Not implemented")

        # Load adapters following PR # 24096
        elif _is_peft_model(model):
            # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
            # TODO: in the future support only specific min PEFT versions
            if (hasattr(model, "active_adapter") or hasattr(model, "active_adapters")) and hasattr(
                model, "load_adapter"
            ):
                if os.path.exists(resume_from_checkpoint):
                    # For BC for older PEFT versions
                    if hasattr(model, "active_adapters"):
                        active_adapters = model.active_adapters
                        if len(active_adapters) > 1:
                            logger.warning("Multiple active adapters detected will only consider the first adapter")
                        active_adapter = active_adapters[0]
                    else:
                        active_adapter = model.active_adapter

                    if adapter_subdirs:
                        for subdir_name in adapter_subdirs:
                            peft_id = os.path.join(resume_from_checkpoint, subdir_name)
                            model.load_adapter(peft_id, subdir_name, is_trainable=(subdir_name == active_adapter))
                        model.set_adapter(active_adapter)
                    else:
                        model.load_adapter(resume_from_checkpoint, active_adapter, is_trainable=True)
                else:
                    logger.warning(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                        "Check some examples here: https://github.com/huggingface/peft/issues/96"
                    )
            else:
                logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
        else:
            # We load the sharded checkpoint
            load_result = load_sharded_checkpoint(
                model, resume_from_checkpoint, strict=is_sagemaker_mp_enabled(), prefer_safe=self.args.save_safetensors
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return LargeScaleWeightedRandomSampler(self.training_ds_sample_weights, len(self.train_dataset))
