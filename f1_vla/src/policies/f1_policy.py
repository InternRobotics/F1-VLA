import os
import logging
import packaging
from pathlib import Path
from collections import deque

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

import safetensors
from safetensors.torch import load_model as load_model_as_safetensor
from safetensors.torch import save_model as save_model_as_safetensor

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.pi0.modeling_pi0 import resize_with_pad, pad_vector

from transformers import AutoTokenizer
from transformers.utils import logging

from f1_vla.src.models.modeling_f1 import F1FlowMatching
from f1_vla.src.models.wm.vqvae import VQVAE
from f1_vla.src.models.configuration_f1 import F1Config


logger = logging.get_logger(__name__)


class F1_VLA(nn.Module):
    config_class = F1Config
    cache_action_steps = 5

    def __init__(
        self,
        config: F1Config,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.use_world_model = config.use_world_model

        self.language_tokenizer = AutoTokenizer.from_pretrained(config.language_tokenizer_path)

        pn = config.gen_expert_config.pn
        patch_nums = tuple(map(int, pn.replace('-', '_').split('_')))

        self.vae = VQVAE(
            vocab_size=config.gen_expert_config.vae.vocab_size, 
            z_channels=config.gen_expert_config.vae.z_channels, 
            ch=config.gen_expert_config.vae.ch, 
            test_mode=config.gen_expert_config.vae.test_mode, 
            share_quant_resi=config.gen_expert_config.vae.share_quant_resi, 
            v_patch_nums=patch_nums
        )
        if os.path.exists(config.gen_expert_config.vae.vae_ckpt):
            vae_ckpt = torch.load(config.gen_expert_config.vae.vae_ckpt, map_location='cpu', weights_only=False)
            self.vae.load_state_dict(vae_ckpt, strict=True)
            del vae_ckpt
        for param in self.vae.parameters():
            param.requires_grad = False

        self.use_only_3rd_hist_image = True
        self.last_l = patch_nums[-1] * patch_nums[-1]

        self.gen_loss_fct = nn.CrossEntropyLoss(reduction="none")

        self.model = F1FlowMatching(config, patch_nums, self.vae, **kwargs)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.cache_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad
    def select_action_with_world_model(
        self, 
        batch: dict[str, Tensor], 
        noise: Tensor | None = None, 
        top_k: int = 900,
        top_p: float = 0.95,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        **kwargs,
    ) -> Tensor:
        self.eval()

        if len(self._action_queue) == 0:
            images, image_masks = self.prepare_mix_images(batch)
            world_model_images = self.prepare_mix_history_images(batch)
            state = self.prepare_state(batch)
            lang_tokens, lang_masks = self.prepare_language(batch)

            B, T, C, H, W = world_model_images.shape
            world_model_images = world_model_images.reshape(B * T, C, H, W)

            world_model_indices_list = self.model.vae.img_to_idxBl(world_model_images)
            world_model_input_embs = self.model.vae.quantize.idxBl_to_var_input(world_model_indices_list)
            world_model_input_embs = world_model_input_embs.reshape(B, T, *world_model_input_embs.shape[1:])

            action_output = self.model.sample_actions_with_world_model(
                images=images, 
                image_masks=image_masks, 
                lang_tokens=lang_tokens, 
                lang_masks=lang_masks, 
                state=state, 
                world_model_input_embs=world_model_input_embs, 
                predict_action_only=False, 
                noise=noise,
                top_k=top_k, top_p=top_p, num_samples=num_samples, rng=rng,
            )
            actions = action_output.actions

            # Unpad actions
            original_action_dim = 7
            actions = actions[:, :, :original_action_dim]

            return actions

    def forward_with_world_model(
        self, 
        batch: dict[str, Tensor], 
        noise: Tensor | None = None, 
        time: Tensor | None = None, 
        cur_n_obs_img_steps: int | None = None, 
        cur_n_pred_img_steps: int | None = None,  
        train_gen_expert_only: bool = False, 
        gen_out_loss_ratio: float = 0.1
    ) -> dict[str, Tensor]:

        #########################################################
        # prepare the inputs
        #########################################################

        images, img_masks = self.prepare_mix_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        action_is_pad = batch.get("action_is_pad")

        world_model_images = self.prepare_mix_history_images(batch)
        B, T, C, H, W = world_model_images.shape
        world_model_images = world_model_images.reshape(B * T, C, H, W)
        world_model_image_indices = self.model.vae.img_to_idxBl(world_model_images)
        # prepare the output of world model
        gt_world_model_indices = torch.cat(world_model_image_indices, dim=1).reshape(B, T, -1)[:, cur_n_obs_img_steps: cur_n_obs_img_steps + cur_n_pred_img_steps].contiguous()
        # prepare the input of world model
        world_model_embs = self.model.vae.quantize.idxBl_to_var_input(world_model_image_indices)
        world_model_embs = world_model_embs.reshape(B, T, *world_model_embs.shape[1:])
        world_model_input_embs = world_model_embs[:, :cur_n_obs_img_steps]
        world_model_output_embs = world_model_embs[:, cur_n_obs_img_steps:cur_n_obs_img_steps + cur_n_pred_img_steps]
        if len(world_model_output_embs.shape) == 4:
            world_model_output_embs = world_model_output_embs.reshape(B, -1, world_model_output_embs.shape[3])

        #########################################################
        # Forward and compute the loss
        #########################################################
        action_losses, gen_logits = self.model.forward_with_world_model(images, img_masks, lang_tokens, lang_masks, state, world_model_input_embs, world_model_output_embs, actions, noise, time)

        gen_token_len = gen_logits.shape[1]
        gt_world_model_indices = gt_world_model_indices.reshape(B, -1)[:, :gen_token_len]
        gen_loss = self.gen_loss_fct(gen_logits.reshape(-1, gen_logits.shape[-1]), gt_world_model_indices.reshape(-1)).view(B, -1)
        gen_loss = gen_loss.mean()

        loss_dict = {}
        loss_dict["wm_acc_mean"] = (gen_logits.argmax(dim=-1) == gt_world_model_indices).float().mean()
        last_resolution_token_len = self.model.num_resolutions * self.model.num_resolutions
        loss_dict["wm_acc_tail"] = (gen_logits[:, -last_resolution_token_len:].argmax(dim=-1) == gt_world_model_indices[:, -last_resolution_token_len:]).float().mean()

        if train_gen_expert_only:
            loss_dict["loss"] = gen_loss
            loss_dict["wm_loss"] = gen_loss
            return loss_dict

        loss_dict["action_losses_after_forward"] = action_losses.clone()

        if action_is_pad is not None:
            in_episode_bound = ~action_is_pad
            if action_losses.shape == in_episode_bound.shape:
                action_losses = action_losses * in_episode_bound
            else:
                action_losses = action_losses * in_episode_bound.unsqueeze(-1)
            loss_dict["action_losses_after_in_ep_bound"] = action_losses.clone()

        # Remove padding
        action_losses = action_losses[:, :, : self.config.max_action_dim]
        loss_dict["action_losses_after_rm_padding"] = action_losses.clone()
        loss_dict["action_loss"] = action_losses.mean().clone()
        loss_dict["wm_loss"] = gen_loss.clone()

        loss_dict["loss"] = loss_dict["action_loss"] + gen_out_loss_ratio * loss_dict["wm_loss"]

        return loss_dict

    def prepare_mix_images(self, batch):
        images = []
        image_masks = []

        # Hack
        img_keys = [
            "observation.images.image0",
            "observation.images.image1",
            "observation.images.image2",
        ]

        for key in img_keys:
            if key not in batch:
                img = torch.zeros_like(batch["observation.images.image0"])
                mask = torch.zeros_like(batch["observation.images.image0_mask"])
                if self.config.resize_imgs_with_padding is not None:
                    img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
                images.append(img)
                image_masks.append(mask)
                continue
            img = batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            img = img * 2.0 - 1.0

            images.append(img)
            image_masks.append(batch[f"{key}_mask"])
        
        # delete the empty images
        for i in range(len(images) - 1, -1, -1):
            if images[i].sum() == 0:
                images.pop(i)
                image_masks.pop(i)

        return images, image_masks

    def prepare_mix_history_images(self, batch):
        images = []
        img_keys = ["observation.images.image0_history"]

        # Preprocess image features present in the batch
        for key in img_keys:
            hist_img = batch[key]

            # assert len(hist_img.shape) == 5
            images.append(hist_img)

        return images[0]

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch[OBS_STATE].device
        tasks = batch["task"]

        # PaliGemma prompt has to end with a new line
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding="max_length",
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
            truncation=True,
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: F1Config | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ):
        """
        The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are
        deactivated). To train it, you should first set it back in training mode with `policy.train()`.
        """
        if config is None:
            config = F1Config.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)

        if model_id.endswith(".json"):
            model_id = "/".join(model_id.split("/")[:-1])

        if os.path.isdir(model_id):
            print(f"Loading weights from local directory: {model_id}")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, "cpu", strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, "cpu", strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        if packaging.version.parse(safetensors.__version__) < packaging.version.parse("0.4.3"):
            load_model_as_safetensor(model, model_file, strict=strict)
            if map_location != "cpu":
                logger.warning(
                    "Loading model weights on other devices than 'cpu' is not supported natively in your version of safetensors."
                    " This means that the model is loaded on 'cpu' first and then copied to the device."
                    " This leads to a slower loading time."
                    " Please update safetensors to version 0.4.3 or above for improved performance."
                )
                model.to(map_location)
        else:
            safetensors.torch.load_model(model, model_file, strict=strict, device=map_location)
        return model

    def _save_pretrained(self, save_directory: Path) -> None:
        self.config.save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))
