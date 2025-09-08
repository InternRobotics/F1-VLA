import os
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
from typing import Union

import torch
from torch.utils.data import Sampler

from f1_vla.src.policies.f1_policy import F1_VLA


def save_training_args(training_args, policy_config, config):
    os.makedirs(training_args.output_dir, exist_ok=True)
    policy_config.save_pretrained(Path(training_args.output_dir))

    if not os.path.exists(Path(training_args.output_dir) / "config.yaml"):
        OmegaConf.save(config, Path(training_args.output_dir) / "config.yaml")


def clean_overrides(override_args):
    cleaned_args = []
    for arg in override_args:
        if arg.startswith("--"):
            cleaned_args.append(arg[2:])
        else:
            cleaned_args.append(arg)
    return cleaned_args


def load_ckpt(policy, config):
    if config.exp.load_ckpt is not None:
        F1_VLA._load_as_safetensor(policy, config.exp.load_ckpt, "cpu", False)
        
    return policy


def set_policy_config(policy_config, src_config):
    """
    Set the policy config from the config file
    Args:
        policy_config: The policy config to set which is used to initialize the policy
        src_config: The policy config from the local config file
    """
    policy_config.pretrained_path = src_config.path
    policy_config.language_tokenizer_path = src_config.language_tokenizer_path

    policy_config.use_world_model = src_config.use_world_model

    if policy_config.use_world_model:
        policy_config.gen_expert_config.pn = src_config.pn
        policy_config.gen_expert_config.temporal_conv_kernel_size = src_config.temporal_conv_kernel_size
        policy_config.gen_expert_config.temporal_conv_stride = src_config.temporal_conv_stride
        policy_config.gen_expert_config.num_resolutions = src_config.num_resolutions
        policy_config.gen_expert_config.vae.vae_ckpt = src_config.vae_ckpt

    policy_config.resize_imgs_with_padding = eval(src_config.resize_imgs_with_padding)

    policy_config.attention_implementation = src_config.attention_implementation
    policy_config.chunk_size = src_config.chunk_size

    return policy_config


class LargeScaleWeightedRandomSampler(Sampler):
    def __init__(
        self, 
        weights: Union[torch.Tensor, list, np.ndarray], 
        num_samples: int, 
        replacement: bool = True, 
        max_block: int = 2**24 - 1
    ):
        if isinstance(weights, list):
            weights = torch.tensor(weights)
        elif isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
        self.max_block = max_block

    def __iter__(self):
        return iter(self._sample_indices().tolist())

    def _sample_indices(self) -> torch.Tensor:
        weights = self.weights
        total_weight = weights.sum()
        indices = []
        n = len(weights)
        num_blocks = (n + self.max_block - 1) // self.max_block

        for i in range(num_blocks):
            start = i * self.max_block
            end = min((i + 1) * self.max_block, n)
            block_weights = weights[start:end].float()
            block_weight_sum = block_weights.sum()

            if block_weight_sum == 0:
                continue

            block_prob = block_weight_sum / total_weight
            block_sample_count = int(round(self.num_samples * block_prob.item()))
            sampled = torch.multinomial(block_weights, block_sample_count, self.replacement)
            indices.append(sampled + start)

        return torch.cat(indices)[:self.num_samples]  # truncate in case of rounding error

    def __len__(self):
        return self.num_samples


def convert_ds_stats_to_dict(ds_stats):
    for k, v in ds_stats.items():
        for _k, _v in v.items():
            if isinstance(_v, np.ndarray):
                ds_stats[k][_k] = _v.tolist()
    return ds_stats
