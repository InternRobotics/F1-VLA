import numpy as np
from dataclasses import dataclass
from typing import SupportsIndex, Sequence

import torch
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.transforms import (
    ImageTransforms, 
    ImageTransformsConfig, 
)

from transformers.configuration_utils import PretrainedConfig
from f1_vla.src.configs.base_config import DataConfig
import f1_vla.src.utils.transforms as _transforms
from f1_vla.src.processors.train_processors.policy_trainer import PolicyTrainingArguments


class TransformedDataset(Dataset):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex):
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


def transform_dataset(dataset: Dataset, dataset_config: DataConfig):
    return TransformedDataset(
        dataset,
        [
            *dataset_config.data_transforms.inputs,
            *dataset_config.repack_transforms.inputs,
            *dataset_config.model_transforms.inputs,
        ],
    )


@dataclass
class CollateFn:
    max_state_dim: int = 50
    max_action_dim: int = 50
    suffix: str = "history"
    image_size: tuple[int, int] = (224, 224)

    def __call__(self, items):
        dataset_idx = [x[0] for x in items]
        items = [x[1] for x in items]

        batch = {"dataset_idx": torch.tensor(dataset_idx, dtype=torch.long)}

        # observation.images
        max_num_images, image_shapes = self._compute_max_num_images(items, prefix="observation.images.image", suffix=self.suffix)
        for num_image_idx in range(max_num_images):
            img_key = f"observation.images.image{num_image_idx}"
            imgs, img_masks = [], []
            for sample in items:
                if img_key in sample.keys():
                    imgs.append(sample[img_key])
                    img_masks.append(1)
                else:
                    imgs.append(torch.zeros(image_shapes[img_key]))
                    img_masks.append(0)
            batch[img_key] = torch.stack(imgs)
            batch[f"{img_key}_mask"] = torch.tensor(img_masks, dtype=torch.bool)

        # observation.state
        state = [x["observation.state"] for x in items]
        padded_states = []
        state_masks = []
        for s in state:
            length = s.shape[0]
            if length < self.max_state_dim:
                pad_size = self.max_state_dim - length
                padded = torch.cat([s, torch.zeros(pad_size, dtype=s.dtype, device=s.device)])
                mask = torch.cat([torch.ones(length, dtype=torch.bool, device=s.device),
                                torch.zeros(pad_size, dtype=torch.bool, device=s.device)])
            else:
                padded = s[:self.max_state_dim]
                mask = torch.ones(self.max_state_dim, dtype=torch.bool, device=s.device)
            padded_states.append(padded)
            state_masks.append(mask)

        batch["observation.state"] = torch.stack(padded_states)
        batch["observation.state_mask"] = torch.stack(state_masks)

        # action
        actions = [x["action"] for x in items]
        padded_actions = []
        action_masks = []
        for a in actions:
            action_dim = a.shape[1]
            if action_dim < self.max_action_dim:
                pad_size = self.max_action_dim - action_dim
                padded = torch.cat([
                    a, 
                    torch.zeros(a.shape[0], pad_size, dtype=a.dtype, device=a.device)
                ], dim=-1)
                mask = torch.cat([
                    torch.ones(a.shape[0], action_dim, dtype=torch.bool, device=a.device),
                    torch.zeros(a.shape[0], pad_size, dtype=torch.bool, device=a.device)
                ], dim=-1)
            else:
                padded = a[:, :self.max_action_dim]
                mask = torch.ones(a.shape[0], self.max_action_dim, dtype=torch.bool, device=a.device)
            padded_actions.append(padded)
            action_masks.append(mask)

        batch["action"] = torch.stack(padded_actions)
        batch["action_mask"] = torch.stack(action_masks)

        action_is_pad = [x["action_is_pad"] for x in items]
        batch["action_is_pad"] = torch.stack(action_is_pad)

        # generation expert inputs
        gen_obs_key = f"observation.images.image0_{self.suffix}"
        gen_obs = []
        for item in items:
            gen_obs.append(item[gen_obs_key])
        batch[gen_obs_key] = torch.stack(gen_obs)
        
        # task instruction
        batch["task"] = [x["task"] for x in items]

        return batch

    def _compute_max_num_images(self, items, prefix, suffix):
        max_num = -1
        image_shapes = {}
        for item in items:
            num = 0
            for key in item.keys():
                if key.startswith(prefix) and not key.endswith(suffix) and not key.endswith("pad"):
                    if key in image_shapes:
                        assert image_shapes[key] == item[key].shape
                    else:
                        image_shapes[key] = item[key].shape
                    num += 1
            max_num = max(max_num, num)

        return max_num, image_shapes


class MixtureDataset(Dataset):
    def __init__(
        self, 
        datasets: Sequence[Dataset], 
        datasets_name: Sequence[str], 
        datasets_meta: Sequence[LeRobotDatasetMetadata], 
        datasets_weights: Sequence[float] = None,
        is_eval: bool = False,
        num_eval_episodes: int = None,
        stage: str = "stage1_pretrain_wm",
        shuffle: bool = True,
    ):
        self.datasets = datasets
        self.datasets_name = datasets_name
        self.meta = datasets_meta
        self.num_episodes = [info["num_episodes"] for info in datasets_meta]
        self.num_frames = [info["num_frames"] for info in datasets_meta]
        self.stage = stage
        
        assert stage in [
            "stage1_pretrain_wm", 
            "stage1_infer_wm", 
            "stage2_pretrain_vla",
            "stage2_infer_vla",
            "stage3_finetune_vla",
            "stage3_infer_vla",
        ]

        self._compute_len(is_eval)
        self._get_weights(datasets_weights)

        if not is_eval:
            assert len(self.flat_sample_map) == len(self.sample_weights)

            if shuffle:
                # shuffle the flat_sample_map and sample_weights in the same order
                indices = np.random.permutation(len(self.flat_sample_map))
                self.flat_sample_map = [self.flat_sample_map[i] for i in indices]
                self.sample_weights = self.sample_weights[indices]
        else:
            selected_indices = np.random.choice(
                len(self.flat_sample_map),
                size=num_eval_episodes,
                replace=False,
                p=self.sample_weights
            )
            self.flat_sample_map = [self.flat_sample_map[i] for i in selected_indices]
            self.sample_weights = self.sample_weights[selected_indices]
            self.sample_weights = self.sample_weights / np.sum(self.sample_weights)
            with open("flat_sample_map.txt", "w") as f:
                for item in self.flat_sample_map:
                    f.write(str(item) + "\n")

    def __len__(self):
        return len(self.flat_sample_map)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.flat_sample_map):
            raise IndexError(f"Index {index} is out of bounds for the dataset.")

        dataset_idx, sample_idx = self.flat_sample_map[index]

        return (dataset_idx, self.datasets[dataset_idx][sample_idx])

    def _compute_len(self, is_eval=False):
        self.all_sample_indices = []

        for i, (ds, meta) in enumerate(zip(self.datasets, self.meta)):
            actual_ds = ds._dataset

            # Only stage1 uses limited sampling via num_indices, other stages use all data
            num_indices = meta["num_indices"] if self.stage.startswith("stage1") else None

            if isinstance(actual_ds, MultiLeRobotDataset):
                indices_list = []

                for sub_ds in actual_ds._datasets:
                    _from = sub_ds.episode_data_index["from"]
                    _to = sub_ds.episode_data_index["to"]

                    indices = self._sample_indices(_from, _to, num_indices, is_eval=is_eval, dataset_name=self.datasets_name[i])
                    indices_list.append(indices)

                self.all_sample_indices.append(indices_list)

            elif isinstance(actual_ds, LeRobotDataset):
                _from = actual_ds.episode_data_index["from"]
                _to = actual_ds.episode_data_index["to"]

                indices = self._sample_indices(_from, _to, num_indices, is_eval=is_eval, dataset_name=self.datasets_name[i])
                self.all_sample_indices.append(indices)

        self.flat_sample_map = self._create_flat_sample_map()

    def _create_flat_sample_map(self):
        flat_map = []
        for dataset_idx, sample_group in enumerate(self.all_sample_indices):
            if isinstance(sample_group, list) and len(sample_group) > 0 and isinstance(sample_group[0], list):
                for sub_group in sample_group:
                    for tensor in sub_group:
                        for i in range(tensor.numel()):
                            flat_map.append((dataset_idx, tensor[i].item()))
            elif isinstance(sample_group, list) and len(sample_group) > 0:
                for tensor in sample_group:
                    for i in range(tensor.numel()):
                        flat_map.append((dataset_idx, tensor[i].item()))
            elif isinstance(sample_group, torch.Tensor):
                for i in range(sample_group.numel()):
                    flat_map.append((dataset_idx, sample_group[i].item()))
        
        return flat_map

    def _sample_indices(self, start, end, num_frames, random_pad=False, is_eval=False, dataset_name=None):
        all_indices = []
        for _start, _end in zip(start, end):

            # shift the indices by 1 if the dataset is eval
            if is_eval and self.stage.startswith("stage1"):
                _start += 1
                if _start >= _end:
                    continue

            frame_count = _end - _start

            # Determine sampling strategy based on stage
            if self.stage.startswith("stage1"):
                assert num_frames is not None, "num_frames should be provided for stage 1"
                target_frames = num_frames
            else:
                assert num_frames is None, "num_frames should not be provided for stage 2, 3"
                # For stage 2/3, use different strategies per dataset
                if "agibot" in dataset_name.lower() and self.stage.startswith("stage2"):
                    # AgiBotWorld uses 1/3 sampling
                    target_frames = max(1, frame_count // 3)
                else:
                    target_frames = frame_count

            if frame_count >= target_frames:
                indices = torch.linspace(_start, _end - 1, steps=target_frames).long()
            else:
                if random_pad:
                    pad_size = target_frames - frame_count
                    indices = torch.arange(_start, _end)
                    pad_indices = indices[torch.randint(0, frame_count, (pad_size,))]
                    indices = torch.cat([indices, pad_indices])
                    indices = indices[torch.randperm(target_frames)]
                else:
                    indices = torch.arange(_start, _end)

            all_indices.append(indices)

        return all_indices

    def _get_weights(self, datasets_weights: dict):
        """Assign normalized sampling weights to each sample."""
        self.sample_weights = []
        self.datasets_weight_map = {}
        self.sample_weights = []

        for ds_name, sample_group in zip(self.datasets_name, self.all_sample_indices):
            if isinstance(sample_group, list) and isinstance(sample_group[0], torch.Tensor):
                flattened = sample_group
            elif isinstance(sample_group, list):
                flattened = [x for sublist in sample_group for x in sublist]
            else:
                flattened = [sample_group]

            num_indices = sum(x.numel() for x in flattened)
            weight = datasets_weights[ds_name]

            if ds_name not in self.datasets_weight_map:
                self.datasets_weight_map[ds_name] = [weight] * num_indices
            else:
                self.datasets_weight_map[ds_name].extend([weight] * num_indices)
            
            self.sample_weights.extend([weight] * num_indices)

        total_weights = sum(self.sample_weights)
        for k, v in self.datasets_weight_map.items():
            self.datasets_weight_map[k] = sum(v) / total_weights

        self.sample_weights = np.array(self.sample_weights, dtype=np.float32)
        self.sample_weights = self.sample_weights / np.sum(self.sample_weights)

    def __str__(self):
        RESET = "\033[0m"
        BOLD = "\033[1m"
        CYAN = "\033[96m"
        YELLOW = "\033[93m"
        GREEN = "\033[92m"
        MAGENTA = "\033[95m"

        lines = [
            f"{BOLD}{MAGENTA}############ 👈 Weight map: ###########{RESET}"
        ]
        max_key_len = max(len(k) for k in self.datasets_weight_map.keys()) + 2
        for k, v in self.datasets_weight_map.items():
            lines.append(f"{CYAN}{k:<{max_key_len}} : {v*100:>10.2f}{RESET}")
        lines.append("---------------------------------------")
        lines.append(f"{CYAN}{'Total episodes':<{max_key_len}}{RESET} : {YELLOW}{sum(self.num_episodes):>10.2f}{RESET}")
        lines.append(f"{BOLD}{MAGENTA}#######################################{RESET}")
        
        return "\n".join(lines)


def create_data(
    policy_config: PretrainedConfig, 
    dataset_config: DataConfig, 
    training_args: PolicyTrainingArguments, 
    stage: str,
    max_eval_samples: int = None,
):
    # create image transforms for understanding expert
    img_trans_cfg = ImageTransformsConfig(
        enable=training_args.image_transforms_enabled,
        max_num_transforms=training_args.image_transforms_max_num_transforms,
        random_order=training_args.image_transforms_random_order,
    )
    filtered_tfs = {
        name: tf for name, tf in img_trans_cfg.tfs.items() if name in training_args.image_transforms_type
    }
    img_trans_cfg.tfs = filtered_tfs
    image_transforms = ImageTransforms(img_trans_cfg)

    cur_n_obs_img_steps = -1
    cur_n_pred_img_steps = -1

    # create training datasets
    all_training_datasets = []
    all_training_datasets_name = []
    all_training_datasets_meta = []
    all_training_datasets_weight = {}
    for ds_name, ds_config in dataset_config.items():
        if isinstance(ds_config.local_path, str):
            # to handle the case that the data path is a single path
            ds_meta = LeRobotDatasetMetadata(ds_name, root=ds_config.local_path)
            delta_timestamps = {
                **{
                    key: 
                    [
                        (t - (ds_config.n_obs_img_steps - ds_config.n_pred_img_steps)) / ds_meta.fps 
                        for t in range(
                            0, ds_config.n_obs_img_steps + ds_config.n_pred_img_steps, ds_config.obs_img_stride
                        )
                    ] for key in ds_config.camera_keys
                },
                **{
                    key: [t / ds_meta.fps for t in range(policy_config.chunk_size)]
                    for key in ds_config.action_keys
                }
            }
            training_dataset = LeRobotDataset(
                ds_config.local_path,
                delta_timestamps=delta_timestamps,
                # stage=stage,
                video_backend="pyav"
            )
            meta_info = {
                "num_episodes": training_dataset.num_episodes, 
                "num_frames": training_dataset.num_frames,
                "fps": training_dataset.fps,
                "num_indices": ds_config.num_indices,
            }
        else:
            # to handle the case that the data path is a list of paths
            local_paths = ds_config.local_path
            ds_meta = LeRobotDatasetMetadata(ds_name, root=local_paths[0])
            # Note: a2d dataset has a different action timestamp with the observation timestamp
            shift = 1 if "agibot" in ds_name else 0
            delta_timestamps = {
                **{
                    key: 
                    [
                        (t - (ds_config.n_obs_img_steps - ds_config.n_pred_img_steps)) / ds_meta.fps 
                        for t in range(
                            0, ds_config.n_obs_img_steps + ds_config.n_pred_img_steps, ds_config.obs_img_stride
                        )
                    ] for key in ds_config.camera_keys
                },
                **{
                    key: [(t + shift) / ds_meta.fps for t in range(policy_config.chunk_size)]
                    for key in ds_config.action_keys
                }
            }
            training_dataset = MultiLeRobotDataset(
                local_paths, 
                delta_timestamps=delta_timestamps, 
                # stage=stage
            )
            meta_info = {
                "num_episodes": training_dataset.num_episodes, 
                "num_frames": training_dataset.num_frames,
                "fps": training_dataset.fps,
                "num_indices": ds_config.num_indices,
            }

        training_dataset = transform_dataset(training_dataset, ds_config)

        all_training_datasets_name.append(ds_name)
        all_training_datasets_meta.append(meta_info)
        all_training_datasets_weight[ds_name] = ds_config.weight
        all_training_datasets.append(training_dataset)

        # to ensure the consistency of n_obs_img_steps and n_pred_img_steps across all datasets
        if cur_n_obs_img_steps == -1:
            cur_n_obs_img_steps = ds_config.n_obs_img_steps // ds_config.obs_img_stride
        if cur_n_pred_img_steps == -1:
            cur_n_pred_img_steps = ds_config.n_pred_img_steps // ds_config.obs_img_stride

        if cur_n_obs_img_steps != ds_config.n_obs_img_steps // ds_config.obs_img_stride:
            raise ValueError(f"n_obs_img_steps is not consistent across datasets. {cur_n_obs_img_steps} != {ds_config.n_obs_img_steps // ds_config.obs_img_stride}")
        if cur_n_pred_img_steps != ds_config.n_pred_img_steps // ds_config.obs_img_stride:
            raise ValueError(f"n_pred_img_steps is not consistent across datasets. {cur_n_pred_img_steps} != {ds_config.n_pred_img_steps // ds_config.obs_img_stride}")

    training_dataset = MixtureDataset(
        all_training_datasets, 
        all_training_datasets_name,
        all_training_datasets_meta,
        all_training_datasets_weight,
        stage=stage,
    )
    training_ds_weights = training_dataset.sample_weights

    return training_dataset, image_transforms, training_ds_weights, cur_n_obs_img_steps, cur_n_pred_img_steps
