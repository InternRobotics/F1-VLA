import dataclasses
import numpy as np
from collections.abc import Sequence, Callable

import torchvision.transforms.functional as TF

from torchvision.transforms import transforms as tv_transforms

from typing import (
    Protocol,
    Dict,
    List,
    Any,
    Tuple,
)

from PIL import Image

import torch

import f1_vla.src.utils.image_tools as image_tools

from lerobot.configs.policies import PreTrainedConfig


@dataclasses.dataclass
class TensorSpec:
    shape: Tuple[int, ...]
    dtype: torch.dtype


class DataTransformFn(Protocol):
    def __call__(self, data: Dict) -> Dict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass
class CompositeTransform(DataTransformFn):
    transforms: Sequence[DataTransformFn]

    def __call__(self, data: Dict) -> Dict:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclasses.dataclass
class RepackTransform:
    def __init__(self, structure: Dict[str, Any]):
        self.structure = structure

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flat_item = flatten_dict(data)
        return {new_key: flat_item[old_key] for new_key, old_key in self._flatten_structure(self.structure).items()}

    def _flatten_structure(self, structure: Dict[str, Any], parent_key: str = '', sep: str = '/') -> Dict[str, str]:
        """Flatten the structure dictionary."""
        items = []
        for k, v in structure.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_structure(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


@dataclasses.dataclass
class InjectHistoryObservation:
    n_obs_img_steps: int
    n_pred_img_steps: int
    obs_img_stride: int
    suffix: str = "history"

    def __post_init__(self):
        if self.n_obs_img_steps % self.obs_img_stride != 0:
            raise ValueError(f"n_obs_img_steps {self.n_obs_img_steps} must be divisible by obs_img_stride {self.obs_img_stride}")
        if self.n_pred_img_steps % self.obs_img_stride != 0:
            raise ValueError(f"n_pred_img_steps {self.n_pred_img_steps} must be divisible by obs_img_stride {self.obs_img_stride}")

        self.cur_n_obs_img_steps = self.n_obs_img_steps // self.obs_img_stride
        self.cur_n_pred_img_steps = self.n_pred_img_steps // self.obs_img_stride
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sample = {}
        for key, value in data.items():
            if key.startswith("observation") and "image" in key and self.suffix not in key:
                sample[key] = value[-self.cur_n_pred_img_steps-1]
            else:
                sample[key] = value
        return sample


@dataclasses.dataclass
class TransformGenObservation:
    gen_obs_transforms: list
    suffix: str = "history"

    def __call__(self, data: Dict) -> Dict:
        for k, v in data.items():
            if k.endswith(self.suffix):
                v = [image_tools.convert_to_uint8(x) for x in v.numpy()]
                v = [Image.fromarray(x.transpose(1, 2, 0)) for x in v]
                for t in self.gen_obs_transforms:
                    if isinstance(t, ConsistentRandomCrop):
                        v = t(v)
                    else:
                        v = [t(x) for x in v]
                data[k] = torch.stack(v)
        return data


class ConsistentRandomCrop(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = tuple(tv_transforms._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))

    def forward(self, img_list: List[Image.Image]):
        img = img_list[0]

        i, j, h_crop, w_crop = tv_transforms.RandomCrop.get_params(img, output_size=self.size)

        cropped_frames = [TF.crop(f, i, j, h_crop, w_crop) for f in img_list]
        return cropped_frames


@dataclasses.dataclass
class Normalize(DataTransformFn):
    norm_method: str
    norm_stats: dict
    norm_keys: list[str]

    def __call__(self, data: Dict) -> Dict:
        if self.norm_stats is None:
            if self.norm_method is not None and self.norm_method != "none":
                raise ValueError(f"norm_method is {self.norm_method} but norm_stats is None")
            return data

        if self.norm_method == "mean_std":
            for key in self.norm_keys:
                data[key] = (data[key] - self.norm_stats[key]["mean"]) / (self.norm_stats[key]["std"] + 1e-8)
        elif self.norm_method == "min_max":
            for key in self.norm_keys:
                data[key] = (data[key] - self.norm_stats[key]["min"]) / (self.norm_stats[key]["max"] - self.norm_stats[key]["min"] + 1e-8)
        elif self.norm_method == "quantile":
            for key in self.norm_keys:
                data[key] = (data[key] - self.norm_stats[key]["q01"]) / (self.norm_stats[key]["q99"] - self.norm_stats[key]["q01"] + 1e-8) * 2.0 - 1.0
        else:
            raise ValueError(f"norm_method is {self.norm_method} but norm_stats is None")
        
        return data


@dataclasses.dataclass
class Unnormalize(DataTransformFn):
    norm_method: str
    norm_stats: dict
    norm_keys: list[str]

    def __call__(self, data: Dict) -> Dict:
        if self.norm_stats is None:
            if self.norm_method is not None and self.norm_method != "none":
                raise ValueError(f"norm_method is {self.norm_method} but norm_stats is None")
            return data

        if self.norm_method == "mean_std":
            for key in self.norm_keys:
                data[key] = data[key] * (self.norm_stats[key]["std"] + 1e-8) + self.norm_stats[key]["mean"]
        elif self.norm_method == "min_max":
            for key in self.norm_keys:
                data[key] = data[key] * (self.norm_stats[key]["max"] - self.norm_stats[key]["min"] + 1e-8) + self.norm_stats[key]["min"]
        elif self.norm_method == "quantile":
            for key in self.norm_keys:
                data[key] = (data[key] + 1.0) / 2.0 * (self.norm_stats[key]["q99"] - self.norm_stats[key]["q01"] + 1e-8) + self.norm_stats[key]["q01"]
        else:
            raise ValueError(f"norm_method is {self.norm_method} but norm_stats is None")
        
        return data


@dataclasses.dataclass
class ResizeImages(DataTransformFn):
    height: int
    width: int
    suffix: str = "history"

    def __call__(self, data: Dict) -> Dict:
        for key, value in data.items():
            if self.suffix in key or "pad" in key: continue
            if key.startswith("observation.images."):
                data[key] = image_tools.resize_with_pad(value, self.width, self.height, pad_value=0)

        return data


@dataclasses.dataclass
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: Dict) -> Dict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass
class DeltaActions(DataTransformFn):
    """Convert absolute actions to delta actions (action - current_state).
    
    This transform converts actions from absolute values to relative changes
    from the current state: delta_action = action - current_state
    """
    state_key: str = "observation.state"
    action_key: str = "action"

    def __call__(self, data: Dict) -> Dict:
        if self.action_key not in data:
            raise ValueError(f"Action key {self.action_key} not found in data")

        state, actions = data[self.state_key], data[self.action_key]
        abs_pre_actions = torch.cat((state[None, :], actions[:-1]), dim=0)
        abs_post_actions = actions.clone()
        actions = abs_post_actions - abs_pre_actions
        
        data[self.action_key] = actions

        return data


def _assert_quantile_stats(norm_stats) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )


def flatten_dict(tree: dict, parent_key: str = '', sep: str = '/') -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    items = []
    for k, v in tree.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, return_mask: bool = False) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        x = np.pad(x, pad_width)
        if return_mask:
            mask = np.ones_like(x)
            mask[..., :current_dim] = 0
            return x, mask
        return x
    return x, np.zeros_like(x)


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def unflatten_dict(tree: dict, sep: str = '/') -> dict:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    result = {}
    for key, value in tree.items():
        parts = key.split(sep)
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def apply_tree(
    tree: Dict, selector: Dict, fn: Callable[[Any, Any], Any], *, strict: bool = False
) -> Dict:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: Any) -> Any:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


class ModelTransformGroup(Group):
    default_prompt: str | None = None

    def __init__(
        self, 
        policy_config: PreTrainedConfig, 
        n_obs_img_steps: int, 
        n_pred_img_steps: int, 
        obs_img_stride: int,
        gen_obs_transforms: tv_transforms.Compose,
    ) -> Group:
        super().__init__()
        match policy_config.model_type:
            case "f1" | "pi0":
                self.inputs = [
                    InjectHistoryObservation(
                        n_obs_img_steps=n_obs_img_steps,
                        n_pred_img_steps=n_pred_img_steps,
                        obs_img_stride=obs_img_stride,
                    ),
                    TransformGenObservation(
                        gen_obs_transforms=gen_obs_transforms,
                    ),
                ]
