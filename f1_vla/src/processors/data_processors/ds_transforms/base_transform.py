import dataclasses
import torch
import numpy as np
from abc import abstractmethod
from f1_vla.src.utils.transforms import Normalize

@dataclasses.dataclass
class BaseTransform:
    action_dim: int = 9

    camera_keys: list[str] = dataclasses.field(default_factory=lambda: ["observation.images.image_0"])
    state_keys: list[str] = dataclasses.field(default_factory=lambda: ["observation.states"])
    action_keys: list[str] = dataclasses.field(default_factory=lambda: ["action"])
    keys_to_keep: list[str] = dataclasses.field(default_factory=lambda: ["timestamp", "index", "frame_index", "episode_index", "task_index", "task"])

    do_normalize: bool = True
    norm_method: str = "mean_std"
    norm_stats: dict | None = None

    def __post_init__(self):
        if self.do_normalize:
            if self.norm_stats is None:
                raise ValueError("norm_stats must be provided if do_normalize is True")
            # Convert numpy arrays to torch tensors
            for key, value in self.norm_stats.items():
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        self.norm_stats[key][k] = torch.from_numpy(v).to(torch.float32)
                    elif isinstance(v, list):
                        self.norm_stats[key][k] = torch.from_numpy(np.array(v)).to(torch.float32)
                    elif isinstance(v, torch.Tensor):
                        self.norm_stats[key][k] = v
                    else:
                        raise ValueError(f"Unsupported type: {type(v)}")

            self.normalize_state = Normalize(self.norm_method, self.norm_stats, self.state_keys)
            self.normalize_action = Normalize(self.norm_method, self.norm_stats, self.action_keys)
        
        valid_camera_keys = set(self.camera_keys)
        valid_camera_keys.update(k + "_is_pad" for k in self.camera_keys)

        valid_state_keys = set(self.state_keys)

        valid_action_keys = set(self.action_keys)
        valid_action_keys.update(k + "_is_pad" for k in self.action_keys)

        self.valid_keys = valid_camera_keys | valid_state_keys | valid_action_keys | set(self.keys_to_keep)


@dataclasses.dataclass
class BaseTransformOutputs:
    action_dim: int = 9

    @abstractmethod
    def __call__(self, data: dict) -> dict:
        pass
