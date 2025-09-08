import dataclasses
import numpy as np

from f1_vla.src.processors.data_processors.ds_transforms.base_transform import (
    BaseTransform,
    BaseTransformOutputs
)


@dataclasses.dataclass
class FractalInputs(BaseTransform):

    def __call__(self, data: dict) -> dict:

        keys_to_delete = [k for k in data.keys() if k not in self.valid_keys]
        for k in keys_to_delete:
            del data[k]

        if self.do_normalize:
            data = self.normalize_state(data)
            data = self.normalize_action(data)

        return data


@dataclasses.dataclass
class FractalOutputs(BaseTransformOutputs):
    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"action": np.asarray(data["action"][:, :self.action_dim])}
