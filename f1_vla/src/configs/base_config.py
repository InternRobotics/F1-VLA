import dataclasses
from abc import abstractmethod

from collections.abc import Sequence

import f1_vla.src.utils.transforms as _transforms
from f1_vla.src.utils.transforms import ModelTransformGroup


@dataclasses.dataclass
class DataConfig:
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    camera_names: Sequence[str] = None
    state_keys: Sequence[str] = None
    action_keys: Sequence[str] = None
    predict_img_keys: Sequence[str] = None

    local_path: str | None = None
    weight: float = 1

    n_obs_img_steps: int = 12
    n_pred_img_steps: int = 3
    obs_img_stride: int = 3

    num_indices: int = 50

    image_size: tuple[int, int] = (224, 224)


@dataclasses.dataclass
class LeRobotDataConfig(DataConfig):
    """Base class for LeRobot dataset configurations with common initialization logic."""
    
    def initialize(
        self, 
        policy_config, 
        camera_keys: list[str], 
        state_keys: list[str],
        action_keys: list[str],
        predict_img_keys: list[str],
        image_size: dict,
        gen_obs_suffix: str,
        local_path: str, 
        ds_stats: dict,
        weight: int = 1,
        do_normalize: bool = True,
        norm_method: str | None = None,
        n_obs_img_steps: int | None = None, 
        n_pred_img_steps: int | None = None, 
        obs_img_stride: int | None = None,
        num_indices: int | None = None,
        gen_obs_transforms: list | None = None,
    ):
        # Store basic attributes
        self.local_path = local_path
        self.camera_keys = camera_keys
        self.state_keys = state_keys
        self.action_keys = action_keys
        self.predict_img_keys = predict_img_keys
        self.image_size = image_size
        self.suffix = gen_obs_suffix
        self.weight = weight

        # Set optional attributes if provided
        if n_obs_img_steps: self.n_obs_img_steps = n_obs_img_steps
        if n_pred_img_steps: self.n_pred_img_steps = n_pred_img_steps
        if obs_img_stride: self.obs_img_stride = obs_img_stride
        if num_indices: self.num_indices = num_indices
        if gen_obs_transforms: self.gen_obs_transforms = gen_obs_transforms

        # Create dataset-specific transforms (to be implemented by subclasses)
        self.data_transforms = self._create_data_transforms(
            policy_config=policy_config, 
            ds_stats=ds_stats, 
            do_normalize=do_normalize, 
            norm_method=norm_method
        )

        # Create common repack transforms
        self.repack_transforms = self._create_repack_transforms(
            camera_keys=camera_keys, 
            predict_img_keys=predict_img_keys, 
            gen_obs_suffix=gen_obs_suffix
        )

        # Create model transforms
        self.model_transforms = ModelTransformGroup(
            policy_config=policy_config, 
            n_obs_img_steps=self.n_obs_img_steps, 
            n_pred_img_steps=self.n_pred_img_steps, 
            obs_img_stride=self.obs_img_stride,
            gen_obs_transforms=self.gen_obs_transforms
        )

        return self

    @abstractmethod
    def _create_data_transforms(self, policy_config, ds_stats, do_normalize, norm_method):
        """Create dataset-specific data transforms. To be implemented by subclasses."""
        pass

    def _create_repack_transforms(self, camera_keys, predict_img_keys, gen_obs_suffix):
        """Create common repack transforms."""
        repack_images_keys = {f"observation.images.image{idx}": key for idx, key in enumerate(camera_keys)}
        return _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        **repack_images_keys,
                        # only support single predict_img_key
                        f"observation.images.image0_{gen_obs_suffix}": f"{predict_img_keys[0]}",
                        f"observation.images.image0_{gen_obs_suffix}_is_pad": f"{predict_img_keys[0]}_is_pad",
                        "observation.state": "observation.state",

                        "action": "action",
                        "action_is_pad": "action_is_pad",

                        "task": "task",
                    }
                ),
                _transforms.ResizeImages(
                    height=self.image_size.height, 
                    width=self.image_size.width, 
                    suffix=self.suffix
                ),
            ]
        )

    @abstractmethod
    def _apply_delta_actions(self):
        """Apply delta actions if needed. Can be overridden by subclasses."""
        pass
        
