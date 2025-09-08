from dataclasses import dataclass

from f1_vla.src.configs.base_config import LeRobotDataConfig

import f1_vla.src.utils.transforms as _transforms

from f1_vla.src.processors.data_processors.ds_transforms.libero_transform import LiberoInputs, LiberoOutputs
from f1_vla.src.processors.data_processors.ds_transforms.bridge_transform import BridgeV2Inputs, BridgeV2Outputs
from f1_vla.src.processors.data_processors.ds_transforms.agibotworld_transform import AgiBotWorldInputs, AgiBotWorldOutputs
from f1_vla.src.processors.data_processors.ds_transforms.fractal_transform import FractalInputs, FractalOutputs


@dataclass
class LeRobotLiberoDataConfig(LeRobotDataConfig):

    def _create_data_transforms(self, policy_config, ds_stats, do_normalize, norm_method):
        """Create Libero-specific data transforms."""
        return _transforms.Group(
            inputs=[
                LiberoInputs(
                    action_dim=policy_config.max_action_dim, 
                    camera_keys=self.camera_keys,
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    norm_stats=ds_stats,
                    do_normalize=do_normalize,
                    norm_method=norm_method,
                ),
            ],
            outputs=[LiberoOutputs()],
        )


@dataclass
class LeRobotAgiBotWorldDataConfig(LeRobotDataConfig):

    def _create_data_transforms(self, policy_config, ds_stats, do_normalize, norm_method):
        """Create AgiBotWorld-specific data transforms."""
        return _transforms.Group(
            inputs=[
                AgiBotWorldInputs(
                    action_dim=policy_config.max_action_dim, 
                    camera_keys=self.camera_keys,
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    norm_stats=ds_stats,
                    do_normalize=do_normalize,
                    norm_method=norm_method,
                ),
            ],
            outputs=[AgiBotWorldOutputs()],
        )


@dataclass
class LeRobotBridgeV2DataConfig(LeRobotDataConfig):

    def _create_data_transforms(self, policy_config, ds_stats, do_normalize, norm_method):
        """Create BridgeV2-specific data transforms."""
        return _transforms.Group(
            inputs=[
                BridgeV2Inputs(
                    action_dim=policy_config.max_action_dim, 
                    camera_keys=self.camera_keys,
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    norm_stats=ds_stats,
                    do_normalize=do_normalize,
                    norm_method=norm_method,
                ),
            ],
            outputs=[BridgeV2Outputs()],
        )


@dataclass
class LeRobotFractalDataConfig(LeRobotDataConfig):

    def _create_data_transforms(self, policy_config, ds_stats, do_normalize, norm_method):
        """Create Fractal-specific data transforms."""
        return _transforms.Group(
            inputs=[
                FractalInputs(
                    action_dim=policy_config.max_action_dim, 
                    camera_keys=self.camera_keys,
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    norm_stats=ds_stats,
                    do_normalize=do_normalize,
                    norm_method=norm_method,
                ),
            ],
            outputs=[FractalOutputs()],
        )
