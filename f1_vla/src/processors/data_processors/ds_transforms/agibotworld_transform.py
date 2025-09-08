import dataclasses
import torch
import numpy as np

from f1_vla.src.processors.data_processors.ds_transforms.base_transform import (
    BaseTransform,
    BaseTransformOutputs
)


@dataclasses.dataclass
class AgiBotWorldInputs(BaseTransform):
    single_arm_dim: int = 7

    def __call__(self, data: dict) -> dict:

        keys_to_delete = [k for k in data.keys() if k not in self.valid_keys]
        for k in keys_to_delete:
            del data[k]

        if self.do_normalize:
            data = self.normalize_state(data)
            data = self.normalize_action(data)

        # reformulate the task
        data["task"] = data["task"].split("|")[0].strip()

        # reformulate state and action
        # state
        state_left_arm_gripper = torch.cat(
            [
                data["observation.states.joint.position"][:self.single_arm_dim],
                data["observation.states.effector.position"][0].unsqueeze(-1),
            ],
            dim=-1
        )
        state_right_arm_gripper = torch.cat(
            [
                data["observation.states.joint.position"][self.single_arm_dim:2*self.single_arm_dim],
                data["observation.states.effector.position"][1].unsqueeze(-1),
            ],
            dim=-1
        )
        state_arm_gripper = torch.cat(
            [
                state_left_arm_gripper,
                state_right_arm_gripper,
            ],
            dim=-1
        )
        # state_head = data["observation.states.head.position"]
        # state_waist = data["observation.states.waist.position"]
        state = torch.cat(
            [
                state_arm_gripper,
                # state_head, 
                # state_waist
            ], 
            dim=-1
        )
        assert state.shape[0] == 16, f"state shape is {state.shape[0]}"
        data["observation.state"] = state
        for k in self.state_keys:
            del data[k]

        # action
        action_left_arm_gripper = torch.cat(
            (
                data["actions.joint.position"][:, :self.single_arm_dim],
                data["actions.effector.position"][:, 0].unsqueeze(-1),
            ),
            dim=-1
        )
        action_right_arm_gripper = torch.cat(
            (
                data["actions.joint.position"][:, self.single_arm_dim:2*self.single_arm_dim],
                data["actions.effector.position"][:, 1].unsqueeze(-1),
            ),
            dim=-1
        )
        action_arm_gripper = torch.cat(
            [
                action_left_arm_gripper,
                action_right_arm_gripper,
            ],
            dim=-1
        )
        action = torch.cat(
            [
                action_arm_gripper,
                # data["actions.head.position"],
                # data["actions.waist.position"],
            ], 
            dim=-1
        )
        # assert action.shape[-1] == 8 * 2 + 2 + 2, f"action shape is {action.shape[-1]}"
        data["action"] = action
        data["action_is_pad"] = data["actions.joint.position_is_pad"]
        for k in self.action_keys:
            del data[k]
            del data[f"{k}_is_pad"]    

        return data


@dataclasses.dataclass
class AgiBotWorldOutputs(BaseTransformOutputs):
    action_dim: int = 9

    def __call__(self, data: dict) -> dict:
        return {"action": np.asarray(data["action"][:, :self.action_dim])}
