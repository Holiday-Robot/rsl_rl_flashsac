"""Runnable check for the palm-frame orientation error used by the XHand repose observations.

``isaaclab.utils.math`` needs a running Isaac Sim, so the module boots a headless app first
(the Isaac Lab test convention). Run with the Isaac Lab venv:

    OMNI_KIT_ACCEPT_EULA=YES python -m pytest tests/test_xhand_repose.py
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest  # noqa: E402
import torch  # noqa: E402

from isaaclab_flashsac.mdp.obs.inhand import rotation_error_6d  # noqa: E402

IDENTITY_PALM = [1.0, 0.0, 0.0, 0.0]
TRAINING_PALM = [0.5, 0.5, -0.5, -0.5]  # (w, x, y, z), palm up with the fingers along world +x


@pytest.mark.parametrize("palm", [IDENTITY_PALM, TRAINING_PALM])
def test_rotation_error_6d_is_identity_when_aligned(palm: list[float]) -> None:
    quat = torch.tensor([[0.7071, 0.0, 0.7071, 0.0]])
    error = rotation_error_6d(quat, quat, torch.tensor([palm]))
    assert error.shape == (1, 6)
    assert torch.allclose(error, torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]), atol=1e-6)


def test_rotate_about_axis_goal_is_90deg_about_the_palm_normal() -> None:
    import gymnasium as gym
    from isaaclab.utils.math import axis_angle_from_quat, quat_apply_inverse, quat_conjugate, quat_mul
    from isaaclab_tasks.utils import parse_env_cfg

    import isaaclab_flashsac.envs  # noqa: F401

    task = "Isaac-Repose-Cube-XHand-Play-v0"
    env = gym.make(task, cfg=parse_env_cfg(task, device="cuda:0", num_envs=4))
    env.reset()
    base = env.unwrapped
    command = base.command_manager.get_term("object_pose")
    palm_quat = base.scene["robot"].data.root_quat_w
    object_quat = base.scene["object"].data.root_quat_w
    # goal * object^-1 as a rotation vector in the palm frame
    rotvec = quat_apply_inverse(
        palm_quat, axis_angle_from_quat(quat_mul(command.quat_command_w, quat_conjugate(object_quat)))
    )
    expected = torch.tensor([0.0, torch.pi / 2, 0.0], device=rotvec.device).expand_as(rotvec)
    env.close()
    assert torch.allclose(rotvec, expected, atol=1e-3)
