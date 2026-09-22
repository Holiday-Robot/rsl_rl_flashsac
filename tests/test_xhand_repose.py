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
