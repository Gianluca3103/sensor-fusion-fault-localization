import numpy as np
import torch

from PFS_Radar.pfs_radar_model import PFSRadarReliabilityModel, parameter_breakdown
from PFS_Radar.radar_data import pose_matrix, project_radar_bev
from PFS_Radar.train_pfs_radar import localization_surrogate_loss


def test_pose_matrix_uses_xyzw_quaternion_order():
    half_sqrt = np.sqrt(0.5)
    transform = pose_matrix(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([0.0, 0.0, half_sqrt, half_sqrt]),
    )
    point = transform @ np.asarray([1.0, 0.0, 0.0, 1.0])
    assert np.allclose(point[:3], [1.0, 3.0, 3.0])


def test_radar_projection_uses_expected_channels():
    points = np.asarray(
        [
            [10.0, 0.0, 0.0, -5.0, 10.0, 128.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 8.0, 10.0, 255.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    bev = project_radar_bev(points, np.eye(4), (0.0, 64.0), (-32.0, 32.0), 0.2)
    assert bev.shape == (4, 320, 320)
    assert bev[0].sum() == 1.0
    assert np.isclose(bev[1].max(), 1.0)
    assert np.isclose(bev[2].max(), 8.0 / 30.0)
    assert np.isclose(bev[3].max(), 1.0)


def test_pfs_radar_model_shapes_and_parameter_accounting():
    model = PFSRadarReliabilityModel(base_channels=4)
    model.eval()
    with torch.no_grad():
        output = model(
            torch.zeros(1, 3, 64, 64),
            torch.zeros(1, 4, 64, 64),
            return_features=True,
        )
    assert output["logits"].shape == (1, 1, 64, 64)
    assert output["pfs_reliability"].shape == (1, 1, 4, 4)
    breakdown = parameter_breakdown(model)
    assert breakdown["total"] == sum(value for key, value in breakdown.items() if key != "total")


def test_localization_loss_penalizes_broad_predictions():
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 7:9, 7:9] = 1.0
    correct = torch.full_like(target, -6.0)
    correct[:, :, 7:9, 7:9] = 6.0
    broad = torch.full_like(target, 2.0)

    correct_loss = localization_surrogate_loss(correct, target, radius_cells=1)
    broad_loss = localization_surrogate_loss(broad, target, radius_cells=1)

    assert broad_loss > correct_loss


def test_localization_loss_has_finite_gradients():
    logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 4:8, 4:8] = 1.0

    loss = localization_surrogate_loss(logits, target, radius_cells=1)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
