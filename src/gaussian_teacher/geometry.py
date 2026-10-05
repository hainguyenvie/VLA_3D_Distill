"""Oracle geometry from the simulator (privileged 3D, level A): cameras, back-projection, reprojection.

Frames. MuJoCo renders bottom-up, so `upright = raw[::-1]`. The policy is trained on `raw[::-1, ::-1]`
(a 180-degree rotation), i.e. the upright image mirrored left-right. All geometry here is done in the
upright frame (pixel u = column to the right, v = row downwards, OpenCV camera axes); helpers convert
dense maps to and from the policy frame so targets stay aligned with what the policy sees.
"""
import numpy as np


def camera_matrices(sim, camera_name: str, height: int, width: int):
    """Intrinsics K (3x3) and camera-to-world pose T_wc (4x4, x right / y down / z forward)."""
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

    return get_camera_intrinsic_matrix(sim, camera_name, height, width), get_camera_extrinsic_matrix(sim, camera_name)


def policy_to_upright(x: np.ndarray) -> np.ndarray:
    """Dense map (..., H, W[, C]) from the policy frame to the upright frame (and back: it is an involution)."""
    return x[..., ::-1, :] if x.ndim >= 3 and x.shape[-1] in (1, 3) else x[..., ::-1]


def backproject(depth: np.ndarray, K: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    """Upright metric depth (H, W) -> world points (H, W, 3)."""
    h, w = depth.shape
    # K puts the principal point at W/2 in pixel-edge coordinates, so pixel index i has its centre at i + 0.5
    u, v = np.meshgrid(np.arange(w, dtype=np.float64) + 0.5, np.arange(h, dtype=np.float64) + 0.5)
    x = (u - K[0, 2]) * depth / K[0, 0]
    y = (v - K[1, 2]) * depth / K[1, 1]
    cam = np.stack([x, y, depth, np.ones_like(depth)], axis=-1)
    return (cam @ T_wc.T)[..., :3]


def project(points: np.ndarray, K: np.ndarray, T_wc: np.ndarray):
    """World points (..., 3) -> upright pixel coordinates (..., 2) as (u, v) and camera depth (...).

    Coordinates are continuous with pixel centres at i + 0.5; the pixel index is floor(u), floor(v).
    """
    T_cw = np.linalg.inv(T_wc)
    cam = points @ T_cw[:3, :3].T + T_cw[:3, 3]
    z = cam[..., 2]
    uv = cam[..., :2] / z[..., None] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    return uv, z


def reprojection_error(depth_a, K_a, T_a, depth_b, K_b, T_b, tol: float = 0.02):
    """Back-project view A, project into view B and compare with B's (bilinearly sampled) depth.

    Returns (median, mean) absolute depth error in metres over the pixels of A that land inside B and are
    not occluded there, and the fraction of A's pixels that are. A correct camera model gives
    millimetre-level error; a wrong convention gives centimetres or an almost empty overlap.
    """
    pts = backproject(depth_a, K_a, T_a).reshape(-1, 3)
    uv, z = project(pts, K_b, T_b)
    h, w = depth_b.shape
    u, v = uv[:, 0] - 0.5, uv[:, 1] - 0.5  # continuous index coordinates for bilinear sampling
    inside = (z > 0) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    u, v, z = u[inside], v[inside], z[inside]
    u0, v0 = np.clip(np.floor(u).astype(int), 0, w - 2), np.clip(np.floor(v).astype(int), 0, h - 2)
    fu, fv = u - u0, v - v0
    d = (depth_b[v0, u0] * (1 - fu) * (1 - fv) + depth_b[v0, u0 + 1] * fu * (1 - fv)
         + depth_b[v0 + 1, u0] * (1 - fu) * fv + depth_b[v0 + 1, u0 + 1] * fu * fv)
    err = np.abs(d - z)
    visible = err < tol * z + tol  # larger error = the point is occluded in B
    if not visible.any():
        return float("nan"), float("nan"), 0.0
    return float(np.median(err[visible])), float(err[visible].mean()), float(visible.sum() / len(pts))
