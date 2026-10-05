"""Gate for the oracle-3D pipeline: camera model and depth conventions must be self-consistent.

1. Synthetic: a tilted plane seen by two cameras (analytic depth) -> round trip and cross-view error ~ 0.
2. Simulator: for a LIBERO task, back-project the agent-view depth and reproject into the wrist view (and
   back); the depth disagreement on mutually visible pixels must be at the millimetre level.
    python scripts/check_geometry.py --suite libero_object --task 0 [--out outputs/week1/geometry_check]
"""
import argparse
import os

import numpy as np

from src.gaussian_teacher.geometry import backproject, camera_matrices, project, reprojection_error


def synthetic():
    K = np.array([[300.0, 0, 128], [0, 300.0, 128], [0, 0, 1]])
    Ta, Tb = np.eye(4), np.eye(4)
    Tb[:3, 3] = [0.2, 0, 0]
    u, v = np.meshgrid(np.arange(256) + 0.5, np.arange(256) + 0.5)  # pixel centres

    def plane_depth(T):  # plane z = 2 + 0.1 x, camera looking along +z from T's origin
        o = T[:3, 3]
        return (2 + 0.1 * o[0] - o[2]) / (1 - 0.1 * (u - 128) / 300)

    da, db = plane_depth(Ta), plane_depth(Tb)
    pts = backproject(da, K, Ta)
    uv, z = project(pts, K, Ta)
    assert np.abs(uv[..., 0] - u).max() < 1e-6 and np.abs(uv[..., 1] - v).max() < 1e-6 and np.abs(z - da).max() < 1e-9
    assert np.abs(pts[..., 2] - (2 + 0.1 * pts[..., 0])).max() < 1e-9
    med, mean, vis = reprojection_error(da, K, Ta, db, K, Tb)
    assert mean < 1e-4 and vis > 0.8, (med, mean, vis)
    print(f"synthetic ok: cross-view depth error {mean * 1000:.3f} mm on {vis:.0%} of pixels")


def simulator(suite_name, task_id, out):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import get_real_depth_map

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                             camera_heights=256, camera_widths=256, camera_depths=True)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(task_id)[0])
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    cams = {}
    for name in ("agentview", "robot0_eye_in_hand"):
        depth = get_real_depth_map(env.sim, obs[f"{name}_depth"])[..., 0][::-1]  # upright
        K, T = camera_matrices(env.sim, name, 256, 256)
        cams[name] = (depth, K, T, obs[f"{name}_image"][::-1])
        print(f"{name}: depth range {depth.min():.3f}..{depth.max():.3f} m")
    a, w = cams["agentview"], cams["robot0_eye_in_hand"]
    ok = True
    for src, dst, tag in ((a, w, "agent->wrist"), (w, a, "wrist->agent")):
        med, mean, vis = reprojection_error(src[0], src[1], src[2], dst[0], dst[1], dst[2])
        print(f"{tag}: depth error median {med * 1000:.2f} mm / mean {mean * 1000:.2f} mm on {vis:.1%} of source pixels")
        ok = ok and med < 3e-3
    # the end effector must project inside the agent view at a depth equal to the rendered depth there
    eef = obs["robot0_eef_pos"]
    uv, z = project(eef[None], a[1], a[2])
    print(f"eef projects to pixel {uv[0].round(1)} at {z[0]:.3f} m; rendered depth there {a[0][int(uv[0, 1]), int(uv[0, 0])]:.3f} m")
    if out:
        import imageio

        os.makedirs(out, exist_ok=True)
        for name, (depth, _, _, rgb) in cams.items():
            d = np.clip((depth - depth.min()) / (np.percentile(depth, 99) - depth.min() + 1e-6), 0, 1)
            imageio.imwrite(os.path.join(out, f"{name}_rgb_depth.png"),
                            np.concatenate([rgb, np.repeat((d * 255).astype(np.uint8)[..., None], 3, -1)], 1))
        mark = a[3].copy()
        ui, vi = int(uv[0, 0]), int(uv[0, 1])
        mark[max(vi - 3, 0) : vi + 4, max(ui - 3, 0) : ui + 4] = [255, 0, 0]
        imageio.imwrite(os.path.join(out, "agentview_eef_marker.png"), mark)
    env.close()
    assert ok, "cross-view depth error above 3 mm: camera model or depth convention is off"
    print("CHECK_GEOMETRY_OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    synthetic()
    simulator(args.suite, args.task, args.out)
