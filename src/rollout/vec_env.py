"""Vectorised LIBERO environments, one env per worker process.

Every env follows the SimpleVLA-RL / OpenVLA-OFT episode protocol: fixed benchmark initial state,
`num_steps_wait` no-op steps, then open-loop execution of action chunks. Workers never import TF or CUDA.
Rendering goes through MUJOCO_GL (EGL on the GPU named by MUJOCO_EGL_DEVICE_ID, or OSMesa on the CPU) and
happens once per action chunk, at exactly the simulator instant at which upstream stepping renders.

One env per process is required: robosuite renders garbage when a process holds two EGL contexts.
"""
import contextlib
import multiprocessing as mp
import os
import re
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]

# Visual perturbations for training rollouts (`LiberoVecEnv(perturb=...)`, switched on per reset). One draw per
# episode: with probability p_clean nothing; otherwise each of camera / light / sensor is active with probability
# 0.6 (at least one). The camera orbits the point it looks at on the table and is re-aimed by a small jitter.
VIEW_AUG = dict(p_clean=0.25, azimuth=75.0, elevation=15.0, distance=(1.0, 2.0), aim=10.0,  # degrees / distance factor
                light=(0.3, 1.7), light_shift=1.0, tint=0.15,  # intensity factor, metres, per-channel fraction
                noise=0.08, blur=2.0)  # Gaussian noise std (of 1.0) and blur radius (pixels), upper bounds


_PERT_RNG = None


def mirror_quat(q_wxyz: np.ndarray) -> np.ndarray:
    """Orientation of a rigid object in the scene reflected through the world xz-plane (M = diag(1, -1, 1)).

    A reflection is not a rotation, so the object is rotated into the pose of its mirror image using one of its own
    symmetry planes S: R' = M R S, with S the reflection of the body axis that is neither the object's vertical axis
    the one most aligned with the world y axis (cans, bottles and boxes are symmetric under it). Keeps the
    object upright and mirrors its yaw."""
    import mujoco

    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q_wxyz, dtype=np.float64))
    R = R.reshape(3, 3)
    up_b = R.T @ np.array([0.0, 0.0, 1.0])
    k_up = int(np.argmax(np.abs(up_b)))
    y_b = np.abs(R.T @ np.array([0.0, 1.0, 0.0]))
    y_b[k_up] = -1
    S = np.eye(3)
    S[int(np.argmax(y_b)), int(np.argmax(y_b))] = -1
    Rm = np.diag([1.0, -1.0, 1.0]) @ R @ S
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Rm.reshape(-1))
    return q

# Counterfactual worlds for training rollouts (`LiberoVecEnv(counterfactual=...)`, switched on per reset): at every
# rendered query the frames are also rendered with the target object displaced horizontally by a fresh random
# vector of this length range (metres); the simulator state itself is not changed.
COUNTERFACTUAL = dict(mode="shift", delta=(0.02, 0.08), theta=(0.1, 0.35), coshift=(0.08, 0.3), p_swap=0.5)
# mode "mirror": the whole scene is reflected through the vertical plane of the robot base (object positions and
# orientations, arm joints q1 q3 q5 negated and q7 reflected about `c7`); the correct action is the nominal one
# reflected (dy, rx, rz negated), so the label is exact and the target sits at a different place.
# mode "rotate": every movable object is rotated about the vertical axis of the robot's first joint by a fresh random
# angle (|theta| in `theta`, radians, either sign) and so is the arm (q1 += theta); the camera, the table and the
# robot base stay. The correct action is the nominal one rotated by theta (dx dy and rx ry), so the label is exact and
# needs no symmetric scene, no renamed instruction and no knowledge of which object is the target.
# mode "coshift": the target object and the robot's hand are moved together by the same horizontal vector (the hand by
# inverse kinematics, keeping its orientation), to a free spot of the area the objects occupy or, with probability
# `p_swap`, onto another object's spot (that object takes the target's). The hand-target geometry is unchanged, so
# the nominal chunk is the correct action until it starts carrying the target (the training masks the rest); the
# other objects and the container stay, so the target is somewhere else relative to everything the policy could have
# memorised. Only pre-grasp queries are valid (`cf_valid`); needs the target's identity, nothing else.
# mode "swap": the target object and another movable object (not the container) exchange their horizontal positions;
# the counterfactual instruction names the object now standing where the target was (`cf_target_name`), so the
# correct action is unchanged and known exactly (it is the action of the nominal world).


def sample_perturbation(rng: np.random.Generator, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Concrete perturbation of one episode, or None for the nominal scene."""
    if rng.random() < cfg["p_clean"]:
        return None
    on = rng.random(3) < 0.6
    if not on.any():
        on[rng.integers(3)] = True
    out: Dict[str, Any] = {}
    if on[0]:
        out["camera"] = dict(azimuth=rng.uniform(-cfg["azimuth"], cfg["azimuth"]), elevation=rng.uniform(0, cfg["elevation"]),
                             distance=rng.uniform(*cfg["distance"]), aim=rng.uniform(-cfg["aim"], cfg["aim"], size=2))
    if on[1]:
        out["light"] = dict(diffuse=rng.uniform(*cfg["light"]), ambient=rng.uniform(*cfg["light"]),
                            tint=1 + rng.uniform(-cfg["tint"], cfg["tint"], size=3),
                            shift=rng.uniform(-cfg["light_shift"], cfg["light_shift"], size=3))
    if on[2]:
        out["sensor"] = dict(noise=rng.uniform(0, cfg["noise"]), blur=rng.uniform(0, cfg["blur"]), seed=int(rng.integers(2**31)))
    return out


def _axis_quat(axis, degrees: float) -> np.ndarray:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, dtype=np.float64), np.radians(degrees))
    return q


def perturbed_camera(pos: np.ndarray, quat: np.ndarray, plane_z: float, c: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Orbit a camera (world pose, MuJoCo convention: it looks along its -z) about the point where its optical axis
    meets the horizontal plane z = plane_z: azimuth about the vertical, elevation, distance factor; then re-aim it."""
    import mujoco

    def rot(q, v):
        out = np.zeros(3)
        mujoco.mju_rotVecQuat(out, np.asarray(v, dtype=np.float64), q)
        return out

    def mul(a, b):
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, a, b)
        return out

    pos, quat = np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)
    look = rot(quat, [0, 0, -1])
    t = (plane_z - pos[2]) / look[2] if look[2] < -1e-3 else 1.0
    pivot = pos + t * look
    q = _axis_quat([0, 0, 1], c["azimuth"])
    d = rot(q, pos - pivot)
    side = np.cross([0, 0, 1], d)
    side = side / max(np.linalg.norm(side), 1e-9)
    q = mul(_axis_quat(-side, c["elevation"]), q)  # positive elevation raises the camera
    new_pos = pivot + c["distance"] * rot(q, pos - pivot)
    new_quat = mul(q, quat)
    for axis, deg in zip(([1, 0, 0], [0, 1, 0]), c["aim"]):  # pitch / yaw about the camera's own axes
        new_quat = mul(new_quat, _axis_quat(axis, deg))
    return new_pos, new_quat / np.linalg.norm(new_quat)


def _target_pos(robo) -> np.ndarray:
    """Privileged world position of the task's target (first object of interest of the BDDL goal); NaN if none."""
    names = getattr(robo, "obj_of_interest", None) or []
    if not names or names[0] not in robo.obj_body_id:
        return np.full(3, np.nan, dtype=np.float32)
    return np.array(robo.sim.data.body_xpos[robo.obj_body_id[names[0]]], dtype=np.float32)


def _pack_obs(env, obs, cfg) -> Dict[str, Any]:
    out = {
        # rotate 180 degrees to match the training-data preprocessing
        "rgb": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "eef_pos": obs["robot0_eef_pos"].astype(np.float32),
        "eef_quat": obs["robot0_eef_quat"].astype(np.float32),
        "gripper_qpos": obs["robot0_gripper_qpos"].astype(np.float32),
        "sim_state": env.get_sim_state().astype(np.float64),
        "target_pos": _target_pos(env.env),
    }
    if cfg.get("wrist"):
        out["wrist_rgb"] = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    if cfg.get("depth"):
        from robosuite.utils.camera_utils import get_real_depth_map

        d = get_real_depth_map(env.sim, obs["agentview_depth"])[..., 0]  # metres
        out["depth"] = np.ascontiguousarray(d[::-1, ::-1]).astype(np.float32)
    return out


class EnvRunner:
    """One LIBERO env that renders only the frames the policy consumes.

    robosuite renders every camera at every control step, but the policy only looks at the observation
    returned by the last step of each action chunk. Camera observables are therefore disabled for all other
    steps (physics is unaffected) and enabled just for that step, with their sampling timer restored, so
    the frame is taken at the same physics substep as in upstream stepping (the 24th of 25, a consequence of
    the two forced observation updates in `reset` + `set_init_state`).
    The steps that render run under `lock`, which serialises GPU rendering across worker processes
    (`lock` is a no-op context for the CPU renderer, where workers render in parallel).
    """

    def __init__(self, bddl: str, cfg: Dict[str, Any], lock):
        from libero.libero.envs import OffScreenRenderEnv

        self.cfg, self.lock = cfg, lock
        with lock:
            self.env = OffScreenRenderEnv(
                bddl_file_name=bddl,
                camera_heights=cfg["resolution"],
                camera_widths=cfg["resolution"],
                camera_depths=bool(cfg.get("depth")),
                camera_names=["agentview"] + (["robot0_eye_in_hand"] if cfg.get("wrist") else []),
            )
            self.env.seed(0)  # upstream: the seed affects object positions even with a fixed initial state
        self.t, self.done, self.closed = 0, False, False
        self.pert, self.cf = None, None

    def _cameras(self, enabled: bool) -> None:
        robo = self.env.env
        for name, ob in robo._observables.items():
            if name.endswith(("_image", "_depth")):
                robo.modify_observable(name, "enabled", enabled)
                if enabled:  # enabling zeroes the sampling timer; put it back in the phase upstream stepping has
                    ob._time_since_last_sample, ob._sampled = self._phase[name]

    def _obs(self, obs) -> Dict[str, Any]:
        """Packed observation; in a perturbed episode `rgb` (and `depth`) show the perturbed view the student is fed,
        and `rgb_clean` keeps the nominal frame for the teacher."""
        out = _pack_obs(self.env, obs, self.cfg)
        if self.pert:
            out["rgb_clean"] = out["rgb"]
            out.update(self._stash)
        if self.cf:
            out.update(self._stash_cf)
        return out

    def _hook_camera(self) -> None:
        """Make the agent-view sensor also render the perturbed view, at the very simulator instant it samples (the
        sensor runs inside the control step, under the render lock held by the caller)."""
        robo = self.env.env
        sensor = robo._observables["agentview_image"]._sensor

        def both(obs_cache):
            img = sensor(obs_cache)
            if self.pert:
                self._stash = self._perturbed_view()
            if self.cf:
                self._stash_cf = self._counterfactual_view()
            return img

        both.__modality__ = sensor.__modality__
        robo.modify_observable("agentview_image", "sensor", both)

    def _start_perturbation(self, fixed: Optional[Dict[str, Any]] = None) -> None:
        """Draw this episode's perturbation (or take `fixed`) and precompute the camera / light parameters it puts
        into the model."""
        global _PERT_RNG
        if _PERT_RNG is None:  # one stream per worker process, across the envs it builds
            _PERT_RNG = np.random.default_rng([os.getpid(), int.from_bytes(os.urandom(4), "little")])
        self.pert = fixed if fixed is not None else sample_perturbation(_PERT_RNG, self.cfg["perturb"])
        if not self.pert:
            return
        robo = self.env.env
        m, d = robo.sim.model._model, robo.sim.data._data
        cid = robo.sim.model.camera_name2id("agentview")
        names = ("cam_pos", "cam_quat", "light_pos", "light_dir", "light_diffuse", "light_ambient")
        self._nominal = {k: getattr(m, k).copy() for k in names}
        self._nominal["head"] = (m.vis.headlight.diffuse.copy(), m.vis.headlight.ambient.copy())
        new = {k: v.copy() for k, v in self._nominal.items() if k != "head"}
        new["head"] = tuple(v.copy() for v in self._nominal["head"])
        if "camera" in self.pert:  # the camera hangs off the world body, so its model pose is its world pose
            tgt = _target_pos(robo)
            plane_z = float(tgt[2]) if np.isfinite(tgt).all() else float(d.cam_xpos[cid][2]) - 0.5
            new["cam_pos"][cid], new["cam_quat"][cid] = perturbed_camera(m.cam_pos[cid], m.cam_quat[cid], plane_z, self.pert["camera"])
        if "light" in self.pert:
            li = self.pert["light"]
            new["light_diffuse"] = np.clip(new["light_diffuse"] * li["diffuse"] * li["tint"], 0, 1)
            new["light_ambient"] = np.clip(new["light_ambient"] * li["ambient"], 0, 1)
            new["light_pos"] = new["light_pos"] + li["shift"]
            new["head"] = (np.clip(new["head"][0] * li["diffuse"] * li["tint"], 0, 1), np.clip(new["head"][1] * li["ambient"], 0, 1))
        self._perturbed = new

    def _set_visuals(self, values) -> None:
        import mujoco

        robo = self.env.env
        m, d = robo.sim.model._model, robo.sim.data._data
        for k, v in values.items():
            if k == "head":
                m.vis.headlight.diffuse[:], m.vis.headlight.ambient[:] = v
            else:
                getattr(m, k)[:] = v
        mujoco.mj_camlight(m, d)  # world poses of cameras and lights only; the physics state is not touched

    def _coshift_setup(self, robo) -> None:
        """Addresses for the coshift counterfactual: target and other free objects, arm joints, horizontal radii."""
        m, d = robo.sim.model._model, robo.sim.data._data
        names = getattr(robo, "obj_of_interest", None) or []
        if not names or names[0] not in robo.obj_body_id:
            return
        robot = robo.robots[0]
        jid = [robo.sim.model.joint_name2id(j) for j in robot.robot_joints]
        self._cs_arm_q = np.array([m.jnt_qposadr[j] for j in jid])
        self._cs_arm_v = np.array([m.jnt_dofadr[j] for j in jid])
        self._cs_lo, self._cs_hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
        self._cs_site = robot.eef_site_id

        def qadr(n):
            try:
                a = robo.sim.model.get_joint_qpos_addr(f"{n}_joint0")
            except Exception:  # fixed object
                return None
            return int(a[0] if isinstance(a, (tuple, list, np.ndarray)) else a)

        def radius(root):  # horizontal reach of the subtree's collision geoms around the root body's position
            r = 0.0
            for g in range(m.ngeom):
                b = int(m.geom_bodyid[g])
                if not (b == root or self._is_descendant(m, b, root)) or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                    continue
                c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
                w = corners @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                r = max(r, float(np.linalg.norm(w[:, :2] - d.xpos[root][:2], axis=1).max()))
            return r

        self._cs_name, self._cs_body = names[0], robo.obj_body_id[names[0]]
        self._cs_target = qadr(names[0])
        self._cs_container = names[1] if len(names) > 1 else None
        self._cs_others = {n: qadr(n) for n in robo.objects_dict if n != names[0] and n in robo.obj_body_id and qadr(n) is not None}
        self._cs_r = {n: radius(robo.obj_body_id[n]) for n in [names[0]] + list(self._cs_others)}
        if self._cs_target is not None:
            self.cf = True

    def _counterfactual_view(self) -> Dict[str, Any]:
        """Frames of the same instant with the target object moved horizontally by a fresh random vector.

        Only the rendered poses of the target's bodies, geoms and sites are shifted (and shifted back): after
        mj_step the kinematic arrays still describe the qpos of the last substep's start, so recomputing kinematics
        would also move the fingers and change the nominal frames; shifting the poses leaves everything else intact.
        """
        import robosuite.macros as macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

        global _PERT_RNG
        if _PERT_RNG is None:
            _PERT_RNG = np.random.default_rng([os.getpid(), int.from_bytes(os.urandom(4), "little")])
        robo, res = self.env.env, self.cfg["resolution"]
        conv = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        d = robo.sim.data._data
        if self.cfg["counterfactual"].get("mode") == "mirror":
            return self._mirror_view(robo, res, conv)
        if self.cfg["counterfactual"].get("mode") == "coshift":
            return self._coshift_view(robo, res, conv)
        if self.cfg["counterfactual"].get("mode") == "rotate":
            lo, hi = self.cfg["counterfactual"].get("theta", COUNTERFACTUAL["theta"])
            return self._rotate_view(robo, res, conv, float(_PERT_RNG.choice([-1, 1]) * _PERT_RNG.uniform(lo, hi)))
        moves = []  # (ids of a subtree, horizontal shift)
        out = {}
        if self.cfg["counterfactual"].get("mode", "shift") == "swap" and self._cf_others:
            name = self._cf_others[int(_PERT_RNG.integers(len(self._cf_others)))]
            other_ids, other_body = self._cf_other_ids[name]
            dxy = d.xpos[other_body][:2] - d.xpos[self._cf_root][:2]
            shift = np.array([dxy[0], dxy[1], 0.0])
            moves = [(self._cf_ids, shift), (other_ids, -shift)]
            out.update(cf_target_name=np.array(name), cf_source_name=np.array(self._cf_name))
        else:
            lo, hi = self.cfg["counterfactual"]["delta"]
            ang, mag = _PERT_RNG.uniform(0, 2 * np.pi), _PERT_RNG.uniform(lo, hi)
            shift = np.array([mag * np.cos(ang), mag * np.sin(ang), 0.0])
            moves = [(self._cf_ids, shift)]
        out["cf_delta"] = shift[:2].astype(np.float32)

        def apply(sign):
            for (bodies, geoms, sites), sh in moves:
                for arr, ids in ((d.xpos, bodies), (d.xipos, bodies), (d.geom_xpos, geoms), (d.site_xpos, sites)):
                    arr[ids] += sign * sh

        apply(1)
        try:
            for cam, key in (("agentview", "rgb_cf"), ("robot0_eye_in_hand", "wrist_rgb_cf")):
                if cam == "robot0_eye_in_hand" and not self.cfg.get("wrist"):
                    continue
                img = robo.sim.render(camera_name=cam, width=res, height=res, depth=False)
                out[key] = np.ascontiguousarray(img[::conv][::-1, ::-1])
        finally:
            apply(-1)
        out["target_pos_cf"] = _target_pos(robo) + shift.astype(np.float32)
        return out

    _KIN = ("qpos", "xpos", "xquat", "xmat", "xipos", "ximat", "geom_xpos", "geom_xmat", "site_xpos", "site_xmat",
            "cam_xpos", "cam_xmat", "light_xpos", "light_xdir", "xanchor", "xaxis")

    def _mirror_view(self, robo, res, conv) -> Dict[str, Any]:
        """Frames of the reflected scene at this instant, and the proprio the robot has there. The simulator's
        positions and kinematic arrays are restored exactly afterwards (no physics step happens in between)."""
        import mujoco

        m, d = robo.sim.model._model, robo.sim.data._data
        saved = {k: getattr(d, k).copy() for k in self._KIN}
        y0, c7 = self._mirror_y0, float(self.cfg["counterfactual"].get("c7", np.pi / 4))
        out = {}
        try:
            for a in self._mirror_obj_addrs:  # free joints: x y z, then quaternion w x y z
                d.qpos[a + 1] = 2 * y0 - d.qpos[a + 1]
                d.qpos[a + 3 : a + 7] = mirror_quat(d.qpos[a + 3 : a + 7])
            q = self._mirror_arm_addrs
            for i in (0, 2, 4):
                d.qpos[q[i]] *= -1
            d.qpos[q[6]] = 2 * c7 - d.qpos[q[6]]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
            for cam, key in (("agentview", "rgb_cf"), ("robot0_eye_in_hand", "wrist_rgb_cf")):
                if cam == "robot0_eye_in_hand" and not self.cfg.get("wrist"):
                    continue
                img = robo.sim.render(camera_name=cam, width=res, height=res, depth=False)
                out[key] = np.ascontiguousarray(img[::conv][::-1, ::-1])
            robot = robo.robots[0]
            out["cf_eef_pos"] = np.array(d.site_xpos[robot.eef_site_id], dtype=np.float32)
            w, x, y, z = d.xquat[robo.sim.model.body_name2id(robot.robot_model.eef_name)]
            out["cf_eef_quat"] = np.array([x, y, z, w], dtype=np.float32)  # xyzw, as robosuite reports it
        finally:
            for k, v in saved.items():
                getattr(d, k)[:] = v
        out["cf_delta"] = np.zeros(2, dtype=np.float32)
        out["target_pos_cf"] = _target_pos(robo) * np.array([1, -1, 1], dtype=np.float32) + np.array([0, 2 * y0, 0], dtype=np.float32)
        return out

    def _render_cf(self, robo, res, conv, out) -> None:
        """Render the (already posed) counterfactual world and read the end effector's pose there."""
        d = robo.sim.data._data
        for cam, key in (("agentview", "rgb_cf"), ("robot0_eye_in_hand", "wrist_rgb_cf")):
            if cam == "robot0_eye_in_hand" and not self.cfg.get("wrist"):
                continue
            img = robo.sim.render(camera_name=cam, width=res, height=res, depth=False)
            out[key] = np.ascontiguousarray(img[::conv][::-1, ::-1])
        robot = robo.robots[0]
        out["cf_eef_pos"] = np.array(d.site_xpos[robot.eef_site_id], dtype=np.float32)
        w, x, y, z = d.xquat[robo.sim.model.body_name2id(robot.robot_model.eef_name)]
        out["cf_eef_quat"] = np.array([x, y, z, w], dtype=np.float32)  # xyzw, as robosuite reports it

    def _rotate_view(self, robo, res, conv, theta: float) -> Dict[str, Any]:
        """Frames of the scene rotated by `theta` about the first joint's vertical axis (movable objects and arm), and
        the proprio the robot has there; the simulator's arrays are restored exactly afterwards."""
        import mujoco

        m, d = robo.sim.model._model, robo.sim.data._data
        saved = {k: getattr(d, k).copy() for k in self._KIN}
        c, s, ctr = np.cos(theta), np.sin(theta), self._rot_center
        qz = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])
        out = {}
        try:
            for a in self._mirror_obj_addrs:  # free joints: x y z, then quaternion w x y z
                x, y = d.qpos[a] - ctr[0], d.qpos[a + 1] - ctr[1]
                d.qpos[a], d.qpos[a + 1] = ctr[0] + c * x - s * y, ctr[1] + s * x + c * y
                q = np.zeros(4)
                mujoco.mju_mulQuat(q, qz, d.qpos[a + 3 : a + 7].copy())
                d.qpos[a + 3 : a + 7] = q
            d.qpos[self._mirror_arm_addrs[0]] += theta
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
            self._render_cf(robo, res, conv, out)
        finally:
            for k, v in saved.items():
                getattr(d, k)[:] = v
        tp = _target_pos(robo)
        x, y = tp[0] - ctr[0], tp[1] - ctr[1]
        out["target_pos_cf"] = np.array([ctr[0] + c * x - s * y, ctr[1] + s * x + c * y, tp[2]], dtype=np.float32)
        out["cf_delta"] = (out["target_pos_cf"] - tp)[:2].astype(np.float32)
        out["cf_theta"] = np.float32(theta)
        return out

    _COM = ("subtree_com", "cdof", "cinert")  # also written by mj_comPos (needed for Jacobians)

    def _ik_shift(self, m, d, dxy) -> float:
        """Move the arm so that the end-effector site is displaced by (dx, dy, 0) with its orientation unchanged
        (damped least squares on the arm joints, from the current configuration). Returns the remaining error
        (metres + radians)."""
        import mujoco

        sid, qa, va = self._cs_site, self._cs_arm_q, self._cs_arm_v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p_goal = d.site_xpos[sid] + np.array([dxy[0], dxy[1], 0.0])
        R_goal = d.site_xmat[sid].reshape(3, 3).copy()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        err = np.inf
        for _ in range(100):
            e_p = p_goal - d.site_xpos[sid]
            R_err = R_goal @ d.site_xmat[sid].reshape(3, 3).T  # world-frame rotation still to do
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, R_err.reshape(-1))
            e_r = np.zeros(3)
            mujoco.mju_quat2Vel(e_r, q, 1.0)
            err = float(np.linalg.norm(e_p) + np.linalg.norm(e_r))
            if np.linalg.norm(e_p) < 2e-5 and np.linalg.norm(e_r) < 2e-4:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, sid)
            J = np.vstack([jacp[:, va], jacr[:, va]])
            e = np.concatenate([e_p, e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            d.qpos[qa] = np.clip(d.qpos[qa] + dq, self._cs_lo, self._cs_hi)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
        return err

    def _coshift_place(self, d):
        """Draw where the target goes: (shift of the target, other object that takes its spot or None), or None."""
        cfg = self.cfg["counterfactual"]
        lo, hi = cfg.get("coshift", COUNTERFACTUAL["coshift"])
        ta = self._cs_target
        xy_t = d.qpos[ta : ta + 2].copy()
        others = {n: d.qpos[a : a + 2].copy() for n, a in self._cs_others.items()}
        allxy = np.array([xy_t] + list(others.values()))
        box_lo, box_hi = allxy.min(0) - 0.05, allxy.max(0) + 0.05  # the area the objects occupy now
        movable = [n for n in others if n != self._cs_container]
        for _ in range(40):
            if movable and _PERT_RNG.random() < cfg.get("p_swap", 0.0):
                y = movable[int(_PERT_RNG.integers(len(movable)))]
                new, swap = others[y], y
            else:
                ang, mag = _PERT_RNG.uniform(0, 2 * np.pi), _PERT_RNG.uniform(lo, hi)
                new, swap = xy_t + mag * np.array([np.cos(ang), np.sin(ang)]), None
            dist = np.linalg.norm(new - xy_t)
            if not (lo <= dist <= hi) or (new < box_lo).any() or (new > box_hi).any():
                continue
            clear = all(np.linalg.norm(new - (xy_t if n == swap else xy)) >= self._cs_r[self._cs_name] + self._cs_r[n] + 0.01
                        for n, xy in others.items() if n != swap)
            if clear:
                return new - xy_t, swap
        return None

    def _coshift_view(self, robo, res, conv) -> Dict[str, Any]:
        """Frames of the world in which the target and the hand moved together (see COUNTERFACTUAL), and the
        proprio there; the simulator's arrays are restored exactly afterwards."""
        global _PERT_RNG
        if _PERT_RNG is None:
            _PERT_RNG = np.random.default_rng([os.getpid(), int.from_bytes(os.urandom(4), "little")])
        import mujoco

        m, d = robo.sim.model._model, robo.sim.data._data
        saved = {k: getattr(d, k).copy() for k in self._KIN + self._COM}
        out = {}
        try:
            place = None if self.closed else self._coshift_place(d)  # pre-grasp only
            shift, swap = place if place is not None else (np.zeros(2), None)
            ta = self._cs_target
            if swap is not None:
                a = self._cs_others[swap]
                d.qpos[a : a + 2] = d.qpos[ta : ta + 2]
            d.qpos[ta : ta + 2] += shift
            err = self._ik_shift(m, d, shift)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
            self._render_cf(robo, res, conv, out)
            out["target_pos_cf"] = np.array(d.xpos[self._cs_body], dtype=np.float32)
        finally:
            for k, v in saved.items():
                getattr(d, k)[:] = v
        out["cf_delta"] = shift.astype(np.float32)
        out["cf_valid"] = bool(place is not None and err < 1e-3)
        out["cf_target_name"] = np.array(swap or "")  # the object that took the target's spot, if any
        return out

    @staticmethod
    def _is_descendant(m, b: int, root: int) -> bool:
        while b > 0:
            b = int(m.body_parentid[b])
            if b == root:
                return True
        return False

    def _perturbed_view(self) -> Dict[str, Any]:
        import robosuite.macros as macros
        from robosuite.utils.camera_utils import get_real_depth_map
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING  # the flip the camera sensor applies

        robo, res, depth = self.env.env, self.cfg["resolution"], bool(self.cfg.get("depth"))
        conv = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        self._set_visuals(self._perturbed)
        img = robo.sim.render(camera_name="agentview", width=res, height=res, depth=depth)
        self._set_visuals(self._nominal)
        out = {}
        if depth:
            img, dep = img
            dmap = get_real_depth_map(robo.sim, np.expand_dims(dep[::conv], axis=-1))[..., 0]
            out["depth"] = np.ascontiguousarray(dmap[::-1, ::-1]).astype(np.float32)
        rgb = np.ascontiguousarray(img[::conv][::-1, ::-1])
        if "sensor" in self.pert:
            se = self.pert["sensor"]
            if se["blur"] > 0.05:
                from PIL import Image, ImageFilter

                rgb = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(se["blur"])))
            noise = np.random.default_rng([se["seed"], self.t + 10**6]).normal(0, se["noise"] * 255, rgb.shape)
            rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        out["rgb"] = rgb
        return out

    def _run(self, actions, stop_at_end: bool = True) -> Dict[str, Any]:
        """Execute actions until the episode ends; only the last one renders."""
        obs, rendered = None, False
        for k, a in enumerate(actions):
            if k == len(actions) - 1:
                with self.lock:
                    self._cameras(True)
                    obs, _, self.done, _ = self.env.step(a)
                    self._cameras(False)
                rendered = True
            else:  # robosuite-level step: the LIBERO-Plus wrapper post-processes the frame, which is absent here
                obs, _, self.done, _ = self.env.env.step(a)
            self.t += 1
            self.closed = self.closed or a[-1] > 0  # a gripper-close command has been executed in this episode
            if stop_at_end and (self.done or self.t >= self.cfg["max_steps"]):
                break
        if not rendered:  # the episode ended mid-chunk: render its final state (only used for logging)
            with self.lock:
                self._cameras(True)
                obs = self.env.env._get_observations(force_update=True)
                self._cameras(False)
        return dict(self._obs(obs), t=self.t, done=bool(self.done), closed_before=bool(self.closed),
                    active=not (self.done or self.t >= self.cfg["max_steps"]))

    def reset(self, init_state, perturb=False, counterfactual: bool = False) -> Dict[str, Any]:
        """Start an episode. `perturb`: True draws this episode's visual perturbation from the configured ranges; a
        dict (as returned by `sample_perturbation`) applies exactly that one. `counterfactual`: also render every
        query with the target object displaced (see COUNTERFACTUAL)."""
        robo = self.env.env
        cams = [name for name in robo._observables if name.endswith(("_image", "_depth"))]
        with self.lock:  # a (hard) reset rebuilds the simulator and its render context
            # cameras must be enabled across reset + set_init_state, exactly as upstream: these two calls reset the
            # sampling timers and force two observation updates, which is what puts the timers in their phase
            for name in cams:
                robo.modify_observable(name, "enabled", True)
            self.env.reset()
            self.pert, self.cf = None, None
            if self.cfg.get("perturb") or self.cfg.get("counterfactual"):
                self._hook_camera()
            self.env.set_init_state(init_state)
            # sampling-timer state of each camera at the start of a control step (identical at every step)
            self._phase = {name: (robo._observables[name]._time_since_last_sample, robo._observables[name]._sampled)
                           for name in cams}
            self._cameras(False)
        if isinstance(perturb, dict):
            self._start_perturbation(perturb)
        elif perturb and self.cfg.get("perturb"):
            self._start_perturbation()
        if counterfactual and self.cfg.get("counterfactual") and self.cfg["counterfactual"].get("mode") in ("mirror", "rotate"):
            robot = robo.robots[0]
            self._mirror_y0 = float(robo.sim.data._data.xpos[robo.sim.model.body_name2id(robot.robot_model.root_body)][1])
            j1 = robo.sim.model.joint_name2id(robot.robot_joints[0])
            self._rot_center = np.array(robo.sim.data._data.xanchor[j1][:2], dtype=np.float64)  # first joint's axis
            self._mirror_arm_addrs = [int(a) for a in robot._ref_joint_pos_indexes]
            self._mirror_obj_addrs = []
            for n in robo.objects_dict:
                try:
                    a = robo.sim.model.get_joint_qpos_addr(f"{n}_joint0")
                except Exception:  # fixed object
                    continue
                self._mirror_obj_addrs.append(int(a[0] if isinstance(a, (tuple, list, np.ndarray)) else a))
            self.cf = True
        elif counterfactual and self.cfg.get("counterfactual") and self.cfg["counterfactual"].get("mode") == "coshift":
            self._coshift_setup(robo)
        elif counterfactual and self.cfg.get("counterfactual"):
            names = getattr(robo, "obj_of_interest", None) or []
            if names and names[0] in robo.obj_body_id:
                m = robo.sim.model._model

                def subtree(root):
                    bodies = [b for b in range(m.nbody) if b == root or self._is_descendant(m, b, root)]
                    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies]
                    sites = [x for x in range(m.nsite) if m.site_bodyid[x] in bodies]
                    return (np.array(bodies), np.array(geoms, dtype=int), np.array(sites, dtype=int))

                self._cf_name, self._cf_root = names[0], robo.obj_body_id[names[0]]
                self._cf_ids = subtree(self._cf_root)
                # swap partners: the other free objects of the scene, except the container of the task
                container = names[1] if len(names) > 1 else None
                self._cf_others, self._cf_other_ids = [], {}
                for n in robo.objects_dict:
                    if n in (names[0], container) or n not in robo.obj_body_id:
                        continue
                    try:
                        robo.sim.model.get_joint_qpos_addr(f"{n}_joint0")
                    except Exception:  # fixed object
                        continue
                    self._cf_others.append(n)
                    self._cf_other_ids[n] = (subtree(robo.obj_body_id[n]), robo.obj_body_id[n])
                self.cf = True
        self.t = -self.cfg["num_steps_wait"]
        out = self._run([DUMMY_ACTION] * self.cfg["num_steps_wait"], stop_at_end=False)  # upstream ignores `done` here
        self.t, self.done, self.closed = 0, False, False
        return dict(out, t=0, done=False, active=True, closed_before=False)

    def restore(self, sim_state, t0: int, gripper_cmd: float) -> Dict[str, Any]:
        """Continue from a logged mid-episode simulator state at step `t0` (no wait steps).

        The first frame is a forced render of exactly that state. Two pieces of robosuite state are not part of
        the simulator state and are rebuilt here: the gripper's integrated command target (a reset leaves it
        half-open; `gripper_cmd` is the last command executed before this state, -1 open, +1 close) and the arm
        controller's cached end-effector pose and goal.
        """
        robo = self.env.env
        cams = [name for name in robo._observables if name.endswith(("_image", "_depth"))]
        with self.lock:
            for name in cams:
                robo.modify_observable(name, "enabled", True)
            self.env.reset()
            obs = self.env.set_init_state(sim_state)
            self._phase = {name: (robo._observables[name]._time_since_last_sample, robo._observables[name]._sampled)
                           for name in cams}
            self._cameras(False)
        self.pert, self.cf = None, None
        robot = robo.robots[0]
        # PandaGripper.format_action accumulates [-1, 1] * 0.01 * sign(cmd) per substep and saturates within 4 steps
        robot.gripper.current_action = np.array([-1.0, 1.0]) * float(np.sign(gripper_cmd))
        # The arm controller cached the end-effector pose of the reset (home) configuration and would steer the
        # first control step from there; refresh it from the restored state and hold the current pose as goal.
        robot.controller.update(force=True)
        robot.controller.reset_goal()
        self.t, self.done, self.closed = t0, False, gripper_cmd > 0
        return dict(_pack_obs(self.env, obs, self.cfg), t=t0, done=False, active=True, closed_before=bool(self.closed))

    def step(self, actions) -> Dict[str, Any]:
        return self._run([a.tolist() for a in actions])  # (k, 7) actions, already gripper-post-processed

    def close(self) -> None:
        with self.lock:
            self.env.close()


def _worker(conn, cfg: Dict[str, Any], lock) -> None:
    import faulthandler

    faulthandler.enable()  # a crash inside MuJoCo / EGL otherwise kills the worker without any message
    try:
        runner, cur_task = None, None
        lock = lock if lock is not None else contextlib.nullcontext()
        while True:
            cmd, arg = conn.recv()
            if cmd in ("reset", "restore"):
                task_id, bddl, state, t0, gripper_cmd, perturb, cf = arg
                if task_id != cur_task:
                    if runner is not None:
                        runner.close()
                    runner, cur_task = EnvRunner(bddl, cfg, lock), task_id  # imports only the env stack, never torch
                conn.send(("ok", runner.reset(state, perturb, cf) if cmd == "reset" else runner.restore(state, t0, gripper_cmd)))
            elif cmd == "step":
                conn.send(("ok", runner.step(arg)))
            elif cmd == "close":
                conn.send(("ok", None))
                conn.close()
                os._exit(0)  # skip the slow env / EGL teardown
    except Exception:  # surface the traceback in the parent instead of dying silently
        conn.send(("error", traceback.format_exc()))


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return float("inf")


class LiberoVecEnv:
    def __init__(self, suite: str, num_envs: int, max_steps: int, num_steps_wait: int = 10, resolution: int = 256,
                 depth: bool = False, wrist: bool = False, perturb: Optional[Dict[str, Any]] = None,
                 counterfactual: Optional[Dict[str, Any]] = None):
        from libero.libero import benchmark, get_libero_path

        self.cfg = dict(suite=suite, max_steps=max_steps, num_steps_wait=num_steps_wait, resolution=resolution,
                        depth=depth, wrist=wrist, perturb=perturb,  # perturb: ranges as in VIEW_AUG, used per reset
                        counterfactual=counterfactual)  # as in COUNTERFACTUAL, used per reset
        self.suite = benchmark.get_benchmark_dict()[suite]()
        self._bddl_root, self._init_cache = get_libero_path("bddl_files"), {}
        self.num_envs = num_envs
        ctx = mp.get_context("spawn")
        # GPU rendering is serialised across workers: concurrent EGL rendering from several processes gets them
        # killed by the driver on the H200 machine (NVRM Xid 31). The CPU renderer needs no lock.
        lock = ctx.Lock() if os.environ.get("MUJOCO_GL", "egl") == "egl" else None
        self.lock = lock  # kept alive here: a spawned worker rebuilds the semaphore when it starts, which may be
        # after the constructor returned; if the parent had dropped it by then the worker dies with FileNotFoundError
        self.conns, self.procs = [], []
        for _ in range(num_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, self.cfg, lock), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)

    def reset(self, i: int, task_id: int, trial_id: int, perturb: bool = False, counterfactual: bool = False) -> None:
        """Ask env i to start an episode from benchmark initial state `trial_id` of task `task_id`."""
        self.conns[i].send(("reset", (task_id, self._bddl(task_id), self.init_states(task_id)[trial_id], 0, -1.0, perturb,
                                      counterfactual)))

    def restore(self, i: int, task_id: int, sim_state: np.ndarray, t0: int, gripper_cmd: float) -> None:
        """Ask env i to continue an episode of task `task_id` from a logged simulator state at step `t0`.

        `gripper_cmd` is the last gripper command executed before that state (-1 open, +1 close).
        """
        self.conns[i].send(("restore", (task_id, self._bddl(task_id), sim_state, t0, gripper_cmd, False, False)))

    def step(self, i: int, actions: np.ndarray) -> None:
        self.conns[i].send(("step", actions))

    def ready(self, i: int) -> bool:
        """True if env i has answered its pending request (so `recv` will not block)."""
        return self.conns[i].poll(0)

    def recv(self, i: int, timeout: float = 900.0) -> Any:
        if not self.conns[i].poll(timeout):
            raise TimeoutError(f"env {i} did not answer within {timeout}s")
        status, payload = self.conns[i].recv()
        if status != "ok":
            raise RuntimeError(f"env worker {i} failed:\n{payload}")
        return payload

    def _bddl(self, task_id: int) -> str:
        task = self.suite.get_task(task_id)
        return os.path.join(self._bddl_root, task.problem_folder, task.bddl_file)

    def init_states(self, task_id: int) -> np.ndarray:
        if task_id not in self._init_cache:
            self._init_cache[task_id] = np.asarray(self.suite.get_task_init_states(task_id))
        return self._init_cache[task_id]

    def task_language(self, task_id: int) -> str:
        """Instruction for the prompt.

        LIBERO-Plus derives `task.language` from the file name, so the perturbation id ("table 1", "view 0 0 100
        2 4 initstate 0", "light 3", ...) leaks into the instruction (LIBERO-plus issue #64). Those suffixes are
        stripped here; only the "language" perturbations keep their rewritten instruction.
        """
        task = self.suite.get_task(task_id)
        if "_language_" in task.name:
            return task.language
        base = re.sub(r"_(table|tb|add|light|level)_?\d+.*$|_view_.*$", "", task.name)
        return task.language if base == task.name else " ".join(base.split("_"))

    def task_info(self) -> List[Tuple[int, str, int]]:
        """(task id, instruction, number of benchmark initial states) for every task of the suite."""
        return [(i, self.task_language(i), len(self.init_states(i))) for i in range(self.suite.n_tasks)]

    def close(self) -> None:
        for conn, p in zip(self.conns, self.procs):
            try:
                conn.send(("close", None))
                if conn.poll(20):
                    conn.recv()
            except Exception:
                pass
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def postprocess_actions(actions: np.ndarray) -> np.ndarray:
    """Policy gripper output in [0, 1] (1 = open) -> env convention {-1 = open, +1 = close}, binarised."""
    a = actions.copy()
    g = 2 * a[..., -1] - 1
    a[..., -1] = -np.sign(g)
    return a
