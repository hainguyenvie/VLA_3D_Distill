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

def reflect_quat(q_wxyz: np.ndarray, axis: int, front=None) -> np.ndarray:
    """`mirror_quat` for a reflection of world axis `axis` (0: x, through the yz-plane; 1: y, through the xz-plane):
    R' = M R S, S the reflection of one horizontal body axis, under which the object is taken to be symmetric. By default
    the body axis most aligned with the reflected world axis (objects facing the robot: mirrors their yaw); with `front`
    (a horizontal world direction the object faces, e.g. furniture facing the work area), the horizontal body axis most
    perpendicular to it, so that the mirrored object faces the mirrored direction."""
    import mujoco

    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q_wxyz, dtype=np.float64))
    R = R.reshape(3, 3)
    up_b = R.T @ np.array([0.0, 0.0, 1.0])
    k_up = int(np.argmax(np.abs(up_b)))
    if front is None:
        a_b = np.abs(R.T @ np.eye(3)[axis])
    else:
        f = np.array([front[0], front[1], 0.0]) / (np.linalg.norm(front[:2]) + 1e-9)
        a_b = 1.0 - np.abs(R.T @ f)
    a_b[k_up] = -1
    S = np.eye(3)
    S[int(np.argmax(a_b)), int(np.argmax(a_b))] = -1
    M = np.eye(3)
    M[axis, axis] = -1
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (M @ R @ S).reshape(-1))
    return q


def rotate_world(m, d, center, theta: float, free_addrs, arm_q1: int, fixed_bodies, qvel_too: bool = False) -> None:
    """Rotate a LIBERO world by `theta` about the vertical axis through `center` (x, y): free objects (qpos, and their
    world-frame linear velocity if `qvel_too`), the fixed bodies hanging off the world (furniture, tables: their model
    pose), and the arm by its first joint. Kinematics are not recomputed here."""
    import mujoco

    c, s = np.cos(theta), np.sin(theta)
    qz = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])

    def rot_xy(p):
        x, y = p[0] - center[0], p[1] - center[1]
        p[0], p[1] = center[0] + c * x - s * y, center[1] + s * x + c * y

    q = np.zeros(4)
    for a in free_addrs:  # free joints: x y z, then quaternion w x y z
        rot_xy(d.qpos[a : a + 3])
        mujoco.mju_mulQuat(q, qz, d.qpos[a + 3 : a + 7].copy())
        d.qpos[a + 3 : a + 7] = q
        if qvel_too:
            j = [k for k in range(m.njnt) if m.jnt_qposadr[k] == a][0]
            v = d.qvel[m.jnt_dofadr[j] : m.jnt_dofadr[j] + 2]
            v[0], v[1] = c * v[0] - s * v[1], s * v[0] + c * v[1]  # angular velocity of a free joint is body-local
    for b in fixed_bodies:
        rot_xy(m.body_pos[b])
        mujoco.mju_mulQuat(q, qz, m.body_quat[b].copy())
        m.body_quat[b] = q
    d.qpos[arm_q1] += theta


def world_fixed_bodies(robo) -> List[int]:
    """Bodies attached directly to the world that do not move on their own and are not the robot or the floor:
    furniture and tables. They are part of the scene a rotation counterfactual has to rotate."""
    m = robo.sim.model._model
    robot_root = robo.sim.model.body_name2id(robo.robots[0].robot_model.root_body)
    free_bodies = {int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j] == 0}
    out = []
    for b in range(1, m.nbody):
        name = robo.sim.model.body_id2name(b) or ""
        if m.body_parentid[b] == 0 and b != robot_root and b not in free_bodies and not name.startswith("floor"):
            out.append(b)
    return out


# Scene transforms of the ECT baseline (`EnvRunner.ect_replay`; Table 20 of arXiv 2609.39971: Spatial and Goal mirrors,
# Object the y-mirror and the x / xy mirrors with a shift, Long shifts). The paper does not give the shift sizes; these
# are ours. "identity" is the control of the replay controller.
ECT_TRANSFORMS = {
    "identity": {},
    "ymirror": {"my": 1},
    "xmirror_shift": {"mx": 1, "shift": (-0.08, 0.0)},
    "xymirror_shift": {"mx": 1, "my": 1, "shift": (-0.06, 0.06)},
    "shift": {"shift": (-0.08, 0.06)},
}

# Counterfactual worlds for training rollouts (`LiberoVecEnv(counterfactual=...)`, switched on per reset): at every
# rendered query the frames are also rendered with the target object displaced horizontally by a fresh random
# vector of this length range (metres); the simulator state itself is not changed.
COUNTERFACTUAL = dict(mode="shift", delta=(0.02, 0.08), theta=(0.1, 0.35), coshift=(0.08, 0.3), p_swap=0.5,
                      p_coshift=0.5, coshift_phase="pre", coshift_post=(0.08, 0.4), unique=False)
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
# With `coshift_phase` "post" or "both", queries after the grasp (target held) move the hand, the held target and the
# container (the place target) together by a vector of length in `coshift_post`, the other objects stay: the hand-
# container geometry is unchanged, so the nominal chunk is the correct action for carrying and placing. This does not
# move the object the instruction may describe by its relations before it is picked up.
# mode "relocate": after the grasp (target held), the container alone is moved to a free spot (render only; hand and held
# object stay), so the hand-to-goal relation changes; the label is the chunk of a privileged scripted placer that knows
# where the container is (src/rollout/scripted.py), returned as `cf_label` (env convention). The only counterfactual
# here whose correct action differs from a transformed nominal one.
# mode "retarget": before the grasp, the target alone is moved (to a free spot, or onto another object's spot with
# probability `p_swap`, that object taking the target's), the hand stays: the hand-to-target relation changes; the label
# is a privileged scripted approach to above the target (`cf_label`, valid for the first `cf_len` steps; the grasp is left
# to the policy). For suites that name the target (Object, Goal), not Spatial (which describes it by its relations).
# `unique`: no counterfactual moves an object (target, container, fixture) that has a twin of the same kind in the scene,
# since the instruction can then only refer to it by where it is (this turns retarget off on Spatial by itself).
# mode "coreloc": "coshift" before the grasp (hand + target moved, nominal label) and "relocate" while carrying (container
# moved, scripted-placer label `cf_label`).
# mode "mix": at every query one of "rotate" and "coshift" (probability `p_coshift` for coshift, which falls back to
# rotate after the grasp or when no placement is found); `cf_kind` says which (0 rotate, 1 coshift).
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
        self._geo_z0 = None

    def _cameras(self, enabled: bool) -> None:
        robo = self.env.env
        for name, ob in robo._observables.items():
            if name.endswith(("_image", "_depth")):
                robo.modify_observable(name, "enabled", enabled)
                if enabled:  # enabling zeroes the sampling timer; put it back in the phase upstream stepping has
                    ob._time_since_last_sample, ob._sampled = self._phase[name]

    def _geo_d(self) -> np.ndarray:
        """Privileged 3D displacement from the hand to the current sub-goal, as in 2606.27663 ("Direct Action-Head
        Injection of a Grounded 3D Point"): the target object (first object of interest) until it has risen 1 cm above
        its height at the start of the episode, then its place target from the task goal (BDDL `on` / `in`: a movable
        object's position or a region site's position; the last object of interest if the goal names none). NaN-free:
        zeros when the task has no movable target."""
        robo = self.env.env
        d = robo.sim.data._data
        names = getattr(robo, "obj_of_interest", None) or []
        if not names or names[0] not in robo.obj_body_id:
            return np.zeros(3, dtype=np.float32)
        tgt = names[0]
        p_t = d.xpos[robo.obj_body_id[tgt]].copy()
        if getattr(self, "_geo_z0", None) is None:
            self._geo_z0 = float(p_t[2])
            self._geo_place = None
            try:
                for pred in robo.parsed_problem["goal_state"]:
                    if len(pred) == 3 and str(pred[0]).lower() in ("on", "in") and pred[1] == tgt:
                        self._geo_place = pred[2]
                        break
            except Exception:
                pass
            if self._geo_place is None and len(names) > 1:
                self._geo_place = names[-1]
        goal = p_t
        if p_t[2] - self._geo_z0 > 0.01 and self._geo_place is not None:
            place = self._geo_place
            if place in robo.obj_body_id:
                goal = d.xpos[robo.obj_body_id[place]].copy()
            else:
                try:
                    goal = d.site_xpos[robo.sim.model.site_name2id(place)].copy()
                except Exception:
                    goal = p_t
        eef = d.site_xpos[robo.robots[0].eef_site_id]
        return (goal - eef).astype(np.float32)

    def _obs(self, obs) -> Dict[str, Any]:
        """Packed observation; in a perturbed episode `rgb` (and `depth`) show the perturbed view the student is fed,
        and `rgb_clean` keeps the nominal frame for the teacher."""
        out = _pack_obs(self.env, obs, self.cfg)
        out["geo_d"] = self._geo_d()
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
                # the privileged teacher's chunk in the factual world (zeros when the view has no scripted label)
                self._stash_cf.setdefault("cf_label_nom", np.zeros((50, 7), dtype=np.float32))
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

    def _tint_objects(self, robo, p: float) -> None:
        """Multiply the material (or geom) colour of each movable object, with probability `p` per object, by a random
        colour (per channel in [0.25, 1]), so that objects of the same kind are seen in several colours (LIBERO-PRO's
        object cell recolours the target). The model is rebuilt at every reset, so nothing needs undoing."""
        global _PERT_RNG
        if _PERT_RNG is None:
            _PERT_RNG = np.random.default_rng([os.getpid(), int.from_bytes(os.urandom(4), "little")])
        m = robo.sim.model._model
        for name, root in robo.obj_body_id.items():
            if _PERT_RNG.random() >= p:
                continue
            tint = _PERT_RNG.uniform(0.25, 1.0, 3)
            tint /= tint.max()  # keep the brightest channel: a hue shift more than a darkening
            mats = set()
            for g in range(m.ngeom):
                b = int(m.geom_bodyid[g])
                if not (b == root or self._is_descendant(m, b, root)):
                    continue
                if m.geom_matid[g] >= 0:
                    mats.add(int(m.geom_matid[g]))
                else:
                    m.geom_rgba[g, :3] *= tint
            for mid in mats:
                m.mat_rgba[mid, :3] *= tint

    def _displace_distractor(self, robo, dist: float) -> None:
        """Move one random movable object that is neither the target nor the container by about `dist` metres to a free
        spot of the object area (in the physics), so that it is the only object off its usual spot."""
        import mujoco

        global _PERT_RNG
        if _PERT_RNG is None:
            _PERT_RNG = np.random.default_rng([os.getpid(), int.from_bytes(os.urandom(4), "little")])
        self._coshift_setup(robo)
        self.cf = None
        if not self._cs_ok:
            return
        m, d = robo.sim.model._model, robo.sim.data._data
        cands = [n for n in self._cs_others if n != self._cs_container]
        if not cands:
            return
        name = cands[int(_PERT_RNG.integers(len(cands)))]
        a = self._cs_others[name]
        xy = d.qpos[a : a + 2].copy()
        others = {n: d.qpos[b : b + 2].copy() for n, b in list(self._cs_others.items()) + [(self._cs_name, self._cs_target)] if n != name}
        allxy = np.array([xy] + list(others.values()))
        lo_b, hi_b = allxy.min(0) - 0.05, allxy.max(0) + 0.05
        for _ in range(100):
            ang, mag = _PERT_RNG.uniform(0, 2 * np.pi), _PERT_RNG.uniform(0.8 * dist, 1.2 * dist)
            new = xy + mag * np.array([np.cos(ang), np.sin(ang)])
            if (new < lo_b).any() or (new > hi_b).any():
                continue
            if all(np.linalg.norm(new - o) >= self._cs_r[name] + self._cs_r.get(n, 0.04) + 0.01 for n, o in others.items()):
                d.qpos[a : a + 2] = new
                mujoco.mj_forward(m, d)
                self.displaced = name
                return

    def _coshift_setup(self, robo) -> None:
        """Addresses for the coshift counterfactual: target and other free objects, arm joints, horizontal radii."""
        m, d = robo.sim.model._model, robo.sim.data._data
        names = getattr(robo, "obj_of_interest", None) or []
        self._cs_ok = False
        if not names or names[0] not in robo.obj_body_id:
            return
        robot = robo.robots[0]
        jid = [robo.sim.model.joint_name2id(j) for j in robot.robot_joints]
        self._cs_arm_q = np.array([m.jnt_qposadr[j] for j in jid])
        self._cs_arm_v = np.array([m.jnt_dofadr[j] for j in jid])
        self._cs_lo, self._cs_hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
        self._cs_site = robot.eef_site_id
        self._cs_fingers = np.array(robot._ref_gripper_joint_pos_indexes)

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
        self._cs_z0 = float(d.qpos[self._cs_target + 2]) if self._cs_target is not None else 0.0
        # objects the task carries (multi-object tasks carry several): free objects of interest before the container
        self._cs_carry = {n: (qadr(n), float(d.qpos[qadr(n) + 2])) for n in names[:-1]
                          if n in robo.obj_body_id and qadr(n) is not None}
        self._cs_container = names[-1] if len(names) > 1 else None  # the place target is named last (Object, Spatial, Long)
        self._cs_others = {n: qadr(n) for n in robo.objects_dict if n != names[0] and n in robo.obj_body_id and qadr(n) is not None}
        self._cs_r = {n: radius(robo.obj_body_id[n]) for n in [names[0]] + list(self._cs_others)}
        from src.rollout.scripted import rim_height

        self._cs_ctop = (rim_height(m, d, robo.obj_body_id[self._cs_container], self._is_descendant)
                         if self._cs_container in self._cs_others else 0.0)
        # a place target that is a region of a fixture (top of the cabinet, stove burner, rack): its site, and the fixture's
        # root body, which a relocation moves as a whole
        self._cs_cfix = None
        if self._cs_container and self._cs_container not in self._cs_others:
            try:
                sid = robo.sim.model.site_name2id(self._cs_container)
            except Exception:
                sid = None
            if sid is not None:
                root = int(m.site_bodyid[sid])
                while m.body_parentid[root] != 0:
                    root = int(m.body_parentid[root])
                if root != robo.sim.model.body_name2id(robot.robot_model.root_body):
                    self._cs_cfix = (sid, root, radius(root))
        # goal-driven place targets (BDDL goal "on" / "in" predicates): for each object to carry, what has to be moved to
        # relocate its place target, and where on it to aim. Kinds: ("free", qpos addr, site or None, container name)
        # for a movable container (or a region on one), ("fixture", root body, site) for a region on furniture; regions
        # on the table itself cannot be relocated and are left out.
        self._cs_place = {}
        try:
            goal = robo.parsed_problem["goal_state"]
        except Exception:
            goal = []
        robot_root = robo.sim.model.body_name2id(robot.robot_model.root_body)
        free_roots = {robo.obj_body_id[n]: n for n in robo.obj_body_id if qadr(n) is not None}
        for pred in goal:
            if len(pred) != 3 or str(pred[0]).lower() not in ("on", "in") or pred[1] not in robo.obj_body_id or qadr(pred[1]) is None:
                continue
            tgt = pred[2]
            if tgt in robo.obj_body_id and qadr(tgt) is not None:
                self._cs_place[pred[1]] = ("free", qadr(tgt), None, tgt)
                continue
            try:
                sid = robo.sim.model.site_name2id(tgt)
            except Exception:
                continue
            body = int(m.site_bodyid[sid])
            root = body
            while m.body_parentid[root] != 0:
                root = int(m.body_parentid[root])
            owner = next((free_roots[b] for b in free_roots if b == body or self._is_descendant(m, body, b)), None)
            rname = robo.sim.model.body_id2name(root) or ""
            if owner is not None:
                self._cs_place[pred[1]] = ("free", qadr(owner), sid, owner)
            elif root != robot_root and not rname.endswith("table") and not rname.startswith("floor"):
                self._cs_place[pred[1]] = ("fixture", root, sid, rname, radius(root))
        self._cs_carry.update({n: (qadr(n), float(d.qpos[qadr(n) + 2])) for n in self._cs_place if n not in self._cs_carry})
        for n in self._cs_place:  # radii of carried objects and of movable containers (clearance checks)
            if n not in self._cs_r:
                self._cs_r[n] = radius(robo.obj_body_id[n])
        self._cs_rims = {}  # rim height of movable containers
        from src.rollout.scripted import rim_height as _rim

        for pl in self._cs_place.values():
            if pl[0] == "free" and pl[3] not in self._cs_rims:
                self._cs_rims[pl[3]] = _rim(m, d, robo.obj_body_id[pl[3]], self._is_descendant)
                self._cs_r.setdefault(pl[3], radius(robo.obj_body_id[pl[3]]))
        # furniture standing on the table (not the table itself): a moved container must keep clear of it
        self._cs_fixed = [d.xpos[b][:2].copy() for b in world_fixed_bodies(robo)
                          if not (robo.sim.model.body_id2name(b) or "").endswith("table")]
        # objects the instruction can only pick out by where they are, because the scene holds another of the same kind
        # (the two black bowls of LIBERO-Spatial, the left / right plates of LIBERO-Long): with `unique`, a counterfactual
        # never moves them (it would contradict the instruction)
        named = list(robo.objects_dict.items()) + list(getattr(robo, "fixtures_dict", {}).items())
        kinds = [type(o).__name__ for _, o in named]
        self._cs_ambiguous = {n for n, o in named if kinds.count(type(o).__name__) > 1}
        if self._cs_target is not None:
            self.cf, self._cs_ok = True, True

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
        if self.cfg["counterfactual"].get("mode") == "relocate":
            return self._relocate_view(robo, res, conv)
        if self.cfg["counterfactual"].get("mode") == "retarget":
            return self._retarget_view(robo, res, conv)
        if self.cfg["counterfactual"].get("mode") == "full":  # before the grasp retarget or coshift, while carrying relocate
            if self.closed:
                return dict(self._relocate_view(robo, res, conv), cf_len=np.int16(0), cf_target_name=np.array(""),
                            cf_kind=np.int8(2), cf_eef_pos=np.zeros(3, dtype=np.float32), cf_eef_quat=np.zeros(4, dtype=np.float32),
                            cf_post=True, target_pos_cf=_target_pos(robo))
            if _PERT_RNG.random() < self.cfg["counterfactual"].get("p_coshift", 0.5):
                out = self._coshift_view(robo, res, conv)
                return dict(out, cf_label=np.zeros((50, 7), dtype=np.float32), cf_len=np.int16(0), cf_kind=np.int8(1),
                            cf_post=False)
            return dict(self._retarget_view(robo, res, conv), cf_kind=np.int8(0), cf_post=False,
                        cf_eef_pos=np.zeros(3, dtype=np.float32), cf_eef_quat=np.zeros(4, dtype=np.float32),
                        target_pos_cf=_target_pos(robo))
        if self.cfg["counterfactual"].get("mode") == "rr":  # retarget before the grasp, relocate while carrying
            if not self.closed:
                return self._retarget_view(robo, res, conv)
            return dict(self._relocate_view(robo, res, conv), cf_len=np.int16(0), cf_target_name=np.array(""))
        if self.cfg["counterfactual"].get("mode") == "coreloc":  # coshift before the grasp, relocate while carrying
            robot = robo.robots[0]
            if not self.closed:
                out = self._coshift_view(robo, res, conv)
                out["cf_label"] = np.zeros((50, 7), dtype=np.float32)
            else:
                out = self._relocate_view(robo, res, conv)
                d = robo.sim.data._data
                out["cf_eef_pos"] = np.array(d.site_xpos[robot.eef_site_id], dtype=np.float32)  # the hand does not move
                w, x, y, z = d.xquat[robo.sim.model.body_name2id(robot.robot_model.eef_name)]
                out["cf_eef_quat"] = np.array([x, y, z, w], dtype=np.float32)
                out["cf_target_name"], out["cf_post"] = np.array(""), True
                out["target_pos_cf"] = _target_pos(robo)
            return out
        if self.cfg["counterfactual"].get("mode") == "mix":
            if self._cs_ok and _PERT_RNG.random() < self.cfg["counterfactual"].get("p_coshift", 0.5) and not self.closed:
                out = self._coshift_view(robo, res, conv)
                if out["cf_valid"]:
                    return dict(out, cf_theta=np.float32(0.0), cf_kind=np.int8(1))
            lo, hi = self.cfg["counterfactual"].get("theta", COUNTERFACTUAL["theta"])
            out = self._rotate_view(robo, res, conv, float(_PERT_RNG.choice([-1, 1]) * _PERT_RNG.uniform(lo, hi)))
            return dict(out, cf_valid=True, cf_target_name=np.array(""), cf_kind=np.int8(0))
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
        """Frames of the scene rotated by `theta` about the first joint's vertical axis (movable objects, furniture and
        tables, and the arm), and the proprio the robot has there; the simulator's and model's arrays are restored
        exactly afterwards."""
        import mujoco

        m, d = robo.sim.model._model, robo.sim.data._data
        saved = {k: getattr(d, k).copy() for k in self._KIN}
        fb = self._rot_fixed
        saved_model = (m.body_pos[fb].copy(), m.body_quat[fb].copy())
        c, s, ctr = np.cos(theta), np.sin(theta), self._rot_center
        out = {}
        try:
            rotate_world(m, d, ctr, theta, self._mirror_obj_addrs, self._mirror_arm_addrs[0], fb)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
            self._render_cf(robo, res, conv, out)
        finally:
            m.body_pos[fb], m.body_quat[fb] = saved_model
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

    def _coshift_place_post(self, d):
        """While carrying: shift for the place target of the held object, or None if nothing of the task is held or its
        place target cannot be moved / has no free spot. The place target comes from the task's goal (BDDL on / in) when
        known, else from the last object of interest; what is moved is stored in `_cs_spec`."""
        # held = the gripper is commanded closed, its fingers are not shut on nothing, and a lifted object of the task sits
        # between them (an object just released into the container is lifted and near the hand, but not held)
        fingers = float(np.abs(d.qpos[self._cs_fingers]).sum())
        if getattr(self, "grip_cmd", 1.0) <= 0 or fingers < 0.004:
            self._cs_why = "not held"
            return None
        held = [(n, a) for n, (a, z0) in self._cs_carry.items()
                if d.qpos[a + 2] - z0 > 0.02 and np.linalg.norm(d.qpos[a : a + 3] - d.site_xpos[self._cs_site]) < 0.08]
        if not held:
            self._cs_why = "not held"
            return None
        name, ta = held[0]
        self._cs_held = ta
        pl = getattr(self, "_cs_place", {}).get(name)
        cfix = getattr(self, "_cs_cfix", None)
        if pl is not None and pl[0] == "free":
            spec = ("free", pl[1], pl[2], pl[3])  # container qpos addr, aim site (or None), container name
        elif pl is not None:
            spec = ("fixture", pl[1], pl[2], pl[4])  # root body, aim site, radius
        elif cfix is not None:
            spec = ("fixture", cfix[1], cfix[0], cfix[2])
        elif self._cs_container in self._cs_others:
            spec = ("free", self._cs_others[self._cs_container], None, self._cs_container)
        else:
            self._cs_why = "no place target"
            return None
        if spec[0] == "free" and spec[1] == ta:
            return None
        if self._ambiguous(spec[3] if spec[0] == "free" else (pl[3] if pl is not None else "")):
            self._cs_why = "place target named by position"
            return None
        self._cs_spec = spec
        lo, hi = self.cfg["counterfactual"].get("coshift_post", COUNTERFACTUAL["coshift_post"])
        if hi <= 0:  # ablation: no move at all (the scripted label in the nominal world)
            self._cs_riders = []
            return np.zeros(2)
        if spec[0] == "free":
            ca, r_c = spec[1], self._cs_r.get(spec[3], 0.1)
            xy_c = d.qpos[ca : ca + 2].copy()
            riders = self._riding(d, ca, r_c, exclude=ta)
        else:
            r_c, xy_c, riders = spec[3], d.xpos[spec[1]][:2].copy(), []
        self._cs_riders = riders
        cont_addr = spec[1] if spec[0] == "free" else None
        others = {n: d.qpos[a : a + 2].copy() for n, a in list(self._cs_others.items()) + [(self._cs_name, self._cs_target)]
                  if a not in (ta, cont_addr) and a not in riders}
        allxy = np.array([xy_c, d.qpos[ta : ta + 2]] + list(others.values()))
        # containers sit at the edge of the object area: let them go a little beyond it, and accept a small overlap of
        # bounding circles (they are conservative) with the objects standing on the table
        box_lo, box_hi = allxy.min(0) - 0.15, allxy.max(0) + 0.15
        fixed = [fx for fx in self._cs_fixed if np.linalg.norm(fx - xy_c) > 1e-3]  # not the moved one
        for _ in range(100):
            ang, mag = _PERT_RNG.uniform(0, 2 * np.pi), _PERT_RNG.uniform(lo, hi)
            new = xy_c + mag * np.array([np.cos(ang), np.sin(ang)])
            if (new < box_lo).any() or (new > box_hi).any():
                continue
            if (all(np.linalg.norm(new - xy) >= 0.8 * (r_c + self._cs_r.get(n, 0.04)) for n, xy in others.items())
                    and all(np.linalg.norm(new - fx) > max(0.2, 0.8 * r_c) for fx in fixed)):
                return new - xy_c
        self._cs_why = "no free spot"
        return None

    def _ambiguous(self, name: str) -> bool:
        """With the `unique` option: the object (or the fixture a root body name belongs to) has a twin in the scene."""
        if not self.cfg["counterfactual"].get("unique") or not name:
            return False
        return any(name == n or name.startswith(n + "_") for n in getattr(self, "_cs_ambiguous", ()))

    def _riding(self, d, ca: int, r_c: float, exclude=None) -> List[int]:
        """qpos addresses of free objects lying in / on the movable container at `ca` (they move with it)."""
        out = []
        for n, a in list(self._cs_others.items()) + [(self._cs_name, self._cs_target)]:
            if a in (ca, exclude, getattr(self, "_cs_held", None)):
                continue
            if np.linalg.norm(d.qpos[a : a + 2] - d.qpos[ca : ca + 2]) < 0.8 * r_c and d.qpos[a + 2] > d.qpos[ca + 2] - 0.01:
                out.append(a)
        return out

    def _riding_container(self, d, exclude=None) -> List[int]:
        """qpos addresses of free objects already lying in / on the container (they move with it)."""
        if self._cs_container not in self._cs_others:
            return []
        return self._riding(d, self._cs_others[self._cs_container], self._cs_r[self._cs_container], exclude)

    def _retarget_view(self, robo, res, conv) -> Dict[str, Any]:
        """Frames with the target moved (hand unchanged) before the grasp, and the scripted approach chunk there."""
        global _PERT_RNG
        if _PERT_RNG is None:
            _PERT_RNG = np.random.default_rng([os.getpid(), int.from_bytes(os.urandom(4), "little")])
        import mujoco

        from src.rollout.scripted import approach_chunk

        m, d = robo.sim.model._model, robo.sim.data._data
        saved = {k: getattr(d, k).copy() for k in self._KIN}
        out, place = {}, None
        try:
            if not self.closed and self._cs_ok and not self._ambiguous(self._cs_name):
                place = self._coshift_place(d)
            if place is not None:
                shift, swap = place
                ta = self._cs_target
                mujoco.mj_kinematics(m, d)  # the same teacher in the factual world, to check it against the expert
                out["cf_label_nom"] = approach_chunk(d.site_xpos[self._cs_site].copy(), d.xpos[self._cs_body].copy())[0]
                if swap is not None:
                    a = self._cs_others[swap]
                    d.qpos[a : a + 2] = d.qpos[ta : ta + 2]
                d.qpos[ta : ta + 2] += shift
                mujoco.mj_kinematics(m, d)
                lab, n = approach_chunk(d.site_xpos[self._cs_site].copy(), d.xpos[self._cs_body].copy())
                out["cf_label"], out["cf_len"] = lab.astype(np.float32), np.int16(n)
            mujoco.mj_camlight(m, d)
            self._render_cf(robo, res, conv, out)
        finally:
            for k, v in saved.items():
                getattr(d, k)[:] = v
        out.setdefault("cf_label", np.zeros((50, 7), dtype=np.float32))
        out.setdefault("cf_len", np.int16(0))
        out["cf_delta"] = (place[0] if place is not None else np.zeros(2)).astype(np.float32)
        out["cf_valid"] = place is not None
        out["cf_target_name"] = np.array((place[1] or "") if place is not None else "")
        return out

    def _relocate_view(self, robo, res, conv) -> Dict[str, Any]:
        """Frames with the container moved to a free spot (hand and held target unchanged), and the scripted placer's
        chunk in that world; outside the carrying phase the nominal world is rendered and the query is invalid."""
        global _PERT_RNG
        if _PERT_RNG is None:
            _PERT_RNG = np.random.default_rng([os.getpid(), int.from_bytes(os.urandom(4), "little")])
        import mujoco

        m, d = robo.sim.model._model, robo.sim.data._data
        if not getattr(self, "_cs_ok", False):  # no movable target in this task: nominal frames, invalid
            out = {}
            self._render_cf(robo, res, conv, out)
            return dict(out, cf_label=np.zeros((50, 7), dtype=np.float32), cf_delta=np.zeros(2, dtype=np.float32),
                        cf_valid=False)
        saved = {k: getattr(d, k).copy() for k in self._KIN}
        moved_root, saved_root = None, None
        out, shift = {}, None
        try:
            if self.closed and self._cs_ok:
                shift = self._coshift_place_post(d)
            if shift is not None:
                spec = self._cs_spec
                mujoco.mj_kinematics(m, d)  # the same teacher in the factual world, to check it against the expert
                out["cf_label_nom"] = self._placer_label(d, spec)
                if spec[0] == "fixture":  # move the furniture as a whole (render only; the model pose is restored below)
                    moved_root, saved_root = spec[1], m.body_pos[spec[1]].copy()
                    m.body_pos[spec[1]][:2] += shift
                else:
                    for a in self._cs_riders + [spec[1]]:
                        d.qpos[a : a + 2] += shift
                mujoco.mj_kinematics(m, d)
                out["cf_label"] = self._placer_label(d, spec)
            mujoco.mj_camlight(m, d)
            self._render_cf(robo, res, conv, out)
        finally:
            if moved_root is not None:
                m.body_pos[moved_root] = saved_root
            for k, v in saved.items():
                getattr(d, k)[:] = v
        out.setdefault("cf_label", np.zeros((50, 7), dtype=np.float32))
        out["cf_delta"] = (shift if shift is not None else np.zeros(2)).astype(np.float32)
        out["cf_valid"] = shift is not None
        return out

    def _placer_label(self, d, spec) -> np.ndarray:
        """The scripted placer's chunk (env convention) for the held object and the place target `spec`, in the world
        the kinematic arrays of `d` describe."""
        from src.rollout.scripted import placer_chunk

        ta = self._cs_held
        eef = d.site_xpos[self._cs_site].copy()
        if spec[0] == "fixture":  # place onto the region (a flat site on the furniture)
            goal, top = d.site_xpos[spec[2]].copy(), 0.0
        else:  # into / onto a movable container, aiming at its region when the goal names one
            ca = spec[1]
            goal = d.qpos[ca : ca + 3].copy()
            if spec[2] is not None:
                goal[:2] = d.site_xpos[spec[2]][:2]
            top = self._cs_rims.get(spec[3], self._cs_ctop) if hasattr(self, "_cs_rims") else self._cs_ctop
        return placer_chunk(eef, d.qpos[ta : ta + 3].copy(), goal, top=top).astype(np.float32)

    def _coshift_view(self, robo, res, conv) -> Dict[str, Any]:
        """Frames of the world in which the target and the hand moved together (see COUNTERFACTUAL), and the
        proprio there; the simulator's arrays are restored exactly afterwards."""
        global _PERT_RNG
        if _PERT_RNG is None:
            _PERT_RNG = np.random.default_rng([os.getpid(), int.from_bytes(os.urandom(4), "little")])
        import mujoco

        m, d = robo.sim.model._model, robo.sim.data._data
        if not getattr(self, "_cs_ok", False):  # no movable target in this task: nominal frames, invalid
            out = {}
            self._render_cf(robo, res, conv, out)
            return dict(out, target_pos_cf=_target_pos(robo), cf_delta=np.zeros(2, dtype=np.float32), cf_valid=False,
                        cf_target_name=np.array(""), cf_post=bool(self.closed))
        saved = {k: getattr(d, k).copy() for k in self._KIN + self._COM}
        out = {}
        try:
            phase = self.cfg["counterfactual"].get("coshift_phase", "pre")
            post = bool(self.closed)
            if not post:
                place = self._coshift_place(d) if phase in ("pre", "both") and not self._ambiguous(self._cs_name) else None
            else:
                sh = self._coshift_place_post(d) if phase in ("post", "both") else None
                place = None if sh is None else (sh, None)
            shift, swap = place if place is not None else (np.zeros(2), None)
            ta = self._cs_held if (post and place is not None) else self._cs_target
            if swap is not None:
                a = self._cs_others[swap]
                d.qpos[a : a + 2] = d.qpos[ta : ta + 2]
            d.qpos[ta : ta + 2] += shift
            if post and place is not None:  # the container (and what already lies in it) moves with the hand and held target
                for a in self._riding_container(d, exclude=ta) + [self._cs_others[self._cs_container]]:
                    d.qpos[a : a + 2] += shift
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
        out["cf_post"] = bool(post)
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
            self.grip_cmd = float(a[-1])  # the gripper command being executed (the render hook reads it)
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
            self._geo_z0 = None
            if self.cfg.get("perturb") or self.cfg.get("counterfactual"):
                self._hook_camera()
            self.env.set_init_state(init_state)
            if self.cfg.get("obj_tint"):  # training: random colours on the movable objects (appearance variation)
                self._tint_objects(robo, float(self.cfg["obj_tint"]))
            if self.cfg.get("displace_distractor"):  # diagnostic: one non-target object starts somewhere unusual
                self._displace_distractor(robo, float(self.cfg["displace_distractor"]))
            if self.cfg.get("q1_offset"):  # diagnostic: the arm starts turned about its first joint, the scene does not
                import mujoco

                robot = robo.robots[0]
                robo.sim.data._data.qpos[robot._ref_joint_pos_indexes[0]] += float(self.cfg["q1_offset"])
                mujoco.mj_forward(robo.sim.model._model, robo.sim.data._data)
                robot.controller.update(force=True)
                robot.controller.reset_goal()
            # sampling-timer state of each camera at the start of a control step (identical at every step)
            self._phase = {name: (robo._observables[name]._time_since_last_sample, robo._observables[name]._sampled)
                           for name in cams}
            self._cameras(False)
        if isinstance(perturb, dict):
            self._start_perturbation(perturb)
        elif perturb and self.cfg.get("perturb"):
            self._start_perturbation()
        if counterfactual and self.cfg.get("counterfactual") and self.cfg["counterfactual"].get("mode") == "mix":
            self._coshift_setup(robo)  # sets self.cf only when the task has a movable target; rotate works regardless
        if counterfactual and self.cfg.get("counterfactual") and self.cfg["counterfactual"].get("mode") in ("mirror", "rotate", "mix"):
            robot = robo.robots[0]
            self._mirror_y0 = float(robo.sim.data._data.xpos[robo.sim.model.body_name2id(robot.robot_model.root_body)][1])
            j1 = robo.sim.model.joint_name2id(robot.robot_joints[0])
            self._rot_center = np.array(robo.sim.data._data.xanchor[j1][:2], dtype=np.float64)  # first joint's axis
            self._rot_fixed = world_fixed_bodies(robo)
            self._mirror_arm_addrs = [int(a) for a in robot._ref_joint_pos_indexes]
            self._mirror_obj_addrs = []
            for n in robo.objects_dict:
                try:
                    a = robo.sim.model.get_joint_qpos_addr(f"{n}_joint0")
                except Exception:  # fixed object
                    continue
                self._mirror_obj_addrs.append(int(a[0] if isinstance(a, (tuple, list, np.ndarray)) else a))
            self.cf = True
        elif counterfactual and self.cfg.get("counterfactual") and self.cfg["counterfactual"].get("mode") in ("coshift", "relocate", "coreloc", "retarget", "rr", "full"):
            self._coshift_setup(robo)
            self.cf = True  # tasks without a movable target (open a drawer, turn on the stove) still render: invalid queries
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

    def _ect_transform(self, robo, spec: Dict[str, Any]):
        """Apply an ECT scene transform to the freshly reset scene (free objects and the furniture standing on the table;
        the robot and the tables stay): reflect x about the objects' centre (`mx`), reflect y about the robot base
        (`my`), then translate by `shift`. Returns (map of world points, sign vector of the rotation commands)."""
        import mujoco

        m, d = robo.sim.model._model, robo.sim.data._data
        robot = robo.robots[0]
        y0 = float(d.xpos[robo.sim.model.body_name2id(robot.robot_model.root_body)][1])
        free = [int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j] == 0]
        xc = float(np.mean([d.qpos[a] for a in free])) if free else 0.0
        mx, my = bool(spec.get("mx")), bool(spec.get("my"))
        sh = np.array(list(spec.get("shift", (0.0, 0.0))) + [0.0])

        def point(p):
            p = np.array(p, dtype=np.float64).copy()
            if mx:
                p[0] = 2 * xc - p[0]
            if my:
                p[1] = 2 * y0 - p[1]
            return p + sh

        def orient(q, front=None):
            if mx and my:  # two reflections = a half turn about the vertical axis
                out = np.zeros(4)
                mujoco.mju_mulQuat(out, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(q, dtype=np.float64))
                return out
            if mx or my:
                return reflect_quat(q, 0 if mx else 1, front)
            return np.asarray(q, dtype=np.float64)

        centre = np.mean([d.qpos[a : a + 2] for a in free], axis=0) if free else np.zeros(2)
        for a in free:
            d.qpos[a : a + 3] = point(d.qpos[a : a + 3])
            d.qpos[a + 3 : a + 7] = orient(d.qpos[a + 3 : a + 7])
        for b in world_fixed_bodies(robo):
            if (robo.sim.model.body_id2name(b) or "").endswith("table"):
                continue
            front = centre - m.body_pos[b][:2]  # furniture faces the work area (drawers, doors, burners)
            m.body_pos[b] = point(m.body_pos[b])
            m.body_quat[b] = orient(m.body_quat[b], front)
        mujoco.mj_forward(m, d)
        # rotation commands are world-frame axis-angle deltas, a pseudo-vector: a reflection M maps them to -M w
        rot = np.array([1.0, 1.0, 1.0])
        if mx and my:
            rot = np.array([-1.0, -1.0, 1.0])
        elif mx:
            rot = np.array([1.0, -1.0, -1.0])
        elif my:
            rot = np.array([-1.0, 1.0, -1.0])
        return point, rot

    def ect_replay(self, init_state, env_actions: np.ndarray, query_ts, spec: Dict[str, Any], hold: int = 40) -> Dict[str, Any]:
        """Counterpart of a successful episode as in ECT (Equivariant Counterfactual Training, arXiv 2609.39971):
        s' = M_s(s), a' = Replay_{s'}(M_a(a)). The episode's end-effector path is recovered by re-executing its actions
        from its initial state; the scene is then transformed (`_ect_transform`, robot and start pose unchanged) and a
        closed-loop tracking controller follows the transformed path (translation: the transformed command plus a
        correction of the position error through the kinematic model of `src/rollout/scripted.py`; rotation commands
        transformed analytically; gripper commands copied), holding the
        last waypoint for up to `hold` steps. Returns the frames and proprio at the episode's query steps `query_ts`, the
        executed commands (env convention) and whether the task succeeded in the transformed scene."""
        from src.rollout.scripted import TRACK_GAIN

        env_actions = np.asarray(env_actions, dtype=np.float64)
        self.reset(init_state)
        robo = self.env.env
        site = robo.robots[0].eef_site_id
        d = robo.sim.data._data
        path, n = [d.site_xpos[site].copy()], 0
        for a in env_actions:  # the original path, without rendering
            robo.step(a)
            path.append(d.site_xpos[site].copy())
            n += 1
            if robo._check_success():
                break
        if not robo._check_success():
            return dict(success=False, orig_success=False)
        first = self.reset(init_state)
        robo = self.env.env
        d = robo.sim.data._data
        point, rot = self._ect_transform(robo, spec)
        robo.robots[0].controller.update(force=True)
        robo.robots[0].controller.reset_goal()
        with self.lock:  # the first frame shows the transformed scene
            self._cameras(True)
            first = self._obs(robo._get_observations(force_update=True))
            self._cameras(False)
        targets = [point(p) for p in path]
        # translation commands map like vectors: the reflected axes change sign (a shift leaves them alone)
        mov = np.array([-1.0 if spec.get("mx") else 1.0, -1.0 if spec.get("my") else 1.0, 1.0])
        qset = {int(t) for t in query_ts}
        frames = {0: first} if 0 in qset else {}
        acts, success = [], False
        for t in range(n + hold):
            # the transformed command as feedforward, plus half of the position error to the transformed path at this
            # step (kinematic model); in the unchanged scene the error stays zero and the episode is reproduced exactly
            a = np.zeros(7)
            src = env_actions[min(t, n - 1)]
            ff = src[:3] * mov if t < n else 0.0
            a[:3] = np.clip(ff + 0.5 * (targets[min(t, n)] - d.site_xpos[site]) / (TRACK_GAIN * 0.05), -1.0, 1.0)
            a[3:6] = src[3:6] * rot if t < n else 0.0
            a[6] = src[6]
            if t + 1 in qset:  # rendered exactly as a query frame of the policy's own episodes
                frames[t + 1] = self._run([a], stop_at_end=False)
            else:
                robo.step(a)
            acts.append(a)
            if robo._check_success():
                success = True
                break
        keys = ("rgb", "wrist_rgb", "eef_pos", "eef_quat", "gripper_qpos")
        qs = sorted(t for t in frames)
        return dict(success=success, orig_success=True, n_orig=n, n_replay=len(acts), actions=np.array(acts, dtype=np.float32),
                    query_ts=np.array(qs), **{k: np.stack([frames[t][k] for t in qs]) for k in keys if k in frames[qs[0]]})

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
        self._geo_z0 = None
        robot = robo.robots[0]
        # PandaGripper.format_action accumulates [-1, 1] * 0.01 * sign(cmd) per substep and saturates within 4 steps
        robot.gripper.current_action = np.array([-1.0, 1.0]) * float(np.sign(gripper_cmd))
        # The arm controller cached the end-effector pose of the reset (home) configuration and would steer the
        # first control step from there; refresh it from the restored state and hold the current pose as goal.
        robot.controller.update(force=True)
        robot.controller.reset_goal()
        self.t, self.done, self.closed = t0, False, gripper_cmd > 0
        return dict(_pack_obs(self.env, obs, self.cfg), geo_d=self._geo_d(), t=t0, done=False, active=True,
                    closed_before=bool(self.closed))

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
            elif cmd == "ect":
                task_id, bddl, state, env_actions, query_ts, spec = arg
                if task_id != cur_task:
                    if runner is not None:
                        runner.close()
                    runner, cur_task = EnvRunner(bddl, cfg, lock), task_id
                conn.send(("ok", runner.ect_replay(state, env_actions, query_ts, spec)))
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
                 counterfactual: Optional[Dict[str, Any]] = None, q1_offset: float = 0.0, displace_distractor: float = 0.0,
                 obj_tint: float = 0.0):
        from libero.libero import benchmark, get_libero_path

        self.cfg = dict(suite=suite, max_steps=max_steps, num_steps_wait=num_steps_wait, resolution=resolution,
                        depth=depth, wrist=wrist, perturb=perturb,  # perturb: ranges as in VIEW_AUG, used per reset
                        counterfactual=counterfactual,  # as in COUNTERFACTUAL, used per reset
                        q1_offset=q1_offset,  # diagnostic: first arm joint turned by this much at every reset (radians)
                        displace_distractor=displace_distractor,  # diagnostic: one non-target object moved this far
                        obj_tint=obj_tint)  # training: probability that a movable object is recoloured in an episode
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

    def ect(self, i: int, task_id: int, trial_id: int, env_actions: np.ndarray, query_ts, spec: Dict[str, Any]) -> None:
        """Ask env i for the ECT counterpart (`EnvRunner.ect_replay`) of an episode of task `task_id` from benchmark initial
        state `trial_id` that executed `env_actions` (env convention)."""
        self.conns[i].send(("ect", (task_id, self._bddl(task_id), self.init_states(task_id)[trial_id], env_actions,
                                    list(query_ts), spec)))

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
