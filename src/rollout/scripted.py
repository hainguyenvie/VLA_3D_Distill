"""Privileged scripted controller for the carry-and-place phase of LIBERO pick-and-place tasks.

It knows where the held object and the place target (container) are and drives the hand so that the object ends up
above the container, lowers it and opens the gripper. Actions are in the env convention of LIBERO's OSC_POSE
controller (deltas in [-1, 1], 1 = 5 cm target offset; gripper -1 open, +1 close); orientation is held.
"""
import numpy as np


class ScriptedPlacer:
    def __init__(self, cruise: float = 0.15, release: float = 0.03, tol_xy: float = 0.02, gain: float = 0.6,
                 vmax: float = 0.9, top: float = 0.0):
        """`top`: height of the container's rim above its body origin. A flat container (plate, < 3 cm) is approached
        at `cruise` above its origin and the object lowered to `release`; one with walls (basket, bowl) is approached
        at `cruise` above its rim and the object dropped from 6 cm above the rim."""
        self.cruise, self.release, self.tol_xy, self.gain, self.vmax = cruise, release, tol_xy, gain, vmax
        self.walls = top >= 0.03
        self.top = top
        self.opened = 0

    def act(self, eef: np.ndarray, obj: np.ndarray, cont: np.ndarray) -> np.ndarray:
        """eef: hand site position; obj: held object's position; cont: container position (all world, metres)."""
        a = np.zeros(7)
        goal_xy = cont[:2] - (obj[:2] - eef[:2])  # where the hand must be for the object to be over the container
        e_xy = goal_xy - eef[:2]
        above = np.linalg.norm(e_xy) < self.tol_xy
        if self.opened:  # released: go up and stay open
            a[2], a[6] = self.vmax, -1.0
            self.opened += 1
            return a
        base = cont[2] + (self.top if self.walls else 0.0)  # rim of a container with walls, origin of a flat one
        drop = 0.06 if self.walls else self.release
        if not above:  # carry at cruise height above the container
            z_goal = eef[2] + (base + self.cruise - obj[2])
        else:  # lower the object onto / into the container
            z_goal = eef[2] + (base + drop - obj[2])
        a[:2] = np.clip(self.gain * e_xy / 0.05, -self.vmax, self.vmax)
        a[2] = np.clip(self.gain * (z_goal - eef[2]) / 0.05, -self.vmax, self.vmax)
        a[6] = 1.0
        if above and obj[2] - base < drop + 0.015:
            a[6], self.opened = -1.0, 1
        return a


# Tracking of LIBERO's OSC_POSE controller, fitted on logged pi0.5 rollouts (Spatial): hand displacement per control step
# ~ GAIN * action * 0.05 m per axis (x, y, z); residual std 0.9-3.5 cm over a 10-step chunk.
TRACK_GAIN = np.array([0.19, 0.25, 0.13])


def placer_chunk(eef: np.ndarray, obj: np.ndarray, cont: np.ndarray, horizon: int = 50, top: float = 0.0) -> np.ndarray:
    """The placer's next `horizon` actions from this state, rolled out on the fitted kinematic model (the held object
    moves with the hand until the gripper opens). `top`: container rim above its origin. Env convention, (horizon, 7)."""
    placer = ScriptedPlacer(top=top)
    eef, obj = np.array(eef, dtype=np.float64), np.array(obj, dtype=np.float64)
    out = np.zeros((horizon, 7))
    for k in range(horizon):
        a = placer.act(eef, obj, cont)
        out[k] = a
        step = TRACK_GAIN * a[:3] * 0.05
        eef = eef + step
        if not placer.opened:
            obj = obj + step
    return out


def rim_height(m, d, root: int, is_descendant) -> float:
    """Height of the highest collision-geom point of body `root`'s subtree above the body's origin (its rim)."""
    zmax = None
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if not (b == root or is_descendant(m, b, root)) or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
        z = (corners @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].max()
        zmax = z if zmax is None else max(zmax, z)
    return 0.0 if zmax is None else float(zmax - d.xpos[root][2])
