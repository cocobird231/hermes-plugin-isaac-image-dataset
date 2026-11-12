"""Explicit body/optical pose conversions; world is right-handed Z-up metres."""
import math

import numpy as np


def optical_rotation(yaw_deg, pitch_deg, roll_deg):
    """R_world_optical; body X-forward Y-left Z-up, optical X-right Y-down Z-forward."""
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    if not np.isfinite([yaw, pitch, roll]).all():
        raise ValueError("Finite angles required")
    cy, sy, cp, sp, cr, sr = (math.cos(yaw), math.sin(yaw),
                             math.cos(-pitch), math.sin(-pitch),
                             math.cos(roll), math.sin(roll))
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.]])
    ry = np.array([[cp, 0, sp], [0, 1., 0], [-sp, 0, cp]])
    rx = np.array([[1., 0, 0], [0, cr, -sr], [0, sr, cr]])
    body_from_optical = np.array([[0., 0, 1], [-1., 0, 0], [0, -1., 0]])
    return rz @ ry @ rx @ body_from_optical


def matrix_to_xyzw(rotation):
    """Stable proper-rotation conversion with canonical sign w >= 0."""
    r = np.asarray(rotation, dtype=float)
    if (r.shape != (3, 3) or not np.isfinite(r).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(r), 1., atol=1e-8)):
        raise ValueError("A finite proper rotation matrix is required")
    # Symmetric eigenvector method avoids singular trace/180-degree branches.
    k = np.array([
        [r[0, 0]-r[1, 1]-r[2, 2], r[0, 1]+r[1, 0], r[0, 2]+r[2, 0], r[2, 1]-r[1, 2]],
        [r[0, 1]+r[1, 0], r[1, 1]-r[0, 0]-r[2, 2], r[1, 2]+r[2, 1], r[0, 2]-r[2, 0]],
        [r[0, 2]+r[2, 0], r[1, 2]+r[2, 1], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]],
        [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)]]) / 3.
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    q /= np.linalg.norm(q)
    if q[3] < 0 or (q[3] == 0 and q[np.argmax(np.abs(q[:3]))] < 0):
        q *= -1
    return q


def pose_from_angles(position, yaw_deg, pitch_deg, roll_deg):
    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Finite XYZ position required")
    rotation = optical_rotation(yaw_deg, pitch_deg, roll_deg)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, position
    return np.r_[position, matrix_to_xyzw(rotation)].tolist(), transform.tolist()
