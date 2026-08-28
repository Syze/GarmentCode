"""Generate an A-pose -> custom-pose SMPL body-vertex animation sequence.

Output: a (N, 6890, 3) .npy in METRES, with frame 0 shifted so y_min=0 (matching
the apose.obj the garment is draped on). The sim loads this, scales x100 to cm,
and steps the body through it during simulation so the cloth tracks the pose.

Also holds the minimal SMPL forward (LBS) this needs -- betas + pose (72
axis-angle) -> 6890x3 verts -- as a standalone numpy implementation with no
smplx dependency, loading the classic SMPL .pkl (v_template, shapedirs,
posedirs, J_regressor, weights, kintree_table, f). That used to be a separate
`smpl_pose.py`; it had exactly one other caller and nothing else in the repo
imported it, so the two are one file.
"""
import argparse
import pickle

import numpy as np


# --------------------------------------------------------------------------- #
#  Minimal SMPL forward
# --------------------------------------------------------------------------- #
def load_smpl(pkl_path):
    # The classic SMPL pkl embeds chumpy arrays, so unpickling imports chumpy,
    # which fails on numpy>=1.24 where the deprecated builtin aliases
    # (np.bool, np.int, ...) were removed. Restore them first.
    for _name, _val in (('bool', bool), ('int', int), ('float', float),
                        ('complex', complex), ('object', object),
                        ('unicode', str), ('str', str)):
        if not hasattr(np, _name):
            setattr(np, _name, _val)
    d = pickle.load(open(pkl_path, 'rb'), encoding='latin1')
    Jr = d['J_regressor']
    Jr = Jr.toarray() if hasattr(Jr, 'toarray') else np.asarray(Jr)
    return {
        'v_template': np.asarray(d['v_template'], dtype=np.float64),
        'shapedirs': np.asarray(d['shapedirs'], dtype=np.float64),
        'posedirs': np.asarray(d['posedirs'], dtype=np.float64),
        'J_regressor': np.asarray(Jr, dtype=np.float64),
        'weights': np.asarray(d['weights'], dtype=np.float64),
        'kintree_table': np.asarray(d['kintree_table']),
        'f': np.asarray(d['f'], dtype=np.int64),
    }


def _rodrigues(r):
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _with_zeros(x):  # (3,4) -> (4,4)
    return np.vstack([x, [0, 0, 0, 1]])


def smpl_forward(model, betas, pose):
    """betas: (10,), pose: (72,) axis-angle. Returns (6890,3) verts."""
    betas = np.asarray(betas, dtype=np.float64).reshape(-1)
    pose = np.asarray(pose, dtype=np.float64).reshape(24, 3)

    v_shaped = model['v_template'] + model['shapedirs'].dot(betas)   # (6890,3)
    J = model['J_regressor'].dot(v_shaped)                           # (24,3)

    R = np.stack([_rodrigues(pose[i]) for i in range(24)])           # (24,3,3)
    pose_feature = (R[1:] - np.eye(3)).reshape(-1)                   # (207,)
    v_posed = v_shaped + model['posedirs'].dot(pose_feature)         # (6890,3)

    parent = {i: int(model['kintree_table'][0, i]) for i in range(1, 24)}
    G = np.zeros((24, 4, 4))
    G[0] = _with_zeros(np.hstack([R[0], J[0].reshape(3, 1)]))
    for i in range(1, 24):
        local = _with_zeros(np.hstack([R[i], (J[i] - J[parent[i]]).reshape(3, 1)]))
        G[i] = G[parent[i]] @ local
    # subtract rest-pose so T-pose maps to identity
    G_rest = np.zeros((24, 4, 4))
    for i in range(24):
        Jh = np.array([J[i, 0], J[i, 1], J[i, 2], 0.0])
        G_rest[i] = G[i].copy()
        G_rest[i][:, 3] -= G[i] @ Jh
    T = np.tensordot(model['weights'], G_rest, axes=[[1], [0]])      # (6890,4,4)
    vh = np.hstack([v_posed, np.ones((v_posed.shape[0], 1))])
    v = np.einsum('nij,nj->ni', T, vh)[:, :3]
    return v


def slerp_pose(pose_a, pose_b, t):
    """Per-joint interpolation in axis-angle space (linear; fine for our range)."""
    a = np.asarray(pose_a, dtype=np.float64)
    b = np.asarray(pose_b, dtype=np.float64)
    return (1 - t) * a + t * b


# --------------------------------------------------------------------------- #
#  The avatar and the two poses
# --------------------------------------------------------------------------- #
SMPL_FEMALE = "../swan-comfyui/Muse/models/smpl/SMPL_FEMALE.pkl"

BETAS = [0.523972, 0.645996, -1.499363, 2.569221, 0.525446,
         -0.765820, 1.078215, 0.520933, 0.280636, 0.583474]

THETA_A = np.zeros(72)
THETA_A[5] = 0.188496      # L_Hip Z  (leg spread)
THETA_A[8] = -0.188496     # R_Hip Z
THETA_A[50] = -0.785398    # L_Shoulder Z (arm 45 deg down)
THETA_A[53] = 0.785398     # R_Shoulder Z

THETA_CUSTOM = np.array([
    -0.005136, -0.004898, -0.144924, -0.221284, 0.297841, 0.344219,
    0.006012, -0.143297, 0.135718, -0.005167, 0.079620, 0.050350,
    0.225910, 0.051062, -0.021328, -0.041922, 0.063024, 0.073901,
    -0.246364, -0.091100, 0.082613, 0.0, 0.263409, 0.0,
    0.0, -0.154359, 0.0, 0.126302, -0.045083, 0.053375,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.243023, -0.005390, -0.049195, 0.085610, 0.136157, -0.226399,
    0.092618, -0.125227, 0.238328, 0.595916, 0.087281, -0.131291,
    0.155787, -0.123385, -1.360561, 0.212667, 0.074907, 1.079776,
    -0.183054, -0.364018, 0.059605, -0.125191, 0.333325, -0.036981,
    0.624357, 0.114115, -0.009279, 0.053802, -0.189524, 0.168932,
    0.166027, 0.039811, -0.293157, 0.029655, 0.017913, 0.236960])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=60, help='number of frames')
    ap.add_argument('--out', default='./assets/bodies/Hyperdrop/pose_anim_apose_to_custom.npy')
    # The sequence must be generated from the SAME betas as the .obj the garment
    # was draped on, or the cloth starts on one body and is animated onto
    # another. Defaults to the Hyperdrop avatar's betas above.
    ap.add_argument('--betas', type=float, nargs=10, default=BETAS,
                    help='10 SMPL shape betas (default: the Hyperdrop avatar)')
    args = ap.parse_args()
    betas = list(args.betas)

    m = load_smpl(SMPL_FEMALE)
    yshift = smpl_forward(m, betas, THETA_A)[:, 1].min()  # fixed shift -> frame0 y_min=0

    frames = []
    for t in np.linspace(0.0, 1.0, args.n):
        v = smpl_forward(m, betas, slerp_pose(THETA_A, THETA_CUSTOM, t))
        v[:, 1] -= yshift
        frames.append(v.astype(np.float32))
    seq = np.stack(frames)
    np.save(args.out, seq)
    print(f'wrote {seq.shape} -> {args.out}')


if __name__ == '__main__':
    main()
