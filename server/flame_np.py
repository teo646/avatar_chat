"""
FLAME 2023 expression / pose evaluation in plain numpy, matching UIKA's UVFlameHead (uika/models/rendering/flame_model).

UIKA places one Gaussian per valid UV texel. A texel i sits in a mesh triangle with vertex ids v_ik and barycentric
weights b_ik (k = 0..2). Its position in the animated head is

    x_i(expr, pose) = T_i (c_i + sum_k b_ik d_k(expr)) + translation
    T_i             = sum_j wpix_ij A_j(expr, pose)          wpix_ij = sum_k b_ik lbs_weights[v_ik, j]
    d_k(expr)       = shapedirs_expr[v_k] @ expr             expression displacement of mesh vertex k
    A_j             = relative rigid transform of joint j    (root, neck, jaw, left eye, right eye)

c_i is the canonical position of the Gaussian (mean shape + the offset the network predicted). So the server only has to
produce d (V x 3) and A (5 x 4 x 4); the browser does the rest per Gaussian. The Gaussian's rotation and scale are not
changed by the animation in UIKA either.
"""
import numpy as np

N_JOINTS = 5  # root, neck, jaw, left eye, right eye


def rodrigues(rot_vecs: np.ndarray) -> np.ndarray:
    """(N, 3) axis-angle -> (N, 3, 3)"""
    angle = np.linalg.norm(rot_vecs + 1e-8, axis=1, keepdims=True)
    rx, ry, rz = np.split(rot_vecs / angle, 3, axis=1)
    zeros = np.zeros_like(rx)
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1).reshape(-1, 3, 3)
    cos, sin = np.cos(angle)[:, :, None], np.sin(angle)[:, :, None]
    return np.eye(3)[None] + sin * K + (1 - cos) * (K @ K)


class FlameNp:
    def __init__(self, npz_path: str):
        z = np.load(npz_path)
        self.template = z['template'].astype(np.float64)           # (V, 3)
        self.shapedirs_expr = z['shapedirs_expr'].astype(np.float64)  # (V, 3, n_expr)
        self.J_regressor = z['J_regressor'].astype(np.float64)     # (5, V)
        self.parents = z['parents'].astype(np.int64)               # (5,)
        self.lbs_weights = z['lbs_weights'].astype(np.float64)     # (V, 5)
        self.faces = z['faces'].astype(np.int64)                   # (F, 3)
        self.n_expr = self.shapedirs_expr.shape[2]

    def expression_offsets(self, expr: np.ndarray) -> np.ndarray:
        """(n_expr,) -> (V, 3) displacement of every mesh vertex"""
        return self.shapedirs_expr @ np.asarray(expr, dtype=np.float64)

    def joint_transforms(self, expr, rotation, neck, jaw, eyes):
        """Relative transforms A (5, 4, 4) and the rest joints J (5, 3).

        expr (n_expr,), rotation / neck / jaw (3,) axis-angle, eyes (6,) axis-angle of both eyes.
        The mean shape is used (betas = 0), like UIKA's decoder.
        """
        v_shaped = self.template + self.expression_offsets(expr)
        J = self.J_regressor @ v_shaped                              # (5, 3)
        pose = np.concatenate([rotation, neck, jaw, eyes]).reshape(-1, 3)
        rot = rodrigues(pose)                                        # (5, 3, 3)

        rel = J.copy()
        rel[1:] -= J[self.parents[1:]]
        T = np.zeros((N_JOINTS, 4, 4))
        T[:, :3, :3] = rot
        T[:, :3, 3] = rel
        T[:, 3, 3] = 1.0
        chain = [T[0]]
        for i in range(1, N_JOINTS):
            chain.append(chain[self.parents[i]] @ T[i])
        chain = np.stack(chain)                                      # (5, 4, 4) posed joint transforms
        Jh = np.concatenate([J, np.zeros((N_JOINTS, 1))], axis=1)    # (5, 4), homogeneous with w = 0 as in lbs.py
        A = chain.copy()
        A[:, :, 3] -= np.einsum('jab,jb->ja', chain, Jh)
        return A, J

    def animate_vertices(self, expr, rotation=(0, 0, 0), neck=(0, 0, 0), jaw=(0, 0, 0), eyes=(0,) * 6, translation=(0, 0, 0)):
        """Mesh vertices of the animated head (V, 3), for checks and for the previews."""
        d = self.expression_offsets(expr)
        A, _ = self.joint_transforms(expr, np.asarray(rotation, float), np.asarray(neck, float), np.asarray(jaw, float),
                                     np.asarray(eyes, float))
        T = np.einsum('vj,jab->vab', self.lbs_weights, A)
        x = self.template + d
        xh = np.concatenate([x, np.ones((len(x), 1))], axis=1)
        return np.einsum('vab,vb->va', T, xh)[:, :3] + np.asarray(translation)
