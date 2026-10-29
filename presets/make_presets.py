"""
Emotion presets (FLAME 2023 expression + jaw) for the avatar_chat demo, written to emotion_presets.json.

Why this is not an average of tracked frames: the example motions shipped with UIKA (assets/motion/0000-0002, 456 frames)
contain talking and neutral faces only. Scored for mouth corners, brows, eye opening and jaw, the most extreme frames are
still within a few millimeters and a few degrees of neutral, and rendered on the avatar they are not recognizable as
happy / sad / angry / surprised (presets/cand_renders/sheet.png). So the presets are designed in FACS style instead:
for every emotion a displacement is set for a few facial landmarks (mouth corners, brows, eyelids, lips) in millimeters and
the 100 FLAME expression coefficients are fitted to it by ridge regression on the model's own expression blendshapes.
Replace the 'expr' / 'jaw' of a preset by the average of a tracked emotional sequence (NeRSemble EMO-1..4) when available:
the file format stays the same, 'source' tells where a preset came from.

    python make_presets.py            # writes emotion_presets.json and settings_check.json (for render_settings.py)
"""
import json
import pickle
import sys

import numpy as np

sys.path.insert(0, '/home/ty/Codes/avatar_chat/server')
from flame_np import FlameNp

ASSET = '/home/ty/Codes/avatar_chat/web/assets/custom/flame.npz'
OUT = '/home/ty/Codes/avatar_chat/server/emotion_presets.json'
F = FlameNp(ASSET)
masks = pickle.load(open('/home/ty/Codes/UIKA/model_zoo/human_parametric_models/FLAME_masks.pkl', 'rb'), encoding='latin1')
T = F.template
S = F.shapedirs_expr.reshape(-1, F.n_expr)  # (V*3, n_expr)

lips = np.asarray(masks['lips'])
corner_r = lips[np.argsort(T[lips, 0])[:6]]       # FLAME: +x is the subject's left, so "r" has negative x
corner_l = lips[np.argsort(-T[lips, 0])[:6]]
mid = lips[np.abs(T[lips, 0]) < 0.006]
upper_lip = mid[T[mid, 1] > np.median(T[mid, 1])]
lower_lip = mid[T[mid, 1] <= np.median(T[mid, 1])]


def brow_sets(name):
    ids = np.asarray(masks[name])
    top = ids[T[ids, 1] > np.percentile(T[ids, 1], 70)]
    inner = top[np.abs(T[top, 0]) < np.percentile(np.abs(T[top, 0]), 40)]
    outer = top[np.abs(T[top, 0]) > np.percentile(np.abs(T[top, 0]), 60)]
    return top, inner, outer


def eye_sets(eye, region):
    c = T[np.asarray(masks[eye])].mean(0)
    ids = np.asarray(masks[region])
    near = ids[np.linalg.norm(T[ids] - c, axis=1) < 0.022]
    return near[T[near, 1] > c[1]], near[T[near, 1] <= c[1]]


brow_top, brow_inner, brow_outer = [], [], []
for name in ('left_eye_region', 'right_eye_region'):
    t, i, o = brow_sets(name)
    brow_top.append(t), brow_inner.append(i), brow_outer.append(o)
brow_top, brow_inner, brow_outer = map(np.concatenate, (brow_top, brow_inner, brow_outer))
ul, dl = eye_sets('left_eyeball', 'left_eye_region')
ur, dr = eye_sets('right_eyeball', 'right_eye_region')
upper_lid, lower_lid = np.concatenate([ul, ur]), np.concatenate([dl, dr])


def field(spec, sigma=0.009):
    """spec: list of (vertex ids, (dx, dy, dz) in mm, flip_x). The displacement falls off with the distance to the ids.
    dx > 0 means 'towards the face center' (mirrored on both sides)."""
    t = np.zeros_like(T)
    for ids, d, mirror in spec:
        d = np.asarray(d, float) / 1000.0
        # distance of every vertex to the nearest id of the set
        dist = np.min(np.linalg.norm(T[:, None, :] - T[ids][None], axis=2), axis=1)
        w = np.exp(-0.5 * (dist / sigma) ** 2)
        vec = np.tile(d, (len(T), 1))
        if mirror:  # first component points to the face center
            vec[:, 0] = -np.sign(T[:, 0]) * d[0]
        t += w[:, None] * vec
    return t


# the lips must not pucker or protrude by accident: their depth (z) is held at the target with a high weight
WEIGHT = np.ones_like(T)
WEIGHT[lips, 2] = 8.0
WEIGHT[lips, :2] = 2.0


def fit(target, lam=0.01, limit=3.0):
    """weighted ridge regression of the expression coefficients on the target displacement of all vertices"""
    W = WEIGHT.reshape(-1)
    # later components are noisier and less natural: the penalty grows with the component index
    pen = 1.0 + (np.arange(S.shape[1]) / 8.0) ** 2
    A = S.T @ (S * W[:, None]) + lam * np.diag(pen) * (S ** 2).sum() / S.shape[1]
    e = np.linalg.solve(A, S.T @ (W * target.reshape(-1)))
    return np.clip(e, -limit, limit)


def report(name, e, target):
    d = F.shapedirs_expr @ e
    err = np.abs(d - target).max() * 1000
    print(f'{name:10s} |e|={np.linalg.norm(e):5.2f}  max coefficient {np.abs(e).max():4.2f}  max vertex error {err:4.1f} mm')


EMOTIONS = {
    'neutral': dict(spec=[], jaw_deg=0.0),
    'happy': dict(spec=[
        (np.concatenate([corner_r, corner_l]), (-7.0, 10.0, 0.0), True),    # corners up and out (negative: away from the face center)
        (lower_lid, (0.0, 4.0, 0.0), False),                                # cheeks push the lower lids up
        (brow_outer, (0.0, 1.0, 0.0), False),
    ], jaw_deg=5.0),
    'sad': dict(spec=[
        (np.concatenate([corner_r, corner_l]), (0.0, -7.0, 0.0), True),     # corners down
        (brow_inner, (2.0, 8.0, 0.0), True),                                # inner brows up and together
        (brow_outer, (0.0, -1.5, 0.0), False),
        (upper_lid, (0.0, -1.5, 0.0), False),                               # heavy lids
        (lower_lip, (0.0, 2.5, 0.0), False),                                # chin pushes the lower lip up
    ], jaw_deg=0.0),
    'angry': dict(spec=[
        (brow_inner, (5.0, -9.0, 0.0), True),                               # inner brows down and together
        (brow_outer, (0.0, -3.5, 0.0), False),
        (upper_lid, (0.0, -2.5, 0.0), False),                               # eyes narrow
        (lower_lid, (0.0, 1.5, 0.0), False),
        (upper_lip, (0.0, -1.5, 0.0), False),                               # lips pressed
        (lower_lip, (0.0, 1.5, 0.0), False),
        (np.concatenate([corner_r, corner_l]), (-1.0, -1.5, 0.0), True),
    ], jaw_deg=0.0),
    'surprised': dict(spec=[
        (brow_top, (0.0, 9.0, 0.0), False),                                 # brows up
        (upper_lid, (0.0, 3.5, 0.0), False),                                # eyes wide
        (lower_lid, (0.0, -1.0, 0.0), False),
        (np.concatenate([corner_r, corner_l]), (1.5, 0.5, 0.0), True),      # mouth rounds
    ], jaw_deg=14.0),
}


def main():
    presets, check = {}, []
    for name, cfg in EMOTIONS.items():
        target = field(cfg['spec']) if cfg['spec'] else np.zeros_like(T)
        e = fit(target) if cfg['spec'] else np.zeros(F.n_expr)
        report(name, e, target)
        presets[name] = {
            'expr': np.round(e, 4).tolist(),
            'jaw': [round(float(np.radians(cfg['jaw_deg'])), 4), 0.0, 0.0],
            'eyes': [0.0] * 6,
            'neck': [0.0, 0.0, 0.0],
            'source': 'synthetic FACS-style landmark targets fitted to the FLAME 2023 expression space (see make_presets.py)',
        }
        check.append({'name': name, 'expr': (0.8 * e).tolist(), 'jaw': [0.8 * np.radians(cfg['jaw_deg']), 0, 0]})
        check.append({'name': name + '_full', 'expr': e.tolist(), 'jaw': [np.radians(cfg['jaw_deg']), 0, 0]})

    # blink: the upper lids go down to the lower lids
    gap = np.zeros_like(T)
    open_amount = float(np.mean(T[upper_lid, 1]) - np.mean(T[lower_lid, 1]))
    blink_t = field([(upper_lid, (0.0, -0.92 * open_amount * 1000, 0.0), False), (lower_lid, (0.0, 0.08 * open_amount * 1000, 0.0), False)],
                    sigma=0.006)
    blink = fit(blink_t, lam=2e-3, limit=4.0)
    report('blink', blink, blink_t)
    check.append({'name': 'blink', 'expr': blink.tolist()})
    out = {'intensity_default': 0.8, 'n_expr': int(F.n_expr), 'presets': presets, 'blink': np.round(blink, 4).tolist(),
           'notes': 'jaw is an axis-angle (rad) rotation about x; eyes/neck are zero. blink is an expression vector, scale 0..1.'}
    json.dump(out, open(OUT, 'w'), indent=1)
    json.dump(check, open('/home/ty/Codes/avatar_chat/presets/settings_check.json', 'w'))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
