"""Score the tracked frames of UIKA's example motions for facial expressions (needs only numpy + the exported flame.npz)."""
import glob, pickle, sys
import numpy as np
sys.path.insert(0, '/home/ty/Codes/avatar_chat/server')
from flame_np import FlameNp

F = FlameNp('/home/ty/Codes/avatar_chat/web/assets/custom/flame.npz')
masks = pickle.load(open('/home/ty/Codes/UIKA/model_zoo/human_parametric_models/FLAME_masks.pkl', 'rb'), encoding='latin1')
T = F.template
lips = np.asarray(masks['lips'])
# mouth corners: the most lateral lip vertices on each side
lx = T[lips, 0]
corner_r = lips[np.argsort(lx)[:6]]
corner_l = lips[np.argsort(-lx)[:6]]
# brows: highest 30% of each eye region in y, split into inner / outer
def brow_sets(name):
    ids = np.asarray(masks[name])
    top = ids[T[ids, 1] > np.percentile(T[ids, 1], 70)]
    inner = top[np.abs(T[top, 0]) < np.percentile(np.abs(T[top, 0]), 40)]
    outer = top[np.abs(T[top, 0]) > np.percentile(np.abs(T[top, 0]), 60)]
    return top, inner, outer
tl, il, ol = brow_sets('left_eye_region'); tr, ir, orr = brow_sets('right_eye_region')
# eye opening: vertices near the eyeball, above / below its center
def eye_sets(eye, region):
    c = T[np.asarray(masks[eye])].mean(0)
    ids = np.asarray(masks[region])
    near = ids[np.linalg.norm(T[ids] - c, axis=1) < 0.022]
    return near[T[near, 1] > c[1]], near[T[near, 1] < c[1]]
ul, dl = eye_sets('left_eyeball', 'left_eye_region'); ur, dr = eye_sets('right_eyeball', 'right_eye_region')

def features(expr, jaw):
    d = F.expression_offsets(expr)
    f = {}
    f['corner_up'] = (d[corner_r, 1].mean() + d[corner_l, 1].mean()) / 2 * 1000       # mm
    f['mouth_w'] = (d[corner_l, 0].mean() - d[corner_r, 0].mean()) * 1000
    f['brow_up'] = (d[tl, 1].mean() + d[tr, 1].mean()) / 2 * 1000
    f['brow_inner_up'] = ((d[il, 1].mean() - d[ol, 1].mean()) + (d[ir, 1].mean() - d[orr, 1].mean())) / 2 * 1000
    f['brow_pinch'] = -((d[il, 0].mean()) - (d[ir, 0].mean())) * 1000                  # inner brows move together
    f['eye_open'] = ((d[ul, 1].mean() - d[dl, 1].mean()) + (d[ur, 1].mean() - d[dr, 1].mean())) / 2 * 1000
    f['jaw_deg'] = np.degrees(jaw[0])
    return f

rows = []
for seq in ('0000', '0001', '0002'):
    for p in sorted(glob.glob(f'/home/ty/Codes/UIKA/assets/motion/{seq}/flame_param/*.npz')):
        z = np.load(p)
        f = features(z['expr'].reshape(-1), z['jaw_pose'].reshape(-1))
        rows.append((seq, int(p.split('/')[-1][:5]), f))
names = list(rows[0][2].keys())
X = np.array([[r[2][n] for n in names] for r in rows])
if __name__ == '__main__':
    print('frames', len(rows))
    for i, n in enumerate(names):
        print(f'{n:14s} min {X[:, i].min():7.2f} p10 {np.percentile(X[:, i], 10):7.2f} median {np.median(X[:, i]):7.2f} p90 {np.percentile(X[:, i], 90):7.2f} max {X[:, i].max():7.2f}')
    for seq in ('0000', '0001', '0002'):
        sel = [k for k, r in enumerate(rows) if r[0] == seq]
        print(seq, 'mean', {n: round(float(X[sel, i].mean()), 2) for i, n in enumerate(names)})
        print(seq, 'std ', {n: round(float(X[sel, i].std()), 2) for i, n in enumerate(names)})
    np.save('/home/ty/Codes/avatar_chat/presets/motion_features.npy', X)
