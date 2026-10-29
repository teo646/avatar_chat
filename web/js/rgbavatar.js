// RGBAvatar (Li et al., CVPR 2025) Gaussian head in the browser: reduced Gaussian blendshapes bound to a FLAME mesh.
//
// Everything the browser needs comes from export_rgbavatar_web.py (RGBAvatar/): neither the MLP nor the FLAME model is sent.
// For a blend weight psi (K = 20) and the posed mesh vertices v, every frame the Gaussians are rebuilt exactly as RGBAvatar does it
// (model/gaussian.py, model/binding.py, cuda_utils/face_tbn.cu, cuda_utils/mesh_binding.cu):
//
//     blended   = G0 + sum_k psi_k dG_k                    xyz, rot, rgb are blended; scale and opacity are not
//     rot       = normalize(blended rot)                   activations come after the blending
//     rgb       = max(blended rgb, 0)
//     TBN       = [normalize(tangent) normalize(bitangent) normalize(normal)] of the Gaussian's triangle (from v and the UVs)
//     xyz_world = TBN xyz + sum_k bary_k v_k
//     quat      = quat(TBN) * rot
//
// An emotion is a pair (psi, v) measured from the person's own video. The face moves between emotions by interpolating both.
import * as THREE from '../vendor/three.module.js';
import { SplatAvatar, quatToCov4 } from './splat.js';

const smooth = (t) => { t = Math.min(1, Math.max(0, t)); return t * t * (3 - 2 * t); };

export async function loadBlendAvatar(baseUrl, { maxGaussians = 0 } = {}) {
  const [meta, buf] = await Promise.all([
    fetch(`${baseUrl}/meta.json`).then((r) => r.json()),
    fetch(`${baseUrl}/rgbavatar.bin`).then((r) => r.arrayBuffer()),
  ]);
  const a = {};
  for (const [name, info] of Object.entries(meta.arrays)) {
    const n = info.shape.reduce((p, q) => p * q, 1);
    a[name] = /u(int)?32|u4/.test(info.dtype) ? new Uint32Array(buf, info.offset, n) : new Float32Array(buf, info.offset, n);
  }
  const K = meta.n_blend;
  let N = meta.count;
  let keep = null;   // optional reduction for weak devices, as in splat.js
  if (maxGaussians > 0 && maxGaussians < N) {
    const imp = new Float32Array(N);
    for (let i = 0; i < N; i++) imp[i] = a.opacity[i] * a.scale[3 * i] * a.scale[3 * i + 1] * a.scale[3 * i + 2];
    keep = Uint32Array.from(Array.from({ length: N }, (_, i) => i).sort((p, q) => imp[q] - imp[p]).slice(0, maxGaussians).sort((p, q) => p - q));
    N = keep.length;
  }
  const pick = (arr, k) => {
    if (!keep) return arr;
    const out = new arr.constructor(N * k);
    for (let i = 0; i < N; i++) for (let c = 0; c < k; c++) out[i * k + c] = arr[keep[i] * k + c];
    return out;
  };
  const d = {
    N, K, meta, sizeBoost: keep ? Math.pow(meta.count / N, 1 / 3) * 0.9 : 1,
    xyz0: pick(a.xyz0, 3), rot0: pick(a.rot0, 4), rgb0: pick(a.rgb0, 3), scale: pick(a.scale, 3), opacity: pick(a.opacity, 1),
    dg: pick(a.dg, K * 10), faceId: pick(a.face_id, 1), bary: pick(a.bary, 3),
    faces: a.faces, faceUvs: a.face_uvs, psi: a.psi, verts: a.verts, xIn: a.x_in, vroot: a.vroot, w0: a.w0, joint: a.joint, posedirs: a.posedirs_neck,
    mlp: { w1: a.mlp_w1, b1: a.mlp_b1, w2: a.mlp_w2, b2: a.mlp_b2, w3: a.mlp_w3, b3: a.mlp_b3 },
  };
  return new BlendAvatar(d);
}

const FIELDS = ['x', 'verts', 'vroot', 'joint'];   // what a face state is made of, interpolated and blended together
const N_IN = 129, MLP_H = 128;   // MLP input (expr 100, neck 6, jaw 6, eyes 12, eyelids 2, translation 3) and hidden width

export class BlendAvatar extends SplatAvatar {
  constructor(d) {
    super({ N: d.N, meta: { n_vertices: 0 }, xyz: d.xyz0, bary: d.bary, vidx: new Uint32Array(0), wpix: new Float32Array(0),
            color: d.rgb0, opacity: d.opacity, scale: d.scale, quat: d.rot0, sizeBoost: d.sizeBoost });
    this.b = d;
    this.meta = d.meta;
    this.K = d.K;
    this.nFaces = d.faces.length / 3;
    this.tbn = new Float32Array(this.nFaces * 9);       // per triangle: tangent, bitangent, normal
    this.qFace = new Float32Array(this.nFaces * 4);     // quaternion (w, x, y, z) of that matrix
    this.quat = new Float32Array(this.N * 4);           // world rotation of every Gaussian (kept for the self test)
    this.act = new Int32Array(d.K); this.actW = new Float32Array(d.K);
    this.nVerts = d.meta.n_vertices;
    this.emotions = {};
    d.meta.emotions.forEach((name, e) => {
      this.emotions[name] = { psi: d.psi.slice(e * d.K, (e + 1) * d.K), x: d.xIn.slice(e * N_IN, (e + 1) * N_IN), joint: d.joint.slice(e * 3, e * 3 + 3), vroot: d.vroot.slice(e * this.nVerts * 3, (e + 1) * this.nVerts * 3), verts: d.verts.slice(e * this.nVerts * 3, (e + 1) * this.nVerts * 3) };
    });
    this.w0 = d.w0;   // skinning weight of the root joint per vertex (the head turn, see BlendDriver.turnHead)
    this.pd = d.posedirs; this.delta = new Float32Array(d.meta.n_vertices * 3);   // pose correctives of the neck rotation (9, V*3)
    this.m = d.mlp; this.h1 = new Float32Array(MLP_H); this.h2 = new Float32Array(MLP_H);
    this.mesh.matrixAutoUpdate = false;                  // the head sway rotates the whole mesh object about the neck
    this.setState(this.emotions.neutral.psi, this.emotions.neutral.verts);
  }

  /** The MLP of RGBAvatar (129 -> 128 -> 128 -> K, ReLU): blend weights psi for the inputs x (expression, neck, jaw, eyes, eyelids, translation). */
  mlp(x, out) {
    const { w1, b1, w2, b2, w3, b3 } = this.m, h1 = this.h1, h2 = this.h2, K = this.K;
    for (let i = 0; i < MLP_H; i++) { let a = b1[i]; const o = i * N_IN; for (let j = 0; j < N_IN; j++) a += w1[o + j] * x[j]; h1[i] = a > 0 ? a : 0; }
    for (let i = 0; i < MLP_H; i++) { let a = b2[i]; const o = i * MLP_H; for (let j = 0; j < MLP_H; j++) a += w2[o + j] * h1[j]; h2[i] = a > 0 ? a : 0; }
    for (let i = 0; i < K; i++) { let a = b3[i]; const o = i * MLP_H; for (let j = 0; j < MLP_H; j++) a += w3[o + j] * h2[j]; out[i] = a; }
    return out;
  }

  /** TBN and its quaternion for every triangle (RGBAvatar: compute_face_tbn, MatrixToQuaternion of mesh_binding.cu). */
  computeTbn(v) {
    const { faces, faceUvs } = this.b, tbn = this.tbn, qf = this.qFace;
    for (let f = 0; f < this.nFaces; f++) {
      const i0 = faces[3 * f] * 3, i1 = faces[3 * f + 1] * 3, i2 = faces[3 * f + 2] * 3;
      const e1x = v[i1] - v[i0], e1y = v[i1 + 1] - v[i0 + 1], e1z = v[i1 + 2] - v[i0 + 2];
      const e2x = v[i2] - v[i0], e2y = v[i2 + 1] - v[i0 + 1], e2z = v[i2 + 2] - v[i0 + 2];
      const u = f * 6;
      const d1u = faceUvs[u + 2] - faceUvs[u], d1v = faceUvs[u + 3] - faceUvs[u + 1];
      const d2u = faceUvs[u + 4] - faceUvs[u], d2v = faceUvs[u + 5] - faceUvs[u + 1];
      const k = 1 / (d1u * d2v - d1v * d2u);
      let tx = (e1x * d2v - e2x * d1v) * k, ty = (e1y * d2v - e2y * d1v) * k, tz = (e1z * d2v - e2z * d1v) * k;
      let bx = (e2x * d1u - e1x * d2u) * k, by = (e2y * d1u - e1y * d2u) * k, bz = (e2z * d1u - e1z * d2u) * k;
      let nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
      let l = Math.hypot(tx, ty, tz); tx /= l; ty /= l; tz /= l;
      l = Math.hypot(bx, by, bz); bx /= l; by /= l; bz /= l;
      l = Math.hypot(nx, ny, nz); nx /= l; ny /= l; nz /= l;
      if (!(Number.isFinite(tx + ty + tz + bx + by + bz + nx + ny + nz))) { tx = by = nz = 1; ty = tz = bx = bz = nx = ny = 0; }   // degenerate triangle, unused
      const o = 9 * f;
      tbn[o] = tx; tbn[o + 1] = ty; tbn[o + 2] = tz; tbn[o + 3] = bx; tbn[o + 4] = by; tbn[o + 5] = bz; tbn[o + 6] = nx; tbn[o + 7] = ny; tbn[o + 8] = nz;
      // matrix with the columns t, b, n: m[r][c]; branches of MatrixToQuaternion
      const m00 = tx, m01 = bx, m02 = nx, m10 = ty, m11 = by, m12 = ny, m20 = tz, m21 = bz, m22 = nz;
      const m = [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]];
      const tr = m00 + m11 + m22;
      let choice = 0, mx = m00;
      if (m11 > mx) { choice = 1; mx = m11; }
      if (m22 > mx) { choice = 2; mx = m22; }
      if (tr > mx) choice = 3;
      let w, x, y, z;
      if (choice < 3) {
        const i = choice, j = (i + 1) % 3, kk = (j + 1) % 3;
        const q = [0, 0, 0, 0];
        q[i + 1] = 1 - tr + 2 * m[i][i];
        q[j + 1] = m[j][i] + m[i][j];
        q[kk + 1] = m[kk][i] + m[i][kk];
        q[0] = m[kk][j] - m[j][kk];
        [w, x, y, z] = q;
      } else {
        x = m[2][1] - m[1][2]; y = m[0][2] - m[2][0]; z = m[1][0] - m[0][1]; w = 1 + tr;
      }
      const n = Math.hypot(w, x, y, z) || 1;
      qf[4 * f] = w / n; qf[4 * f + 1] = x / n; qf[4 * f + 2] = y / n; qf[4 * f + 3] = z / n;
    }
  }

  /** Rebuild all Gaussians for the blend weight psi (K) and the posed mesh vertices verts (V*3). */
  setState(psi, verts) {
    const { N, K, tbn, qFace, pos, quat, act, actW } = this, d = this.b;
    this.computeTbn(verts);
    let na = 0;
    for (let k = 0; k < K; k++) if (Math.abs(psi[k]) > 1e-7) { act[na] = k; actW[na] = psi[k]; na++; }
    const A = this.texA.image.data, B = this.texB.image.data, C = this.texColor.image.data, P = this.texCenter.image.data;
    const cov = this.covTmp || (this.covTmp = new Float32Array(6));
    const sb = d.sizeBoost, K10 = K * 10, { xyz0, rot0, rgb0, scale, opacity, dg, faceId, bary, faces } = d;
    for (let i = 0; i < N; i++) {
      const g = 10 * i, o = i * K10;
      let x = xyz0[3 * i], y = xyz0[3 * i + 1], z = xyz0[3 * i + 2];
      let rw = rot0[4 * i], rx = rot0[4 * i + 1], ry = rot0[4 * i + 2], rz = rot0[4 * i + 3];
      let cr = rgb0[3 * i], cg = rgb0[3 * i + 1], cb = rgb0[3 * i + 2];
      for (let a = 0; a < na; a++) {
        const w = actW[a], p = o + act[a] * 10;
        x += w * dg[p]; y += w * dg[p + 1]; z += w * dg[p + 2];
        rw += w * dg[p + 3]; rx += w * dg[p + 4]; ry += w * dg[p + 5]; rz += w * dg[p + 6];
        cr += w * dg[p + 7]; cg += w * dg[p + 8]; cb += w * dg[p + 9];
      }
      const rn = Math.hypot(rw, rx, ry, rz) || 1;
      rw /= rn; rx /= rn; ry /= rn; rz /= rn;
      const f = faceId[i], t = 9 * f;
      const i0 = faces[3 * f] * 3, i1 = faces[3 * f + 1] * 3, i2 = faces[3 * f + 2] * 3;
      const b0 = bary[3 * i], b1 = bary[3 * i + 1], b2 = bary[3 * i + 2];
      const px = tbn[t] * x + tbn[t + 3] * y + tbn[t + 6] * z + b0 * verts[i0] + b1 * verts[i1] + b2 * verts[i2];
      const py = tbn[t + 1] * x + tbn[t + 4] * y + tbn[t + 7] * z + b0 * verts[i0 + 1] + b1 * verts[i1 + 1] + b2 * verts[i2 + 1];
      const pz = tbn[t + 2] * x + tbn[t + 5] * y + tbn[t + 8] * z + b0 * verts[i0 + 2] + b1 * verts[i1 + 2] + b2 * verts[i2 + 2];
      const q = 4 * f, pw = qFace[q], px_ = qFace[q + 1], py_ = qFace[q + 2], pz_ = qFace[q + 3];   // quat(TBN) * rot
      const qw = pw * rw - px_ * rx - py_ * ry - pz_ * rz;
      const qx = pw * rx + rw * px_ + py_ * rz - pz_ * ry;
      const qy = pw * ry + rw * py_ + pz_ * rx - px_ * rz;
      const qz = pw * rz + rw * pz_ + px_ * ry - py_ * rx;
      pos[3 * i] = px; pos[3 * i + 1] = py; pos[3 * i + 2] = pz;
      quat[4 * i] = qw; quat[4 * i + 1] = qx; quat[4 * i + 2] = qy; quat[4 * i + 3] = qz;
      P[4 * i] = px; P[4 * i + 1] = py; P[4 * i + 2] = pz; P[4 * i + 3] = 1;
      quatToCov4(qw, qx, qy, qz, scale[3 * i] * sb, scale[3 * i + 1] * sb, scale[3 * i + 2] * sb, cov, 0);
      A[4 * i] = cov[0]; A[4 * i + 1] = cov[1]; A[4 * i + 2] = cov[2]; A[4 * i + 3] = cov[3];
      B[4 * i] = cov[4]; B[4 * i + 1] = cov[5]; B[4 * i + 2] = 0; B[4 * i + 3] = 0;
      C[4 * i] = cr > 0 ? cr : 0; C[4 * i + 1] = cg > 0 ? cg : 0; C[4 * i + 2] = cb > 0 ? cb : 0; C[4 * i + 3] = opacity[i];
    }
    this.texCenter.needsUpdate = this.texA.needsUpdate = this.texB.needsUpdate = this.texColor.needsUpdate = true;
    this.sortDirty = true;
  }

  sort(camera) { super.sort(camera, this.mesh.matrixWorld); }

  /** Compare the browser result with what RGBAvatar's CUDA code gave for the same emotion (test_ref.json of the export). */
  selfTest(ref) {
    const out = {};
    for (const [name, r] of Object.entries(ref.emotions)) {
      const s = this.emotions[name];
      this.setState(s.psi, s.verts);
      let ex = 0, eq = 0, ec = 0;
      ref.indices.forEach((gi, j) => {
        for (let c = 0; c < 3; c++) {
          ex = Math.max(ex, Math.abs(this.pos[3 * gi + c] - r.xyz[j][c]));
          ec = Math.max(ec, Math.abs(this.texColor.image.data[4 * gi + c] - r.rgb[j][c]));
        }
        const sg = this.quat[4 * gi] < 0 ? -1 : 1;   // q and -q are one rotation
        for (let c = 0; c < 4; c++) eq = Math.max(eq, Math.abs(sg * this.quat[4 * gi + c] - r.quat[j][c]));
      });
      out[name] = { xyz: ex, quat: eq, rgb: ec };
    }
    this.setState(this.emotions.neutral.psi, this.emotions.neutral.verts);
    const psi = new Float32Array(this.K); out._mlp = 0;   // the MLP in JS against the blend weights RGBAvatar computed
    for (const s of Object.values(this.emotions)) { this.mlp(s.x, psi); for (let k = 0; k < this.K; k++) out._mlp = Math.max(out._mlp, Math.abs(psi[k] - s.psi[k])); }
    return out;
  }
}

/** Face state: moves between emotions in 0.3 s, adds the blink, the sway and the nod while waiting for the answer, and the head turn
 *  of the mouse drag. The face is kept as MLP inputs x, a mesh and the part of the mesh that follows the root; the blend weights come from
 *  the MLP every time the state changes, with the neck rotation and the translation of the current head pose in the input like in training. */
export class BlendDriver {
  constructor(avatar, meta) {
    this.av = avatar;
    this.K = avatar.K; this.nv = avatar.nVerts * 3;
    this.pivot = new THREE.Vector3(...meta.neck_pivot);
    const mk = () => ({ x: new Float32Array(N_IN), verts: new Float32Array(this.nv), vroot: new Float32Array(this.nv), joint: new Float32Array(3) });
    this.from = mk(); this.to = mk(); this.cur = mk(); this.eff = mk(); this.turned = new Float32Array(this.nv);
    const n = avatar.emotions.neutral;
    for (const s of [this.from, this.to, this.cur, this.eff]) FIELDS.forEach((f) => s[f].set(n[f]));
    this.xp = new Float32Array(N_IN); this.psi = new Float32Array(this.K);
    this.t0 = -10; this.dur = 0.3; this.settled = true;
    this.nextBlink = 2; this.blinkStart = -10; this.lastBlink = 0;
    this.thinking = false; this.thinkAmp = 0; this.lastT = 0;
    this.forceBlink = null;
    this.lookTarget = { yaw: 0, pitch: 0 }; this.look = { yaw: 0, pitch: 0 }; this.dragging = false;   // mouse drag turns the head (radians)
    this.pose = { yaw: 0, pitch: 0 };   // the head pose the mesh and the blend weights were computed for
  }
  progress(now) { return smooth((now - this.t0) / this.dur); }
  /** p: {emotion, intensity} (intensity 1 = the emotion as it was recorded) or {weights: {emotion: w}}; the face moves there in `dur` seconds.
   *  Target = neutral + sum_i w_i (emotion_i - neutral), over x, verts, vroot and joint; the MLP then runs on the mixed x. */
  setTarget(p, now, dur = 0.3) {
    FIELDS.forEach((f) => this.from[f].set(this.cur[f]));
    const n = this.av.emotions.neutral;
    const terms = (p.weights ? Object.entries(p.weights) : [[p.emotion, p.intensity ?? 1]])
      .filter(([name]) => this.av.emotions[name] && name !== 'neutral')
      .map(([name, w]) => [this.av.emotions[name], Math.min(1.5, Math.max(0, w))]);
    for (const f of FIELDS) {
      const a = this.to[f], b = n[f];
      a.set(b);
      for (const [e, k] of terms) { if (k === 0) continue; const c = e[f]; for (let i = 0; i < a.length; i++) a[i] += k * (c[i] - b[i]); }
    }
    this.t0 = now; this.dur = Math.max(dur, 1e-3); this.settled = false;
  }
  /** Rotation of the head (yaw about y, then pitch about x; pitch > 0 looks down), a THREE.Matrix4 */
  static headMatrix(yaw, pitch) {
    return new THREE.Matrix4().makeRotationY(yaw).multiply(new THREE.Matrix4().makeRotationX(pitch));
  }
  /** The head turn as FLAMEDataset has it: R goes into the FLAME neck joint, the neck is skinned and the shoulders stay.
   *  With w0 the weight of the root joint, t = R j - j (j = neck joint), v_rest the posed rest vertices, d the pose corrective of the
   *  neck rotation (rest pose offset, (R - I) times the posedirs of the neck) and v the mesh of the emotion (neck at rest):
   *    v' = w0 (v_rest + t) + R (v - w0 v_rest) + w0 d + (1 - w0) R d
   *  Left out: the jaw rotation in the skinning of d (a few hundredths of a millimeter). */
  turnHead(src, e, out) {
    const av = this.av, w0 = av.w0, v = src.verts, vr = src.vroot, j = src.joint, n = this.nv / 3, nv = this.nv, d = av.delta, pd = av.pd;
    const tx = e[0] * j[0] + e[4] * j[1] + e[8] * j[2] - j[0], ty = e[1] * j[0] + e[5] * j[1] + e[9] * j[2] - j[1], tz = e[2] * j[0] + e[6] * j[1] + e[10] * j[2] - j[2];
    d.fill(0);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {   // flat index r*3+c of R - I, column major elements: R[r][c] = e[c*4+r]
      const f = e[c * 4 + r] - (r === c ? 1 : 0), o = (r * 3 + c) * nv;
      if (f !== 0) for (let i = 0; i < nv; i++) d[i] += f * pd[o + i];
    }
    for (let i = 0; i < n; i++) {
      const a = 3 * i, dx = v[a] - vr[a], dy = v[a + 1] - vr[a + 1], dz = v[a + 2] - vr[a + 2], w = w0[i], q = 1 - w;
      const cx = d[a], cy = d[a + 1], cz = d[a + 2];
      out[a] = vr[a] + w * tx + e[0] * dx + e[4] * dy + e[8] * dz + w * cx + q * (e[0] * cx + e[4] * cy + e[8] * cz);
      out[a + 1] = vr[a + 1] + w * ty + e[1] * dx + e[5] * dy + e[9] * dz + w * cy + q * (e[1] * cx + e[5] * cy + e[9] * cz);
      out[a + 2] = vr[a + 2] + w * tz + e[2] * dx + e[6] * dy + e[10] * dz + w * cz + q * (e[2] * cx + e[6] * cy + e[10] * cz);
    }
    return out;
  }

  update(now) {
    const dt = Math.min(0.1, now - this.lastT); this.lastT = now;
    const s = this.progress(now);
    const moving = !this.settled;
    if (s >= 1) this.settled = true;
    if (moving) for (const f of FIELDS) { const c = this.cur[f], a = this.from[f], b = this.to[f]; for (let i = 0; i < c.length; i++) c[i] = a[i] + (b[i] - a[i]) * s; }
    let blink = 0;
    if (now >= this.nextBlink && this.blinkStart < 0) this.blinkStart = now;
    if (this.blinkStart >= 0) {
      const t = now - this.blinkStart;
      blink = t < 0.07 ? t / 0.07 : 1 - (t - 0.07) / 0.09;
      if (t > 0.16) { blink = 0; this.blinkStart = -1; this.nextBlink = now + 3 + Math.random() * 2; }
    }
    if (this.forceBlink != null) blink = this.forceBlink;   // test hook

    // head turn of the mouse drag: follows the mouse quickly, returns to the front slowly
    const k = Math.min(1, dt * (this.dragging ? 12 : 4));
    this.look.yaw += (this.lookTarget.yaw - this.look.yaw) * k;
    this.look.pitch += (this.lookTarget.pitch - this.look.pitch) * k;
    if (!this.dragging && this.lookTarget.yaw === 0 && this.lookTarget.pitch === 0 && Math.abs(this.look.yaw) < 0.003 && Math.abs(this.look.pitch) < 0.003) { this.look.yaw = 0; this.look.pitch = 0; }
    const dPose = Math.max(Math.abs(this.look.yaw - this.pose.yaw), Math.abs(this.look.pitch - this.pose.pitch));
    const poseMoved = dPose > 0.003 || (dPose > 0 && this.look.yaw === 0 && this.look.pitch === 0);

    if (blink !== this.lastBlink || moving || this.dirty !== false || poseMoved) {
      let src = this.cur;
      if (blink > 0) {   // blink = neutral + eyes closed: add the difference to whatever is on the face
        const n = this.av.emotions.neutral, b = this.av.emotions.blink;
        for (const f of FIELDS) { const o = this.eff[f], c = this.cur[f], bb = b[f], nn = n[f]; for (let i = 0; i < o.length; i++) o[i] = c[i] + blink * (bb[i] - nn[i]); }
        src = this.eff;
      }
      if (poseMoved) { this.pose.yaw = this.look.yaw; this.pose.pitch = this.look.pitch; }
      // neck rotation (first two rows of R, 6D) and the translation R j - j are MLP inputs; the mesh gets the same rotation in the neck joint
      const e = BlendDriver.headMatrix(this.pose.yaw, this.pose.pitch).elements;   // column major
      const x = this.xp; x.set(src.x);
      x[100] = e[0]; x[101] = e[4]; x[102] = e[8]; x[103] = e[1]; x[104] = e[5]; x[105] = e[9];
      const j = src.joint;
      x[126] = e[0] * j[0] + e[4] * j[1] + e[8] * j[2] - j[0];
      x[127] = e[1] * j[0] + e[5] * j[1] + e[9] * j[2] - j[1];
      x[128] = e[2] * j[0] + e[6] * j[1] + e[10] * j[2] - j[2];
      const verts = this.pose.yaw === 0 && this.pose.pitch === 0 ? src.verts : this.turnHead(src, e, this.turned);
      this.av.setState(this.av.mlp(x, this.psi), verts);
      this.lastBlink = blink; this.dirty = false;
    }
    // sway (a fraction of a degree) and the nod while waiting for the answer: the whole head turns about the neck
    this.thinkAmp += ((this.thinking ? 1 : 0) - this.thinkAmp) * Math.min(1, dt * 5);
    const deg = Math.PI / 180;
    const yaw = (0.7 * Math.sin(now * 2 * Math.PI / 6.3) + 0.3 * Math.sin(now * 2 * Math.PI / 2.9)) * deg;
    const pitch = (0.5 * Math.sin(now * 2 * Math.PI / 4.7 + 1.0)) * deg + this.thinkAmp * 3.5 * deg * (0.5 - 0.5 * Math.cos(now * 2 * Math.PI * 0.9));
    const roll = (0.4 * Math.sin(now * 2 * Math.PI / 7.9 + 2.0)) * deg;
    const m = this.av.mesh;
    m.matrix.makeTranslation(this.pivot.x, this.pivot.y, this.pivot.z)
      .multiply(new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(pitch, yaw, roll, 'ZYX')))
      .multiply(new THREE.Matrix4().makeTranslation(-this.pivot.x, -this.pivot.y, -this.pivot.z));
    m.matrixWorldNeedsUpdate = true;
    this.av.sortDirty = true;
  }
}
