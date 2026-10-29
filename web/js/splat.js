// Gaussian splat avatar for three.js: Gaussians that follow a FLAME head.
//
// The projection of a 3D Gaussian to a screen ellipse and the back-to-front sorted, premultiplied-alpha blending follow
// the usual real-time 3DGS web renderers (e.g. antimatter15/splat, MIT). What is new here is that the centers move:
// every frame, the position of Gaussian i is computed from the FLAME state the server sent (see server/flame_np.py):
//
//     x_i = T_i (c_i + sum_k b_ik d[v_ik]),     T_i = sum_j w_ij A_j
//
// c_i  canonical position (mean shape + predicted offset)     v_ik, b_ik  mesh triangle and barycentric weights of the texel
// d    expression offset of every mesh vertex (V x 3)         A_j        relative transform of joint j (root, neck, jaw, eyes)
// Rotation and scale of a Gaussian do not change (same as UIKA's own renderer).
import * as THREE from '../vendor/three.module.js';

export const TEX_W = 2048;

export const VERT = /* glsl */ `
precision highp float;
precision highp int;
uniform highp sampler2D uCenter, uCovA, uCovB, uColor;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform vec2 uViewport;
in vec2 position;
in float aIndex;
out vec4 vColor;
out vec2 vPos;
void main() {
  int idx = int(aIndex + 0.5);
  ivec2 uv = ivec2(idx & ${TEX_W - 1}, idx / ${TEX_W});
  vec4 c = texelFetch(uCenter, uv, 0);
  vec4 cam = modelViewMatrix * vec4(c.xyz, 1.0);
  vec4 clip = projectionMatrix * cam;
  float lim = 1.25 * clip.w;
  if (clip.z < -clip.w || abs(clip.x) > lim || abs(clip.y) > lim) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  vec4 a = texelFetch(uCovA, uv, 0);
  vec4 b = texelFetch(uCovB, uv, 0);
  mat3 V = mat3(a.x, a.y, a.z,  a.y, a.w, b.x,  a.z, b.x, b.y);
  float fx = projectionMatrix[0][0] * uViewport.x * 0.5;
  float fy = projectionMatrix[1][1] * uViewport.y * 0.5;
  float z = cam.z;                                   // negative in front of the camera
  // screen coordinates in pixels, y up: u = fx x / -z, w = fy y / -z
  mat3 J = mat3(-fx / z, 0.0, 0.0,   0.0, -fy / z, 0.0,   fx * cam.x / (z * z), fy * cam.y / (z * z), 0.0);
  mat3 T = J * mat3(modelViewMatrix);
  mat3 cov = T * V * transpose(T);
  float ca = cov[0][0] + 0.3, cb = cov[0][1], cd = cov[1][1] + 0.3;
  float mid = 0.5 * (ca + cd);
  float rad = length(vec2(0.5 * (ca - cd), cb));
  float l1 = mid + rad, l2 = max(mid - rad, 0.05);
  vec2 dir = normalize(vec2(cb, l1 - ca) + vec2(1e-9, 0.0));
  vec2 major = min(sqrt(2.0 * l1), 1024.0) * dir;
  vec2 minor = min(sqrt(2.0 * l2), 1024.0) * vec2(dir.y, -dir.x);
  vColor = texelFetch(uColor, uv, 0);
  vPos = position;
  vec2 ndc = clip.xy / clip.w + (position.x * major + position.y * minor) * 2.0 / uViewport;
  gl_Position = vec4(ndc, clip.z / clip.w, 1.0);
}`;

export const FRAG = /* glsl */ `
precision highp float;
in vec4 vColor;
in vec2 vPos;
out vec4 outColor;
void main() {
  float A = -dot(vPos, vPos);
  if (A < -4.0) discard;
  float B = exp(A) * vColor.a;
  outColor = vec4(B * vColor.rgb, B);
}`;

export function quatToCov4(w, x, y, z, s0, s1, s2, out, o) {
  // q = (w, x, y, z) not necessarily normalized, s = scales; covariance = R S S R^T, 6 upper triangle values
  const n = Math.hypot(w, x, y, z) || 1;
  w /= n; x /= n; y /= n; z /= n;
  const r00 = 1 - 2 * (y * y + z * z), r01 = 2 * (x * y - w * z), r02 = 2 * (x * z + w * y);
  const r10 = 2 * (x * y + w * z), r11 = 1 - 2 * (x * x + z * z), r12 = 2 * (y * z - w * x);
  const r20 = 2 * (x * z - w * y), r21 = 2 * (y * z + w * x), r22 = 1 - 2 * (x * x + y * y);
  const sx = s0 * s0, sy = s1 * s1, sz = s2 * s2;
  out[o] = r00 * r00 * sx + r01 * r01 * sy + r02 * r02 * sz;
  out[o + 1] = r00 * r10 * sx + r01 * r11 * sy + r02 * r12 * sz;
  out[o + 2] = r00 * r20 * sx + r01 * r21 * sy + r02 * r22 * sz;
  out[o + 3] = r10 * r10 * sx + r11 * r11 * sy + r12 * r12 * sz;
  out[o + 4] = r10 * r20 * sx + r11 * r21 * sy + r12 * r22 * sz;
  out[o + 5] = r20 * r20 * sx + r21 * r21 * sy + r22 * r22 * sz;
}

function quatToCov(q, s, out, o) { quatToCov4(q[0], q[1], q[2], q[3], s[0], s[1], s[2], out, o); }

export async function loadAvatar(baseUrl, { maxGaussians = 0 } = {}) {
  const [meta, buf] = await Promise.all([
    fetch(`${baseUrl}/meta.json`).then((r) => r.json()),
    fetch(`${baseUrl}/gaussians.bin`).then((r) => r.arrayBuffer()),
  ]);
  const a = {};
  for (const [name, info] of Object.entries(meta.arrays)) {
    const n = info.shape.reduce((p, q) => p * q, 1);
    a[name] = /u(int)?32|u4/.test(info.dtype) ? new Uint32Array(buf, info.offset, n) : new Float32Array(buf, info.offset, n);
  }
  let N = meta.count;
  // optional reduction for weak devices: keep the Gaussians that matter most (opacity x volume), make the rest a bit bigger
  let keep = null;
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
  const sizeBoost = keep ? Math.pow(meta.count / N, 1 / 3) * 0.9 : 1;
  const data = {
    N, meta, xyz: pick(a.xyz, 3), bary: pick(a.bary, 3), vidx: pick(a.vidx, 3), wpix: pick(a.wpix, 5),
    color: pick(a.color, 3), opacity: pick(a.opacity, 1), scale: pick(a.scale, 3), quat: pick(a.quat, 4), sizeBoost,
  };
  return new SplatAvatar(data);
}

export class SplatAvatar {
  constructor(d) {
    this.d = d;
    this.N = d.N;
    const N = this.N;
    this.nVerts = d.meta.n_vertices;
    this.dg = new Float32Array(N * 3);       // sum_k b_k d[v_k]
    this.pos = new Float32Array(N * 3);      // animated centers
    this.depth = new Float32Array(N);
    this.order = new Float32Array(N);
    this.counts = new Uint32Array(65536);
    this.keys = new Uint16Array(N);
    this.tmpOrder = new Uint32Array(N);

    const rows = Math.ceil(N / TEX_W);
    const mk = () => new THREE.DataTexture(new Float32Array(TEX_W * rows * 4), TEX_W, rows, THREE.RGBAFormat, THREE.FloatType);
    this.texCenter = mk(); this.texA = mk(); this.texB = mk(); this.texColor = mk();
    for (const t of [this.texCenter, this.texA, this.texB, this.texColor]) {
      t.minFilter = t.magFilter = THREE.NearestFilter; t.generateMipmaps = false; t.needsUpdate = true;
    }
    const cov = new Float32Array(6);
    for (let i = 0; i < N; i++) {
      const s = [d.scale[3 * i] * d.sizeBoost, d.scale[3 * i + 1] * d.sizeBoost, d.scale[3 * i + 2] * d.sizeBoost];
      quatToCov([d.quat[4 * i], d.quat[4 * i + 1], d.quat[4 * i + 2], d.quat[4 * i + 3]], s, cov, 0);
      const A = this.texA.image.data, B = this.texB.image.data, C = this.texColor.image.data;
      A.set([cov[0], cov[1], cov[2], cov[3]], 4 * i);
      B.set([cov[4], cov[5], 0, 0], 4 * i);
      C.set([d.color[3 * i], d.color[3 * i + 1], d.color[3 * i + 2], d.opacity[i]], 4 * i);
    }
    this.texA.needsUpdate = this.texB.needsUpdate = this.texColor.needsUpdate = true;

    const geo = new THREE.InstancedBufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), 2));
    geo.setIndex([0, 1, 2, 0, 2, 3]);
    this.indexAttr = new THREE.InstancedBufferAttribute(this.order, 1);
    this.indexAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('aIndex', this.indexAttr);
    geo.instanceCount = N;
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 10);   // 'position' is a 2D quad corner, not a 3D position
    this.material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: VERT, fragmentShader: FRAG,
      uniforms: {
        uCenter: { value: this.texCenter }, uCovA: { value: this.texA }, uCovB: { value: this.texB },
        uColor: { value: this.texColor }, uViewport: { value: new THREE.Vector2(1, 1) },
      },
      transparent: true, depthTest: false, depthWrite: false, side: THREE.DoubleSide,
      blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
      blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
    });
    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.frustumCulled = false;
    for (let i = 0; i < N; i++) { this.order[i] = i; }
  }

  /** dVerts: Float32Array(V*3) expression offset of every mesh vertex */
  setExpression(dVerts) {
    const { N } = this, { bary, vidx } = this.d, dg = this.dg;
    for (let i = 0; i < N; i++) {
      const i3 = 3 * i;
      const a = vidx[i3] * 3, b = vidx[i3 + 1] * 3, c = vidx[i3 + 2] * 3;
      const w0 = bary[i3], w1 = bary[i3 + 1], w2 = bary[i3 + 2];
      dg[i3] = w0 * dVerts[a] + w1 * dVerts[b] + w2 * dVerts[c];
      dg[i3 + 1] = w0 * dVerts[a + 1] + w1 * dVerts[b + 1] + w2 * dVerts[c + 1];
      dg[i3 + 2] = w0 * dVerts[a + 2] + w1 * dVerts[b + 2] + w2 * dVerts[c + 2];
    }
  }

  /** A: Float32Array(80), 5 joints x 4x4 row-major */
  setPose(A) {
    const { N } = this, { xyz, wpix } = this.d, dg = this.dg, pos = this.pos;
    for (let i = 0; i < N; i++) {
      const i3 = 3 * i, i5 = 5 * i;
      let t00 = 0, t01 = 0, t02 = 0, t03 = 0, t10 = 0, t11 = 0, t12 = 0, t13 = 0, t20 = 0, t21 = 0, t22 = 0, t23 = 0;
      for (let j = 0; j < 5; j++) {
        const w = wpix[i5 + j];
        if (w === 0) continue;
        const o = j * 16;
        t00 += w * A[o]; t01 += w * A[o + 1]; t02 += w * A[o + 2]; t03 += w * A[o + 3];
        t10 += w * A[o + 4]; t11 += w * A[o + 5]; t12 += w * A[o + 6]; t13 += w * A[o + 7];
        t20 += w * A[o + 8]; t21 += w * A[o + 9]; t22 += w * A[o + 10]; t23 += w * A[o + 11];
      }
      const x = xyz[i3] + dg[i3], y = xyz[i3 + 1] + dg[i3 + 1], z = xyz[i3 + 2] + dg[i3 + 2];
      pos[i3] = t00 * x + t01 * y + t02 * z + t03;
      pos[i3 + 1] = t10 * x + t11 * y + t12 * z + t13;
      pos[i3 + 2] = t20 * x + t21 * y + t22 * z + t23;
    }
    const c = this.texCenter.image.data;
    for (let i = 0; i < this.N; i++) { c[4 * i] = pos[3 * i]; c[4 * i + 1] = pos[3 * i + 1]; c[4 * i + 2] = pos[3 * i + 2]; c[4 * i + 3] = 1; }
    this.texCenter.needsUpdate = true;
    this.sortDirty = true;
  }

  /** back-to-front order for the current camera (counting sort on 16 bit depth keys) */
  sort(camera, objectMatrix = null) {
    const m = objectMatrix ? new THREE.Matrix4().multiplyMatrices(camera.matrixWorldInverse, objectMatrix).elements : camera.matrixWorldInverse.elements;
    const { N, pos, depth, keys, counts } = this;
    let zmin = Infinity, zmax = -Infinity;
    for (let i = 0; i < N; i++) {
      const z = m[2] * pos[3 * i] + m[6] * pos[3 * i + 1] + m[10] * pos[3 * i + 2] + m[14];
      depth[i] = z;
      if (z < zmin) zmin = z;
      if (z > zmax) zmax = z;
    }
    const scale = 65535 / Math.max(zmax - zmin, 1e-6);
    counts.fill(0);
    for (let i = 0; i < N; i++) { const k = ((depth[i] - zmin) * scale) | 0; keys[i] = k; counts[k]++; }
    let acc = 0;
    for (let k = 0; k < 65536; k++) { const c = counts[k]; counts[k] = acc; acc += c; }   // ascending z: far (most negative) first
    const out = this.order;
    for (let i = 0; i < N; i++) out[counts[keys[i]]++] = i;
    this.indexAttr.needsUpdate = true;
    this.sortDirty = false;
  }

  setViewport(w, h) { this.material.uniforms.uViewport.value.set(w, h); }
}
