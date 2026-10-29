import * as THREE from '../vendor/three.module.js';
import { loadAvatar } from './splat.js';
import { loadBlendAvatar, BlendDriver } from './rgbavatar.js';

const q = new URLSearchParams(location.search);
const isMobile = /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent) || matchMedia('(pointer: coarse)').matches;
const MAX_GAUSSIANS = q.has('max') ? parseInt(q.get('max'), 10) : (isMobile ? 50000 : 0);
const HOLD_SECONDS = q.has('hold') ? parseFloat(q.get('hold')) : 10;   // after this long without a message the face relaxes to neutral (0: never)
const SESSION = 'web-' + Math.random().toString(36).slice(2, 8);
const $ = (id) => document.getElementById(id);

const api = {
  async get(path) { const r = await fetch(path); if (!r.ok) throw new Error(path + ' ' + r.status); return r.json(); },
  async post(path, body) {
    const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error(path + ' ' + r.status);
    return r.json();
  },
};

function decodeF32(b64) {
  const bin = atob(b64), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}
const smooth = (t) => { t = Math.min(1, Math.max(0, t)); return t * t * (3 - 2 * t); };

// 4x4 row-major helpers
function mul4(a, b, out, o) {
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    out[o + r * 4 + c] = a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] + a[r * 4 + 2] * b[8 + c] + a[r * 4 + 3] * b[12 + c];
  }
}
function rotationAbout(pivot, yaw, pitch, roll) {   // R = Rz Ry Rx about the pivot, as a 4x4
  const [cx, sx, cy, sy, cz, sz] = [Math.cos(pitch), Math.sin(pitch), Math.cos(yaw), Math.sin(yaw), Math.cos(roll), Math.sin(roll)];
  const Rx = [1, 0, 0, 0, cx, -sx, 0, sx, cx], Ry = [cy, 0, sy, 0, 1, 0, -sy, 0, cy], Rz = [cz, -sz, 0, sz, cz, 0, 0, 0, 1];
  const m3 = (a, b) => { const o = new Array(9); for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c]; return o; };
  const R = m3(Rz, m3(Ry, Rx));
  const t = [0, 1, 2].map((r) => pivot[r] - (R[r * 3] * pivot[0] + R[r * 3 + 1] * pivot[1] + R[r * 3 + 2] * pivot[2]));
  return new Float32Array([R[0], R[1], R[2], t[0], R[3], R[4], R[5], t[1], R[6], R[7], R[8], t[2], 0, 0, 0, 1]);
}

/** Holds the face state, blends 0.3 s transitions and adds the idle behavior (blink, sway, thinking nod). */
class Driver {
  constructor(avatar, nVerts) {
    this.av = avatar;
    this.V = nVerts;
    const zeroA = new Float32Array(80);
    for (let j = 0; j < 5; j++) for (let k = 0; k < 4; k++) zeroA[j * 16 + k * 5] = 1;
    this.dFrom = new Float32Array(nVerts * 3); this.dTo = new Float32Array(nVerts * 3); this.dCur = new Float32Array(nVerts * 3);
    this.dEff = new Float32Array(nVerts * 3);
    this.AFrom = zeroA.slice(); this.ATo = zeroA.slice(); this.ACur = zeroA.slice(); this.AEff = zeroA.slice();
    this.pivot = [0, 0, 0];
    this.t0 = -10; this.dur = 0.3;
    this.blinkD = null; this.nextBlink = 2; this.blinkStart = -10;
    this.thinking = false; this.thinkAmp = 0; this.thinkClock = 0;
    this.lastT = 0;
    this.expressionDirty = true;
    this.pending = zeroA.slice();
  }
  progress(now) { return smooth((now - this.t0) / this.dur); }
  /** payload: {d (base64), A, pivot}; the face moves there in `dur` seconds */
  setTarget(p, now, dur = 0.3) {
    const s = this.progress(now);
    for (let i = 0; i < this.dCur.length; i++) this.dFrom[i] = this.dCur[i];
    for (let i = 0; i < 80; i++) this.AFrom[i] = this.ACur[i];
    this.dTo = decodeF32(p.d); this.ATo = Float32Array.from(p.A); this.pivot = p.pivot;
    this.t0 = now; this.dur = Math.max(dur, 1e-3); this.settled = false;
  }
  update(now) {
    const dt = Math.min(0.1, now - this.lastT); this.lastT = now;
    const s = this.progress(now);
    const moving = !this.settled;   // stays true until the last frame of the transition has been applied (also at low fps)
    if (s >= 1) this.settled = true;
    if (moving || this.expressionDirty) {
      for (let i = 0; i < this.dCur.length; i++) this.dCur[i] = this.dFrom[i] + (this.dTo[i] - this.dFrom[i]) * s;
      for (let i = 0; i < 80; i++) this.ACur[i] = this.AFrom[i] + (this.ATo[i] - this.AFrom[i]) * s;
    }
    // blink
    let blink = 0;
    if (now >= this.nextBlink && this.blinkStart < 0) this.blinkStart = now;
    if (this.blinkStart >= 0) {
      const t = now - this.blinkStart;
      blink = t < 0.07 ? t / 0.07 : 1 - (t - 0.07) / 0.09;
      if (t > 0.16) { blink = 0; this.blinkStart = -1; this.nextBlink = now + 3 + Math.random() * 2; }
    }
    if (this.forceBlink != null) blink = this.forceBlink;   // test hook
    if (blink !== this.lastBlink || moving || this.expressionDirty) {
      if (this.blinkD && blink > 0) for (let i = 0; i < this.dEff.length; i++) this.dEff[i] = this.dCur[i] + blink * this.blinkD[i];
      else this.dEff.set(this.dCur);
      this.av.setExpression(this.dEff);
      this.lastBlink = blink;
    }
    // sway (a fraction of a degree) and the nod while waiting for the answer
    this.thinkAmp += ((this.thinking ? 1 : 0) - this.thinkAmp) * Math.min(1, dt * 5);
    this.thinkClock += dt * (0.3 + 0.7 * this.thinkAmp) * 1.0;
    const deg = Math.PI / 180;
    const yaw = (0.7 * Math.sin(now * 2 * Math.PI / 6.3) + 0.3 * Math.sin(now * 2 * Math.PI / 2.9)) * deg;
    const pitch = (0.5 * Math.sin(now * 2 * Math.PI / 4.7 + 1.0)) * deg + this.thinkAmp * 3.5 * deg * (0.5 - 0.5 * Math.cos(now * 2 * Math.PI * 0.9));
    const roll = (0.4 * Math.sin(now * 2 * Math.PI / 7.9 + 2.0)) * deg;
    const P = rotationAbout(this.pivot, yaw, pitch, roll);
    this.AEff.set(this.ACur.subarray(0, 16), 0);
    for (let j = 1; j < 5; j++) mul4(P, this.ACur.subarray(j * 16, j * 16 + 16), this.AEff, j * 16);
    this.av.setPose(this.AEff);
    this.expressionDirty = false;
  }
}

async function main() {
  const status = (t) => { $('status').textContent = t; };
  status('아바타 불러오는 중…');
  const info = await api.get('/api/meta');
  const rgba = info.type === 'rgbavatar';   // RGBAvatar blendshapes (rgbavatar.js) or Gaussians that follow a FLAME head (splat.js)
  const base = '/assets/' + (info.avatar || 'custom');
  const avatar = rgba ? await loadBlendAvatar(base, { maxGaussians: MAX_GAUSSIANS }) : await loadAvatar(base, { maxGaussians: MAX_GAUSSIANS });
  const driver = rgba ? new BlendDriver(avatar, info) : new Driver(avatar, info.n_vertices);
  if (!rgba) {
    api.get('/api/blink').then((b) => { driver.blinkD = decodeF32(b.d); });
    const neutral = await api.get('/api/pose?emotion=neutral');
    driver.setTarget(neutral, -10, 0.001);
    driver.dFrom.set(driver.dTo); driver.AFrom.set(driver.ATo);
    driver.pivot = neutral.pivot;
  }

  // ---- three.js ----
  const canvas = $('avatar');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true, premultipliedAlpha: true, powerPreference: 'high-performance' });
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  scene.add(avatar.mesh);
  const camera = new THREE.PerspectiveCamera(info.fov_y_deg, 1, 0.05, 5);
  const target = new THREE.Vector3(...(avatar.meta && avatar.meta.target ? avatar.meta.target : [0, -0.01, 0]));
  const cam = { yaw: parseFloat(q.get('yaw') || 0) * Math.PI / 180, pitch: parseFloat(q.get('pitch') || 0) * Math.PI / 180, zoom: 1 };
  let camDirty = true;

  // ?res=256x320 renders at a fixed size (the size the model was trained at) and the browser scales it to the panel
  const fixedRes = /^(\d+)x(\d+)$/.exec(q.get('res') || '');
  if (fixedRes) canvas.style.objectFit = 'contain';

  function resize() {
    const dpr = fixedRes ? 1 : Math.min(window.devicePixelRatio || 1, isMobile ? 1.5 : 2);
    const w = fixedRes ? +fixedRes[1] : canvas.clientWidth, h = fixedRes ? +fixedRes[2] : canvas.clientHeight;
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // the whole head stays in view also in a narrow (phone) panel
    const halfV = Math.tan(info.fov_y_deg * Math.PI / 360);
    camera.fov = Math.atan(Math.max(halfV, halfV / camera.aspect)) * 360 / Math.PI;
    camera.updateProjectionMatrix();
    const sz = renderer.getDrawingBufferSize(new THREE.Vector2());
    avatar.setViewport(sz.x, sz.y);
    camDirty = true;
  }
  addEventListener('resize', resize);
  resize();

  let drag = null;
  canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointerup', () => { drag = null; driver.dragging = false; if (rgba) driver.lookTarget.yaw = driver.lookTarget.pitch = 0; });
  const LOOK_YAW = 25 * Math.PI / 180, LOOK_PITCH = 15 * Math.PI / 180;   // inside the head poses the model was trained on
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    if (rgba) {   // the head turns about the neck, the camera stays
      const t = driver.lookTarget; driver.dragging = true;
      t.yaw = Math.max(-LOOK_YAW, Math.min(LOOK_YAW, t.yaw + (e.clientX - drag.x) * 0.006));
      t.pitch = Math.max(-LOOK_PITCH, Math.min(LOOK_PITCH, t.pitch + (e.clientY - drag.y) * 0.006));
      drag = { x: e.clientX, y: e.clientY };
      return;
    }
    cam.yaw = Math.max(-0.7, Math.min(0.7, cam.yaw - (e.clientX - drag.x) * 0.005));
    cam.pitch = Math.max(-0.4, Math.min(0.4, cam.pitch + (e.clientY - drag.y) * 0.005));
    drag = { x: e.clientX, y: e.clientY }; camDirty = true;
  });
  canvas.addEventListener('wheel', (e) => { e.preventDefault(); cam.zoom = Math.max(0.5, Math.min(1.6, cam.zoom * Math.exp(e.deltaY * 0.001))); camDirty = true; }, { passive: false });
  canvas.addEventListener('dblclick', () => { cam.yaw = cam.pitch = 0; cam.zoom = 1; camDirty = true; });

  function placeCamera() {
    const dist = info.camera_distance * cam.zoom;
    camera.position.set(target.x + dist * Math.sin(cam.yaw) * Math.cos(cam.pitch), target.y + dist * Math.sin(cam.pitch), target.z + dist * Math.cos(cam.yaw) * Math.cos(cam.pitch));
    camera.lookAt(target);
    camera.updateMatrixWorld(); camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
    scene.updateMatrixWorld();
  }

  // ---- chat ----
  let holdTimer = null;
  // RGBAvatar: the model scores every emotion 0..1 and the face is their weighted sum; the panel shows the weights the face uses
  const MIX = rgba;
  const MIX_NAMES = { happy: '행복', sad: '슬픔', angry: '화남', surprised: '놀람', neutral: '중립' };
  const bars = {};
  if (MIX) {
    const box = $('mixbars'); box.style.display = 'block';
    for (const [e, label] of Object.entries(MIX_NAMES)) {
      const row = document.createElement('div'); row.className = 'row';
      row.innerHTML = `<span class="name">${label}</span><span class="track"><span class="fill"></span></span><span class="val">0.00</span>`;
      box.appendChild(row); bars[e] = { fill: row.querySelector('.fill'), val: row.querySelector('.val') };
    }
  }
  function showWeights(w) {   // w: {emotion: 0..1}, neutral is what is left over when it is not given
    if (!MIX) return;
    const rest = Math.max(0, 1 - Object.entries(w).reduce((a, [e, v]) => a + (e === 'neutral' ? 0 : v), 0));
    for (const e of Object.keys(bars)) { const v = e === 'neutral' ? (w.neutral ?? rest) : (w[e] || 0); bars[e].fill.style.width = (Math.min(1, v) * 100) + '%'; bars[e].val.textContent = v.toFixed(2); }
  }
  async function applyEmotion(pose, emotion, hold = HOLD_SECONDS) {
    driver.setTarget(pose, performance.now() / 1000, 0.3);
    $('emotion').textContent = emotion;
    showWeights(pose.weights || (pose.emotion ? { [pose.emotion]: pose.intensity ?? 1 } : {}));
    clearTimeout(holdTimer);
    if (hold > 0 && emotion !== 'neutral') holdTimer = setTimeout(async () => {
      driver.setTarget(await api.get('/api/pose?emotion=neutral'), performance.now() / 1000, 1.0); $('emotion').textContent = 'neutral';
      showWeights({});
    }, hold * 1000);
  }
  // the answer is shown in the speech bubble above the avatar; it stays until the next question
  function showBubble(text, cls = '') {
    const b = $('bubble'); b.textContent = text; b.className = (cls + ' show').trim();
  }
  function hideBubble() { $('bubble').classList.remove('show'); }
  let busy = false;
  async function showAnswer(r) {
    showBubble(r.reply);
    await applyEmotion(r.pose, r.emotion);
    if (MIX) showWeights({ ...r.pose.weights, neutral: r.emotions.neutral });   // the model's own neutral score, the face has none
    status(MIX ? '감정: ' + Object.entries(r.emotions).filter(([, v]) => v > 0).map(([e, v]) => `${e} ${v.toFixed(2)}`).join(' · ') + ` · 응답 ${r.latency_s}s`
               : `감정: ${r.emotion} · 응답 ${r.latency_s}s`);
  }
  async function greet() {   // the avatar speaks first (a new conversation that starts with its greeting)
    try { await showAnswer(await api.post('/api/greet', { message: '', session: SESSION })); }
    catch (e) { showBubble('오류: ' + e.message, 'err'); }
  }
  async function send(text) {
    text = text.trim();
    if (!text || busy) return;
    busy = true; $('send').disabled = true; $('input').value = '';
    driver.thinking = true; showBubble('…', 'pending');
    try {
      const r = await api.post('/api/chat', { message: text, session: SESSION });
      driver.thinking = false;
      await showAnswer(r);
    } catch (e) {
      driver.thinking = false; showBubble('오류: ' + e.message, 'err');
    } finally { busy = false; $('send').disabled = false; $('input').focus(); }
  }
  $('form').addEventListener('submit', (e) => { e.preventDefault(); send($('input').value); });
  $('reset').addEventListener('click', async () => { hideBubble(); await greet(); });   // a new conversation starts with the greeting again

  // ---- debug sliders (step 3): the server computes the pose, the browser moves the Gaussians ----
  const dbg = { values: {} };
  const sliderDefs = [
    ...Array.from({ length: 8 }, (_, i) => [`expr${i}`, `표정 ${i}`, -3, 3, 0, 0.05]),
    ['jaw', '턱 열기 (°)', 0, 25, 0, 0.5], ['neck_yaw', '고개 좌우 (°)', -30, 30, 0, 0.5],
    ['neck_pitch', '고개 위아래 (°)', -20, 20, 0, 0.5], ['neck_roll', '고개 기울기 (°)', -20, 20, 0, 0.5],
  ];
  let inflight = false, again = false;
  async function pushDebug() {
    if (inflight) { again = true; return; }
    inflight = true;
    const v = dbg.values, rad = Math.PI / 180;
    const expr = Array.from({ length: 8 }, (_, i) => v[`expr${i}`] || 0);
    try {
      const p = await api.post('/api/pose', { expr, jaw: [(v.jaw || 0) * rad, 0, 0], neck: [(v.neck_pitch || 0) * rad, (v.neck_yaw || 0) * rad, (v.neck_roll || 0) * rad] });
      driver.setTarget(p, performance.now() / 1000, 0.05);
    } finally { inflight = false; if (again) { again = false; pushDebug(); } }
  }
  const box = $('sliders');
  if (rgba) { box.style.display = 'none'; $('dbgReset').style.display = 'none'; }   // free FLAME values do not exist for RGBAvatar
  else for (const [key, label, lo, hi, def, step] of sliderDefs) {
    const row = document.createElement('label'); row.className = 'srow';
    row.innerHTML = `<span>${label}</span><input type="range" min="${lo}" max="${hi}" step="${step}" value="${def}"><output>${def}</output>`;
    const input = row.querySelector('input'), out = row.querySelector('output');
    dbg.values[key] = def;
    input.addEventListener('input', () => { dbg.values[key] = parseFloat(input.value); out.textContent = input.value; pushDebug(); });
    box.appendChild(row);
  }
  $('dbgReset').addEventListener('click', () => { box.querySelectorAll('input').forEach((i) => { i.value = i.defaultValue; i.dispatchEvent(new Event('input')); }); });
  for (const e of (rgba ? [] : info.emotions)) {   // RGBAvatar always shows the mix of the model's scores, no single emotion buttons
    const b = document.createElement('button'); b.type = 'button'; b.textContent = e;
    b.addEventListener('click', async () => applyEmotion(await api.get(`/api/pose?emotion=${e}`), e));
    $('emoButtons').appendChild(b);
  }
  if (q.has('debug')) $('debug').classList.add('open');

  // ---- loop ----
  const stats = { frames: 0, t: performance.now(), fps: 0, ms: 0 };
  function frame() {
    const t0 = performance.now(), now = t0 / 1000;
    driver.update(now);
    if (avatar.sortDirty || camDirty) { placeCamera(); avatar.sort(camera); camDirty = false; }
    renderer.render(scene, camera);
    stats.frames++; stats.ms += performance.now() - t0;
    if (t0 - stats.t > 1000) {
      stats.fps = stats.frames * 1000 / (t0 - stats.t); $('fps').textContent = `${stats.fps.toFixed(0)} fps · ${(stats.ms / stats.frames).toFixed(1)} ms · ${avatar.N.toLocaleString()} gaussians`;
      stats.frames = 0; stats.ms = 0; stats.t = t0;
    }
    requestAnimationFrame(frame);
  }
  window.__avatar = { driver, base, applyEmotion, camera, cam, send, avatar, renderer, scene, placeCamera };
  if (rgba) window.__selfTest = async () => avatar.selfTest(await fetch(base + '/test_ref.json').then((r) => r.json()));
  status(`준비 완료 · 모델 ${info.llm_model}`);
  if (MIX && q.has('w')) {   // ?w=happy:0.6,sad:0.4 shows a mix without the chatbot
    const w = Object.fromEntries(q.get('w').split(',').map((t) => t.split(':')).map(([e, v]) => [e, parseFloat(v)]));
    applyEmotion({ weights: w }, 'mix', 0);
  }
  if (q.has('emotion')) applyEmotion(await api.get(`/api/pose?emotion=${q.get('emotion')}`), q.get('emotion'));
  if (q.has('noidle')) { driver.nextBlink = 1e9; }
  requestAnimationFrame(frame);
  if (!q.has('nogreet')) greet();
  window.__ready = true;
}

main().catch((e) => { console.error(e); document.getElementById('status').textContent = '오류: ' + e.message; window.__error = String(e); });
