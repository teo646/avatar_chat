import puppeteer from 'puppeteer-core';
const S = process.argv[2];
const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: false, args: ['--no-sandbox', '--use-gl=angle', '--ignore-gpu-blocklist', '--disable-renderer-backgrounding', '--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows', '--disable-features=CalculateNativeWinOcclusion', '--disable-frame-rate-limit', '--disable-gpu-vsync', '--user-data-dir=' + S + '/chrome_prof', '--window-size=1300,900'] });
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 800 }); await page.bringToFront();
await page.goto('http://localhost:8000/?max=' + (process.env.MAX || 0), { waitUntil: 'load' });
await page.waitForFunction('window.__ready || window.__error', { timeout: 120000 });
await new Promise((r) => setTimeout(r, 2000));
const res = await page.evaluate(() => new Promise((resolve) => {
  const a = window.__avatar; const t = []; let last = performance.now(); let n = 0;
  const gl = document.getElementById('avatar').getContext('webgl2');
  function f() { const now = performance.now(); t.push(now - last); last = now; if (++n < 120) requestAnimationFrame(f); else resolve({ meanMs: t.slice(10).reduce((s, x) => s + x, 0) / (t.length - 10), maxMs: Math.max(...t.slice(10)) }); }
  requestAnimationFrame(f);
}));
const gpu = await page.evaluate(() => { const a = window.__avatar; const gl = a.renderer.getContext(); const run = (n) => { const t0 = performance.now(); for (let i = 0; i < n; i++) { a.driver.update(performance.now() / 1000 + i * 0.016); a.placeCamera(); a.avatar.sort(a.camera); a.renderer.render(a.scene, a.camera); gl.finish(); } return (performance.now() - t0) / n; }; run(5); return { fullFrameMs: run(60) }; });
const parts = await page.evaluate(() => { const a = window.__avatar; const gl = a.renderer.getContext(); const T = (f, n = 40) => { f(); gl.finish(); const t0 = performance.now(); for (let i = 0; i < n; i++) { f(); gl.finish(); } return +((performance.now() - t0) / n).toFixed(2); };
  a.placeCamera();
  return { drawOnly: T(() => a.renderer.render(a.scene, a.camera)), updateOnly: T(() => a.driver.update(performance.now() / 1000)), sortOnly: T(() => a.avatar.sort(a.camera)),
           uploadAndDraw: T(() => { a.avatar.texCenter.needsUpdate = true; a.renderer.render(a.scene, a.camera); }),
           sortUploadDraw: T(() => { a.avatar.sort(a.camera); a.renderer.render(a.scene, a.camera); }), canvas: [a.renderer.domElement.width, a.renderer.domElement.height] }; });
console.log('parts (ms):', JSON.stringify(parts));
console.log('full frame (update + sort + draw + gl.finish):', JSON.stringify(gpu));
const cpu = await page.evaluate(() => { const a = window.__avatar; const t0 = performance.now(); for (let i = 0; i < 20; i++) { a.driver.update(performance.now() / 1000 + i); } const u = (performance.now() - t0) / 20; const t1 = performance.now(); for (let i = 0; i < 20; i++) a.avatar.sort(a.camera); return { updateMs: u, sortMs: (performance.now() - t1) / 20 }; });
console.log('frame interval:', JSON.stringify(res), '| cpu per step:', JSON.stringify(cpu), '| renderer:', await page.evaluate(() => { const g = document.getElementById('avatar').getContext('webgl2'); return g.getParameter(g.getExtension('WEBGL_debug_renderer_info').UNMASKED_RENDERER_WEBGL); }));
await browser.close();
