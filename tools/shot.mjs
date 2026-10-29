// node shot.mjs "<url>" out.png [width height] [waitMs] [clickSelector...]   headless Chrome screenshot with console log
import puppeteer from 'puppeteer-core';
const [url, out, w = '1280', h = '800', wait = '3000'] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: '/usr/bin/google-chrome', headless: process.env.HEADFUL ? false : 'new',
  args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=' + (process.env.ANGLE || 'swiftshader'), '--ignore-gpu-blocklist', ...(process.env.EXTRA ? process.env.EXTRA.split(' ') : [])],
});
const page = await browser.newPage();
await page.setViewport({ width: +w, height: +h, deviceScaleFactor: 1 });
page.on('console', (m) => console.log('[console]', m.text().slice(0, 300)));
page.on('pageerror', (e) => console.log('[pageerror]', String(e).slice(0, 400)));
await page.goto(url, { waitUntil: 'load' });
await page.waitForFunction('window.__ready || window.__error', { timeout: 120000 });
await new Promise((r) => setTimeout(r, +wait));
console.log('gl:', await page.evaluate(() => { const c = document.getElementById('avatar'); const g = c.getContext('webgl2'); return g ? g.getParameter(g.VERSION) + ' | ' + (g.getExtension('WEBGL_debug_renderer_info') ? g.getParameter(g.getExtension('WEBGL_debug_renderer_info').UNMASKED_RENDERER_WEBGL) : '') + ' | float tex: ' + !!g.getExtension('EXT_color_buffer_float') : 'no webgl2'; }));
console.log('state:', await page.evaluate(() => { const a = window.__avatar; const p = a.avatar.pos; let mn = [1e9,1e9,1e9], mx = [-1e9,-1e9,-1e9]; for (let i = 0; i < a.avatar.N; i++) for (let c = 0; c < 3; c++) { mn[c] = Math.min(mn[c], p[3*i+c]); mx[c] = Math.max(mx[c], p[3*i+c]); } return JSON.stringify({ mn, mx, cam: a.camera.position.toArray(), fov: a.camera.fov, nan: p.some(Number.isNaN) }); }));
console.log('fps text:', await page.$eval('#fps', (e) => e.textContent));
await page.screenshot({ path: out });
await browser.close();
