// Self test of the browser port (incl. the MLP) and a real mouse drag: the head turns, the blend weights follow the head pose.
// node drag_test.mjs <outdir>     (URL=http://localhost:8001/?noidle picks the server)
import puppeteer from 'puppeteer-core';
const [outdir = '.'] = process.argv.slice(2);
const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new', args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader'] });
const page = await browser.newPage();
await page.setViewport({ width: 1100, height: 760 });
page.on('pageerror', (e) => console.log('[pageerror]', String(e).slice(0, 400)));
await page.goto(process.env.URL || 'http://localhost:8001/?noidle&hold=0', { waitUntil: 'load' });
await page.waitForFunction('window.__ready || window.__error', { timeout: 120000 });
console.log('self test', JSON.stringify(await page.evaluate(() => window.__selfTest())));
const st = () => page.evaluate(() => { const d = window.__avatar.driver; return { look: [d.look.yaw, d.look.pitch].map((v) => +(v * 180 / Math.PI).toFixed(1)), pose: [d.pose.yaw, d.pose.pitch].map((v) => +(v * 180 / Math.PI).toFixed(1)), psi0: +d.psi[0].toFixed(4) }; });
const el = await page.$('#avatar'); const box = await el.boundingBox();
const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
const shot = async (n) => { await new Promise((r) => setTimeout(r, 800)); await el.screenshot({ path: `${outdir}/${n}.png` }); console.log(n, JSON.stringify(await st())); };
await shot('drag_0_front');
await page.mouse.move(cx, cy); await page.mouse.down();
for (const [dx, dy] of [[80, 0]]) { for (let i = 1; i <= 8; i++) await page.mouse.move(cx + dx * i / 8, cy + dy * i / 8); }
await shot('drag_1_right');
for (let i = 1; i <= 8; i++) await page.mouse.move(cx + 80 - 200 * i / 8, cy);
await shot('drag_2_left');
for (let i = 1; i <= 8; i++) await page.mouse.move(cx - 120 + 120 * i / 8, cy + 60 * i / 8);
await shot('drag_3_down');
await page.mouse.up(); await new Promise((r) => setTimeout(r, 2500));
await shot('drag_4_released');
await browser.close();
