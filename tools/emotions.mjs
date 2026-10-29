// Screenshot of the avatar canvas for every emotion (and the sway-free neutral), written to <outdir>/<name>.png
// URL=http://localhost:8001/?noidle&hold=0 picks another server, NAMES=neutral,happy,... the emotions (plus blink)
import puppeteer from 'puppeteer-core';
const [outdir, w = '640', h = '720'] = process.argv.slice(2);
const names = (process.env.NAMES || 'neutral,happy,sad,angry,surprised').split(',');
const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new', args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader'] });
const page = await browser.newPage();
await page.setViewport({ width: 1100, height: +h + 40 });
page.on('pageerror', (e) => console.log('[pageerror]', String(e).slice(0, 400)));
await page.goto(process.env.URL || 'http://localhost:8000/?noidle&hold=0', { waitUntil: 'load' });
await page.waitForFunction('window.__ready || window.__error', { timeout: 120000 });
for (const n of names) {
  // NAMES may contain 'blink': the eyes are held closed (driver.forceBlink) on top of neutral
  await page.evaluate(async (n) => {
    window.__avatar.driver.forceBlink = n === 'blink' ? 1 : null;
    const p = await (await fetch('/api/pose?emotion=' + (n === 'blink' ? 'neutral' : n))).json(); window.__avatar.applyEmotion(p, n);
  }, n);
  await new Promise((r) => setTimeout(r, 3500));
  const el = await page.$('#avatar');
  await el.screenshot({ path: `${outdir}/${n}.png` });
  console.log('saved', n);
}
await browser.close();
