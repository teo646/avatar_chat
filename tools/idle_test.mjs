import puppeteer from 'puppeteer-core';
const [out] = process.argv.slice(2);
const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new', args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader'] });
const page = await browser.newPage();
await page.setViewport({ width: 1100, height: 760 });
await page.goto('http://localhost:8000/?noidle&hold=0', { waitUntil: 'load' });
await page.waitForFunction('window.__ready || window.__error', { timeout: 120000 });
const shot = async (name) => { await new Promise((r) => setTimeout(r, 3500)); await (await page.$('#avatar')).screenshot({ path: `${out}/${name}.png` }); };
await shot('idle_open');
await page.evaluate(() => { window.__avatar.driver.forceBlink = 1; });
await shot('idle_blink');
await page.evaluate(() => { const d = window.__avatar.driver; d.forceBlink = 0; d.thinking = true; });
// thinking nod: sample the pitch rotation of the neck matrix over time
const samples = await page.evaluate(() => new Promise((res) => { const d = window.__avatar.driver; const s = []; const t0 = performance.now(); const f = () => { s.push([+(((performance.now() - t0) / 1000)).toFixed(2), +d.AEff[22].toFixed(4)]); if (performance.now() - t0 < 4000) setTimeout(f, 200); else res(s); }; f(); }));
console.log('A[neck] [1][2] element (pitch coupling) while thinking:', JSON.stringify(samples.map((x) => x[1])));
await browser.close();
