import puppeteer from 'puppeteer-core';
const [url, expr] = process.argv.slice(2);
const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new', args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader'] });
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('[pageerror]', String(e).slice(0, 400)));
await page.goto(url, { waitUntil: 'load' });
await page.waitForFunction('window.__ready || window.__error', { timeout: 120000 });
console.log(JSON.stringify(await page.evaluate(expr), null, 0));
await browser.close();
