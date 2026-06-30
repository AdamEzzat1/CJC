const puppeteer = require('puppeteer');
const path = require('path');

const files = [
  'tidyview_v3_hero',
  'tidyview_v3_benches',
  'tidyview_v3_variance',
  'tidyview_v3_why_stable',
  'tidyview_v3_architecture',
];

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 630, deviceScaleFactor: 2 });

  for (const name of files) {
    const html = 'file:///' + path.resolve(__dirname, name + '.html').replace(/\\/g, '/');
    const outJpg = path.resolve(__dirname, name + '.jpg');
    const outPng = path.resolve(__dirname, name + '.png');

    await page.goto(html, { waitUntil: 'networkidle0' });
    await page.screenshot({ path: outJpg, type: 'jpeg', quality: 95, clip: { x:0, y:0, width:1200, height:630 } });
    await page.screenshot({ path: outPng, type: 'png',  clip: { x:0, y:0, width:1200, height:630 } });
    console.log('wrote', outJpg, '+', outPng);
  }

  await browser.close();
})();
