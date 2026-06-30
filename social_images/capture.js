const puppeteer = require('puppeteer');
const path = require('path');

const files = [
  'tidyview_pipeline',
  'regex_cjclang',
  'tidyview_operations',
];

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 630, deviceScaleFactor: 2 });

  for (const name of files) {
    const html = 'file:///' + path.resolve(__dirname, name + '.html').replace(/\\/g, '/');
    const out  = path.resolve(__dirname, name + '.jpg');

    await page.goto(html, { waitUntil: 'networkidle0' });
    await page.screenshot({ path: out, type: 'jpeg', quality: 95, clip: { x:0, y:0, width:1200, height:630 } });
    console.log('wrote', out);
  }

  await browser.close();
})();
