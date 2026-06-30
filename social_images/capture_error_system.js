// Captures the two-fold error system social images to PNG.
//
// Run from the social_images/ directory:
//   node capture_error_system.js
//
// Produces:
//   error_system_linkedin.png    (1200x630, deviceScaleFactor 2 -> 2400x1260 actual)
//   error_system_instagram.png   (1080x1080, deviceScaleFactor 2 -> 2160x2160 actual)
//
// Both PNGs are ready to upload to LinkedIn / Instagram respectively.

const puppeteer = require('puppeteer');
const path = require('path');

const targets = [
  { name: 'error_system_linkedin',   width: 1200, height: 630  },
  { name: 'error_system_instagram',  width: 1080, height: 1080 },
];

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  for (const t of targets) {
    await page.setViewport({
      width: t.width,
      height: t.height,
      deviceScaleFactor: 2,
    });

    const htmlPath = 'file:///' + path.resolve(__dirname, t.name + '.html').replace(/\\/g, '/');
    const outPng   = path.resolve(__dirname, t.name + '.png');
    const outJpg   = path.resolve(__dirname, t.name + '.jpg');

    await page.goto(htmlPath, { waitUntil: 'networkidle0' });

    await page.screenshot({
      path: outPng,
      type: 'png',
      clip: { x: 0, y: 0, width: t.width, height: t.height },
    });

    await page.screenshot({
      path: outJpg,
      type: 'jpeg',
      quality: 95,
      clip: { x: 0, y: 0, width: t.width, height: t.height },
    });

    console.log('wrote', outPng, '+', outJpg);
  }

  await browser.close();
})();
