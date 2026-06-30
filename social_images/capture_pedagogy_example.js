// Captures the "actual pedagogy example" social images to PNG.
//
// Run from social_images/:
//   node capture_pedagogy_example.js
//
// Produces:
//   pedagogy_example_linkedin.png    (1200x630, 2x DPR)
//   pedagogy_example_instagram.png   (1080x1080, 2x DPR)

const puppeteer = require('puppeteer');
const path = require('path');

const targets = [
  { name: 'pedagogy_example_linkedin',   width: 1200, height: 630  },
  { name: 'pedagogy_example_instagram',  width: 1080, height: 1080 },
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
