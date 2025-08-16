// record.js
import fs from 'fs';
import path from 'path';
import puppeteer from 'puppeteer';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const jsonFile = process.argv[2];
if (!jsonFile) {
  console.error("Usage: node record.js <json_filename.json>");
  process.exit(1);
}

// 스크린샷 저장 폴더
const framesDir = path.join(__dirname, 'frames');
if (!fs.existsSync(framesDir)) fs.mkdirSync(framesDir);

(async () => {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: [
      '--no-sandbox',
      '--headless=new',
      '--use-angle=vulkan',
      '--enable-features=Vulkan',
      '--disable-vulkan-surface',
      '--enable-unsafe-webgpu',
      '--window-size=1280,720'
    ]
  });
  
  const page = await browser.newPage();
  page.on('console', msg => console.log('BROWSER LOG:', msg.text()));
  page.on('pageerror', err => console.log('BROWSER ERROR:', err));
  page.on('requestfailed', req => console.log('REQUEST FAILED:', req.url(), req.failure()));
  await page.goto(
    `http://localhost:9999/viewer_record_webm.html?file=${encodeURIComponent(jsonFile)}`,
    { waitUntil: 'networkidle2' }
  );

  // 총 프레임 수 확인
  await page.waitForFunction('window.dataReady === true');
  const totalFrames = await page.evaluate(() => window.getTotalFrames());
  console.log(`총 프레임: ${totalFrames}`);

  // 프레임별로 렌더링 및 스크린샷 저장
  for (let i = 0; i < totalFrames; i++) {
    await page.evaluate(idx => window.renderFrame(idx), i);
    await page.screenshot({
      path: path.join(framesDir, `frame_${String(i).padStart(5, '0')}.png`)
    });
    if (i % 10 === 0) console.log(`프레임 ${i}/${totalFrames} 저장`);
  }

  await browser.close();

  console.log('모든 프레임 스크린샷 저장 완료!');
  console.log('다음 명령어로 동영상으로 변환하세요:');
  console.log(`ffmpeg -framerate 30 -i frames/frame_%05d.png -c:v libvpx-vp9 -pix_fmt yuva420p output.webm`);
})();
