// wait_record.js
import puppeteer from 'puppeteer';

(async () => {
  const jsonFile = process.argv[2];
  if (!jsonFile) {
    console.error('Usage: node wait_record.js <json_filename.json>');
    process.exit(1);
  }

  // headless: false 로 띄워야 window.close()가 실제로 페이지를 닫습니다.
  const browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage();

  // viewer_record_webm.html 단일 모드로 로드
  await page.goto(
    `http://localhost:9999/viewer_record_webm.html?file=${encodeURIComponent(jsonFile)}`,
    { waitUntil: 'networkidle0' }
  );

  // 페이지의 콘솔에서 'RECORD_FINISHED' 로그를 기다림
  await new Promise((resolve) => {
    page.on('console', (msg) => {
      if (msg.text() === 'RECORD_FINISHED') {
        resolve();
      }
    });
  });

  console.log('✅ 완료');
  await browser.close();
})();