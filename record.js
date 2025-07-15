import fs from 'fs';
import path from 'path';
import puppeteer from 'puppeteer';

import { fileURLToPath } from 'url';
// ESM environment에서 __dirname 정의
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 단일 JSON 파일명을 명령 인자로 전달받음
const jsonFile = process.argv[2];
if (!jsonFile) {
  console.error("Usage: node record.js <json_filename.json>");
  process.exit(1);
}
// (stray top-level await puppeteer.launch removed)
(async () => {
  const browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  const page = await browser.newPage();
  // 페이지에 파일명 파라미터를 포함하여 로드, 네트워크 안정화까지 대기
  await page.goto(`http://localhost:9999/viewer_record_webm.html?file=${encodeURIComponent(jsonFile)}`, { waitUntil: 'networkidle0' });
  // 페이지 초기화 함수가 실행될 시간을 잠시 대기
  await new Promise(resolve => setTimeout(resolve, 2000));
  // mediaRecorder가 정의되고 초기 상태인 'inactive'이 될 때까지 대기
  await page.waitForFunction(
    () => typeof window.mediaRecorder !== 'undefined' && window.mediaRecorder.state === 'inactive',
    { timeout: 60000 }
  );
  // 1) 청크 배열 준비
  const recordedChunks = [];

  // 2) 브라우저→Node 콜백 등록
  // 녹화된 데이터(청크)를 Node.js 환경으로 받기 위한 함수
  await page.exposeFunction('onChunkAvailable', chunkAsUint8Array => {
    recordedChunks.push(Buffer.from(chunkAsUint8Array));
  });
  // 3) 직접 녹화 시작 및 이벤트 핸들러 설정
  await page.evaluate(() => {
    recordedChunks = [];
    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) {
        e.data.arrayBuffer().then(buffer => {
          window.onChunkAvailable(new Uint8Array(buffer));
        });
      }
    };
    return new Promise(resolve => {
      mediaRecorder.onstop = resolve;
      mediaRecorder.start();
      setTimeout(() => mediaRecorder.stop(), 5000);
    });
  });

  // 5) 파일 저장
  const outputPath = path.resolve(__dirname, 'output.webm');
  fs.writeFileSync(outputPath, Buffer.concat(recordedChunks));
  console.log('✅ output.webm 생성 완료:', outputPath);

  await browser.close();
})();
