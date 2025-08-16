import React, { useEffect, useRef, useState } from 'react';
import './WebcamViewer.css';

const WebcamViewer = () => {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const [webcamStatus, setWebcamStatus] = useState('연결 중...');
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [handLandmarker, setHandLandmarker] = useState(null);
  const [webcamCtx, setWebcamCtx] = useState(null);
  const [webcamStream, setWebcamStream] = useState(null);

  useEffect(() => {
    initWebcam();
    return () => {
      // 정리 작업
      if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const initWebcam = async () => {
    try {
      console.log('웹캠 초기화 시작...');
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 }
        }, 
        audio: false 
      });
      
      setWebcamStream(stream);
      videoRef.current.srcObject = stream;
      
      // 비디오 로드 완료 대기
      await new Promise((resolve) => {
        videoRef.current.onloadedmetadata = () => {
          console.log('웹캠 비디오 로드 완료:', videoRef.current.videoWidth, 'x', videoRef.current.videoHeight);
          resolve();
        };
      });
      
      videoRef.current.play();
      console.log('웹캠 재생 시작');
      
      setWebcamStatus('연결됨');
      setIsWebcamActive(true);
      
      // MediaPipe Tasks Vision 초기화
      await initMediaPipeHands();
    } catch (error) {
      console.error('웹캠 접근 실패:', error);
      setWebcamStatus('접근 실패');
      if (videoRef.current) {
        videoRef.current.style.display = 'none';
      }
    }
  };

  const initMediaPipeHands = async () => {
    try {
      console.log('MediaPipe Tasks Vision 모듈 로드 시작...');
      // MediaPipe Tasks Vision 모듈 로드
      const { HandLandmarker, FilesetResolver } = await import("@mediapipe/tasks-vision");
      console.log('MediaPipe 모듈 로드 완료');
      
      console.log('FilesetResolver 초기화...');
      const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
      console.log('FilesetResolver 초기화 완료');
      
      console.log('HandLandmarker 생성 시작...');
      const landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
      });
      console.log('HandLandmarker 생성 완료');
      
      setHandLandmarker(landmarker);
      
      // 웹캠 오버레이 캔버스 설정
      const overlay = overlayRef.current;
      const ctx = overlay.getContext('2d');
      setWebcamCtx(ctx);
      console.log('웹캠 오버레이 캔버스 설정 완료');
      
      // 고정된 웹캠 해상도 설정
      const webcamWidth = 640;
      const webcamHeight = 480;
      
      // 비디오 크기 설정
      console.log('비디오 크기 설정:', webcamWidth, 'x', webcamHeight);
      overlay.width = webcamWidth;
      overlay.height = webcamHeight;
      overlay.style.width = webcamWidth + 'px';
      overlay.style.height = webcamHeight + 'px';
      
      // 비디오 요소 크기 조정
      videoRef.current.style.width = webcamWidth + 'px';
      videoRef.current.style.height = webcamHeight + 'px';
      
      setWebcamStatus('손 감지 활성화');
      console.log('MediaPipe Hands 초기화 완료');
      
      // 즉시 손 감지 시작
      console.log('손 감지 즉시 시작...');
      setTimeout(() => {
        startHandDetection();
      }, 100);
      
    } catch (error) {
      console.error('MediaPipe Hands 초기화 실패:', error);
      setWebcamStatus('손 감지 초기화 실패');
    }
  };

  const startHandDetection = () => {
    console.log('=== 손 감지 시작 함수 호출 ===');
    console.log('handLandmarker 존재:', !!handLandmarker);
    console.log('videoRef.current 존재:', !!videoRef.current);
    console.log('videoRef.current readyState:', videoRef.current ? videoRef.current.readyState : 'N/A');
    console.log('videoRef.current videoWidth:', videoRef.current ? videoRef.current.videoWidth : 'N/A');
    console.log('videoRef.current videoHeight:', videoRef.current ? videoRef.current.videoHeight : 'N/A');
    console.log('overlayRef.current 존재:', !!overlayRef.current);
    console.log('webcamCtx 존재:', !!webcamCtx);
    
    // 조건 확인
    const hasHandLandmarker = !!handLandmarker;
    const hasVideo = !!videoRef.current;
    const videoReady = hasVideo && videoRef.current.readyState >= 2;
    const hasOverlay = !!overlayRef.current && !!webcamCtx;
    
    console.log('조건 확인:', {
      hasHandLandmarker,
      hasVideo,
      videoReady,
      hasOverlay
    });
    
    if (hasHandLandmarker && hasVideo && hasOverlay) {
      console.log('모든 조건 충족! 손 감지 루프 시작');
      detectHands();
    } else {
      console.log('조건 불충족, 1초 후 재시도...');
      // 조건이 충족되지 않으면 1초 후 다시 시도
      setTimeout(startHandDetection, 1000);
    }
  };

  const detectHands = async () => {
    try {
      console.log('손 감지 실행 중... (프레임:', Date.now(), ')');
      
      // 비디오가 준비되지 않았으면 대기
      if (videoRef.current.readyState < 2) {
        console.log('비디오가 아직 준비되지 않음, 다음 프레임에서 재시도');
        requestAnimationFrame(detectHands);
        return;
      }
      
      const results = await handLandmarker.detectForVideo(videoRef.current, performance.now());
      console.log('손 감지 결과:', results);
      
      // 오버레이 캔버스 클리어
      webcamCtx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
      
      // 손 랜드마크 그리기
      if (results && results.landmarks && results.landmarks.length > 0) {
        console.log('감지된 손 개수:', results.landmarks.length);
        console.log('손 구분 정보:', results.handedness);
        
        results.landmarks.forEach((landmarks, handIndex) => {
          // handedness 정보를 사용하여 실제 왼손/오른손 구분
          const handedness = results.handedness && results.handedness[handIndex] ? results.handedness[handIndex][0] : null;
          let color, handType;
          
          if (handedness) {
            if (handedness.categoryName === 'Left') {
              color = '#00FF00'; // 왼손: 초록색
              handType = '왼손';
            } else if (handedness.categoryName === 'Right') {
              color = '#FF0000'; // 오른손: 빨간색
              handType = '오른손';
            } else {
              color = '#FFFF00'; // 알 수 없음: 노란색
              handType = '알 수 없음';
            }
          } else {
            // handedness 정보가 없으면 기존 방식 사용
            color = handIndex === 0 ? '#00FF00' : '#FF0000';
            handType = handIndex === 0 ? '첫 번째 손' : '두 번째 손';
          }
          
          console.log(`손 ${handIndex + 1}: ${handType} (${color})`);
          drawHandLandmarks(landmarks, color);
        });
      } else {
        console.log('감지된 손이 없습니다');
      }
      
      // 다음 프레임 처리
      requestAnimationFrame(detectHands);
    } catch (error) {
      console.error('손 감지 오류:', error);
      // 오류 발생 시에도 계속 시도
      setTimeout(() => {
        requestAnimationFrame(detectHands);
      }, 100);
    }
  };

  const drawHandLandmarks = (landmarks, color) => {
    // 손 연결선 정의
    const HAND_CONNECTIONS = [
      [0,1],[1,2],[2,3],[3,4],
      [0,5],[5,6],[6,7],[7,8],
      [0,9],[9,10],[10,11],[11,12],
      [0,13],[13,14],[14,15],[15,16],
      [0,17],[17,18],[18,19],[19,20],
      [5,9],[9,13],[13,17],[5,17]
    ];
    
    // 연결선 그리기
    HAND_CONNECTIONS.forEach(([start, end]) => {
      const startPoint = landmarks[start];
      const endPoint = landmarks[end];
      
      if (startPoint && endPoint) {
        const startX = startPoint.x * overlayRef.current.width;
        const startY = startPoint.y * overlayRef.current.height;
        const endX = endPoint.x * overlayRef.current.width;
        const endY = endPoint.y * overlayRef.current.height;
        
        webcamCtx.strokeStyle = color;
        webcamCtx.lineWidth = 2;
        webcamCtx.beginPath();
        webcamCtx.moveTo(startX, startY);
        webcamCtx.lineTo(endX, endY);
        webcamCtx.stroke();
      }
    });
    
    // 랜드마크 점 그리기
    landmarks.forEach(landmark => {
      const x = landmark.x * overlayRef.current.width;
      const y = landmark.y * overlayRef.current.height;
      
      webcamCtx.fillStyle = color;
      webcamCtx.beginPath();
      webcamCtx.arc(x, y, 3, 0, 2 * Math.PI);
      webcamCtx.fill();
    });
  };

  return (
    <div className="right-panel">
      <div className="panel-title">실시간 웹캠</div>
      
      <div className="webcam-container">
        <video 
          ref={videoRef} 
          id="webcam-video" 
          autoPlay 
          muted 
          playsInline
        />
        <canvas 
          ref={overlayRef} 
          id="webcam-overlay" 
          style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
        />
        <div className="webcam-overlay">
          <div>웹캠 스트림</div>
          <div>{webcamStatus}</div>
        </div>
        <div className={`status-indicator ${isWebcamActive ? 'active' : ''}`}></div>
      </div>
    </div>
  );
};

export default WebcamViewer; 