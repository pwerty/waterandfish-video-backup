import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import './LandmarkViewer.css';

const LandmarkViewer = ({ data, currentFrame, showCylinders, showLeftHand, showRightHand }) => {
  const canvasRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const poseObjectsRef = useRef([]);
  const leftHandObjectsRef = useRef([]);
  const rightHandObjectsRef = useRef([]);
  const poseLinesRef = useRef([]);
  const leftHandLinesRef = useRef([]);
  const rightHandLinesRef = useRef([]);
  const animationIdRef = useRef(null);

  // 색상 상수
  const POSE_COLOR = 0x00b894;
  const LEFT_COLOR = 0x0984e3;
  const RIGHT_COLOR = 0xd63031;

  // 연결 구조
  const POSE_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],[12,14],[14,16],[16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],[27,29],[28,30],[29,31],[30,32],[27,31],[28,32]
  ];
  const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17],[5,17]
  ];

  // 카메라 컨트롤 변수
  const cameraInitPos = { x: -0.388, y: 0.334, z: -0.655 };
  const cameraInitRot = { x: 0, y: 0, z: 0 };
  const yawRef = useRef(-10.3 * Math.PI / 180);
  const pitchRef = useRef(178.7 * Math.PI / 180);
  const rollRef = useRef(0 * Math.PI / 180);

  // 초기화
  useEffect(() => {
    initLandmarkViewer();
    
    // 전역 함수로 카메라 리셋 함수 등록
    window.resetLandmarkCamera = resetCamera;

    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      // 정리 작업
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, []);

  // 프레임이 변경될 때마다 랜드마크 업데이트
  useEffect(() => {
    if (data && sceneRef.current) {
      loadFrame(currentFrame);
    }
  }, [data, currentFrame, showCylinders, showLeftHand, showRightHand]);

  const initLandmarkViewer = () => {
    const canvas = canvasRef.current;
    const container = canvas.parentElement;

    // 씬 생성
    sceneRef.current = new THREE.Scene();
    sceneRef.current.background = new THREE.Color(0xf0f0f0);

    // 컨테이너 크기에 맞춘 반응형 해상도 설정
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;

    // 카메라 생성
    cameraRef.current = new THREE.PerspectiveCamera(75, containerWidth / containerHeight, 0.001, 1000);
    cameraRef.current.position.set(cameraInitPos.x, cameraInitPos.y, cameraInitPos.z);
    cameraRef.current.rotation.set(cameraInitRot.x, cameraInitRot.y, cameraInitRot.z);
    updateCameraDirection();

    // 렌더러 생성
    rendererRef.current = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });

    // 디바이스 픽셀 비율 고려하여 해상도 설정
    const pixelRatio = Math.min(window.devicePixelRatio, 2);
    rendererRef.current.setPixelRatio(pixelRatio);

    // 반응형 캔버스 크기 설정
    canvas.width = containerWidth * pixelRatio;
    canvas.height = containerHeight * pixelRatio;
    canvas.style.width = '100%';
    canvas.style.height = '100%';

    rendererRef.current.setSize(containerWidth, containerHeight);
    rendererRef.current.setClearColor(0xf0f0f0, 1);

    // 조명 설정
    const ambientLight = new THREE.AmbientLight(0x404040, 0.8);
    sceneRef.current.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
    directionalLight.position.set(1, 1, 1);
    sceneRef.current.add(directionalLight);
    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight2.position.set(-1, 1, -1);
    sceneRef.current.add(directionalLight2);

    // 애니메이션 루프 시작
    animate();
  };

  const animate = () => {
    animationIdRef.current = requestAnimationFrame(animate);
    if (rendererRef.current && sceneRef.current && cameraRef.current) {
      rendererRef.current.render(sceneRef.current, cameraRef.current);
    }
  };

  const updateCameraDirection = () => {
    if (cameraRef.current) {
      cameraRef.current.rotation.order = 'YXZ';
      cameraRef.current.rotation.y = yawRef.current;
      cameraRef.current.rotation.x = pitchRef.current;
      cameraRef.current.rotation.z = rollRef.current;
    }
  };

  const resetCamera = () => {
    if (cameraRef.current) {
      cameraRef.current.position.set(cameraInitPos.x, cameraInitPos.y, cameraInitPos.z);
      yawRef.current = -10.3 * Math.PI / 180;
      pitchRef.current = 178.7 * Math.PI / 180;
      rollRef.current = 0 * Math.PI / 180;
      updateCameraDirection();
      logCameraState('카메라 리셋');
    }
  };

  const logCameraState = (action) => {
    const yawDegrees = (yawRef.current * 180 / Math.PI).toFixed(1);
    const pitchDegrees = (pitchRef.current * 180 / Math.PI).toFixed(1);
    const rollDegrees = (rollRef.current * 180 / Math.PI).toFixed(1);
    console.log(`[${action}] 카메라 상태:`, {
      위치: {
        x: cameraRef.current.position.x.toFixed(3),
        y: cameraRef.current.position.y.toFixed(3),
        z: cameraRef.current.position.z.toFixed(3)
      },
      각도: {
        yaw: yawDegrees + '°',
        pitch: pitchDegrees + '°',
        roll: rollDegrees + '°'
      }
    });
  };

  const clearScene = () => {
    if (!sceneRef.current) return;

    [...poseObjectsRef.current, ...leftHandObjectsRef.current, ...rightHandObjectsRef.current].forEach(obj => {
      sceneRef.current.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    });
    [...poseLinesRef.current, ...leftHandLinesRef.current, ...rightHandLinesRef.current].forEach(line => {
      sceneRef.current.remove(line);
      if (line.geometry) line.geometry.dispose();
      if (line.material) line.material.dispose();
    });
    poseObjectsRef.current = [];
    leftHandObjectsRef.current = [];
    rightHandObjectsRef.current = [];
    poseLinesRef.current = [];
    leftHandLinesRef.current = [];
    rightHandLinesRef.current = [];
  };

  const drawConnectionsCylinder = (landmarks, connections, color, lineArray) => {
    connections.forEach(connection => {
      if (connection[0] < landmarks.length && connection[1] < landmarks.length) {
        const start = landmarks[connection[0]];
        const end = landmarks[connection[1]];
        const startVec = new THREE.Vector3(start[0], start[1], start[2]);
        const endVec = new THREE.Vector3(end[0], end[1], end[2]);
        const distance = startVec.distanceTo(endVec);

        // 메인 원기둥 (더 진한 색상)
        const geometry = new THREE.CylinderGeometry(0.008, 0.008, distance, 12);
        const material = new THREE.MeshLambertMaterial({ 
          color: color, 
          transparent: true, 
          opacity: 0.9,
          emissive: color,
          emissiveIntensity: 0.1
        });
        const cylinder = new THREE.Mesh(geometry, material);
        cylinder.position.copy(startVec).add(endVec).multiplyScalar(0.5);
        const up = new THREE.Vector3(0, 1, 0);
        const quaternion = new THREE.Quaternion().setFromUnitVectors(up, new THREE.Vector3().subVectors(endVec, startVec).normalize());
        cylinder.quaternion.copy(quaternion);
        sceneRef.current.add(cylinder);
        lineArray.push(cylinder);

        // 외곽선 원기둥 (더 큰 반지름, 어두운 색상)
        const outlineGeometry = new THREE.CylinderGeometry(0.012, 0.012, distance, 12);
        const outlineMaterial = new THREE.MeshBasicMaterial({ 
          color: 0x000000, 
          transparent: true, 
          opacity: 0.6,
          side: THREE.BackSide
        });
        const outlineCylinder = new THREE.Mesh(outlineGeometry, outlineMaterial);
        outlineCylinder.position.copy(cylinder.position);
        outlineCylinder.quaternion.copy(cylinder.quaternion);
        sceneRef.current.add(outlineCylinder);
        lineArray.push(outlineCylinder);
      }
    });
  };

  const loadFrame = (frameIndex) => {
    if (!data || !sceneRef.current) return;
    console.log("data loading!");
    clearScene();

    // 랜드마크 회전 함수 (Y축 중심 시계방향 180도)
    const rotateLandmark = (lm) => {
      return [-lm[0], lm[1], -lm[2]]; // x와 z 좌표 반전
    };

    // Left Hand
    if (data.left_hand && data.left_hand[frameIndex] && showLeftHand) {
      const rotatedLeftHand = data.left_hand[frameIndex].map(rotateLandmark);
      if (showCylinders) {
        drawConnectionsCylinder(rotatedLeftHand, HAND_CONNECTIONS, LEFT_COLOR, leftHandLinesRef.current);
      }
    }

    // Right Hand
    if (data.right_hand && data.right_hand[frameIndex] && showRightHand) {
      const rotatedRightHand = data.right_hand[frameIndex].map(rotateLandmark);
      if (showCylinders) {
        drawConnectionsCylinder(rotatedRightHand, HAND_CONNECTIONS, RIGHT_COLOR, rightHandLinesRef.current);
      }
    }
  };

  return (
    <div className="left-panel">
      <div className="panel-title">3D 랜드마크 뷰어</div>
      <div className="canvas-container">
        <canvas ref={canvasRef} id="landmark-canvas"></canvas>
      </div>
    </div>
  );
};

export default LandmarkViewer; 