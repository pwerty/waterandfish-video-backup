import React, { useState, useEffect } from 'react';
import LandmarkViewer from './LandmarkViewer';

const SimpleDataViewer = () => {
  const [data, setData] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);

  // 데이터 로드
  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      // 첫 번째 JSON 파일만 로드
      const response = await fetch('/result/KETI_SL_0000000414_landmarks.json');
      const landmarkData = await response.json();
      setData(landmarkData);
    } catch (error) {
      console.error('데이터 로드 실패:', error);
    }
  };

  return (
    <div style={{ width: '100%', height: '100vh' }}>
      <LandmarkViewer
        data={data}
        currentFrame={currentFrame}
        showCylinders={true}
        showLeftHand={true}
        showRightHand={true}
      />
    </div>
  );
};

export default SimpleDataViewer; 