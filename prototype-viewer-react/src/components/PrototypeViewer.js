import React, { useState, useEffect, useRef } from 'react';
import './PrototypeViewer.css';
import Header from './Header';
import LandmarkViewer from './LandmarkViewer';
import WebcamViewer from './WebcamViewer';

const PrototypeViewer = () => {
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [animationSpeed, setAnimationSpeed] = useState(5);
  const [selectedVideo, setSelectedVideo] = useState('');
  const [allData, setAllData] = useState(null);
  const [data, setData] = useState(null);
  const [showCylinders, setShowCylinders] = useState(true);
  const [showLeftHand, setShowLeftHand] = useState(true);
  const [showRightHand, setShowRightHand] = useState(true);
  
  const animationIntervalRef = useRef(null);

  // 데이터 로드
  useEffect(() => {
    loadLandmarkData();
  }, []);

  // 선택된 비디오가 변경될 때 데이터 업데이트
  useEffect(() => {
    if (allData && selectedVideo && allData[selectedVideo]) {
      setData(allData[selectedVideo]);
    }
  }, [selectedVideo, allData]);

  // 애니메이션 재생/정지 처리
  useEffect(() => {
    if (isPlaying) {
      animationIntervalRef.current = setInterval(() => {
        if (data && currentFrame < data.pose.length - 1) {
          setCurrentFrame(prev => prev + 1);
        } else {
          setCurrentFrame(0);
        }
      }, 1000 / animationSpeed);
    } else {
      if (animationIntervalRef.current) {
        clearInterval(animationIntervalRef.current);
        animationIntervalRef.current = null;
      }
    }

    return () => {
      if (animationIntervalRef.current) {
        clearInterval(animationIntervalRef.current);
      }
    };
  }, [isPlaying, animationSpeed, data, currentFrame]);

  const loadLandmarkData = async () => {
    try {
      // 파일 목록을 files.json에서 불러옴
      const response = await fetch('/result/files.json');
      const jsonFiles = await response.json();

      if (jsonFiles.length === 0) {
        throw new Error('result 디렉토리에 JSON 파일이 없습니다.');
      }

      // 모든 JSON 파일들을 로드하여 allData에 저장
      const loadPromises = jsonFiles.map(filename =>
        fetch(`/result/${filename}`)
          .then(response => response.json())
          .then(data => ({ filename, data }))
      );

      const results = await Promise.all(loadPromises);

      // allData를 파일명을 키로 하는 객체로 구성
      const newAllData = {};
      results.forEach(({ filename, data }) => {
        const baseName = filename.replace('_landmarks.json', '');
        newAllData[baseName] = data;
      });

      setAllData(newAllData);

      // 첫 번째 파일을 기본 선택
      const firstFile = Object.keys(newAllData)[0];
      setSelectedVideo(firstFile);

    } catch (error) {
      console.error('데이터 로드 실패:', error);
    }
  };

  const handlePreviousFrame = () => {
    if (currentFrame > 0) {
      setCurrentFrame(currentFrame - 1);
    }
  };

  const handleNextFrame = () => {
    if (data && currentFrame < data.pose.length - 1) {
      setCurrentFrame(currentFrame + 1);
    }
  };

  const handlePlayAnimation = () => {
    setIsPlaying(!isPlaying);
  };

  const handleSliderChange = (value) => {
    setCurrentFrame(parseInt(value));
  };

  const handleSpeedChange = (value) => {
    setAnimationSpeed(parseInt(value));
  };

  const handleVideoChange = (videoName) => {
    setSelectedVideo(videoName);
  };

  const handleResetCamera = () => {
    // LandmarkViewer에서 카메라 리셋 처리
    if (window.resetLandmarkCamera) {
      window.resetLandmarkCamera();
    }
  };

  return (
    <div className="prototype-viewer">
      <Header
        selectedVideo={selectedVideo}
        currentFrame={currentFrame}
        isPlaying={isPlaying}
        animationSpeed={animationSpeed}
        allData={allData}
        showCylinders={showCylinders}
        showLeftHand={showLeftHand}
        showRightHand={showRightHand}
        onPreviousFrame={handlePreviousFrame}
        onNextFrame={handleNextFrame}
        onPlayAnimation={handlePlayAnimation}
        onSliderChange={handleSliderChange}
        onSpeedChange={handleSpeedChange}
        onVideoChange={handleVideoChange}
        onResetCamera={handleResetCamera}
        onShowCylindersChange={setShowCylinders}
        onShowLeftHandChange={setShowLeftHand}
        onShowRightHandChange={setShowRightHand}
      />
      
      
      <div className="main-container">
        <LandmarkViewer
          data={data}
          currentFrame={currentFrame}
          showCylinders={showCylinders}
          showLeftHand={showLeftHand}
          showRightHand={showRightHand}
        />
        <WebcamViewer />
      </div>
    </div>
  );
};

export default PrototypeViewer; 