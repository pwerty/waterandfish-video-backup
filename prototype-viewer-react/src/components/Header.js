import React from 'react';
import './Header.css';

const Header = ({
  selectedVideo,
  currentFrame,
  isPlaying,
  animationSpeed,
  allData,
  showCylinders,
  showLeftHand,
  showRightHand,
  onPreviousFrame,
  onNextFrame,
  onPlayAnimation,
  onSliderChange,
  onSpeedChange,
  onVideoChange,
  onResetCamera,
  onShowCylindersChange,
  onShowLeftHandChange,
  onShowRightHandChange
}) => {
  const maxFrame = allData && allData[selectedVideo] ? allData[selectedVideo].pose.length - 1 : 0;

  return (
    <div className="header">
      <div className="header-content">
        <div className="header-title">
          <h1>Prototype Viewer</h1>
          <p>랜드마크 애니메이션 vs 실시간 웹캠 비교</p>
        </div>
        <div className="header-controls">
          <div className="control-group">
            <label>비디오:</label>
            <select 
              value={selectedVideo} 
              onChange={(e) => onVideoChange(e.target.value)}
            >
              {allData && Object.keys(allData).map(filename => (
                <option key={filename} value={filename}>
                  {filename}
                </option>
              ))}
            </select>
          </div>
          <div className="control-group">
            <label>프레임:</label>
            <span>{currentFrame}</span>
          </div>
          <div className="control-buttons">
            <button onClick={onPreviousFrame}>◀</button>
            <button onClick={onPlayAnimation}>
              {isPlaying ? '⏸ 정지' : '▶ 재생'}
            </button>
            <button onClick={onNextFrame}>▶</button>
            <button onClick={onResetCamera}>🔄</button>
          </div>
          <div className="control-group">
            <label>프레임:</label>
            <input 
              type="range" 
              min="0" 
              max={maxFrame} 
              value={currentFrame} 
              style={{ width: '120px' }} 
              onChange={(e) => onSliderChange(e.target.value)}
            />
          </div>
          <div className="control-group">
            <label>속도:</label>
            <input 
              type="range" 
              min="1" 
              max="30" 
              value={animationSpeed} 
              onChange={(e) => onSpeedChange(e.target.value)} 
              style={{ width: '80px' }}
            />
            <span>{animationSpeed}</span>
          </div>
          <div className="control-toggles">
            <label>
              <input 
                type="checkbox" 
                checked={showCylinders} 
                onChange={(e) => onShowCylindersChange(e.target.checked)}
              /> 
              원기둥(선)
            </label>
            <label>
              <input 
                type="checkbox" 
                checked={showLeftHand} 
                onChange={(e) => onShowLeftHandChange(e.target.checked)}
              /> 
              좌측 손
            </label>
            <label>
              <input 
                type="checkbox" 
                checked={showRightHand} 
                onChange={(e) => onShowRightHandChange(e.target.checked)}
              /> 
              우측 손
            </label>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Header; 