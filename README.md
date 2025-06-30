# Team 5 - Water and Fish Video Project

3D 랜드마크 시각화 및 실시간 웹캠 비교 프로젝트입니다.

## 프로젝트 구조

```
team5-waterandfish-Video/
├── prototypeViewer.html          # 기존 HTML 버전
├── prototype-viewer-react/       # React 컴포넌트 버전
│   ├── src/
│   │   ├── components/           # React 컴포넌트들
│   │   ├── App.js               # 메인 앱 컴포넌트
│   │   └── index.js             # 앱 진입점
│   ├── package.json             # React 프로젝트 설정
│   └── README.md                # React 프로젝트 설명
├── result/                      # 랜드마크 데이터 JSON 파일들
├── unified_landmark_pipeline.py # Python 랜드마크 처리 파이프라인
├── requirements.txt             # Python 의존성
└── README.md                    # 이 파일
```

## 주요 기능

### 1. 3D 랜드마크 뷰어
- Three.js를 사용한 3D 손 랜드마크 시각화
- 프레임별 애니메이션 재생
- 카메라 컨트롤 (회전, 줌)
- 왼손/오른손 구분 표시

### 2. 실시간 웹캠
- MediaPipe를 사용한 실시간 손 감지
- 웹캠 스트림과 랜드마크 오버레이
- 왼손/오른손 자동 구분

### 3. 애니메이션 컨트롤
- 프레임별 재생/정지
- 재생 속도 조절
- 슬라이더를 통한 프레임 이동
- 비디오 파일 선택

## 버전별 특징

### HTML 버전 (prototypeViewer.html)
- 단일 파일로 구성된 독립 실행형 애플리케이션
- CDN을 통한 라이브러리 로드
- 브라우저에서 직접 실행 가능

### React 버전 (prototype-viewer-react/)
- 컴포넌트 기반 아키텍처
- 모듈화된 코드 구조
- 상태 관리 최적화
- 유지보수성 향상

## 설치 및 실행

### HTML 버전
```bash
# 브라우저에서 직접 열기
open prototypeViewer.html
```

### React 버전
```bash
cd prototype-viewer-react
npm install
npm start
```

### Python 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 사용된 기술

### Frontend
- **HTML/CSS/JavaScript**: 기본 웹 기술
- **React**: UI 라이브러리 (React 버전)
- **Three.js**: 3D 그래픽 라이브러리
- **MediaPipe**: 손 감지 및 랜드마크 추출

### Backend
- **Python**: 데이터 처리 및 분석
- **OpenCV**: 컴퓨터 비전 라이브러리
- **NumPy**: 수치 계산 라이브러리

## 데이터 형식

랜드마크 데이터는 JSON 형식으로 저장됩니다:

```json
{
  "pose": [
    [[x, y, z], [x, y, z], ...],  // 프레임 0
    [[x, y, z], [x, y, z], ...],  // 프레임 1
    ...
  ],
  "left_hand": [
    [[x, y, z], [x, y, z], ...],  // 프레임 0
    [[x, y, z], [x, y, z], ...],  // 프레임 1
    ...
  ],
  "right_hand": [
    [[x, y, z], [x, y, z], ...],  // 프레임 0
    [[x, y, z], [x, y, z], ...],  // 프레임 1
    ...
  ]
}
```

## 브라우저 지원

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## 개발 팀

Team 5 - Water and Fish

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 