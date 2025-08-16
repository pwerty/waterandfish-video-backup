# Prototype Viewer React

기존 HTML 기반의 Prototype Viewer를 React 컴포넌트로 변환한 프로젝트입니다.

## 주요 기능

- **3D 랜드마크 뷰어**: Three.js를 사용한 3D 손 랜드마크 시각화
- **실시간 웹캠**: MediaPipe를 사용한 실시간 손 감지 및 랜드마크 표시
- **애니메이션 컨트롤**: 프레임별 재생, 속도 조절, 일시정지 기능
- **반응형 디자인**: 다양한 화면 크기에 대응하는 반응형 레이아웃

## 컴포넌트 구조

```
src/
├── components/
│   ├── PrototypeViewer.js      # 메인 컨테이너 컴포넌트
│   ├── PrototypeViewer.css     # 메인 스타일
│   ├── Header.js               # 헤더 컨트롤 컴포넌트
│   ├── Header.css              # 헤더 스타일
│   ├── LandmarkViewer.js       # 3D 랜드마크 뷰어 컴포넌트
│   ├── LandmarkViewer.css      # 랜드마크 뷰어 스타일
│   ├── WebcamViewer.js         # 웹캠 뷰어 컴포넌트
│   └── WebcamViewer.css        # 웹캠 뷰어 스타일
├── App.js                      # 루트 앱 컴포넌트
└── index.js                    # 앱 진입점
```

## 설치 및 실행

### 필수 요구사항
- Node.js (v14 이상)
- npm 또는 yarn

### 설치
```bash
# 의존성 설치
npm install

# 개발 서버 실행
npm start
```

### 빌드
```bash
# 프로덕션 빌드
npm run build
```

## 사용된 라이브러리

- **React**: UI 라이브러리
- **Three.js**: 3D 그래픽 라이브러리
- **@mediapipe/tasks-vision**: MediaPipe 손 감지 라이브러리

## 주요 개선사항

### 1. 컴포넌트화
- 기존 단일 HTML 파일을 여러 React 컴포넌트로 분리
- 각 기능별로 독립적인 컴포넌트 구성
- 재사용 가능한 컴포넌트 구조

### 2. 상태 관리
- React Hooks를 사용한 효율적인 상태 관리
- 컴포넌트 간 데이터 전달 최적화
- 애니메이션 상태 관리 개선

### 3. 성능 최적화
- useRef를 사용한 DOM 요소 직접 접근
- useEffect를 통한 생명주기 관리
- 메모리 누수 방지를 위한 정리 작업

### 4. 코드 구조
- 모듈화된 CSS 파일 구조
- 명확한 컴포넌트 책임 분리
- 유지보수 가능한 코드 구조

## 데이터 구조

프로젝트는 `result/` 디렉토리의 JSON 파일들을 자동으로 로드합니다:

```json
{
  "pose": [...],        // 포즈 랜드마크 데이터
  "left_hand": [...],   // 왼손 랜드마크 데이터
  "right_hand": [...]   // 오른손 랜드마크 데이터
}
```

## 브라우저 지원

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
