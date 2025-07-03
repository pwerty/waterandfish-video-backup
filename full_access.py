"""
수어 인식 모델 설정 파일, 모든 파라미터를 통일적으로 관리합니다.
팀장 컴퓨터 연결 후 진행 가능
"""

# 데이터 처리 파라미터
TARGET_SEQ_LENGTH = 30  # 시퀀스 길이 (프레임 수)

# 캐시 설정
CACHE_DIR = "cache"

# MediaPipe 설정
MEDIAPIPE_STATIC_IMAGE_MODE = False
MEDIAPIPE_MODEL_COMPLEXITY = 1
MEDIAPIPE_SMOOTH_LANDMARKS = True
MEDIAPIPE_ENABLE_SEGMENTATION = False
MEDIAPIPE_SMOOTH_SEGMENTATION = True
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5


# 파일 경로
IDENTICAL_VIDEO_ROOT = "'smb://조현호의 MacBook Air._smb._tcp.local/수어 데이터셋/수어 데이터셋'"

# 비디오 루트 디렉토리 매핑
VIDEO_ROOTS = [
    ((1, 3000), f"{IDENTICAL_VIDEO_ROOT}/0001~3000(영상)"),
    ((3001, 6000), f"{IDENTICAL_VIDEO_ROOT}/3001~6000(영상)"),
    ((6001, 8280), f"{IDENTICAL_VIDEO_ROOT}/6001~8280(영상)"),
    ((8381, 9000), f"{IDENTICAL_VIDEO_ROOT}/8381~9000(영상)"),
    ((9001, 9600), f"{IDENTICAL_VIDEO_ROOT}/9001~9600(영상)"),
    ((9601, 10480), f"{IDENTICAL_VIDEO_ROOT}/9601~10480(영상)"),
    ((10481, 12994), f"{IDENTICAL_VIDEO_ROOT}/10481~12994"),
    ((12995, 15508), f"{IDENTICAL_VIDEO_ROOT}/12995~15508"),
    ((15509, 18022), f"{IDENTICAL_VIDEO_ROOT}/15509~18022"),
    ((18023, 20536), f"{IDENTICAL_VIDEO_ROOT}/18023~20536"),
    ((20537, 23050), f"{IDENTICAL_VIDEO_ROOT}/20537~23050"),
    ((23051, 25564), f"{IDENTICAL_VIDEO_ROOT}/23051~25564"),
    ((25565, 28078), f"{IDENTICAL_VIDEO_ROOT}/25565~28078"),
    ((28079, 30592), f"{IDENTICAL_VIDEO_ROOT}/28079~30592"),
    ((30593, 33106), f"{IDENTICAL_VIDEO_ROOT}/30593~33106"),
    ((33107, 35620), f"{IDENTICAL_VIDEO_ROOT}/33107~35620"),
    ((36878, 40027), f"{IDENTICAL_VIDEO_ROOT}/36878~40027"),
    ((40028, 43177), f"{IDENTICAL_VIDEO_ROOT}/40028~43177"),
]

# 지원하는 비디오 확장자
VIDEO_EXTENSIONS = [".MOV", ".MTS", ".MP4", ".AVI", ".mov", ".mts", ".mp4", ".avi"]

# 로깅 설정
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FILE = "training.log"

# 성능 모니터링
ENABLE_MEMORY_MONITORING = True
ENABLE_PROGRESS_BAR = True

# 비디오 파일 경로 캐시 (성능 최적화)
_video_path_cache = {}

def get_video_path_by_id(file_id):
    """
    파일 ID로부터 실제 비디오 경로 찾기 (캐시 사용)
    """
    if file_id in _video_path_cache:
        return _video_path_cache[file_id]
    
    for (start, end), root_path in VIDEO_ROOTS:
        if start <= file_id <= end:
            for ext in VIDEO_EXTENSIONS:
                potential_path = f"{root_path}/KETI_SL_{file_id:010d}{ext}"
                # 실제 파일 존재 확인은 사용 시점에서 수행
                _video_path_cache[file_id] = potential_path
                return potential_path
    
    return None

def get_video_paths_by_word(word_id_list):
    """
    단어의 고유 번호 리스트로부터 비디오 경로들 반환
    """
    paths = []
    for file_id in word_id_list:
        path = get_video_path_by_id(file_id)
        if path:
            paths.append(path)
    return paths

def clear_video_path_cache():
    """
    비디오 경로 캐시 초기화
    """
    global _video_path_cache
    _video_path_cache.clear()

def get_action_index(label, actions):
    return actions.index(label)