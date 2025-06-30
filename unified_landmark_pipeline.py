import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import json
import time
import glob
from datetime import datetime
import sys
import argparse
from scipy.signal import savgol_filter

# 설정
RESULT_DIR = 'result'

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

def print_header(title):
    """섹션 헤더 출력"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_step(step_num, description):
    """단계별 진행상황 출력"""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 40)

def check_video_file(video_path):
    """비디오 파일 존재 및 유효성 확인"""
    print_header("비디오 파일 확인")
    
    if not os.path.exists(video_path):
        print(f"❌ 비디오 파일이 존재하지 않습니다: {video_path}")
        return False
    
    if not video_path.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
        print(f"❌ 지원하지 않는 비디오 형식입니다: {video_path}")
        print("지원 형식: .avi, .mp4, .mov, .mkv")
        return False
    
    # 비디오 파일 열기 테스트
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 비디오 파일을 열 수 없습니다: {video_path}")
        return False
    
    # 비디오 정보 출력
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"✅ 비디오 파일 확인됨: {os.path.basename(video_path)}")
    print(f"📊 비디오 정보:")
    print(f"   - 프레임 수: {frame_count:,}")
    print(f"   - FPS: {fps:.2f}")
    print(f"   - 재생 시간: {duration:.2f}초")
    
    cap.release()
    return True

def extract_landmarks_from_video(video_path):
    """비디오에서 랜드마크 추출"""
    print(f"🎬 랜드마크 추출 시작: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    pose_list, left_hand_list, right_hand_list = [], [], []
    
    # 총 프레임 수 계산
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames, desc="프레임 처리") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            
            def lm_to_np(lm, count):
                if lm:
                    return np.array([[p.x, p.y, p.z] for p in lm.landmark])
                else:
                    return np.zeros((count, 3))
            
            pose_list.append(lm_to_np(results.pose_landmarks, 33))
            left_hand_list.append(lm_to_np(results.left_hand_landmarks, 21))
            right_hand_list.append(lm_to_np(results.right_hand_landmarks, 21))
            
            pbar.update(1)
    
    cap.release()
    
    landmarks = {
        'pose': np.stack(pose_list) if pose_list else np.zeros((0,33,3)),
        'left_hand': np.stack(left_hand_list) if left_hand_list else np.zeros((0,21,3)),
        'right_hand': np.stack(right_hand_list) if right_hand_list else np.zeros((0,21,3)),
    }
    
    # 랜드마크 후처리 적용
    print(f"🔧 랜드마크 후처리 시작...")
    landmarks = postprocess_landmarks(landmarks)
    print(f"✅ 랜드마크 후처리 완료")
    
    print(f"✅ 랜드마크 추출 완료:")
    print(f"   - pose: {len(landmarks['pose'])}프레임")
    print(f"   - left_hand: {len(landmarks['left_hand'])}프레임")
    print(f"   - right_hand: {len(landmarks['right_hand'])}프레임")
    
    return landmarks

def ensure_result_directory():
    """결과 디렉토리 생성"""
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        print(f"📁 결과 디렉토리 생성: {RESULT_DIR}")
    else:
        print(f"📁 결과 디렉토리 확인: {RESULT_DIR}")

def save_landmarks_to_json(landmarks, video_path, suffix=''):
    """랜드마크를 JSON 파일로 저장 (suffix로 파일명 구분)"""
    ensure_result_directory()
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    json_filename = os.path.join(RESULT_DIR, f"{base_name}{suffix}.json")
    json_data = {k: v.tolist() for k, v in landmarks.items()}
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON 파일 저장 완료: {json_filename}")
    return json_filename

def show_results(json_filename):
    """결과 파일 확인"""
    print_step(4, "결과 확인")
    
    if os.path.exists(json_filename):
        print(f"✅ 최종 결과 파일: {json_filename}")
        
        # JSON 파일 내용 확인
        try:
            with open(json_filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"📊 추출된 랜드마크 정보:")
            for key, value in data.items():
                if isinstance(value, list):
                    print(f"   - {key}: {len(value)}프레임")
                else:
                    print(f"   - {key}: {value}")
                    
        except Exception as e:
            print(f"⚠️  결과 파일 분석 중 오류: {e}")
    else:
        print("❌ JSON 파일이 생성되지 않았습니다.")

def postprocess_landmarks(all_landmarks,
                          max_gap=5,          # 보간 허용 프레임 길이
                          ma_win=3,           # 이동평균 윈도우
                          sg_win=7, sg_poly=2 # Savitzky–Golay 파라미터
                          ):
    """
    누락(NaN/0) 보간  ▸  중앙값 이상치 제거  ▸  Savitzky–Golay 스무딩
    """
    def is_missing(pt):
        return (pt == 0).all() or np.isnan(pt).any()

    def linear_interp(seq):
        """seq: (T, D) 배열, 누락(0 또는 NaN)을 선형보간으로 채움"""
        seq = seq.copy()
        T = seq.shape[0]
        valid = ~np.apply_along_axis(is_missing, 1, seq)
        idx = np.arange(T)

        # 누락 전부면 생략
        if not valid.any(): 
            return seq

        # 앞뒤 valid 인덱스 추출
        valid_idx = idx[valid]
        for d in range(seq.shape[1]):
            seq[:, d] = np.interp(idx, valid_idx, seq[valid, d])
        return seq

    processed_landmarks = {}
    
    for key, landmarks in all_landmarks.items():
        if landmarks is None or len(landmarks) == 0:
            processed_landmarks[key] = landmarks
            continue
            
        # (T, N, 3) 형태로 변환
        if len(landmarks.shape) == 2:
            landmarks = landmarks.reshape(1, -1, 3)
        
        T, N, D = landmarks.shape
        processed = np.zeros_like(landmarks)
        
        # 각 랜드마크 포인트별로 처리
        for n in range(N):
            seq = landmarks[:, n, :]  # (T, 3)
            
            # 1. 선형 보간
            seq = linear_interp(seq)
            
            # 2. 중앙값 이상치 제거 (이동평균 윈도우 사용)
            if ma_win > 1 and T > ma_win:
                for d in range(D):
                    # 이동평균 계산
                    ma = np.convolve(seq[:, d], np.ones(ma_win)/ma_win, mode='same')
                    # 중앙값과의 차이 계산
                    diff = np.abs(seq[:, d] - ma)
                    # 이상치 임계값 (표준편차의 2배)
                    threshold = 2 * np.std(diff)
                    # 이상치를 이동평균으로 대체
                    outliers = diff > threshold
                    seq[outliers, d] = ma[outliers]
            
            # 3. Savitzky-Golay 스무딩
            if sg_win > 1 and T > sg_win:
                for d in range(D):
                    try:
                        seq[:, d] = savgol_filter(seq[:, d], sg_win, sg_poly)
                    except:
                        # 스무딩 실패 시 원본 유지
                        pass
            
            processed[:, n, :] = seq
        
        processed_landmarks[key] = processed
    
    return processed_landmarks

def main():
    """메인 파이프라인 실행"""
    parser = argparse.ArgumentParser(
        description="비디오에서 MediaPipe 랜드마크를 추출하여 JSON으로 저장하는 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python unified_landmark_pipeline.py video.avi
  python unified_landmark_pipeline.py /path/to/video.mp4
  python unified_landmark_pipeline.py --help

결과 파일은 'result' 디렉토리에 저장됩니다.
        """
    )
    
    parser.add_argument(
        'video_path',
        help='처리할 비디오 파일의 경로 (.avi, .mp4, .mov, .mkv 지원)'
    )
    
    parser.add_argument(
        '--keep-npz',
        action='store_true',
        help='중간 NPZ 파일을 삭제하지 않고 유지'
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print_header("비디오 랜드마크 추출 파이프라인")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"입력 파일: {args.video_path}")
    print(f"결과 디렉토리: {RESULT_DIR}")
    
    try:
        # Step 1: 비디오 파일 확인
        if not check_video_file(args.video_path):
            print("\n❌ 비디오 파일 확인에 실패했습니다. 프로그램을 종료합니다.")
            return
        
        # Step 2: 랜드마크 추출
        landmarks = extract_landmarks_from_video(args.video_path)
        
        # Step 3-1: 원본(raw) 저장
        raw_json_filename = save_landmarks_to_json(landmarks, args.video_path, suffix='_raw')
        
        # Step 3-2: 후보정(postprocess) 적용 및 저장
        post_landmarks = postprocess_landmarks(landmarks)
        converted_json_filename = save_landmarks_to_json(post_landmarks, args.video_path, suffix='_converted')
        
        # Step 4: 결과 확인 (원본/후보정 모두)
        show_results(raw_json_filename)
        show_results(converted_json_filename)
        
        # 완료 메시지
        total_time = time.time() - start_time
        print_header("파이프라인 완료")
        print(f"✅ 모든 작업이 성공적으로 완료되었습니다!")
        print(f"⏱️  총 소요시간: {total_time:.2f}초")
        print(f"🎯 결과 파일: {raw_json_filename}")
        print(f"📁 결과 디렉토리: {os.path.abspath(RESULT_DIR)}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n❌ 예상치 못한 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 