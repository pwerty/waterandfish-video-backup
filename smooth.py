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
from scipy.interpolate import PchipInterpolator
from pykalman import KalmanFilter
import pandas as pd

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
    #landmarks = postprocess_landmarks(landmarks)
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

def postprocess_landmarks(
    all_landmarks,
    hampel_win=5,
    hampel_sig=3.0,
    kalman_trans_cov=1e-4,
    kalman_obs_cov=1e-2
):
    """
    랜드마크 데이터에 대해 PCHIP 보간, Hampel 필터, Kalman 평활화를 순차적으로 적용합니다.

    Args:
        all_landmarks (dict): {'pose', 'left_hand', 'right_hand'} 키를 가진 랜드마크 딕셔너리.
                              각 값은 (T, N, 3) 형태의 NumPy 배열입니다.
        hampel_win (int): Hampel 필터의 윈도우 크기 (홀수 권장).
        hampel_sig (float): Hampel 필터의 이상치 탐지 시그마 값.
        kalman_trans_cov (float): Kalman 필터의 상태 전이 공분산.
        kalman_obs_cov (float): Kalman 필터의 관측 공분산.

    Returns:
        dict: 후처리된 랜드마크 딕셔너리.
    """

    def is_missing(pt):
        """점이 누락되었는지(모두 0인지) 확인합니다."""
        return (pt == 0).all()

    def pchip_interp_seq(seq, max_gap_to_interpolate=15): # 최대 10프레임까지만 보간
        """
        (T, D) 시퀀스에서 누락된 행을 PCHIP으로 보간하되,
        결측 구간이 max_gap_to_interpolate보다 길면 보간하지 않습니다.
        """
        T, D = seq.shape
        if T == 0: return seq

        idx = np.arange(T)
        valid_mask = ~np.apply_along_axis(is_missing, 1, seq)
        valid_idx = idx[valid_mask]

        if valid_idx.size < 2: return seq.copy()

        # --- [핵심 추가] 결측 구간 길이 체크 ---
        out = seq.copy()
        missing_indices = np.where(~valid_mask)[0]
        
        # 연속된 결측 구간 찾기
        from itertools import groupby
        from operator import itemgetter
        
        gaps = []
        for k, g in groupby(enumerate(missing_indices), lambda ix: ix[0] - ix[1]):
            gaps.append(list(map(itemgetter(1), g)))

        for gap in gaps:
            # 결측 구간이 너무 길면 보간하지 않고 건너뛰기
            if len(gap) > max_gap_to_interpolate:
                continue
            
            start_idx = gap[0] - 1
            end_idx = gap[-1] + 1

            # 앞뒤에 유효한 데이터가 있어야 보간 가능
            if start_idx >= 0 and end_idx < T and valid_mask[start_idx] and valid_mask[end_idx]:
                x_interp = [start_idx, end_idx]
                y_interp = seq[[start_idx, end_idx]]
                
                # 이 구간에 대해서만 보간 수행
                interpolator = PchipInterpolator(x_interp, y_interp, axis=0, extrapolate=False)
                out[gap] = interpolator(gap)
                
        return out


    def hampel_filter_1d(x, win_size, n_sig):
        """
        1D Hampel 필터: 이상치(outlier)를 윈도우의 중앙값으로 대체합니다.
        """
        L = len(x)
        if L < win_size:
            return x

        k = 1.4826  # 정규분포에 대한 스케일 팩터
        out = x.copy()
        
        # Series 객체로 변환하여 롤링 연산 효율화
        s = pd.Series(x)
        rolling_median = s.rolling(window=win_size, center=True, min_periods=1).median()
        
        def mad_func(x):
            return np.median(np.abs(x - np.median(x)))
        
        rolling_mad = s.rolling(window=win_size, center=True, min_periods=1).apply(mad_func, raw=True)
        
        threshold = n_sig * k * rolling_mad
        outliers_mask = np.abs(s - rolling_median) > threshold
        
        out[outliers_mask] = rolling_median[outliers_mask]
        return out

    def kalman_smooth_seq(seq):
        """
        (T, D) 시퀀스를 Kalman 필터로 평활화합니다.
        """
        if seq.shape[0] == 0:
            return seq

        # 각 좌표(x,y,z)를 독립적인 상태(위치, 속도)로 모델링
        T, D = seq.shape
        smoothed_seq = np.zeros_like(seq)
        
        for d in range(D):
            series = seq[:, d]
            kf = KalmanFilter(
                transition_matrices=[[1, 1], [0, 1]], # 위치, 속도 모델
                observation_matrices=[[1, 0]],
                initial_state_mean=[series[0], 0],
                transition_covariance=kalman_trans_cov * np.eye(2),
                observation_covariance=kalman_obs_cov,
            )
            smoothed_states, _ = kf.smooth(series)
            smoothed_seq[:, d] = smoothed_states[:, 0] # 평활화된 위치 값만 사용
        return smoothed_seq

    # --- 메인 처리 로직 ---
    processed_landmarks = {}
    print_step("3-2", "후보정(PCHIP+Hampel+Kalman) 적용")

    for key, landmarks in all_landmarks.items():
        if landmarks is None or len(landmarks) == 0:
            processed_landmarks[key] = landmarks
            print(f"  - [{key}] 데이터 없음, 건너뛰기")
            continue
        
        print(f"  - [{key}] 랜드마크 처리 중...")
        arr = np.array(landmarks, dtype=float)
        T, N, _ = arr.shape
        processed = np.zeros_like(arr)

        # 각 랜드마크 점(N개)에 대해 파이프라인 적용
        for j in tqdm(range(N), desc=f"   {key} 필터링", leave=False):
            seq3d = arr[:, j, :]  # (T, 3) 시퀀스

            # 1단계: PCHIP 보간
            interp_seq = pchip_interp_seq(seq3d)

            # # 2단계: Hampel 필터 (x, y, z 각각 적용)
            # hampel_seq = np.zeros_like(interp_seq)
            # for d in range(3):
            #     hampel_seq[:, d] = hampel_filter_1d(interp_seq[:, d], win_size=hampel_win, n_sig=hampel_sig)

            # # 3단계: Kalman 평활화
            # smoothed_seq = kalman_smooth_seq(hampel_seq)
            
            processed[:, j, :] = interp_seq

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
        raw_json_filename = save_landmarks_to_json(landmarks, args.video_path, suffix='_raw_landmarks')
        
        # Step 3-2: 후보정(postprocess) 적용 및 저장
        post_landmarks = postprocess_landmarks(landmarks)
        converted_json_filename = save_landmarks_to_json(post_landmarks, args.video_path, suffix='_converted_landmarks')
        
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