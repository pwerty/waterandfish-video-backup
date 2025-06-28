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

def save_to_npz(landmarks, video_path):
    """랜드마크를 NPZ 파일로 저장"""
    print_step(1, "NPZ 파일 저장")
    
    # 결과 디렉토리 확인
    ensure_result_directory()
    
    # 출력 파일명 생성
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    npz_filename = os.path.join(RESULT_DIR, f"{base_name}_landmarks.npz")
    
    print(f"💾 NPZ 파일 저장 중: {npz_filename}")
    np.savez_compressed(npz_filename, **landmarks)
    print(f"✅ NPZ 파일 저장 완료: {npz_filename}")
    
    return npz_filename

def convert_npz_to_json(npz_filename):
    """NPZ 파일을 JSON으로 변환"""
    print_step(2, "JSON 변환")
    
    print(f"🔄 NPZ 파일 로드 중: {npz_filename}")
    npz_data = np.load(npz_filename)
    
    json_data = {}
    for key in npz_data.keys():
        json_data[key] = npz_data[key].tolist()
    
    # JSON 파일명 생성 (result 디렉토리에 저장)
    base_name = os.path.splitext(os.path.basename(npz_filename))[0]
    json_filename = os.path.join(RESULT_DIR, f"{base_name}.json")
    
    print(f"💾 JSON 파일 저장 중: {json_filename}")
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ JSON 파일 저장 완료: {json_filename}")
    
    # 파일 크기 정보 출력
    file_size = os.path.getsize(json_filename) / (1024 * 1024)  # MB
    print(f"📊 JSON 파일 크기: {file_size:.2f} MB")
    
    return json_filename

def cleanup_npz_file(npz_filename):
    """NPZ 파일 정리"""
    print_step(3, "중간 파일 정리")
    
    if os.path.exists(npz_filename):
        try:
            os.remove(npz_filename)
            print(f"🗑️  {npz_filename} 제거됨")
        except Exception as e:
            print(f"⚠️  {npz_filename} 제거 실패: {e}")
    else:
        print("정리할 NPZ 파일이 없습니다.")

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
        
        # Step 3: NPZ 파일 저장
        npz_filename = save_to_npz(landmarks, args.video_path)
        
        # Step 4: JSON 변환
        json_filename = convert_npz_to_json(npz_filename)
        
        # Step 5: 중간 파일 정리 (옵션)
        if not args.keep_npz:
            cleanup_npz_file(npz_filename)
        else:
            print(f"💾 NPZ 파일 유지: {npz_filename}")
        
        # Step 6: 결과 확인
        show_results(json_filename)
        
        # 완료 메시지
        total_time = time.time() - start_time
        print_header("파이프라인 완료")
        print(f"✅ 모든 작업이 성공적으로 완료되었습니다!")
        print(f"⏱️  총 소요시간: {total_time:.2f}초")
        print(f"🎯 결과 파일: {json_filename}")
        print(f"📁 결과 디렉토리: {os.path.abspath(RESULT_DIR)}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n❌ 예상치 못한 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 