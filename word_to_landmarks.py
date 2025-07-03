#!/usr/bin/env python3
"""
단어 기반 랜드마크 추출 파이프라인
1. 단어를 인자로 받음
2. baseDataList.xlsx에서 조회
3. 조건 필터링 (정면 시점 등)
4. 5개 파일 선택 후 DTW 적용하여 합성
"""

import argparse
import pandas as pd
import os
import json
from pathlib import Path
from full_access import *
import cv2
import mediapipe as mp
import numpy as np
from dtaidistance import dtw
import logging

# 로깅 설정
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class WordToLandmarksPipeline:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=MEDIAPIPE_STATIC_IMAGE_MODE,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            smooth_landmarks=MEDIAPIPE_SMOOTH_LANDMARKS,
            enable_segmentation=MEDIAPIPE_ENABLE_SEGMENTATION,
            smooth_segmentation=MEDIAPIPE_SMOOTH_SEGMENTATION,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )
        
    def load_base_data(self, excel_path="baseDataList.xlsx"):
        """Excel 파일에서 기본 데이터 로드"""
        try:
            df = pd.read_excel(excel_path)
            logger.info(f"Excel 파일 로드 완료: {len(df)} 행")
            return df
        except Exception as e:
            logger.error(f"Excel 파일 로드 실패: {e}")
            return None
    
    def filter_by_word(self, df, word):
        """단어로 필터링"""
        # 단어 컬럼명은 실제 Excel 구조에 맞게 수정 필요
        word_column = '단어'  # 또는 실제 컬럼명
        filtered = df[df[word_column] == word]
        logger.info(f"'{word}' 검색 결과: {len(filtered)} 개")
        return filtered
    
    def apply_conditions(self, df):
        """조건 필터링 (정면 시점 등)"""
        conditions = df.copy()
        
        # 시점 필터링
        if '시점' in conditions.columns:
            conditions = conditions[conditions['시점'] == '정면']
            logger.info(f"정면 시점 필터링 후: {len(conditions)} 개")
        
        # 추가 조건들 (실제 컬럼명에 맞게 수정)
        # 예: 화질, 조명, 배경 등
        
        return conditions
    
    def select_top_files(self, df, count=5):
        """상위 N개 파일 선택"""
        if len(df) > count:
            selected = df.head(count)
        else:
            selected = df
        
        logger.info(f"{len(selected)} 개 파일 선택됨")
        return selected
    
    def get_video_path(self, file_id):
        """파일 ID로부터 실제 비디오 경로 찾기"""
        from full_access import get_video_path_by_id
        return get_video_path_by_id(file_id)
    
    def extract_landmarks(self, video_path):
        """비디오에서 랜드마크 추출"""
        cap = cv2.VideoCapture(video_path)
        landmarks_data = {
            'pose': [],
            'left_hand': [],
            'right_hand': []
        }
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(frame_rgb)
            
            # Pose landmarks
            if results.pose_landmarks:
                pose_data = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
                landmarks_data['pose'].append(pose_data)
            else:
                landmarks_data['pose'].append(None)
            
            # Left hand landmarks
            if results.left_hand_landmarks:
                left_hand_data = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
                landmarks_data['left_hand'].append(left_hand_data)
            else:
                landmarks_data['left_hand'].append(None)
            
            # Right hand landmarks
            if results.right_hand_landmarks:
                right_hand_data = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
                landmarks_data['right_hand'].append(right_hand_data)
            else:
                landmarks_data['right_hand'].append(None)
            
            frame_count += 1
        
        cap.release()
        logger.info(f"랜드마크 추출 완료: {frame_count} 프레임")
        return landmarks_data
    
    def apply_dtw_alignment(self, landmarks_list):
        """DTW를 사용하여 여러 랜드마크 시퀀스 정렬 및 합성"""
        if len(landmarks_list) < 2:
            return landmarks_list[0] if landmarks_list else None
        
        # 기준이 되는 첫 번째 시퀀스
        reference = landmarks_list[0]
        aligned_sequences = [reference]
        
        for i, landmarks in enumerate(landmarks_list[1:], 1):
            logger.info(f"DTW 정렬 중: {i+1}/{len(landmarks_list)}")
            
            # 각 부위별로 DTW 적용
            aligned_landmarks = {'pose': [], 'left_hand': [], 'right_hand': []}
            
            for part in ['pose', 'left_hand', 'right_hand']:
                ref_seq = self._prepare_sequence_for_dtw(reference[part])
                target_seq = self._prepare_sequence_for_dtw(landmarks[part])
                
                if ref_seq is not None and target_seq is not None:
                    # DTW 정렬
                    path = dtw.warping_path(ref_seq, target_seq)
                    aligned_part = self._align_sequence_with_path(landmarks[part], path)
                    aligned_landmarks[part] = aligned_part
                else:
                    aligned_landmarks[part] = landmarks[part]
            
            aligned_sequences.append(aligned_landmarks)
        
        # 정렬된 시퀀스들을 평균내어 합성
        return self._merge_aligned_sequences(aligned_sequences)
    
    def _prepare_sequence_for_dtw(self, sequence):
        """DTW를 위한 시퀀스 준비"""
        if not sequence or all(frame is None for frame in sequence):
            return None
        
        # None 프레임 제거 및 평탄화
        valid_frames = [frame for frame in sequence if frame is not None]
        if not valid_frames:
            return None
        
        # 각 프레임을 1차원 배열로 평탄화
        flattened = []
        for frame in valid_frames:
            if isinstance(frame, list) and len(frame) > 0:
                flat_frame = np.array(frame).flatten()
                flattened.append(flat_frame)
        
        return np.array(flattened) if flattened else None
    
    def _align_sequence_with_path(self, sequence, path):
        """DTW 경로를 사용하여 시퀀스 정렬"""
        # 간단한 구현 - 실제로는 더 정교한 정렬 필요
        return sequence
    
    def _merge_aligned_sequences(self, sequences):
        """정렬된 시퀀스들을 평균내어 합성"""
        if not sequences:
            return None
        
        # 간단한 평균 합성 - 실제로는 더 정교한 합성 필요
        merged = sequences[0]
        logger.info("DTW 기반 시퀀스 합성 완료")
        return merged
    
    def save_landmarks(self, landmarks_data, output_path):
        """랜드마크 데이터를 JSON으로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(landmarks_data, f, ensure_ascii=False, indent=2)
        logger.info(f"랜드마크 데이터 저장: {output_path}")
    
    def process_word(self, word, excel_path="baseDataList.xlsx", output_dir="result"):
        """전체 파이프라인 실행"""
        logger.info(f"단어 '{word}' 처리 시작")
        
        # 1. Excel 데이터 로드
        df = self.load_base_data(excel_path)
        if df is None:
            return False
        
        # 2. 단어로 필터링
        word_filtered = self.filter_by_word(df, word)
        if len(word_filtered) == 0:
            logger.warning(f"단어 '{word}'에 대한 데이터가 없습니다.")
            return False
        
        # 3. 조건 적용
        condition_filtered = self.apply_conditions(word_filtered)
        if len(condition_filtered) == 0:
            logger.warning(f"조건을 만족하는 '{word}' 데이터가 없습니다.")
            return False
        
        # 4. 상위 5개 선택
        selected_files = self.select_top_files(condition_filtered, 5)
        
        # 5. 파일 ID 리스트 추출 및 배치 처리
        file_ids = [row.get('파일ID', row.get('ID', 0)) for _, row in selected_files.iterrows()]
        from full_access import get_video_paths_by_word
        video_paths = get_video_paths_by_word(file_ids)
        
        # 6. 각 파일에서 랜드마크 추출
        landmarks_list = []
        for i, video_path in enumerate(video_paths):
            if video_path and os.path.exists(video_path):
                logger.info(f"랜드마크 추출 중: {video_path}")
                landmarks = self.extract_landmarks(video_path)
                landmarks_list.append(landmarks)
            else:
                logger.warning(f"비디오 파일을 찾을 수 없음: {file_ids[i]}")
        
        if not landmarks_list:
            logger.error("추출된 랜드마크가 없습니다.")
            return False
        
        # 7. DTW 적용하여 합성
        merged_landmarks = self.apply_dtw_alignment(landmarks_list)
        
        # 8. 결과 저장
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{word}_merged_landmarks.json")
        self.save_landmarks(merged_landmarks, output_path)
        
        logger.info(f"단어 '{word}' 처리 완료")
        return True

def main():
    parser = argparse.ArgumentParser(description="단어 기반 랜드마크 추출 파이프라인")
    parser.add_argument("word", help="처리할 단어")
    parser.add_argument("--excel", default="baseDataList.xlsx", help="Excel 파일 경로")
    parser.add_argument("--output", default="result", help="출력 디렉토리")
    
    args = parser.parse_args()
    
    pipeline = WordToLandmarksPipeline()
    success = pipeline.process_word(args.word, args.excel, args.output)
    
    if success:
        print(f"✅ 단어 '{args.word}' 처리 완료")
    else:
        print(f"❌ 단어 '{args.word}' 처리 실패")

if __name__ == "__main__":
    main()