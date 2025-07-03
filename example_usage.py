#!/usr/bin/env python3
"""
word_to_landmarks.py 사용 예제
"""

from word_to_landmarks import WordToLandmarksPipeline
import os

def main():
    # 파이프라인 인스턴스 생성
    pipeline = WordToLandmarksPipeline()
    
    # 처리할 단어 목록
    words = ["안녕하세요", "감사합니다", "죄송합니다"]
    
    for word in words:
        print(f"\n{'='*50}")
        print(f"단어 '{word}' 처리 시작")
        print(f"{'='*50}")
        
        success = pipeline.process_word(
            word=word,
            excel_path="baseDataList.xlsx",
            output_dir="result"
        )
        
        if success:
            print(f"✅ '{word}' 처리 완료")
            
            # 생성된 파일 확인
            output_file = f"result/{word}_merged_landmarks.json"
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"📁 출력 파일: {output_file} ({file_size:,} bytes)")
        else:
            print(f"❌ '{word}' 처리 실패")

if __name__ == "__main__":
    main()