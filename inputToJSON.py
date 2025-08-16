import argparse
import os
import pandas as pd
import subprocess

# boto3가 없으면 스켈레톤 동작을 위해 대체
try:
    import boto3
except ImportError:
    boto3 = None

def parse_words(input_str: str) -> list:
    """쉼표(,)로 구분된 단어들을 파싱하여 리스트로 반환합니다."""
    return [w.strip() for w in input_str.split(',') if w.strip()]

def transform_file(input_path: str, output_dir: str) -> str:
    """
    unified_landmark_pipeline.py를 실행하여 비디오 파일을 JSON으로 변환합니다.
    """
    print(f"[변환 시작] {input_path}")
    
    try:
        # unified_landmark_pipeline.py 실행
        result = subprocess.run(
            ["python", "unified_landmark_pipeline.py", input_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # 생성된 JSON 파일 경로 추정 (절대 경로)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, "result", f"{base_name}_converted_landmarks.json")
        
        if os.path.exists(json_path):
            print(f"[변환 완료] {input_path} → {json_path}")
            return json_path
        else:
            print(f"[경고] JSON 파일이 생성되지 않았습니다: {json_path}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"[오류] 변환 실패: {e}")
        return None

def process_pipeline(words_str: str, excel_path: str, output_dir: str, s3_bucket: str, s3_prefix: str, 
                    base_path: str = "/Users/woo/Documents/Github/team5-waterandfish-Video/videos", 
                    file_extension: str = ".avi"):
    # 1. 단어 파싱
    words = parse_words(words_str)
    
    # 2. 엑셀 로드
    df = pd.read_excel(excel_path)
    
    for word in words:
        # 2. B열에서 일치하는 행 찾기 (B열은 인덱스 1)
        matches = df[df.iloc[:, 1] == word]
        if matches.empty:
            print(f"[경고] '{word}'에 해당하는 항목을 찾을 수 없습니다.")
            continue
        
        # 3. A열의 파일명 추출 및 input_path 조합 (A열은 인덱스 0)
        file_name = matches.iloc[0, 0]
        input_path = os.path.join(base_path, file_name + file_extension)
        
        # 4. 변환 로직 수행
        converted_path = transform_file(input_path, output_dir)

    print("파이프라인 수행이 완료되었습니다.")

if __name__ == "__main__":
    """메인 파이프라인 실행"""
    parser = argparse.ArgumentParser(
        description="비디오에서 MediaPipe 랜드마크를 추출하여 JSON으로 저장하는 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python inputToJSON.py 김치,김밥,버거,치킨
        """
    )
    
    parser.add_argument(
        'words',
        help='쉼표로 구분된 단어들 (예: 김치,김밥,버거,치킨)'
    )
    
    args = parser.parse_args()
    words_str = args.words
    
    process_pipeline(
        words_str=words_str,
        excel_path="uniqueList.xlsx",
        output_dir="./output",
        s3_bucket="my-bucket",
        s3_prefix="uploads/",
        base_path="/Users/woo/Documents/Github/team5-waterandfish-Video/videos",
        file_extension=".avi"
    )

