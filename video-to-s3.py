import argparse
import os
import pandas as pd
import subprocess

# boto3가 없으면 스켈레톤 동작을 위해 대체
try:
    import boto3
except ImportError:
    boto3 = None

# 파일 경로
IDENTICAL_VIDEO_ROOT = "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋"

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

def parse_words(input_str: str) -> list:
    """쉼표(,)로 구분된 단어들을 파싱하여 리스트로 반환합니다."""
    return [w.strip() for w in input_str.split(',') if w.strip()]

def get_video_root(file_number: int) -> str:
    """파일 번호에 따라 적절한 비디오 루트 디렉토리를 반환합니다."""
    for (start, end), root_path in VIDEO_ROOTS:
        if start <= file_number <= end:
            return root_path
    return None

def transform_file(input_path: str, output_dir: str) -> str:
    """
    smooth.py 를 실행하여 비디오 파일을 JSON으로 변환합니다.
    """
    print(f"[변환 시작] {input_path}")
    
    try:
        result = subprocess.run(
            ["python", "smooth_convert_only.py", input_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # 생성된 JSON 파일 경로 추정 (절대 경로)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, "result", f"{base_name}.json")
        
        if os.path.exists(json_path):
            print(f"[변환 완료] {input_path} → {json_path}")
            return json_path
        else:
            print(f"[경고] JSON 파일이 생성되지 않았습니다: ....")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"[오류] 변환 실패: {e}")
        return None

def upload_to_s3(file_path: str, bucket: str, key_prefix: str):
    """
    Amazon S3에 파일을 업로드합니다.
    boto3가 설치되어 있지 않으면 로그만 출력합니다.
    """
    key = os.path.join(key_prefix, os.path.basename(file_path))
    print(f"[S3 업로드] 파일: {file_path} → s3://{bucket}/{key}")
    
    try:
        s3 = boto3.client('s3', region_name='ap-northeast-2')
        s3.upload_file(file_path, bucket, key)
        print("→ 업로드 완료")
    except Exception as e:
        print("→ 업로드 실패:", e)

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
        
        # 3. A열의 파일명과 C열의 번호 추출
        file_name = matches.iloc[0, 0]  # A열 (인덱스 0)
        file_number_str = str(matches.iloc[0, 2])  # C열 (인덱스 2)
        
        # C열의 숫자를 사용하여 VIDEO_ROOTS에서 경로 찾기
        try:
            file_number = int(file_number_str)
            video_root = get_video_root(file_number)
            if video_root:
                input_path = os.path.join(video_root, file_name + file_extension)
                print(f"[경로 찾기] {word} -> 번호: {file_number} -> {video_root}")
            else:
                print(f"[경고] 파일 번호 {file_number}에 대한 루트 디렉토리를 찾을 수 없습니다.")
                continue
        except (ValueError, TypeError):
            print(f"[경고] C열의 값 '{file_number_str}'을 숫자로 변환할 수 없습니다.")
            continue
        
        # 4. 변환 로직 수행
        converted_path = transform_file(input_path, output_dir)
        
        # 5. S3 업로드 (스켈레톤)
        if converted_path:
            upload_to_s3(converted_path, s3_bucket, s3_prefix)
    
    # 6. 완료 메시지
    print("파이프라인 수행이 완료되었습니다.")

if __name__ == "__main__":
    print("caution : 팀장 하드디스크 선 연결 후 시도 할 것.")
    """메인 파이프라인 실행"""
    parser = argparse.ArgumentParser(
        description="비디오에서 MediaPipe 랜드마크를 추출하여 JSON으로 저장하는 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        사용 예시:
        python video-to-s3.py 김치,김밥,버거,치킨
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
        s3_bucket="waterandfish-s3",
        s3_prefix="animations/",
        base_path="/Users/woo/Documents/Github/team5-waterandfish-Video/videos",
        file_extension=".avi"
    )

