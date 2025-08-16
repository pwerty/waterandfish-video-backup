import pandas as pd
import sys

def parse_keywords(keywords_str):
    """콤마로 구분된 키워드 문자열을 리스트로 변환"""
    return [kw.strip() for kw in keywords_str.split(',')]

def find_one_file(csv_path, keywords_str):
    """CSV 파일에서 조건에 맞는 파일명을 하나만 찾아 반환"""
    try:
        df = pd.read_csv(csv_path)
        keywords = parse_keywords(keywords_str)
        
        # 조건: 한국어가 키워드 중 하나 & 방향이 정면 & 타입이 단어
        filtered_df = df[
            (df['한국어'].isin(keywords)) & 
            (df['방향'] == '정면') & 
            (df['타입(단어/문장)'] == '단어')
        ]
        
        if not filtered_df.empty:
            return filtered_df.iloc[0]['파일명']  # 첫 번째 결과만 반환
        else:
            return None
            
    except FileNotFoundError:
        print(f"오류: '{csv_path}' 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("사용법: python3 parseCSV.py <CSV파일경로> <키워드1, 키워드2, ...>")
        sys.exit(1)
    
    csv_file_path = sys.argv[1]
    keywords_input = sys.argv[2]
    
    result = find_one_file(csv_file_path, keywords_input)
    
    if result:
        print(result)
    else:
        print("조건에 맞는 파일을 찾을 수 없습니다.")
