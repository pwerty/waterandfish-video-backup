from datetime import datetime
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId

# 1) 서버 URI, DB/컬렉션명, 고정 chapter_id, 엑셀 파일 경로 설정
MONGO_URI        = 'mongodb+srv://sehyun5004:qwe123@waterandfish.uxyepd5.mongodb.net/'
DATABASE_NAME    = 'waterandfish'
COLLECTION_NAME  = 'Lessons'
CHAPTER_ID       = ObjectId('68626372cba901ab2b744fb9')
EXCEL_FILE_PATH  = 'uniqueList.xlsx'

# 2) created_at 고정값 (2025-07-04 00:00:00)
CREATED_AT = datetime(2025, 7, 4, 0, 0, 0)

def main():
    # MongoDB 연결
    client     = MongoClient(MONGO_URI)
    db         = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    # 엑셀 파일 읽기 (첫 번째 시트)
    df = pd.read_excel(EXCEL_FILE_PATH)

    # A열(media_url), B열(sign_text) 순서로 순회
    for _, row in df.iterrows():
        media_url = row.iloc[0]
        sign_text = row.iloc[1]

        # 이미 추가된 문서가 있으면 건너뛰기
        if collection.find_one({'sign_text': sign_text}):
            continue

        # 새 문서 생성
        doc = {
            'chapter_id'     : CHAPTER_ID,
            'sign_text'      : sign_text,
            'description'    : "",
            'content_type'   : "word",
            'media_url'      : media_url + ".json",
            'model_data_url' : '',
            'order_index'    : 1,
            'created_at'     : CREATED_AT
        }
        #print(doc)
        # 삽입
        collection.insert_one(doc)

    print('모든 데이터 처리 완료.')

if __name__ == '__main__':
    main()
