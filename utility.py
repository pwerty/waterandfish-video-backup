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

        # Build mapping from Excel: sign_text (B-col) -> media_url base (A-col)
    mapping = { row.iloc[1]: row.iloc[0] for _, row in df.iterrows() }

    # Update existing lessons with null media_url
    for doc in collection.find({ 'media_url': None }):
        sign = doc.get('sign_text')
        if sign in mapping:
            new_url = f"{mapping[sign]}.json"
            collection.update_one(
                { '_id': doc['_id'] },
                { '$set': { 'media_url': new_url } }
            )
    print('모든 데이터 처리 완료.')

if __name__ == '__main__':
    main()
