from pymongo import MongoClient

# MongoDB 연결 정보
MONGO_URI = 'mongodb+srv://sehyun5004:qwe123@waterandfish.uxyepd5.mongodb.net/'
DATABASE_NAME = 'waterandfish'
COLLECTION_NAME = 'Lessons'

def main():
    # MongoDB 연결
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    # media_url이 있는 레코드 찾기
    for doc in collection.find({'media_url': {'$ne': None, '$exists': True}}):
        media_url = doc.get('media_url')
        if media_url and len(media_url) >= 5:
            # 오른쪽 5글자 제거하고 .webm 추가
            new_url = media_url[:-5] + '.webm'
            collection.update_one(
                {'_id': doc['_id']},
                {'$set': {'media_url': new_url}}
            )
            print(f"업데이트: {media_url} → {new_url}")
    
    print('모든 데이터 처리 완료.')

if __name__ == '__main__':
    main()