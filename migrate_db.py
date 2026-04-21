import pandas as pd
from sqlalchemy import create_engine
import time
import re

# ==========================================
# CẤU HÌNH KẾT NỐI
# ==========================================
DB_USER = 'postgres'
DB_PASSWORD = '123456'
DB_HOST = '127.0.0.1'
DB_PORT = '5433'
DB_NAME = 'ml_database'

TABLE_NAME = 'staging_vn_housing'
CSV_FILE_PATH = 'VN_housing_dataset.csv'

# Hàm khử dấu tiếng Việt và chuẩn hóa tên cột
def remove_vietnamese_accents(text):
    text = str(text).lower()
    text = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', text)
    text = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', text)
    text = re.sub(r'[ìíịỉĩ]', 'i', text)
    text = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', text)
    text = re.sub(r'[ùúụủũưừứựửữ]', 'u', text)
    text = re.sub(r'[ỳýỵỷỹ]', 'y', text)
    text = re.sub(r'[đ]', 'd', text)
    text = re.sub(r'[^a-z0-9]', '_', text) # Thay ký tự lạ bằng dấu _
    text = re.sub(r'_+', '_', text).strip('_') # Xóa các dấu _ bị thừa
    return text

def main():
    # SỬ DỤNG pg8000 THAY CHO psycopg2 (Chữ pg8000 nằm ngay sau postgresql)
    connection_string = f"postgresql+pg8000://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(connection_string)

    try:
        start_time = time.time()
        
        print(f"Reading... {CSV_FILE_PATH}")
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Copied {len(df)} data lines.")

        print("Converting to normal text and replacing space into dash...")
        # Áp dụng hàm khử dấu tiếng Việt cho toàn bộ cột
        df.columns = [remove_vietnamese_accents(col) for col in df.columns]
        print(f"Added new column in postgreSQL: {df.columns.tolist()}")

        print(f"Migrating data into table {TABLE_NAME} in postgreSQL...")
        df.to_sql(
            name=TABLE_NAME, 
            con=engine, 
            if_exists='replace', 
            index=False,
            chunksize=2000,
            method='multi'
        )
        
        end_time = time.time()
        print(f"✅ THÀNH CÔNG! Đã load dữ liệu trong {round(end_time - start_time, 2)} giây.")

    except Exception as e:
        print(f"❌ CÓ LỖI XẢY RA:")
        print(e)

if __name__ == "__main__":
    main()