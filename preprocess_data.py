import pandas as pd
from sqlalchemy import create_engine
import re
import time

# ==========================================
# 1. CẤU HÌNH KẾT NỐI (Giống hệt file trước)
# ==========================================
DB_USER = 'postgres'
DB_PASSWORD = '123456'
DB_HOST = '127.0.0.1'
DB_PORT = '5433'
DB_NAME = 'ml_database'
TABLE_NAME = 'staging_vn_housing'

def clean_currency(text):
    """Hàm trích xuất số từ chuỗi giá tiền (VD: '86,96 triệu/m²' -> 86.96)"""
    if pd.isna(text):
        return None
    text = str(text).lower().replace(',', '.') # Đổi phẩy thành chấm cho chuẩn số thập phân
    match = re.search(r'([\d\.]+)', text)
    if match:
        return float(match.group(1))
    return None

def clean_number(text):
    """Hàm trích xuất số nguyên từ chuỗi (VD: '5 phòng' -> 5)"""
    if pd.isna(text):
        return None
    match = re.search(r'(\d+)', str(text))
    if match:
        return float(match.group(1)) # Để float vì mảng có thể chứa NaN
    return None

def main():
    connection_string = f"postgresql+pg8000://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(connection_string)

    try:
        start_time = time.time()
        print("Đang trích xuất dữ liệu từ PostgreSQL...")
        
        # Mẹo: Giai đoạn đầu code tính năng, chúng ta dùng LIMIT 5000 để chạy thử cho nhanh
        # Khi nào model hoàn thiện, bạn xóa chữ 'LIMIT 5000' đi để lấy toàn bộ 82k dòng
        query = f"SELECT * FROM {TABLE_NAME} LIMIT 5000;"
        df = pd.read_sql(query, con=engine)
        print(f"Đã lấy thành công {len(df)} dòng dữ liệu.")

        print("Đang tiến hành Feature Engineering...")
        
        # 1. Bỏ cột unnamed_0 (index thừa từ file CSV gốc)
        if 'unnamed_0' in df.columns:
            df = df.drop(columns=['unnamed_0'])

        # 2. Ép kiểu và làm sạch các cột số liệu
        df['dien_tich_m2'] = df['dien_tich'].apply(clean_currency)
        df['gia_m2_trieu'] = df['gia_m2'].apply(clean_currency)
        df['phong_ngu'] = df['so_phong_ngu'].apply(clean_number)
        df['tang'] = df['so_tang'].apply(clean_number)

        # Xóa bỏ các cột văn bản gốc sau khi đã trích xuất xong số
        df = df.drop(columns=['dien_tich', 'gia_m2', 'so_phong_ngu', 'so_tang'])

        print("\n--- XEM TRƯỚC DỮ LIỆU ĐÃ LÀM SẠCH ---")
        print(df[['dia_chi', 'dien_tich_m2', 'phong_ngu', 'gia_m2_trieu']].head())
        
        # Tới đây, df đã là một DataFrame chứa các con số sạch sẽ,
        # sẵn sàng để đưa vào các bước như Fill NA, One-hot Encoding và fit() vào Model.
        
        end_time = time.time()
        print(f"\n✅ Hoàn tất tiền xử lý trong {round(end_time - start_time, 2)} giây.")

    except Exception as e:
        print(f"❌ CÓ LỖI XẢY RA:\n{e}")

if __name__ == "__main__":
    main()