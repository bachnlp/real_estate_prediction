import pandas as pd
import numpy as np
import re

def process_and_export_data(input_path='VN_housing_dataset.csv', output_path='VN_housing_processed.csv'):
    # 1. Load data and drop 'Unnamed: 0'
    df = pd.read_csv(input_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # 2 & 3. Xử lý các cột Diện tích, Dài, Rộng
    def clean_area_measure(text):
        if pd.isna(text):
            return np.nan
        text = str(text).lower().replace(',', '.').strip()
        match = re.search(r'(\d+\.?\d*)\s*m', text)
        if match:
            return float(match.group(1))
        return np.nan

    for col in ['Diện tích', 'Dài', 'Rộng']:
        if col in df.columns:
            df[col] = df[col].apply(clean_area_measure)

    # 4 & 5. Xử lý cột Giá/m2 (Quy đổi tất cả về chuẩn "Triệu VNĐ / m2")
    def clean_price_per_sqm(text):
        if pd.isna(text):
            return np.nan
        text = str(text).lower().replace(',', '.').strip()
        
        # Xóa dấu chấm phân cách hàng nghìn (VD: 1.000.000 đ -> 1000000 đ)
        if text.count('.') > 1 or (text.count('.') == 1 and 'đ' in text):
            text = text.replace('.', '')
            
        try:
            val_match = re.search(r'([\d.]+)', text)
            if not val_match: return np.nan
            val = float(val_match.group(1))
            
            if 'tỷ' in text:
                return val * 1000         # 1.2 tỷ -> 1200 triệu
            elif 'đ' in text:
                return val / 1000000      # 86000000 đ -> 86 triệu
            elif 'nghìn' in text:
                return val / 1000         # 500 nghìn -> 0.5 triệu
            else:
                return val                # Mặc định là triệu
        except ValueError:
            return np.nan

    df['Giá/m2'] = df['Giá/m2'].apply(clean_price_per_sqm)

    # 6. Convert 'Ngày' to datetime objects
    df['Ngày'] = pd.to_datetime(df['Ngày'], errors='coerce')

    # (MỚI) Tiền xử lý Số tầng & Số phòng ngủ (Chuyển thành số thực thay vì để object)
    if 'Số phòng ngủ' in df.columns:
        df['Số phòng ngủ'] = df['Số phòng ngủ'].astype(str).str.extract(r'(\d+)').astype(float)
    if 'Số tầng' in df.columns:
        df['Số tầng'] = df['Số tầng'].astype(str).str.replace('Nhiều hơn 10', '11')
        df['Số tầng'] = df['Số tầng'].str.extract(r'(\d+)').astype(float)

    # 7 & 8. One-hot encoding (CHỈ chọn các cột phân loại chuẩn, BỎ 'Địa chỉ')
    categorical_cols = ['Quận', 'Huyện', 'Loại hình nhà ở', 'Giấy tờ pháp lý']
    # Giữ lại các cột thực sự tồn tại trong dataframe
    valid_cat_cols = [col for col in categorical_cols if col in df.columns]
    
    # Loại bỏ cột Địa chỉ để tránh tràn RAM
    df = df.drop(columns=['Địa chỉ', 'Ngày'], errors='ignore') 
    
    # Mã hóa (Trả về True/False giúp tiết kiệm RAM)
    df_encoded = pd.get_dummies(df, columns=valid_cat_cols, dummy_na=False, dtype=bool)

    # 9. Điền khuyết (Impute with median) - Cập nhật cú pháp Pandas mới
    numerical_cols = df_encoded.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        median_val = df_encoded[col].median()
        df_encoded[col] = df_encoded[col].fillna(median_val)

    # 10. Xuất file
    df_encoded.to_csv(output_path, index=False)
    print(f"✅ Hoàn tất! Dữ liệu đã được lưu tại: {output_path}")
    print(f"Kích thước ma trận cuối cùng: {df_encoded.shape}")
    
    return df_encoded

# Gọi hàm thực thi
processed_df = process_and_export_data()