import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

def load_raw_data(file_path):
    """Load the raw housing dataset."""
    return pd.read_csv(file_path)

def drop_unnecessary_columns(df):
    """Drop unnecessary columns."""
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.drop(columns=['Địa chỉ', 'Số tầng', 'Dài', 'Rộng'], errors='ignore')
    return df

def process_date_column(df):
    """Process the date column into day, month, year."""
    if 'Ngày' in df.columns:
        df['ngày_temp'] = pd.to_datetime(df['Ngày'], errors='coerce')
        df['ngày'] = df['ngày_temp'].dt.day.astype('Int64')
        df['tháng'] = df['ngày_temp'].dt.month.astype('Int64')
        df['năm'] = df['ngày_temp'].dt.year.astype('Int64')
        df = df.drop(columns=['ngày_temp', 'Ngày'], errors='ignore')
    return df

def clean_price_column(df):
    """Clean the price column and convert to standard numeric."""
    if 'Giá/m2' in df.columns:
        df['Giá/m2'] = df['Giá/m2'].astype(str).str.replace(r'triệu/m²|,\s', '', regex=True)
        df['Giá/m2'] = df['Giá/m2'].str.replace(',', '.').str.replace(r'[^\d.]', '', regex=True)
        df['Giá/m2'] = pd.to_numeric(df['Giá/m2'], errors='coerce') * 1_000_000
        df['Giá/m2'] = df['Giá/m2'].fillna(0).astype(int)
    return df

def clean_bedroom_column(df):
    """Clean the bedroom column."""
    if 'Số phòng ngủ' in df.columns:
        df['Số phòng ngủ'] = df['Số phòng ngủ'].astype(str).str.replace('nhiều hơn 10 phòng', '11')
        df['Số phòng ngủ'] = df['Số phòng ngủ'].str.replace('phòng', '').str.strip()
        df['Số phòng ngủ'] = pd.to_numeric(df['Số phòng ngủ'], errors='coerce').astype('Int64')
    return df

def clean_area_column(df):
    """Clean the area column."""
    if 'Diện tích' in df.columns:
        df['Diện tích'] = df['Diện tích'].astype(str).str.replace(r'm²', '', regex=True)
        df['Diện tích'] = df['Diện tích'].str.replace(',', '.').str.replace(r'[^\d.]', '', regex=True)
        df['Diện tích'] = pd.to_numeric(df['Diện tích'], errors='coerce')
        df['Diện tích'] = df['Diện tích'].fillna(0).astype(int)
    return df

def impute_missing_values(df):
    """Impute missing values for both categorical and numerical columns."""
    cat_cols = ['Quận', 'Huyện', 'Loại hình nhà ở', 'Giấy tờ pháp lý']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    num_cols = ['Số phòng ngủ', 'Diện tích', 'ngày', 'tháng', 'năm']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df

def encode_categorical_columns(df):
    """Encode nominal columns with LabelEncoder and ordinal columns with mapping."""
    le = LabelEncoder()
    
    # 1. Nominal Encoding (Các biến không quan trọng thứ tự)
    nominal_cols = ['Quận', 'Huyện', 'Loại hình nhà ở', 'Giấy tờ pháp lý']
    for col in nominal_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            
    # 2. Ordinal Encoding cho price_level (Bảo toàn thứ tự Budget < Mid-range < Premium)
    if 'price_level' in df.columns:
        # Ép kiểu về string để tránh lỗi nếu dữ liệu đang là Category type
        price_mapping = {'Budget': 0, 'Mid-range': 1, 'Premium': 2}
        df['price_level'] = df['price_level'].astype(str).map(price_mapping)
        
    return df

def filter_data(df):
    """Filter data based on area and price ranges to remove outliers."""
    if 'Diện tích' in df.columns and 'Giá/m2' in df.columns:
        df = df[(df['Diện tích'] > 20) & (df['Diện tích'] < 300)]
        df = df[(df['Giá/m2'] > 20_000_000) & (df['Giá/m2'] < 400_000_000)]
    return df

def feature_engineering(df):
    """Perform feature engineering to create new variables."""
    if all(col in df.columns for col in ['Giá/m2', 'Diện tích', 'Số phòng ngủ']):
        df['price_per_room'] = df['Giá/m2'] * df['Diện tích'] / (df['Số phòng ngủ'] + 1)
        df['area_per_room'] = df['Diện tích'] / (df['Số phòng ngủ'] + 1)
        df['rooms_per_100m2'] = np.where(df['Diện tích'] > 0, (df['Số phòng ngủ'] / df['Diện tích']) * 100, 0)

        # Phân loại mức giá dựa trên phân vị
        price_33 = df['Giá/m2'].quantile(0.33)
        price_67 = df['Giá/m2'].quantile(0.67)
        
        # Tạo cột price_level dạng nhãn chữ trước khi qua bước encode
        df['price_level'] = pd.cut(
            df['Giá/m2'], 
            bins=[0, price_33, price_67, float('inf')], 
            labels=['Budget', 'Mid-range', 'Premium'], 
            ordered=True
        )
    return df

def preprocess_data(input_path, output_path):
    """Complete preprocessing pipeline."""
    df = load_raw_data(input_path)
    
    df = drop_unnecessary_columns(df)
    df = process_date_column(df)
    df = clean_price_column(df)
    df = clean_bedroom_column(df)
    df = clean_area_column(df)
    
    if 'Giá/m2' in df.columns and 'Diện tích' in df.columns:
        df = df.dropna(subset=['Giá/m2', 'Diện tích'])

    df = impute_missing_values(df)
    df = filter_data(df)
    df = feature_engineering(df) # Tạo price_level ở đây
    df = encode_categorical_columns(df) # Encode sau khi đã có price_level
    
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    input_file = "../../notebook/preprocessing/VN_housing_dataset.csv"
    output_file = "../../notebook/preprocessing/VN_housing_processed.csv"
    preprocess_data(input_file, output_file)