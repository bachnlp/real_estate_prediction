import os
import pandas as pd
import sys

# 1. Bổ sung thư mục 'src' vào đường dẫn hệ thống để Python có thể import
# __file__ đang ở thư mục scripts/
# .. sẽ lùi ra thư mục gốc
# 'src' sẽ trỏ thẳng vào thư mục chứa code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.data_validation import validate_raw_data, validate_processed_data

# Lưu ý: Theo cấu trúc của bạn, file preprocess.py đang nằm trong thư mục 'data'
from data.preprocess import (
    drop_unnecessary_columns, process_date_column, clean_price_column,
    clean_bedroom_column, clean_area_column, impute_missing_values,
    filter_data, feature_engineering, encode_categorical_columns
)
from models.train import train_model

def run_full_pipeline(input_path, processed_path, model_path):
    print("🚀 BẮT ĐẦU PIPELINE XỬ LÝ VÀ HUẤN LUYỆN 🚀")
    print("-" * 50)

    # BƯỚC 1: LOAD & VALIDATE RAW DATA
    print("\nStep 1: Loading and Validating Raw Data...")
    raw_df = pd.read_csv(input_path)
    validate_raw_data(raw_df)
    
    # BƯỚC 2: PREPROCESSING
    print("\nStep 2: Preprocessing Data...")
    df = drop_unnecessary_columns(raw_df)
    df = process_date_column(df)
    df = clean_price_column(df)
    df = clean_bedroom_column(df)
    df = clean_area_column(df)
    
    if 'Giá/m2' in df.columns and 'Diện tích' in df.columns:
        df = df.dropna(subset=['Giá/m2', 'Diện tích'])
        
    df = impute_missing_values(df)
    df = filter_data(df)
    df = feature_engineering(df)
    df = encode_categorical_columns(df)

    # BƯỚC 3: VALIDATE PROCESSED DATA
    print("\nStep 3: Validating Processed Data...")
    validate_processed_data(df)
    
    # Lưu dữ liệu sạch
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"✅ Data sạch đã lưu tại: {processed_path}")

    # BƯỚC 4: TRAINING
    print("\nStep 4: Training Model...")
    train_model(processed_path, model_path)
    
    print("\n" + "="*50)
    print("🎉 PIPELINE HOÀN TẤT THÀNH CÔNG! 🎉")
    print("="*50)

if __name__ == "__main__":
    # 2. Cập nhật đường dẫn tương đối (tính từ vị trí thư mục scripts/)
    RAW_DATA = "../notebook/preprocessing/VN_housing_dataset.csv"
    PROCESSED_DATA = "../notebook/preprocessing/VN_housing_processed.csv"
    MODEL_FILE = "../src/models/xgboost.pkl"

    run_full_pipeline(RAW_DATA, PROCESSED_DATA, MODEL_FILE)