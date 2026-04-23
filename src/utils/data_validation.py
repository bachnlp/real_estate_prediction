# data_validation.py
import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema, Int64

# 1. SCHEMA CHO DỮ LIỆU ĐẦU VÀO (Trước khi Preprocess)
# Mục đích: Đảm bảo file CSV/API input có đủ các cột cần thiết.
raw_housing_schema = DataFrameSchema(
    {
        "Ngày": Column(str, nullable=True, required=False), # Có thể thiếu vì model ít khi dùng ngày raw
        "Giá/m2": Column(str, nullable=True, required=True),
        "Số phòng ngủ": Column(str, nullable=True, required=True),
        "Diện tích": Column(str, nullable=True, required=True),
        "Quận": Column(str, nullable=True, required=True),
        "Huyện": Column(str, nullable=True, required=True),
        "Loại hình nhà ở": Column(str, nullable=True, required=True),
        "Giấy tờ pháp lý": Column(str, nullable=True, required=True),
    },
    # strict=False cho phép dataframe có thêm các cột thừa (như 'Địa chỉ', 'Số tầng') 
    # mà không báo lỗi, file preprocess sẽ tự drop chúng sau.
    strict=False 
)

# 2. SCHEMA CHO DỮ LIỆU ĐẦU RA (Sau khi Preprocess - Sẵn sàng cho Model)
# Mục đích: Đảm bảo không có giá trị Null, kiểu dữ liệu chuẩn xác, và các giá trị nằm trong khoảng logic.
processed_housing_schema = DataFrameSchema(
    {
        "Giá/m2": Column(int, Check.in_range(20_000_000, 400_000_000), nullable=False),
        "Diện tích": Column(int, Check.in_range(20, 300), nullable=False),
        "Số phòng ngủ": Column(Int64, Check.ge(0), nullable=False), # Ép >= 0
        
        # Các cột thời gian (sau khi tách)
        "ngày": Column(Int64, Check.in_range(1, 31), nullable=False),
        "tháng": Column(Int64, Check.in_range(1, 12), nullable=False),
        "năm": Column(Int64, Check.ge(2000), nullable=False),
        
        # Cột Categorical đã được encode thành số
        "Quận": Column(int, nullable=False),
        "Huyện": Column(int, nullable=False),
        "Loại hình nhà ở": Column(int, nullable=False),
        "Giấy tờ pháp lý": Column(int, nullable=False),
        "price_level": Column(int, Check.isin([0, 1, 2]), nullable=False), # Phải là 0, 1, hoặc 2
        
        # Feature Engineering columns
        "price_per_room": Column(float, nullable=False),
        "area_per_room": Column(float, nullable=False),
        "rooms_per_100m2": Column(float, nullable=False),
    },
    strict=False
)

def validate_raw_data(df):
    """Hàm gọi để validate input."""
    try:
        validated_df = raw_housing_schema.validate(df)
        print("✅ Raw data validation passed!")
        return validated_df
    except pa.errors.SchemaError as e:
        print(f"❌ Raw Data Schema Error: {e}")
        raise

def validate_processed_data(df):
    """Hàm gọi để validate output trước khi feed vào model."""
    try:
        validated_df = processed_housing_schema.validate(df)
        print("✅ Processed data validation passed! Ready for modeling.")
        return validated_df
    except pa.errors.SchemaError as e:
        print(f"❌ Processed Data Schema Error: {e}")
        raise

if __name__ == "__main__":
    import pandas as pd
    
    # Tạo thử 1 dòng data giả để test
    dummy_data = pd.DataFrame({
        "Giá/m2": ["50,5 triệu/m²"],
        "Số phòng ngủ": ["2 phòng"],
        "Diện tích": ["65m²"],
        "Quận": ["Cầu Giấy"],
        "Huyện": [pd.NA],
        "Loại hình nhà ở": ["Chung cư"],
        "Giấy tờ pháp lý": ["Sổ đỏ"]
    })
    
    print("Đang test chạy độc lập file validation...")
    # Thử ném data giả vào trạm kiểm lâm
    validate_raw_data(dummy_data)