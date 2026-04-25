import sys
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from fastapi import Depends
from src.app.db.database import get_db, engine, Base
from src.app.db.schemas import prediction_history
from src.data.preprocess import encode_categorical_columns

# Chỉ đường cho Python lùi ra 2 cấp để tìm thấy thư mục gốc 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Lệnh này sẽ tự động tạo bảng trong PostgreSQL nếu bảng chưa tồn tại
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Hanoi Real Estate API",
    description="Dự đoán giá nhà với dữ liệu thô (chuỗi)",
    version="1.1.0"
)

# Đường dẫn tới thư mục models
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/xgboost.pkl')
model = None

@app.on_event("startup")
def load_artifacts():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Đã load Model XGBoost thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi load: {e}")

# SCHEMA ĐẦU VÀO: Chỉ nhận dữ liệu tự nhiên bằng chữ từ người dùng
class RawHousingData(BaseModel):
    Diện_tích: int = Field(alias="Diện tích", example=65)
    Số_phòng_ngủ: int = Field(alias="Số phòng ngủ", example=2)
    Ngày: int = Field(alias="ngày", example=15)
    Tháng: int = Field(alias="tháng", example=6)
    Năm: int = Field(alias="năm", example=2026)
    Quận: str = Field(example="Cầu Giấy")
    Huyện: str = Field(example="Không", default="Không")
    Loại_hình_nhà_ở: str = Field(alias="Loại hình nhà ở", example="Chung cư")
    Giấy_tờ_pháp_lý: str = Field(alias="Giấy tờ pháp lý", example="Sổ đỏ")

@app.post("/predict")
def get_prediction(data: RawHousingData, db: Session = Depends(get_db)): # <--- Nhúng DB vào đây
    try:
        input_dict = data.model_dump(by_alias=True)
        import pandas as pd
        df = pd.DataFrame([input_dict])
        
        # 1. TIỀN XỬ LÝ (Feature Engineering & Encode)
        df['area_per_room'] = df['Diện tích'] / (df['Số phòng ngủ'] + 1)
        df['rooms_per_100m2'] = (df['Số phòng ngủ'] / df['Diện tích']) * 100
        df = encode_categorical_columns(df, is_training=False)
        
        expected_cols = model.feature_names_in_
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols] 

        # 2. CHẠY MODEL
        prediction = model.predict(df)
        predicted_price_m2 = round(float(prediction[0]), -4)
        total_price = predicted_price_m2 * input_dict['Diện tích']
        
        # 3. LƯU LỊCH SỬ XUỐNG POSTGRESQL TRƯỚC KHI TRẢ KẾT QUẢ
        new_record = prediction_history(
            dien_tich=input_dict['Diện tích'],
            phong_ngu=input_dict['Số phòng ngủ'],
            quan=input_dict['Quận'],
            loai_hinh=input_dict['Loại hình nhà ở'],
            phap_ly=input_dict['Giấy tờ pháp lý'],
            gia_du_doan_m2=predicted_price_m2,
            tong_gia_tri=total_price
        )
        db.add(new_record)
        db.commit()      # Xác nhận lưu
        db.refresh(new_record) # Cập nhật lại ID vừa tạo
        
        # 4. TRẢ KẾT QUẢ CHO FRONTEND
        return {
            "status": "success",
            "record_id": new_record.id, # Gửi kèm ID vừa lưu trong DB
            "predicted_price_per_m2_vnd": predicted_price_m2,
            "total_estimated_price_vnd": total_price
        }
        
    except Exception as e:
        # Nếu có lỗi gì xảy ra, rollback DB để không lưu data rác
        db.rollback() 
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "ok", "message": "API is ready for raw data!"}