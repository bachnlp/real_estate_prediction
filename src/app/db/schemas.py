from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from src.app.db.database import Base

class prediction_history(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    dien_tich = Column(Float)
    phong_ngu = Column(Integer)
    quan = Column(String)
    loai_hinh = Column(String)
    phap_ly = Column(String)
    
    # Kết quả do model trả ra
    gia_du_doan_m2 = Column(Float)
    tong_gia_tri = Column(Float)
    
    # Thời gian user thực hiện dự đoán
    created_at = Column(DateTime, default=datetime.utcnow)