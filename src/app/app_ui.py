import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="Hanoi Real Estate Price Prediction", layout="centered")
st.title("Hanoi Real Estate Price Prediction")

col1, col2 = st.columns(2)

with col1:
    dien_tich = st.number_input("Square(m2)", min_value=1, max_value=10000, value=65)
    phong_ngu = st.number_input("Bedrooms", min_value=0, max_value=20, value=2)
    geographical_list=[
    "Ba Đình", "Bắc Từ Liêm", "Cầu Giấy", "Đống Đa", "Hà Đông", "Hai Bà Trưng", 
    "Hoàn Kiếm", "Hoàng Mai", "Long Biên", "Nam Từ Liêm", "Tây Hồ", "Thanh Xuân",
    "Ba Vì", "Chương Mỹ", "Đan Phượng", "Đông Anh", "Gia Lâm", "Hoài Đức", 
    "Mê Linh", "Mỹ Đức", "Phú Xuyên", "Phúc Thọ", "Quốc Oai", "Sóc Sơn", 
    "Thạch Thất", "Thanh Oai", "Thanh Trì", "Thường Tín", "Ứng Hòa", "Sơn Tây"
    ]
    quan = st.selectbox("District", geographical_list)
with col2:
    loai_hinh = st.selectbox("Housing Type", ["Chung cư", "Nhà riêng", "Nhà mặt phố", "Biệt thự"])
    giay_to = st.selectbox("Legal Documents", ["Sổ đỏ", "Sổ hồng", "Giấy tờ khác"])
    ngay_dang = st.date_input("Posting Date", value=datetime.today())

st.markdown("---")

if st.button("Predict Price", use_container_width=True):
    paylaoad = {
        "Diện tích": dien_tich,
        "Số phòng ngủ": phong_ngu,
        "ngày": ngay_dang.day,
        "tháng": ngay_dang.month,
        "năm": ngay_dang.year,
        "Quận": quan,
        "Huyện": "Không",  
        "Loại hình nhà ở": loai_hinh,
        "Giấy tờ pháp lý": giay_to
    }

    API_URL = "http://localhost:8000/predict"
    try:
        with st.spinner("Predicting..."):
            response = requests.post(API_URL, json=paylaoad)
            if response.status_code == 200:
                result = response.json()
                gia_m2 = result.get("predicted_price_per_m2_vnd")
                tong_gia = result["total_estimated_price_vnd"]
                st.success(f"Predicted Price per m²: {gia_m2:,.0f} VND")
                st.info(f"Total Predicted Price: {tong_gia:,.0f} VND")
            else:
                st.error("Failed to get prediction from API.")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
