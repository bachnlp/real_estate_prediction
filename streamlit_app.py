import sys
import os
import streamlit as st

# 1. Thêm đường dẫn gốc của dự án vào hệ thống tìm kiếm module của Python
# Điều này đảm bảo các lệnh 'from src.app...' có thể hoạt động chính xác
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path:
    sys.path.append(root_path)

# 2. Cấu hình trang (Tùy chọn)
# Lưu ý: st.set_page_config phải là lệnh Streamlit đầu tiên được chạy.
# Nếu trong file src/app/main.py của bạn đã có lệnh này, hãy xóa/comment ở đây.
st.set_page_config(
    page_title="Hanoi Real Estate Prediction",
    page_icon="🏠",
    layout="wide"
)

# 3. Import và thực thi nội dung từ file main.py cũ của bạn
try:
    # Khi import, toàn bộ mã nguồn trong src/app/main.py sẽ được thực thi
    import src.app.main
except Exception as e:
    st.error("Lỗi khởi chạy ứng dụng. Vui lòng kiểm tra lại cấu trúc thư mục hoặc file logs.")
    st.exception(e)