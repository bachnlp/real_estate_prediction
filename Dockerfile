# Sử dụng Python phiên bản nhẹ (slim)
FROM python:3.12-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho psycopg2 (để kết nối Postgres)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Sao chép file requirements và cài đặt thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Mở cổng 8000 cho API
EXPOSE 8000

# Lệnh khởi chạy API
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]