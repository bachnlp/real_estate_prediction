# train.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import joblib
import os

def train_model(data_path, model_output_path):
    print("1. Đang load dữ liệu đã xử lý...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file dữ liệu tại {data_path}")
        return

    # Xác định đặc trưng (X) và biến mục tiêu (y)
    # Giả định bạn đang muốn mô hình dự đoán Giá/m2.
    target_col = 'Giá/m2'
    
    if target_col not in df.columns:
        print(f"❌ Không tìm thấy cột mục tiêu '{target_col}' trong dữ liệu.")
        return

    # Tách X và y. Bỏ đi các cột không dùng để dự đoán nếu cần.
    X = df.drop(columns=[target_col, 'price_per_room', 'price_level'], errors='ignore')
    y = df[target_col]

    print("2. Phân chia tập huấn luyện (Train) và kiểm thử (Test)...")
    # Tỷ lệ 80% train - 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("3. Khởi tạo và huấn luyện mô hình XGBoost...")
    # Khởi tạo mô hình với các siêu tham số (hyperparameters) cơ bản
    model = xgb.XGBRegressor(
        objective='reg:squarederror', # Dùng cho bài toán hồi quy
        n_estimators=300,             # Số lượng cây quyết định
        learning_rate=0.05,           # Tốc độ học
        max_depth=7,                  # Độ sâu tối đa của cây (tăng lên nếu model underfitting)
        subsample=0.8,                # Lấy mẫu ngẫu nhiên 80% data cho mỗi cây (chống overfitting)
        colsample_bytree=0.8,         # Lấy mẫu ngẫu nhiên 80% features cho mỗi cây
        random_state=42,
        n_jobs=-1                     # Tận dụng tối đa số luồng CPU hiện có
    )
    
    # Bắt đầu training
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],  # Đánh giá trực tiếp trong lúc train
        verbose=50                    # Cứ 50 vòng lặp thì in kết quả ra 1 lần
    )

    print("\n4. Đang đánh giá hiệu suất mô hình trên tập Test...")
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # In báo cáo kết quả
    print("\n" + "="*30)
    print("📊 KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    print("="*30)
    print(f"MAE (Sai số trung bình tuyệt đối): {mae:,.0f} VNĐ/m²")
    print(f"RMSE (Độ lệch chuẩn sai số):      {rmse:,.0f} VNĐ/m²")
    print(f"R2 Score (Độ tin cậy):            {r2:.4f} ({(r2*100):.2f}%)")
    print("="*30 + "\n")

    print("5. Đóng gói và lưu trữ mô hình để Deployment...")
    # Đảm bảo thư mục lưu model đã tồn tại, nếu chưa thì tạo mới
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    # Dùng joblib để lưu model (nhẹ và nhanh hơn pickle cho các mảng numpy lớn)
    joblib.dump(model, model_output_path)
    print(f"✅ HOÀN TẤT! Mô hình đã được lưu an toàn tại: {model_output_path}")

def predict(input_data):
    model_path = os.path.join(os.path.dirname(__file__), 'xgboost.pkl')
    model = joblib.load(model_path)
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return prediction[0]

if __name__ == "__main__":    
    INPUT_FILE = "../../notebook/preprocessing/VN_housing_processed.csv"
    MODEL_OUTPUT = "../../src/models/xgboost.pkl" 
    
    train_model(INPUT_FILE, MODEL_OUTPUT)