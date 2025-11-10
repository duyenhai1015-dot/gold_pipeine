# dashboard.py (V5.1 - ML Nâng cao + Giao diện Theme)
import streamlit as st
import pandas as pd
from pymongo import MongoClient
import plotly.express as px
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import base64 # <-- MỚI: Thêm cho Logo
import numpy as np

# MỚI: Import các thư viện ML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# ==========================
# ⚙️ KẾT NỐI MONGODB (Giữ logic V5: An toàn)
# ==========================
@st.cache_data(ttl=60)
def connect_and_load_data():
    MONGO_URI = os.environ.get("MONGODB_ATLAS_URI")
    if not MONGO_URI:
        st.error("Lỗi: Biến môi trường MONGODB_ATLAS_URI chưa được thiết lập!")
        st.stop()
    
    client = MongoClient(MONGO_URI)
    db = client["gold_pipeline"]
    collection = db["gold_prices"] 
    data = list(collection.find({}, {"_id": 0}))
    
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    
    for col in ["Mua vào", "Bán ra"]:
        df[col] = (
            df[col].astype(str).str.replace(r"[^\d.]", "", regex=True)
            .replace("", "0").astype(float)
        )
    
    df["Ngày"] = pd.to_datetime(df["Ngày"], format="%Y-%m-%d", errors="coerce")
    
    vietnam_tz = ZoneInfo("Asia/Ho_Chi_Minh")
    df["Thời gian cập nhật"] = pd.to_datetime(df["Thời gian cập nhật"], errors='coerce').dt.tz_localize(ZoneInfo("UTC"))
    df["Thời gian cập nhật (VN)"] = df["Thời gian cập nhật"].dt.tz_convert(vietnam_tz)

    df = df.dropna(subset=["Ngày", "Thời gian cập nhật"])
    return df

# ==========================
# 🤖 CÁC HÀM MACHINE LEARNING (Giữ nguyên V5)
# ==========================
def create_features(df):
    """Tạo đặc trưng từ cột Ngày cho mô hình ML."""
    df_feat = df[['Ngày', 'Bán ra']].copy()
    df_feat = df_feat.sort_values("Ngày").drop_duplicates("Ngày", keep="last")
    
    df_feat['ngày_trong_tuần'] = df_feat['Ngày'].dt.dayofweek
    df_feat['tháng'] = df_feat['Ngày'].dt.month
    df_feat['ngày_trong_năm'] = df_feat['Ngày'].dt.dayofyear
    df_feat['giá_trễ_1_ngày'] = df_feat['Bán ra'].shift(1)
    df_feat['giá_trễ_7_ngày'] = df_feat['Bán ra'].shift(7)
    df_feat['tb_trượt_7_ngày'] = df_feat['Bán ra'].rolling(window=7).mean().shift(1)
    df_feat = df_feat.dropna()
    return df_feat

def run_model_evaluation(df_ml, theme_color): # <-- Thêm theme_color
    """Chạy train/test split và đánh giá 3 mô hình."""
    FEATURES = ['ngày_trong_tuần', 'tháng', 'ngày_trong_năm', 'giá_trễ_1_ngày', 'giá_trễ_7_ngày', 'tb_trượt_7_ngày']
    TARGET = 'Bán ra'
    split_index = int(len(df_ml) * 0.8)
    train_df = df_ml.iloc[:split_index]
    test_df = df_ml.iloc[split_index:]
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, early_stopping_rounds=10)
    }
    scores = {}
    test_predictions = {}

    for name, model in models.items():
        if name == "XGBoost":
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        else:
            model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        scores[name] = mae
        test_predictions[name] = preds

    best_model_name = min(scores, key=scores.get)
    best_model_instance = models[best_model_name]
    
    df_plot = pd.DataFrame({'Ngày': test_df['Ngày'], 'Giá trị thực tế': y_test, 'Giá trị dự báo (Tốt nhất)': test_predictions[best_model_name]})
    
    # MỚI: Dùng theme_color cho biểu đồ
    fig = px.line(df_plot, x='Ngày', y=['Giá trị thực tế', 'Giá trị dự báo (Tốt nhất)'], 
                  title=f'So sánh trên tập Test (Mô hình tốt nhất: {best_model_name})',
                  markers=True, color_discrete_map={
                      'Giá trị thực tế': theme_color,
                      'Giá trị dự báo (Tốt nhất)': '#FF5733' # Màu khác
                  })
    
    return scores, best_model_name, best_model_instance, fig

def run_future_forecast(model, df_ml, features_list):
    """Dùng model tốt nhất để dự báo 30 ngày tương lai."""
    recent_data = df_ml.iloc[-30:].copy()
    future_predictions = []
    
    for i in range(30):
        last_row = recent_data.iloc[-1]
        next_date = last_row['Ngày'] + timedelta(days=1)
        next_day_features = {
            'ngày_trong_tuần': next_date.dayofweek,
            'tháng': next_date.month,
            'ngày_trong_năm': next_date.dayofyear,
            'giá_trễ_1_ngày': last_row['Bán ra'],
            'giá_trễ_7_ngày': recent_data.iloc[-6]['Bán ra'],
            'tb_trượt_7_ngày': recent_data.iloc[-7:]['Bán ra'].mean()
        }
        X_future = pd.DataFrame([next_day_features])[features_list]
        next_pred = model.predict(X_future)[0]
        future_predictions.append({'Ngày': next_date, 'Dự báo': next_pred})
        new_row = {'Ngày': next_date, 'Bán ra': next_pred, **next_day_features}
        recent_data = pd.concat([recent_data, pd.DataFrame([new_row])], ignore_index=True)

    df_forecast = pd.DataFrame(future_predictions)
    return df_forecast

# ==========================
# 🎨 CẤU HÌNH GIAO DIỆN
# ==========================
st.set_page_config(page_title="Gold Price Dashboard", layout="wide")
df_all = connect_and_load_data()

if df_all.empty:
    st.warning("⚠️ Chưa có dữ liệu. Vui lòng chạy 'backfill_data.py' và 'scraper.py'.")
    st.stop()

# ==========================
# 🧩 BỘ LỌC SIDEBAR (Lấy Filter 1)
# ==========================
st.sidebar.header("🎛️ Bộ lọc dữ liệu")
available_brands = df_all["Thương hiệu"].unique()
source = st.sidebar.selectbox("🪙 Chọn thương hiệu vàng:", available_brands)

# ==========================
# 🎨 THEME & LOGO (Lấy từ V-Theme)
# ==========================
theme_data = {
    "PNJ": {"color": "#001F3F", "bg_light": "#E6EEF8", "logo": "logopnj.png"}, # Sửa đường dẫn nếu cần
    "DOJI": {"color": "#B22222", "bg_light": "#FCECEC", "logo": "logodoji.png"},
    "SJC": {"color": "#CCAF66", "bg_light": "#FFF9E6", "logo": "logosjc.png"}
}

theme = theme_data.get(source.upper(), {"color": "#2E86C1", "bg_light": "#F4F6F8", "logo": ""})
theme_color = theme["color"]
bg_light = theme["bg_light"]
logo_path = theme["logo"]

# ==========================
# 🖌️ CSS THEME (Lấy từ V-Theme)
# ==========================
st.markdown(f"""
    <style>
    body, .stApp {{
        background-color: {bg_light};
        font-family: 'Segoe UI', sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background-color: {theme_color}10;
        border-right: 2px solid {theme_color}50;
    }}
    [data-testid="stSidebar"] * {{
        color: #000 !important;
        font-weight: 500;
    }}
    .main-header {{
        background: {theme_color};
        padding: 12px 0;
        color: white;
        text-align: center;
        font-size: 36px;
        font-weight: 800;
        border-radius: 0 0 12px 12px;
        letter-spacing: 1px;
    }}
    h2, h3, h4, h5, .stSubheader {{
        color: {theme_color} !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {theme_color} !important;
        font-weight: 700;
        font-size: 26px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {theme_color}15;
        border-radius: 8px;
        margin: 2px;
        color: #333;
        font-weight: 600;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {theme_color}30;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background-color: {theme_color};
        color: white !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================
# 🖼️ LOGO + TIÊU ĐỀ (Lấy từ V-Theme)
# ==========================
def load_logo_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

logo_base64 = load_logo_base64(logo_path)
if logo_base64:
    st.markdown(f"""
        <div class="main-header">
            <img src="data:image/png;base64,{logo_base64}" 
                 style="height:50px; margin-right:10px; vertical-align:middle; border-radius:8px;">
            GOLD PRICE DASHBOARD - VIETNAM 🇻🇳
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"<div class='main-header'>🏆 GOLD PRICE DASHBOARD - VIETNAM 🇻🇳</div>", unsafe_allow_html=True)

# ==========================
# 📂 LỌC DỮ LIỆU (Tiếp tục Filter 2 & 3)
# ==========================
df_brand_filtered = df_all[df_all["Thương hiệu"] == source].copy()
available_types = sorted(df_brand_filtered["Loại vàng"].unique())
gold_type = st.sidebar.selectbox("🎗️ Chọn loại vàng:", available_types)
df_type_filtered = df_brand_filtered[df_brand_filtered["Loại vàng"] == gold_type].copy()

if df_type_filtered.empty:
    st.warning(f"Không tìm thấy dữ liệu cho loại vàng: '{gold_type}'.")
    st.stop()

min_date = df_type_filtered["Ngày"].min().to_pydatetime()
max_date = df_type_filtered["Ngày"].max().to_pydatetime()
date_range = st.sidebar.date_input("🗓️ Chọn khoảng ngày:", (min_date, max_date), min_value=min_date, max_value=max_date)

if len(date_range) != 2:
    st.sidebar.error("Bạn phải chọn khoảng ngày (bắt đầu và kết thúc).")
    st.stop()

start_date, end_date = date_range
df_final = df_type_filtered[
    (df_type_filtered["Ngày"] >= pd.to_datetime(start_date)) &
    (df_type_filtered["Ngày"] <= pd.to_datetime(end_date))
].sort_values(by="Ngày")

if df_final.empty:
    st.warning(f"Không tìm thấy dữ liệu cho '{gold_type}' trong khoảng ngày đã chọn.")
    st.stop()

# ==========================
# 💎 GIÁ MỚI NHẤT (Lấy từ V-Theme)
# ==========================
latest = df_final.sort_values(by="Thời gian cập nhật").iloc[-1]
st.markdown(f"<h2>💎 Dữ liệu mới nhất cho: {gold_type}</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1: st.metric("Ngày", latest['Ngày'].strftime("%d-%m-%Y"))
with col2: st.metric("Giá mua", f"{latest['Mua vào']:,.0f} VND")
with col3: st.metric("Giá bán", f"{latest['Bán ra']:,.0f} VND")

# ==========================
# 📊 TABS (Cấu trúc V5)
# ==========================
df_final["Chênh lệch"] = df_final["Bán ra"] - df_final["Mua vào"]

tab_ml, tab_compare, tab_data, tab_spread = st.tabs([
    "🤖 Model Center (Dự báo Nâng cao)", 
    "📊 So sánh các Thương hiệu", 
    "📋 Dữ liệu chi tiết",
    "📉 Chênh lệch Mua/Bán"
])

# --- MỚI: Tab 1 (Dự báo Nâng cao) ---
with tab_ml:
    st.header(f"Trung tâm Đánh giá & Dự báo Mô hình")
    st.info(f"Đang phân tích dữ liệu 'Bán ra' cho: {gold_type}")
    
    df_ml = create_features(df_final)
    
    if len(df_ml) < 20:
        st.warning("Cần ít nhất 20 ngày dữ liệu (sau khi lọc) để chạy so sánh mô hình.")
    else:
        with st.spinner("Đang huấn luyện 3 mô hình... (Có thể mất 1 phút)"):
            scores, best_name, best_model, test_fig = run_model_evaluation(df_ml, theme_color)
            
            st.subheader("1. Kết quả Đánh giá Mô hình (trên tập Test)")
            st.write("Chỉ số: MAE (Sai số Tuyệt đối Trung bình) - Càng thấp càng tốt.")
            
            df_scores = pd.DataFrame.from_dict(scores, orient='index', columns=['MAE (VND)'])
            df_scores = df_scores.sort_values('MAE (VND)')
            df_scores['MAE (VND)'] = df_scores['MAE (VND)'].map('{:,.0f}'.format)
            st.dataframe(df_scores)
            
            st.success(f"Mô hình tối ưu được chọn: **{best_name}** (MAE: {scores[best_name]:,.0f} VND)")
            st.plotly_chart(test_fig, use_container_width=True)

            st.subheader("2. Dự báo 30 ngày tới (dùng mô hình tốt nhất)")
            
            FEATURES = ['ngày_trong_tuần', 'tháng', 'ngày_trong_năm', 'giá_trễ_1_ngày', 'giá_trễ_7_ngày', 'tb_trượt_7_ngày']
            X_all, y_all = df_ml[FEATURES], df_ml['Bán ra']
            
            if best_name == "XGBoost":
                 best_model.fit(X_all, y_all, eval_set=[(X_all, y_all)], verbose=False)
            else:
                 best_model.fit(X_all, y_all)
            
            df_forecast = run_future_forecast(best_model, df_ml, FEATURES)

            fig_forecast = px.line(df_final, x="Ngày", y="Bán ra", title=f"Giá BÁN (Lịch sử & Dự báo)", markers=True)
            # Thêm màu cho biểu đồ dự báo
            fig_forecast.update_traces(line=dict(color=theme_color), name='Giá thực tế')
            fig_forecast.add_scatter(x=df_forecast['Ngày'], y=df_forecast['Dự báo'], mode='lines', name=f'Dự báo ({best_name})', line=dict(color='#FF5733', dash='dot'))
            st.plotly_chart(fig_forecast, use_container_width=True)

# --- Tab 2: So sánh Thương hiệu (Giữ nguyên V5) ---
with tab_compare:
    st.header("So sánh giá bán giữa các thương hiệu")
    st.info(f"Đang so sánh cho loại vàng: **{gold_type}**")
    df_compare = df_all[(df_all["Loại vàng"] == gold_type) & (df_all["Ngày"] >= pd.to_datetime(start_date)) & (df_all["Ngày"] <= pd.to_datetime(end_date))].copy()
    df_compare = df_compare.sort_values("Thời gian cập nhật").drop_duplicates(["Ngày", "Thương hiệu"], keep="last")
    if df_compare.empty or df_compare['Thương hiệu'].nunique() <= 1:
        st.warning(f"Không có đủ dữ liệu (từ nhiều thương hiệu) để so sánh cho loại vàng '{gold_type}'.")
    else:
        df_pivot = df_compare.pivot_table(index='Ngày', columns='Thương hiệu', values='Bán ra').fillna(method='ffill') 
        fig_compare = px.line(df_pivot, title=f"So sánh giá bán: {gold_type}", markers=True)
        st.plotly_chart(fig_compare, use_container_width=True)

# --- Tab 3: Dữ liệu chi tiết (Giữ nguyên V5) ---
with tab_data:
    st.header(f"Dữ liệu chi tiết (đã lọc cho {source})")
    df_display = df_final.sort_values(by="Thời gian cập nhật", ascending=False)
    # Thêm cột Giờ VN (nếu có)
    if 'Thời gian cập nhật (VN)' in df_display.columns:
        df_display["Giờ VN"] = df_display["Thời gian cập nhật (VN)"].dt.strftime('%d-%m-%Y %H:%M:%S')
        st.dataframe(df_display[["Thương hiệu", "Ngày", "Loại vàng", "Mua vào", "Bán ra", "Giờ VN", "source"]], use_container_width=True)
    else:
        st.dataframe(df_display[["Thương hiệu", "Ngày", "Loại vàng", "Mua vào", "Bán ra", "source"]], use_container_width=True)


# --- Tab 4: Chênh lệch Mua/Bán (Lấy từ V-Theme) ---
with tab_spread:
    st.subheader("Chênh lệch giữa giá Bán và giá Mua")
    fig_spread = px.bar(df_final, x="Ngày", y="Chênh lệch",
                        title=f"Chênh lệch Mua/Bán - {source} ({gold_type})",
                        hover_data=['Mua vào', 'Bán ra'],
                        color_discrete_sequence=[theme_color]) # <-- Thêm màu
    st.plotly_chart(fig_spread, use_container_width=True)
