# dashboard.py (V4 - Nâng cấp Dự báo bằng Prophet)
import streamlit as st
import pandas as pd
from pymongo import MongoClient
import plotly.express as px
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from prophet import Prophet # <-- MỚI: Import Prophet
import os # <-- MỚI: Import os để đọc biến môi trường

# ==========================
# ⚙️ KẾT NỐI MONGODB
# ==========================
@st.cache_data(ttl=60)  # Cache 1 phút
def connect_and_load_data():
    # Đọc từ biến môi trường (an toàn)
    MONGO_URI = os.environ.get("MONGODB_ATLAS_URI")
    if not MONGO_URI:
        st.error("Lỗi: Biến môi trường MONGODB_ATLAS_URI chưa được thiết lập!")
        st.stop()
    
    client = MongoClient(MONGO_URI)
    db = client["gold_pipeline"]
    collection = db["gold_prices"] 
    data = list(collection.find({}, {"_id": 0}))
    
    if not data:
        return pd.DataFrame() # Trả về DataFrame rỗng
        
    df = pd.DataFrame(data)
    
    # --- Xử lý dữ liệu ngay khi tải ---
    for col in ["Mua vào", "Bán ra"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[^\d.]", "", regex=True)
            .replace("", "0")
            .astype(float)
        )
    
    df["Ngày"] = pd.to_datetime(df["Ngày"], format="%Y-%m-%d", errors="coerce")
    
    # --- Chuyển đổi múi giờ ---
    vietnam_tz = ZoneInfo("Asia/Ho_Chi_Minh")
    df["Thời gian cập nhật"] = pd.to_datetime(df["Thời gian cập nhật"], errors='coerce').dt.tz_localize(ZoneInfo("UTC"))
    df["Thời gian cập nhật (VN)"] = df["Thời gian cập nhật"].dt.tz_convert(vietnam_tz)

    df = df.dropna(subset=["Ngày", "Thời gian cập nhật"])
    return df

# ==========================
# 🎨 CẤU HÌNH GIAO DIỆN
# ==========================
st.set_page_config(page_title="Gold Price Dashboard", layout="wide")
st.title("🏆 GOLD PRICE DASHBOARD - VIETNAM 🇻🇳")

# ==========================
# 📊 LẤY DỮ LIỆU
# ==========================
df_all = connect_and_load_data()

if df_all.empty:
    st.warning("⚠️ Chưa có dữ liệu. Vui lòng chạy 'backfill_data.py' và 'scraper.py'.")
    st.stop()

# ==========================
# 🧩 BỘ LỌC SIDEBAR
# ==========================
st.sidebar.header("Bộ lọc chính")

# --- Filter 1: Thương hiệu ---
available_brands = df_all["Thương hiệu"].unique()
source = st.sidebar.selectbox("🪙 Chọn thương hiệu vàng:", available_brands)
df_brand_filtered = df_all[df_all["Thương hiệu"] == source].copy()

# --- Filter 2: Loại vàng ---
available_types = df_brand_filtered["Loại vàng"].unique()
available_types.sort()
gold_type = st.sidebar.selectbox("🎗️ Chọn loại vàng:", available_types)
df_type_filtered = df_brand_filtered[df_brand_filtered["Loại vàng"] == gold_type].copy()

# --- Fix lỗi NaTType ---
if df_type_filtered.empty:
    st.warning(f"Không tìm thấy dữ liệu cho loại vàng: '{gold_type}'.")
    st.stop() 

# --- Filter 3: Khoảng ngày ---
min_date = df_type_filtered["Ngày"].min().to_pydatetime()
max_date = df_type_filtered["Ngày"].max().to_pydatetime()

date_range = st.sidebar.date_input(
    "🗓️ Chọn khoảng ngày:",
    (min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) != 2:
    st.sidebar.error("Bạn phải chọn một khoảng ngày (bắt đầu và kết thúc).")
    st.stop()
    
start_date, end_date = date_range
df_final = df_type_filtered[
    (df_type_filtered["Ngày"] >= pd.to_datetime(start_date)) & 
    (df_type_filtered["Ngày"] <= pd.to_datetime(end_date))
].sort_values(by="Ngày")

# ==========================
# 📈 HIỂN THỊ DỮ LIỆU
# ==========================
st.subheader(f"Dữ liệu cho: {gold_type} ({source})")
if not df_final.empty:
    latest = df_final.sort_values(by="Thời gian cập nhật").iloc[-1]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ngày", latest['Ngày'].strftime("%d-%m-%Y"))
    with col2:
        st.metric("Giá mua", f"{latest['Mua vào']:,.0f} VND")
    with col3:
        st.metric("Giá bán", f"{latest['Bán ra']:,.0f} VND")
else:
    st.info("Không có dữ liệu trong khoảng ngày đã chọn.")

# --- Tạo các Tab chính ---
tab_chart, tab_compare, tab_data, tab_spread = st.tabs([
    "📈 Biểu đồ & Dự báo Xu hướng", 
    "📊 So sánh các Thương hiệu", 
    "📋 Dữ liệu chi tiết",
    "📉 Chênh lệch Mua/Bán" # Thêm lại Tab Chênh lệch
])

# --- Tab 1: Biểu đồ & Dự báo (NÂNG CẤP LÊN PROPHET) ---
with tab_chart:
    st.header(f"Diễn biến giá: {gold_type}")
    
    if df_final.empty:
        st.warning("Không có dữ liệu để vẽ biểu đồ.")
    else:
        # Biểu đồ giá mua
        fig_buy = px.line(df_final, x="Ngày", y="Mua vào", title=f"Giá MUA", markers=True)
        st.plotly_chart(fig_buy, use_container_width=True)
        
        # --- MỚI: Logic Dự báo bằng Prophet (cho Giá Bán) ---
        st.subheader("Dự báo xu hướng giá bán (với Prophet)")
        
        # 1. Chuẩn bị dữ liệu (Prophet cần 'ds' và 'y')
        # Lấy giá trị cuối cùng mỗi ngày để dự báo
        df_prophet = df_final.sort_values("Thời gian cập nhật").drop_duplicates("Ngày", keep="last")
        df_prophet = df_prophet[['Ngày', 'Bán ra']].rename(columns={'Ngày': 'ds', 'Bán ra': 'y'})
        
        if len(df_prophet) > 5: # Prophet cần ít nhất vài điểm dữ liệu
            # 2. Huấn luyện mô hình (tắt bớt 1 số thứ cho nhanh)
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.05)
            m.fit(df_prophet)
            
            # 3. Tạo 30 ngày trong tương lai
            future = m.make_future_dataframe(periods=30, freq='D')
            
            # 4. Dự báo
            forecast = m.predict(future)
            
            # 5. Vẽ biểu đồ (dùng df_final để thấy real-time)
            fig_sell = px.line(df_final, x="Ngày", y="Bán ra", title=f"Giá BÁN (Lịch sử & Real-time)", markers=True)
            
            # Thêm đường dự báo (yhat)
            fig_sell.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Dự báo (Prophet)')
            # Thêm dải không chắc chắn (upper/lower)
            fig_sell.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Dự báo (Cao)', line=dict(color='rgba(255,165,0,0.3)'))
            fig_sell.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Dự báo (Thấp)', line=dict(color='rgba(0,128,0,0.3)'), fill='tonexty', fillcolor='rgba(0,100,80,0.2)')
            
            st.plotly_chart(fig_sell, use_container_width=True)
            st.caption("Lưu ý: Đây là mô hình dự báo Time-series của Prophet, không phải là tư vấn đầu tư.")
        else:
            st.info("Cần ít nhất 6 ngày dữ liệu trong khoảng đã chọn để chạy dự báo Prophet.")

# --- Tab 2: So sánh Thương hiệu ---
with tab_compare:
    st.header("So sánh giá bán giữa các thương hiệu")
    st.info(f"Đang so sánh cho loại vàng: **{gold_type}**")

    df_compare = df_all[
        (df_all["Loại vàng"] == gold_type) &
        (df_all["Ngày"] >= pd.to_datetime(start_date)) & 
        (df_all["Ngày"] <= pd.to_datetime(end_date))
    ].copy()
    
    df_compare = df_compare.sort_values("Thời gian cập nhật").drop_duplicates(["Ngày", "Thương hiệu"], keep="last")
    
    if df_compare.empty or df_compare['Thương hiệu'].nunique() <= 1:
        st.warning(f"Không có đủ dữ liệu (từ nhiều thương hiệu) để so sánh cho loại vàng '{gold_type}'.")
    else:
        df_pivot = df_compare.pivot_table(
            index='Ngày', 
            columns='Thương hiệu', 
            values='Bán ra'
        ).fillna(method='ffill') 
        
        fig_compare = px.line(df_pivot, title=f"So sánh giá bán: {gold_type}", markers=True)
        st.plotly_chart(fig_compare, use_container_width=True)

# --- Tab 3: Dữ liệu chi tiết ---
with tab_data:
    st.header(f"Dữ liệu chi tiết (đã lọc cho {source})")
    df_display = df_final.sort_values(by="Thời gian cập nhật", ascending=False)
    df_display["Giờ VN"] = df_display["Thời gian cập nhật (VN)"].dt.strftime('%d-%m-%Y %H:%M:%S')
    st.dataframe(df_display[[
        "Thương hiệu", "Ngày", "Loại vàng", 
        "Mua vào", "Bán ra", "Giờ VN", "source"
    ]], use_container_width=True)

# --- Tab 4: Chênh lệch Mua/Bán ---
with tab_spread:
    st.subheader("Chênh lệch giữa giá Bán và giá Mua")
    df_final['Chênh lệch'] = df_final['Bán ra'] - df_final['Mua vào']
    
    fig_spread = px.bar(df_final, x="Ngày", y="Chênh lệch",
                        title=f"Chênh lệch Mua/Bán - {source} ({gold_type})",
                        hover_data=['Mua vào', 'Bán ra'])
    st.plotly_chart(fig_spread, use_container_width=True)
