import streamlit as st
import pandas as pd
from pymongo import MongoClient
import plotly.express as px
from prophet import Prophet
from datetime import datetime
import base64

# ==========================
# ⚙️ KẾT NỐI MONGODB
# ==========================
@st.cache_data(ttl=60)
def connect_and_load_data():
    client = MongoClient("mongodb+srv://gold_user:nhom5vuive@cluster0.7zcjpnr.mongodb.net/gold_pipeline?appName=Cluster0")
    db = client["gold_pipeline"]
    collection = db["gold_prices"]
    
    data = list(collection.find({}, {"_id": 0}))
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    for col in ["Mua vào", "Bán ra"]:
        df[col] = (
            df[col].astype(str)
            .str.replace(r"[^\d.]", "", regex=True)
            .replace("", "0")
            .astype(float)
        )
    df["Ngày"] = pd.to_datetime(df["Ngày"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["Ngày"])
    return df

# ==========================
# 🎨 CẤU HÌNH GIAO DIỆN
# ==========================
st.set_page_config(page_title="Gold Price Dashboard", layout="wide")
df_all = connect_and_load_data()

if df_all.empty:
    st.warning("⚠️ Chưa có dữ liệu nào trong MongoDB. Vui lòng chạy backfill_data.py trước!")
    st.stop()

# ==========================
# 🧩 BỘ LỌC SIDEBAR
# ==========================
st.sidebar.header("🎛️ Bộ lọc dữ liệu")
available_brands = df_all["Thương hiệu"].unique()
source = st.sidebar.selectbox("🪙 Chọn thương hiệu vàng:", available_brands)

# ==========================
# 🎨 THEME & LOGO
# ==========================
theme_data = {
    "PNJ": {"color": "#001F3F", "bg_light": "#E6EEF8", "logo": "pipeline/logopnj.png"},
    "DOJI": {"color": "#B22222", "bg_light": "#FCECEC", "logo": "pipeline/logodoji.png"},
    "SJC": {"color": "#CCAF66", "bg_light": "#FFF9E6", "logo": "pipeline/logosjc.png"}
}

theme = theme_data.get(source.upper(), {"color": "#2E86C1", "bg_light": "#F4F6F8", "logo": ""})
theme_color = theme["color"]
bg_light = theme["bg_light"]
logo_path = theme["logo"]

# ==========================
# 🖌️ CSS THEME
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
# 🖼️ LOGO + TIÊU ĐỀ
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
# 📂 LỌC DỮ LIỆU
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
# 💎 GIÁ MỚI NHẤT
# ==========================
latest = df_final.iloc[-1]
st.markdown(f"<h2>💎 Dữ liệu mới nhất cho: {gold_type}</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1: st.metric("Ngày", latest['Ngày'].strftime("%d-%m-%Y"))
with col2: st.metric("Giá mua", f"{latest['Mua vào']:,.0f} VND")
with col3: st.metric("Giá bán", f"{latest['Bán ra']:,.0f} VND")

# ==========================
# 📊 BIỂU ĐỒ GIÁ
# ==========================
df_final["Chênh lệch"] = df_final["Bán ra"] - df_final["Mua vào"]
tab1, tab2, tab3 = st.tabs(["📈 Giá mua", "📊 Giá bán", "📉 Chênh lệch"])

with tab1:
    fig_buy = px.line(df_final, x="Ngày", y="Mua vào", title=f"Diễn biến giá MUA - {source} ({gold_type})",
                      markers=True, color_discrete_sequence=[theme_color])
    st.plotly_chart(fig_buy, use_container_width=True)
with tab2:
    fig_sell = px.line(df_final, x="Ngày", y="Bán ra", title=f"Diễn biến giá BÁN - {source} ({gold_type})",
                       markers=True, color_discrete_sequence=[theme_color])
    st.plotly_chart(fig_sell, use_container_width=True)
with tab3:
    fig_spread = px.bar(df_final, x="Ngày", y="Chênh lệch", title=f"Chênh lệch Mua/Bán - {source} ({gold_type})",
                        hover_data=['Mua vào', 'Bán ra'], color_discrete_sequence=[theme_color])
    st.plotly_chart(fig_spread, use_container_width=True)

# ==========================
# 🔮 DỰ BÁO GIÁ VÀNG & KHUYẾN NGHỊ
# ==========================
st.markdown(f"<h2>🔮 Dự báo giá vàng 7 ngày tới & Khuyến nghị đầu tư</h2>", unsafe_allow_html=True)

# Chuẩn bị dữ liệu cho Prophet
df_forecast = df_final.rename(columns={"Ngày": "ds", "Bán ra": "y"})[["ds", "y"]]
model = Prophet(daily_seasonality=True)
model.fit(df_forecast)
future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)

fig_forecast = px.line(forecast, x="ds", y="yhat",
                       title="Dự báo giá Bán ra (7 ngày tới)",
                       labels={"ds": "Ngày", "yhat": "Giá dự báo (VND)"},
                       color_discrete_sequence=[theme_color])
st.plotly_chart(fig_forecast, use_container_width=True)

# Tính phần trăm thay đổi
next_week_pred = forecast.tail(7)["yhat"].mean()
current_price = latest["Bán ra"]
change_pct = ((next_week_pred - current_price) / current_price) * 100

# Hiển thị khuyến nghị
if change_pct > 1.5:
    st.success(f"📈 Giá dự kiến **tăng {change_pct:.2f}%** → Khuyến nghị **MUA** 💰")
elif change_pct < -1.5:
    st.error(f"📉 Giá dự kiến **giảm {change_pct:.2f}%** → Khuyến nghị **BÁN** ⚠️")
else:
    st.info(f"⚖️ Giá biến động nhẹ ({change_pct:.2f}%) → **NÊN GIỮ**, chờ tín hiệu rõ hơn")

# ==========================
# 📋 BẢNG DỮ LIỆU
# ==========================
with st.expander("📋 Xem dữ liệu chi tiết (đã lọc)"):
    st.dataframe(df_final[["Thương hiệu", "Ngày", "Loại vàng", "Mua vào", "Bán ra", "Chênh lệch"]],
                 use_container_width=True)
