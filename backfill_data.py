from datetime import datetime, timedelta
import pandas as pd
import random
import json
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient

# =============================================
# 🔧 KẾT NỐI MONGODB
# =============================================
def connect_mongo():
    client = MongoClient("mongodb+srv://gold_user:nhom5vuive@cluster0.7zcjpnr.mongodb.net/gold_pipeline?appName=Cluster0")
    db = client["gold_pipeline"]
    collection = db["gold_prices"]
    return collection


# =============================================
# 🟡 PNJ GOLD GENERATOR
# =============================================
def create_pnj_data(start_date, end_date):
    gold_types = [
        "Vàng miếng SJC 999.9", "Nhẫn Trơn PNJ 999.9", "Vàng Kim Bảo 999.9",
        "Vàng Phúc Lộc Tài 999.9", "Vàng PNJ - Phượng Hoàng", "Vàng nữ trang 999.9",
        "Vàng nữ trang 999", "Vàng nữ trang 9920", "Vàng nữ trang 99",
        "Vàng 916 (22K)", "Vàng 750 (18K)", "Vàng 680 (16.3K)",
        "Vàng 650 (15.6K)", "Vàng 610 (14.6K)", "Vàng 585 (14K)",
        "Vàng 416 (10K)", "Vàng 375 (9K)", "Vàng 333 (8K)"
    ]

    base_prices = {
        "Vàng miếng SJC 999.9": {"mua": 14750, "bán": 14950},
        "Nhẫn Trơn PNJ 999.9": {"mua": 14640, "bán": 14940},
        "Vàng Kim Bảo 999.9": {"mua": 14640, "bán": 14940},
        "Vàng Phúc Lộc Tài 999.9": {"mua": 14640, "bán": 14940},
        "Vàng PNJ - Phượng Hoàng": {"mua": 14640, "bán": 14940},
        "Vàng nữ trang 999.9": {"mua": 14620, "bán": 14920},
        "Vàng nữ trang 999": {"mua": 14605, "bán": 14905},
        "Vàng nữ trang 9920": {"mua": 14511, "bán": 14811},
        "Vàng nữ trang 99": {"mua": 14481, "bán": 14781},
        "Vàng 916 (22K)": {"mua": 13377, "bán": 13677},
        "Vàng 750 (18K)": {"mua": 10455, "bán": 11205},
        "Vàng 680 (16.3K)": {"mua": 9411, "bán": 10161},
        "Vàng 650 (15.6K)": {"mua": 8963, "bán": 9713},
        "Vàng 610 (14.6K)": {"mua": 8366, "bán": 9116},
        "Vàng 585 (14K)": {"mua": 7993, "bán": 8743},
        "Vàng 416 (10K)": {"mua": 5472, "bán": 6222},
        "Vàng 375 (9K)": {"mua": 4860, "bán": 5610},
        "Vàng 333 (8K)": {"mua": 4189, "bán": 4939}
    }

    print("🟡 Đang tạo dữ liệu PNJ...")
    data = []
    current_date = start_date

    while current_date <= end_date:
        days_from_start = (current_date - start_date).days
        time_trend = 1 + (days_from_start * 0.001)
        daily_volatility = random.uniform(0.98, 1.02)
        weekday_factor = 1.02 if current_date.weekday() < 5 else 0.99

        for gold_type in gold_types:
            base = base_prices[gold_type]
            mua = int(base["mua"] * time_trend * daily_volatility * weekday_factor)
            ban = int(base["bán"] * time_trend * daily_volatility * weekday_factor)
            if ban <= mua:
                ban = mua + 200
            data.append({
                "Thương hiệu": "PNJ",
                "Ngày": current_date.strftime("%Y-%m-%d"),
                "Loại vàng": gold_type,
                "Mua vào": mua,
                "Bán ra": ban,
                "Thời gian cập nhật": datetime.utcnow()
            })
        current_date += timedelta(days=1)

    return data


# =============================================
# 🟢 SJC GOLD GENERATOR
# =============================================
def create_sjc_data(start_date, end_date):
    sjc_types = [
        "Vàng SJC 1L, 10L, 1KG", "Vàng SJC 5 chỉ", "Vàng SJC 0.5 chỉ, 1 chỉ, 2 chỉ",
        "Vàng nhẫn SJC 99,99% 1 chỉ, 2 chỉ, 5 chỉ", "Nữ trang 99,99%", "Nữ trang 99%",
        "Nữ trang 75%", "Nữ trang 68%", "Nữ trang 61%", "Nữ trang 58,3%", "Nữ trang 41,7%"
    ]
    base_prices = {
        "Vàng SJC 1L, 10L, 1KG": {"mua": 147_500_000, "bán": 149_500_000},
        "Vàng SJC 5 chỉ": {"mua": 147_500_000, "bán": 149_520_000},
        "Vàng SJC 0.5 chỉ, 1 chỉ, 2 chỉ": {"mua": 147_500_000, "bán": 149_530_000},
        "Vàng nhẫn SJC 99,99% 1 chỉ, 2 chỉ, 5 chỉ": {"mua": 146_200_000, "bán": 148_700_000},
        "Nữ trang 99,99%": {"mua": 144_700_000, "bán": 147_700_000},
        "Nữ trang 99%": {"mua": 141_738_000, "bán": 146_238_000},
        "Nữ trang 75%": {"mua": 103_436_000, "bán": 110_936_000},
        "Nữ trang 68%": {"mua": 93_096_000, "bán": 100_596_000},
        "Nữ trang 61%": {"mua": 82_756_000, "bán": 90_256_000},
        "Nữ trang 58,3%": {"mua": 78_768_000, "bán": 86_268_000},
        "Nữ trang 41,7%": {"mua": 54_247_000, "bán": 61_747_000}
    }

    print("🟢 Đang tạo dữ liệu SJC...")
    data = []
    current_date = start_date
    while current_date <= end_date:
        days_from_start = (current_date - start_date).days
        time_trend = 1 + (days_from_start * 0.0005)
        daily_volatility = random.uniform(0.985, 1.015)

        for gold_type in sjc_types:
            base = base_prices[gold_type]
            mua = int(base["mua"] * time_trend * daily_volatility)
            ban = int(base["bán"] * time_trend * daily_volatility)
            if ban <= mua:
                ban = mua + 2_000_000
            data.append({
                "Thương hiệu": "SJC",
                "Ngày": current_date.strftime("%Y-%m-%d"),
                "Loại vàng": gold_type,
                "Mua vào": mua,
                "Bán ra": ban,
                "Thời gian cập nhật": datetime.utcnow()
            })
        current_date += timedelta(days=1)

    return data


# =============================================
# 🔴 DOJI CRAWLER (REAL + SIMULATED)
# =============================================
def get_real_doji_prices():
    print("🔴 Lấy giá thật từ DOJI...")
    url = "https://giavang.doji.vn/"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    table = soup.find("table")
    if not table:
        raise Exception("Không tìm thấy bảng giá trên trang DOJI")

    prices = {}
    for row in table.find_all("tr")[1:]:
        cols = [c.get_text(strip=True).replace(",", "").replace(".", "") for c in row.find_all("td")]
        if len(cols) >= 3:
            try:
                prices[cols[0]] = {"mua": int(cols[1]), "bán": int(cols[2])}
            except:
                continue
    return prices


def create_doji_data(start_date, end_date):
    real_today = get_real_doji_prices()
    print("🔴 Đang tạo dữ liệu DOJI...")
    data = []
    current_date = start_date
    while current_date <= end_date:
        for name, base in real_today.items():
            mua = int(base["mua"] * random.uniform(0.95, 1.05))
            ban = int(base["bán"] * random.uniform(0.95, 1.05))
            if ban <= mua:
                ban = mua + 100
            data.append({
                "Thương hiệu": "DOJI",
                "Ngày": current_date.strftime("%Y-%m-%d"),
                "Loại vàng": name,
                "Mua vào": mua,
                "Bán ra": ban,
                "Thời gian cập nhật": datetime.utcnow()
            })
        current_date += timedelta(days=1)
    return data


# =============================================
# 🚀 MAIN PROCESS
# =============================================
def main():
    start_date = datetime(2025, 1, 1)
    end_date = datetime.now() - timedelta(days=1)
    collection = connect_mongo()

    print("🚀 Bắt đầu tạo & lưu dữ liệu vàng vào MongoDB...")

    all_data = []
    all_data.extend(create_pnj_data(start_date, end_date))
    all_data.extend(create_sjc_data(start_date, end_date))
    all_data.extend(create_doji_data(start_date, end_date))

    if all_data:
        collection.insert_many(all_data)
        print(f"✅ Đã thêm {len(all_data)} bản ghi vào MongoDB collection 'gold_prices'")
    else:
        print("❌ Không có dữ liệu để lưu.")


if __name__ == "__main__":
    main()
