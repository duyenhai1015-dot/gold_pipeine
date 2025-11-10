# scraper.py (Phiên bản GHI THẲNG VÀO MONGO)
import requests, re, os
from bs4 import BeautifulSoup
from pymongo import MongoClient
from datetime import datetime, timezone

# Lấy chuỗi kết nối từ biến môi trường (an toàn cho PaaS)
MONGO_URI = os.environ.get("MONGODB_ATLAS_URI")
if not MONGO_URI:
    print("Lỗi: Biến môi trường MONGODB_ATLAS_URI chưa được thiết lập!")
    exit(1)

DB_NAME = "gold_pipeline"
COLLECTION_NAME = "gold_prices"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def connect_mongo():
    """Kết nối tới MongoDB Atlas."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return client, collection

def fetch_from_doji():
    """Cào giá DOJI (giống hệt code cũ)."""
    print("🔴 Đang lấy giá real-time từ DOJI...")
    url = "https://giavang.doji.vn/"
    data = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")
        if not table:
            return []

        scrape_timestamp = datetime.now(timezone.utc)
        scrape_date = scrape_timestamp.strftime("%Y-%m-%d")

        for row in table.find_all("tr")[1:]:
            cols = [c.get_text(strip=True).replace(",", "").replace(".", "") for c in row.find_all("td")]
            if len(cols) >= 3:
                gold_type = cols[0]
                try:
                    buy_price = float(cols[1])
                    sell_price = float(cols[2])
                    
                    # Tạo record chuẩn (khớp với schema backfill)
                    record = {
                        "Thương hiệu": "DOJI",
                        "Ngày": scrape_date,
                        "Loại vàng": gold_type,
                        "Mua vào": buy_price,
                        "Bán ra": sell_price,
                        "Thời gian cập nhật": scrape_timestamp,
                        "source": "doji.vn"
                    }
                    data.append(record)
                except ValueError:
                    continue
        return data
    except Exception as e:
        print(f"Lỗi khi cào DOJI: {e}")
        return []

def save_to_mongo(records, collection):
    """Lưu dữ liệu vào Mongo, cập nhật nếu đã tồn tại."""
    if not records:
        print("Không có dữ liệu mới để lưu.")
        return 0
    
    count = 0
    for rec in records:
        # QUAN TRỌNG: Dùng 'upsert=True'.
        # Nó sẽ tìm bản ghi có (Ngày, Loại vàng) và CẬP NHẬT (thay thế).
        # Nếu chưa có, nó sẽ THÊM MỚI (insert).
        collection.replace_one(
            {"Ngày": rec["Ngày"], "Loại vàng": rec["Loại vàng"]},
            rec,
            upsert=True
        )
        count += 1
    return count

if __name__ == "__main__":
    print(f"[{datetime.now()}] Bắt đầu chạy scraper (phiên bản đơn giản)...")
    
    client, collection = connect_mongo()
    data = fetch_from_doji()
    
    if data:
        saved_count = save_to_mongo(data, collection)
        print(f"Đã cập nhật/thêm {saved_count} bản ghi vào MongoDB Atlas.")
    
    client.close()
    print("Kết thúc scraper.")