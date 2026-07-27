from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "clean_data.csv"


# Đọc dữ liệu
df = pd.read_excel(RAW_PATH, decimal=',')

# Format lại InvoiceDate
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%y %H:%M')

# Loại bỏ bản ghi có Quantity hoặc UnitPrice âm
df = df[df['Quantity'] > 0].copy()
df = df[df['UnitPrice'] > 0].copy()

# Chỉ giữ lại các hoá đơn mua hàng thông thường
df = df[df['InvoiceNo'].astype(str).str.match(r'^\d+$')].copy()

# Loại bỏ khoảng trắng
df['Description'] = df['Description'].str.strip()
df['Country'] = df['Country'].str.strip()
df['StockCode'] = df['StockCode'].astype(str).str.strip()


# Điền dữ liệu Description bị khuyết
#=====================================
# Bước 1: Tạo bảng ánh xạ StockCode -> Description
# (dùng mode vì 1 StockCode đôi khi có vài cách viết Description hơi khác nhau do nhập liệu)

stockcode_to_desc = (
    df.dropna(subset=['Description'])
      .groupby('StockCode')['Description']
      .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
)
# Bước 2: Điền Description bị thiếu dựa trên StockCode tương ứng
df['Description'] = df['Description'].fillna(df['StockCode'].map(stockcode_to_desc))

# Bước 3: Với số còn lại (thường rất ít), điền "Unknown"
df['Description'] = df['Description'].fillna('Unknown')
#=======================================


# Chỉ giữ lại StockCode hợp lệ
df = df[df['StockCode'].astype(str).str.match(r'^\d{5}[A-Za-z]*$')].copy()

# Bổ sung thêm cột TotalPrice
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Xoá dữ liệu trùng lặp
df = df.drop_duplicates().copy()

# Tách 2 bảng theo CustomerId
# Bảng df_rfm loại bỏ những dòng còn thiếu CustomerId
df_rfm = df.dropna(subset=['CustomerID']).copy()
df_rfm['CustomerID'] = df_rfm['CustomerID'].astype(int)
# Bảng df_full giữ nguyên và những chỗ còn thiếu CustomerId điền Guess
df_full = df.copy()
df_full['CustomerID'] = df_full['CustomerID'].fillna('Guest')


# Xuất file
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_rfm.to_csv(OUTPUT_PATH, index=False)
