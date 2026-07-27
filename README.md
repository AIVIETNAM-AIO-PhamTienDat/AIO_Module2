# E-commerce Customer Analytics and Segmentation Dashboard

Dự án phân tích hành vi khách hàng thương mại điện tử bằng RFM và phân nhóm khách hàng bằng K-Means. Ứng dụng Streamlit cho phép nhập ba chỉ số Recency, Frequency, Monetary để dự đoán nhóm khách hàng.

## Cấu trúc thư mục

```text
.
├── App.py
├── assets/
│   └── ảnh_tiêu_đề.png
├── data/
│   ├── raw/
│   │   └── Online Retail.xlsx
│   └── processed/
│       ├── clean_data.csv
│       ├── rfm_uk_full.csv
│       ├── rfm_uk_segmented.csv
│       ├── cluster_profile.csv
│       └── cluster_action_plan.csv
├── models/
│   ├── scaler.pkl
│   ├── kmeans.pkl
│   └── model_metadata.json
├── notebooks/
│   ├── 01_eda_clean_data.ipynb
│   ├── 02_build_rfm_uk.ipynb
│   ├── 03_eda_rfm_uk.ipynb
│   ├── 04_train_kmeans_rfm.ipynb
│   └── 05_cluster_interpretation.ipynb
├── src/
│   └── data_preprocessing.py
├── docs/
│   └── PROJECT_CONTEXT.md
└── archive/
    └── old_kmeans/
```

## Cách chạy

```bash
pip install -r requirements.txt
streamlit run App.py
```

Model hiện tại được train trên bảng RFM full-period của khách hàng United Kingdom, với K=3 và preprocessing `log1p` cho cả Recency, Frequency, Monetary trước khi chuẩn hóa.
