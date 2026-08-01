# E-commerce Customer Analytics and Segmentation Dashboard

Dự án phân tích hành vi khách hàng thương mại điện tử bằng RFM và phân nhóm khách hàng bằng Gaussian Mixture Model (GMM). Ứng dụng Streamlit cho phép nhập ba chỉ số Recency, Frequency, Monetary để dự đoán nhóm khách hàng.

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
│       ├── rfm_gmm_segmented.csv
│       ├── gmm_technical_cluster_profile.csv
│       ├── gmm_business_group_profile.csv
│       └── gmm_action_plan.csv
├── models/
│   ├── scaler_gmm.pkl
│   ├── gmm.pkl
│   ├── cluster_mapping.pkl
│   ├── gmm_input_limits.pkl
│   ├── gmm_ood_threshold.pkl
│   └── gmm_metadata.json
├── notebooks/
│   ├── 01_eda_clean_data.ipynb
│   ├── 02_build_rfm_uk.ipynb
│   ├── 03_eda_rfm_uk.ipynb
│   └── 04_train_gmm_rfm.ipynb
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
hoặc ấn trực tiếp vào link: https://aiomodule2.streamlit.app

Model hiện tại được train trên bảng RFM full-period của khách hàng United Kingdom. GMM dùng 10 cụm kỹ thuật, sau đó gộp thành 3 nhóm kinh doanh: khách hàng bình thường, khách hàng tiềm năng và khách hàng VIP.

Preprocessing hiện tại: giữ nguyên `Recency`, dùng `log1p` cho `Frequency` và `Monetary`, sau đó chuẩn hóa bằng `StandardScaler`.
