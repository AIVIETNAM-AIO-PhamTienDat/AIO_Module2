from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 1. CẤU HÌNH TRANG
# =========================================================

st.set_page_config(
    page_title="Phân nhóm khách hàng RFM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. ĐƯỜNG DẪN FILE
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

SCALER_PATH = MODEL_DIR / "scaler.pkl"
KMEANS_PATH = MODEL_DIR / "kmeans.pkl"
IMAGE_PATH = BASE_DIR / "assets" / "ảnh_tiêu_đề.png"


# =========================================================
# 3. TẢI SCALER VÀ MÔ HÌNH ĐÃ LƯU
# =========================================================

@st.cache_resource
def load_models():
    """Tải scaler và K-Means đã lưu trong thư mục models."""
    missing_files = []

    if not SCALER_PATH.exists():
        missing_files.append(SCALER_PATH.name)

    if not KMEANS_PATH.exists():
        missing_files.append(KMEANS_PATH.name)

    if missing_files:
        raise FileNotFoundError(
            "Không tìm thấy file: " + ", ".join(missing_files)
        )

    loaded_scaler = joblib.load(SCALER_PATH)
    loaded_kmeans = joblib.load(KMEANS_PATH)

    return loaded_scaler, loaded_kmeans


try:
    scaler, kmeans = load_models()

except Exception as error:
    st.error(
        "Không thể tải mô hình. Hãy bảo đảm `scaler.pkl` và "
        "`kmeans.pkl` nằm trong thư mục `models`."
    )
    st.exception(error)
    st.stop()


# =========================================================
# 4. THÔNG TIN CÁC NHÓM KHÁCH HÀNG
# =========================================================
# Kết quả trung bình cụm:
#
# Nhóm 0:
# Recency ≈ 43 ngày
# Frequency ≈ 3,43 đơn
# Monetary ≈ £1.191
#
# Nhóm 1:
# Recency ≈ 167 ngày
# Frequency ≈ 1,36 đơn
# Monetary ≈ £347
#
# Nhóm 2:
# Recency ≈ 18 ngày
# Frequency ≈ 13,41 đơn
# Monetary ≈ £7.188

group_information = {
    0: {
        "name": "Khách hàng thông thường / có tiềm năng",
        "icon": "🌱",
        "description": (
            "Khách hàng mua tương đối gần đây, với Recency trung bình khoảng "
            "43 ngày. Frequency khoảng 3,43 đơn hàng và Monetary khoảng £1.191. "
            "Nhóm này chưa đạt mức VIP nhưng vẫn có khả năng phát triển nếu "
            "được chăm sóc đúng cách."
        ),
        "strategy": (
            "Doanh nghiệp nên áp dụng ưu đãi cho lần mua tiếp theo, chương trình "
            "tích điểm, gợi ý sản phẩm liên quan hoặc bán theo combo để tăng "
            "tần suất mua hàng và tổng giá trị chi tiêu."
        ),
        "css_class": "potential-card",
    },

    1: {
        "name": "Khách hàng ít hoạt động / có nguy cơ rời bỏ",
        "icon": "⚠️",
        "description": (
            "Khách hàng đã lâu không quay lại mua hàng, với Recency trung bình "
            "khoảng 167 ngày. Frequency khoảng 1,36 đơn hàng và Monetary khoảng "
            "£347, thấp nhất trong ba nhóm."
        ),
        "strategy": (
            "Doanh nghiệp có thể triển khai chương trình tái kích hoạt bằng "
            "mã giảm giá, ưu đãi quay lại hoặc nhắc lại sản phẩm từng mua. "
            "Nếu khách hàng không phản hồi, có thể giảm mức độ ưu tiên chăm sóc."
        ),
        "css_class": "risk-card",
    },

    2: {
        "name": "Khách hàng VIP / trung thành",
        "icon": "👑",
        "description": (
            "Khách hàng mua hàng rất gần đây, với Recency trung bình khoảng "
            "18 ngày. Nhóm này có Frequency cao nhất, khoảng 13,41 đơn hàng, "
            "và Monetary cao nhất, khoảng £7.188."
        ),
        "strategy": (
            "Doanh nghiệp nên ưu tiên chăm sóc riêng, cung cấp ưu đãi độc quyền, "
            "quyền tiếp cận sớm sản phẩm mới, chương trình khách hàng thân thiết "
            "và các đề xuất sản phẩm được cá nhân hóa."
        ),
        "css_class": "vip-card",
    },
}


# =========================================================
# 5. HÀM TIỀN XỬ LÝ DỮ LIỆU ĐẦU VÀO
# =========================================================

def preprocess_customer(recency, frequency, monetary):
    """
    Tiền xử lý khách hàng mới giống hệt lúc huấn luyện:
    1. Logarit hóa Recency bằng np.log1p().
    2. Logarit hóa Frequency bằng np.log1p().
    3. Logarit hóa Monetary bằng np.log1p().
    4. Chuẩn hóa bằng scaler đã lưu.
    """

    customer = pd.DataFrame(
        {
            "Recency": [float(recency)],
            "Frequency": [float(frequency)],
            "Monetary": [float(monetary)],
        }
    )

    # Phải giống đúng quy trình đã dùng khi huấn luyện mô hình.
    customer["Recency"] = np.log1p(customer["Recency"])
    customer["Frequency"] = np.log1p(customer["Frequency"])
    customer["Monetary"] = np.log1p(customer["Monetary"])

    expected_features = getattr(scaler, "n_features_in_", 3)

    if customer.shape[1] != expected_features:
        raise ValueError(
            f"Scaler yêu cầu {expected_features} biến đầu vào, "
            f"nhưng ứng dụng đang cung cấp {customer.shape[1]} biến."
        )

    transformed_customer = scaler.transform(customer)

    kmeans_features = getattr(kmeans, "n_features_in_", 3)

    if transformed_customer.shape[1] != kmeans_features:
        raise ValueError(
            f"K-Means yêu cầu {kmeans_features} biến đầu vào, "
            f"nhưng dữ liệu sau chuẩn hóa có "
            f"{transformed_customer.shape[1]} biến."
        )

    return transformed_customer


# =========================================================
# 6. CSS TRANG TRÍ GIAO DIỆN
# =========================================================

st.markdown(
    """
<style>
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f2f6ff 0%, #ffffff 100%);
        border-right: 1px solid #e2e8f3;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    .hero-card {
        min-height: 330px;
        padding: 38px;
        border-radius: 22px;
        background: linear-gradient(135deg, #edf2ff 0%, #faf7ff 100%);
        border: 1px solid #dfe6f2;
        box-shadow: 0 8px 24px rgba(50, 65, 110, 0.06);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .hero-card h1 {
        margin: 0 0 18px 0;
        font-size: 43px;
        line-height: 1.2;
        color: #242b3d;
    }

    .hero-card p {
        margin: 0;
        font-size: 17px;
        line-height: 1.75;
        color: #596176;
    }

    .result-card {
        padding: 28px 30px;
        border-radius: 18px;
        margin-top: 12px;
        margin-bottom: 22px;
        box-shadow: 0 7px 22px rgba(40, 50, 80, 0.06);
    }

    .result-card h2 {
        margin: 0 0 12px 0;
        color: #252c3e;
    }

    .result-card p {
        margin: 0 0 12px 0;
        font-size: 16px;
        line-height: 1.7;
        color: #50586c;
    }

    .risk-card {
        background-color: #fff1f1;
        border-left: 7px solid #dc4c4c;
    }

    .potential-card {
        background-color: #fff8e8;
        border-left: 7px solid #eda91f;
    }

    .vip-card {
        background-color: #eefaf3;
        border-left: 7px solid #24a267;
    }

    .strategy-card {
        padding: 24px 26px;
        border-radius: 16px;
        background-color: #f6f8fc;
        border: 1px solid #e1e6ef;
        margin-top: 12px;
    }

    .strategy-card h3 {
        margin: 0 0 10px 0;
        color: #283149;
    }

    .strategy-card p {
        margin: 0;
        font-size: 16px;
        line-height: 1.7;
        color: #50586c;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e1e6ef;
        border-radius: 15px;
        padding: 18px;
        box-shadow: 0 4px 14px rgba(40, 50, 80, 0.04);
    }

    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 48px;
        border-radius: 11px;
        font-weight: 600;
    }

    .section-title {
        margin-top: 18px;
        margin-bottom: 12px;
        font-size: 28px;
        font-weight: 700;
        color: #252c3e;
    }
</style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 7. SIDEBAR – NHẬP THÔNG TIN KHÁCH HÀNG
# =========================================================

with st.sidebar:
    st.header("📌 Thông tin khách hàng")

    st.write(
        "Nhập ba chỉ số RFM để mô hình xác định "
        "khách hàng thuộc nhóm nào."
    )

    with st.form("customer_form"):
        R = st.number_input(
            "Recency",
            min_value=0,
            value=30,
            step=1,
            help="Số ngày kể từ lần mua gần nhất.",
        )

        F = st.number_input(
            "Frequency",
            min_value=1,
            value=3,
            step=1,
            help="Tổng số đơn hàng của khách hàng.",
        )

        M = st.number_input(
            "Monetary (£)",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            format="%.0f",
            help="Tổng giá trị chi tiêu, tính bằng bảng Anh.",
        )

        submitted = st.form_submit_button(
            "🔍 Phân nhóm khách hàng"
        )

    st.divider()
    st.subheader("Ý nghĩa các chỉ số")

    st.markdown(
        """
**Recency:** Số ngày kể từ lần mua gần nhất. Giá trị càng thấp, khách hàng mua càng gần đây.

**Frequency:** Tổng số đơn hàng của khách hàng.

**Monetary:** Tổng giá trị chi tiêu của khách hàng, tính bằng bảng Anh (£).
        """
    )

    st.info(
        "Ứng dụng logarit hóa Recency, Frequency và Monetary, "
        "sau đó dùng scaler và K-Means đã lưu để xác định nhóm khách hàng."
    )


# =========================================================
# 8. TRANG CHÍNH – TIÊU ĐỀ VÀ ẢNH
# =========================================================

content_column, image_column = st.columns(
    [1, 1.15],
    gap="large",
)

with content_column:
    st.markdown(
        """
<div class="hero-card">
    <h1>
        📊 Phân nhóm khách hàng bằng<br>
        <span style="white-space: nowrap;">K-Means</span>
    </h1>
    <p>
        Ứng dụng sử dụng mô hình RFM kết hợp thuật toán K-Means
        để xác định giá trị khách hàng, hỗ trợ doanh nghiệp xây dựng
        chiến lược chăm sóc và tiếp thị phù hợp.
    </p>
</div>
        """,
        unsafe_allow_html=True,
    )

with image_column:
    if IMAGE_PATH.exists():
        st.image(
            str(IMAGE_PATH),
            use_container_width=True,
        )
    else:
        st.warning(
            "Không tìm thấy ảnh tiêu đề.\n\n"
            f"Ứng dụng đang tìm ảnh tại:\n\n`{IMAGE_PATH}`"
        )


# =========================================================
# 9. PHÂN NHÓM KHÁCH HÀNG
# =========================================================

st.markdown(
    '<div class="section-title">Kết quả phân nhóm</div>',
    unsafe_allow_html=True,
)

if submitted:
    try:
        new_customer_scaled = preprocess_customer(R, F, M)

        label = int(
            kmeans.predict(new_customer_scaled)[0]
        )

        group = group_information.get(
            label,
            {
                "name": f"Nhóm khách hàng {label}",
                "icon": "👤",
                "description": "Chưa có mô tả cho nhóm khách hàng này.",
                "strategy": "Cần bổ sung chiến lược chăm sóc phù hợp.",
                "css_class": "potential-card",
            },
        )

        st.markdown(
            f"""
<div class="result-card {group['css_class']}">
    <h2>{group['icon']} {group['name']}</h2>
    <p>{group['description']}</p>
    <strong>Kết quả mô hình: Nhóm {label}</strong>
</div>
            """,
            unsafe_allow_html=True,
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric(
                label="Recency",
                value=f"{R:,.0f} ngày",
            )

        with metric_col2:
            st.metric(
                label="Frequency",
                value=f"{F:,.0f} đơn",
            )

        with metric_col3:
            st.metric(
                label="Monetary",
                value=f"£{M:,.0f}",
            )

        st.markdown(
            '<div class="section-title">'
            'Chiến lược chăm sóc đề xuất'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class="strategy-card">
    <h3>💡 Hành động đề xuất</h3>
    <p>{group['strategy']}</p>
</div>
            """,
            unsafe_allow_html=True,
        )

    except ValueError as error:
        st.error(
            "Dữ liệu đầu vào không phù hợp với mô hình. "
            "Hãy kiểm tra lại Recency, Frequency và Monetary."
        )
        st.exception(error)

    except Exception as error:
        st.error(
            "Đã xảy ra lỗi trong quá trình phân nhóm khách hàng."
        )
        st.exception(error)

else:
    st.info(
        "👈 Hãy nhập thông tin khách hàng ở thanh bên trái, "
        "sau đó nhấn nút **Phân nhóm khách hàng**."
    )


# =========================================================
# 10. CHÂN TRANG
# =========================================================

st.divider()

st.caption(
    "RFM Customer Segmentation Dashboard · "
    "K-Means Machine Learning Model"
)
