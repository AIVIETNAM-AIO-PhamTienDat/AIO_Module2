from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =====================================================
# 1. CẤU HÌNH TRANG
# =====================================================

st.set_page_config(
    page_title="Customer Segmentation | GMM 10 cụm",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"


# =====================================================
# 2. CSS GIAO DIỆN
# =====================================================

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 5% 10%, rgba(85, 101, 255, 0.10), transparent 24%),
                radial-gradient(circle at 95% 15%, rgba(22, 194, 187, 0.10), transparent 24%),
                linear-gradient(180deg, #f8faff 0%, #f3f7ff 52%, #ffffff 100%);
        }

        .block-container {
            max-width: 1320px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #eff4ff 0%, #f8fbff 55%, #ffffff 100%);
            border-right: 1px solid #dfe7f6;
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
        }

        h1, h2, h3 {
            color: #10214a;
            letter-spacing: -0.02em;
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(241,247,255,0.98));
            border: 1px solid #dce7fb;
            border-radius: 26px;
            padding: 30px 34px;
            box-shadow: 0 18px 48px rgba(31, 59, 118, 0.11);
            margin-bottom: 18px;
        }

        .hero-eyebrow {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            background: #edf2ff;
            color: #3558dd;
            font-weight: 700;
            font-size: 0.82rem;
            margin-bottom: 12px;
        }

        .hero-title {
            font-size: clamp(2rem, 4vw, 3.6rem);
            line-height: 1.06;
            font-weight: 850;
            color: #0b1d46;
            margin: 0 0 10px 0;
        }

        .hero-subtitle {
            font-size: 1.06rem;
            line-height: 1.7;
            color: #637294;
            margin: 0;
        }

        .soft-card {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid #e1e8f5;
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(38, 64, 116, 0.07);
            min-height: 100%;
        }

        .result-card {
            border-radius: 22px;
            padding: 24px;
            color: #11224a;
            background: linear-gradient(135deg, #ffffff 0%, #edf4ff 100%);
            border: 1px solid #d8e4fa;
            box-shadow: 0 14px 34px rgba(34, 67, 132, 0.10);
            margin-top: 10px;
        }

        .result-label {
            color: #677694;
            font-size: 0.88rem;
            margin-bottom: 5px;
        }

        .result-title {
            color: #173783;
            font-size: 1.65rem;
            font-weight: 850;
            margin-bottom: 8px;
        }

        .segment-normal {
            border-left: 7px solid #7f6df2;
        }

        .segment-potential {
            border-left: 7px solid #2d6cdf;
        }

        .segment-vip {
            border-left: 7px solid #10aaa2;
        }

        .segment-special {
            border-left: 7px solid #f0a43c;
        }

        .segment-warning {
            border-left: 7px solid #e06c75;
        }

        .mini-guide {
            background: #f8faff;
            border: 1px solid #e4eaf6;
            border-radius: 16px;
            padding: 16px 18px;
            height: 100%;
        }

        .mini-guide b {
            color: #193875;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.94);
            border: 1px solid #e0e8f7;
            padding: 14px 16px;
            border-radius: 16px;
            box-shadow: 0 8px 22px rgba(31, 59, 118, 0.06);
        }

        div[data-testid="stMetricLabel"] {
            color: #667594;
        }

        .stButton > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button {
            width: 100%;
            border: 0;
            border-radius: 13px;
            padding: 0.75rem 1rem;
            font-weight: 750;
            color: white;
            background: linear-gradient(90deg, #4169e1 0%, #6d5bea 52%, #16aaa3 100%);
            box-shadow: 0 9px 22px rgba(68, 91, 210, 0.22);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stFormSubmitButton > button:hover {
            filter: brightness(1.04);
            transform: translateY(-1px);
        }

        div[data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(255,255,255,0.68);
            padding: 8px;
            border-radius: 15px;
            border: 1px solid #e1e8f5;
        }

        button[data-baseweb="tab"] {
            border-radius: 11px;
            padding-left: 18px;
            padding-right: 18px;
        }

        [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid #dfe7f4;
        }

        .footer-note {
            color: #7b88a4;
            font-size: 0.84rem;
            text-align: center;
            margin-top: 28px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 3. TẢI MÔ HÌNH
# =====================================================

REQUIRED_MODEL_FILES = {
    "scaler": "scaler_gmm.pkl",
    "gmm": "gmm.pkl",
    "ood_threshold": "gmm_ood_threshold.pkl",
    "input_limits": "gmm_input_limits.pkl",
}


# =====================================================
# PHÂN NHÓM KINH DOANH MỚI
# =====================================================

EXPECTED_GMM_COMPONENTS = 10

CLUSTER_MAPPING = {
    0: "Khách hàng bình thường",
    1: "Khách hàng bình thường",
    2: "Khách hàng bình thường",
    3: "Khách hàng VIP",
    4: "Khách hàng tiềm năng",
    5: "Khách hàng bình thường",
    6: "Khách hàng tiềm năng",
    7: "Khách hàng bình thường",
    8: "Khách hàng tiềm năng",
    9: "Khách hàng bình thường",
}


@st.cache_resource(show_spinner=False)
def load_model_assets() -> dict[str, Any]:
    missing = [
        filename
        for filename in REQUIRED_MODEL_FILES.values()
        if not (MODEL_DIR / filename).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Thiếu các file mô hình: " + ", ".join(missing)
        )

    return {
        key: joblib.load(MODEL_DIR / filename)
        for key, filename in REQUIRED_MODEL_FILES.items()
    }


try:
    assets = load_model_assets()
    scaler = assets["scaler"]
    gmm = assets["gmm"]
    ood_threshold = float(assets["ood_threshold"])
    input_limits = assets["input_limits"]
except Exception as exc:
    st.error("Không thể tải mô hình GMM.")
    st.code(str(exc))
    st.info(
        "Hãy đặt file app này cùng thư mục với 4 file mô hình: "
        "`scaler_gmm.pkl`, `gmm.pkl`, `gmm_ood_threshold.pkl` "
        "và `gmm_input_limits.pkl` trong thư mục `models`."
    )
    st.stop()


# Kiểm tra đúng mô hình GMM 10 cụm
if getattr(gmm, "n_components", None) != EXPECTED_GMM_COMPONENTS:
    st.error(
        "File gmm.pkl đang tải không phải mô hình GMM 10 cụm."
    )
    st.info(
        f"Mô hình hiện tại có {getattr(gmm, 'n_components', 'không xác định')} cụm. "
        "Hãy chạy lại notebook K = 10 và thay file gmm.pkl, scaler cùng các ngưỡng mới."
    )
    st.stop()


# =====================================================
# 4. HÀM HỖ TRỢ
# =====================================================


def format_pound(value: float) -> str:
    return f"£{value:,.2f}"


def segment_css_class(segment: str) -> str:
    text = segment.lower()
    if "vip" in text:
        return "segment-vip"
    if "tiềm năng" in text:
        return "segment-potential"
    if "một lần" in text or "giá trị cao" in text:
        return "segment-special"
    if "chưa phân nhóm" in text or "không hợp lệ" in text:
        return "segment-warning"
    return "segment-normal"


def recommendation_for(segment: str) -> str:
    text = segment.lower()
    if "vip" in text:
        return (
            "Ưu tiên chương trình thành viên, ưu đãi cá nhân hóa và chăm sóc sau mua "
            "để duy trì giá trị vòng đời khách hàng."
        )
    if "tiềm năng" in text:
        return (
            "Khuyến khích mua lại bằng voucher có thời hạn, chương trình tích điểm, "
            "gợi ý sản phẩm liên quan và chiến dịch nâng hạng. "
            "Riêng khách thuộc cụm 8 nên được ưu tiên tái kích hoạt."
        )
    if "một lần" in text:
        return (
            "Theo dõi khả năng mua lại, giới thiệu sản phẩm bổ trợ và chăm sóc riêng "
            "vì đây là khách hàng có giá trị đơn hàng cao."
        )
    if "bình thường" in text:
        return (
            "Duy trì tương tác bằng ưu đãi nhẹ, nội dung nhắc nhớ và các gói mua kèm "
            "để tăng Frequency và Monetary."
        )
    return "Kiểm tra lại dữ liệu đầu vào trước khi đưa ra quyết định kinh doanh."


def safe_limit_check(
    value: float,
    minimum_key: str,
    maximum_key: str,
    label: str,
) -> list[str]:
    reasons: list[str] = []
    minimum = input_limits.get(minimum_key)
    maximum = input_limits.get(maximum_key)

    if minimum is not None and value < float(minimum):
        reasons.append(f"{label} thấp hơn phạm vi thường gặp của dữ liệu huấn luyện.")
    if maximum is not None and value > float(maximum):
        reasons.append(f"{label} cao hơn phạm vi thường gặp của dữ liệu huấn luyện.")

    return reasons


def predict_customer(
    recency: float,
    frequency: float,
    monetary: float,
) -> dict[str, Any]:
    """Phân nhóm một khách hàng và trả về toàn bộ thông tin hiển thị."""

    result: dict[str, Any] = {
        "segment": "Chưa phân nhóm",
        "cluster": None,
        "confidence": None,
        "confidence_gap": None,
        "probabilities": None,
        "average_order_value": None,
        "status": "warning",
        "reasons": [],
        "log_likelihood": None,
    }

    try:
        recency = float(recency)
        frequency = float(frequency)
        monetary = float(monetary)
    except (TypeError, ValueError):
        result["segment"] = "Dữ liệu không hợp lệ"
        result["reasons"] = ["Recency, Frequency và Monetary phải là số."]
        return result

    if not all(np.isfinite([recency, frequency, monetary])):
        result["segment"] = "Dữ liệu không hợp lệ"
        result["reasons"] = ["Dữ liệu không được để trống hoặc chứa giá trị vô hạn."]
        return result

    if recency < 0 or frequency < 1 or monetary <= 0:
        result["segment"] = "Dữ liệu không hợp lệ"
        result["reasons"] = [
            "Recency phải từ 0, Frequency phải từ 1 và Monetary phải lớn hơn 0."
        ]
        return result

    average_order_value = monetary / frequency
    result["average_order_value"] = average_order_value

    # Quy tắc kinh doanh đặc biệt đã có trong app gốc.
    if frequency <= 1 and monetary >= 1000:
        result.update(
            {
                "segment": "Khách hàng mua một lần giá trị cao",
                "status": "success",
                "reasons": ["Kết quả được xác định bằng quy tắc kinh doanh đặc biệt."],
            }
        )
        return result

    abnormal_reasons: list[str] = []
    abnormal_reasons.extend(
        safe_limit_check(recency, "Recency_min", "Recency_max", "Recency")
    )
    abnormal_reasons.extend(
        safe_limit_check(
            frequency,
            "Frequency_min",
            "Frequency_max",
            "Frequency",
        )
    )
    abnormal_reasons.extend(
        safe_limit_check(
            monetary,
            "Monetary_min",
            "Monetary_max",
            "Monetary",
        )
    )
    abnormal_reasons.extend(
        safe_limit_check(
            average_order_value,
            "AverageOrderValue_min",
            "AverageOrderValue_max",
            "Giá trị trung bình mỗi đơn",
        )
    )

    new_customer = pd.DataFrame(
        [[recency, frequency, monetary]],
        columns=["Recency", "Frequency", "Monetary"],
    )
    new_customer["Frequency"] = np.log1p(new_customer["Frequency"])
    new_customer["Monetary"] = np.log1p(new_customer["Monetary"])

    try:
        scaled_customer = scaler.transform(new_customer)
        log_likelihood = float(gmm.score_samples(scaled_customer)[0])
        result["log_likelihood"] = log_likelihood
    except Exception as exc:
        result["segment"] = "Không thể dự đoán"
        result["reasons"] = [f"Lỗi khi xử lý dữ liệu: {exc}"]
        return result

    if log_likelihood < ood_threshold:
        abnormal_reasons.append(
            "Tổ hợp Recency, Frequency và Monetary nằm ngoài phân phối mà mô hình đã học."
        )

    if abnormal_reasons:
        result["segment"] = "Chưa phân nhóm đáng tin cậy"
        result["reasons"] = abnormal_reasons
        return result

    cluster = int(gmm.predict(scaled_customer)[0])
    probabilities = gmm.predict_proba(scaled_customer)[0]
    confidence = float(probabilities[cluster])
    sorted_probabilities = np.sort(probabilities)[::-1]
    confidence_gap = (
        float(sorted_probabilities[0] - sorted_probabilities[1])
        if len(sorted_probabilities) > 1
        else 1.0
    )

    segment = CLUSTER_MAPPING.get(cluster, "Chưa xác định")
    reasons: list[str] = []
    status = "success"

    if confidence < 0.60 or confidence_gap < 0.15:
        status = "caution"
        reasons.append(
            "Khách hàng nằm gần ranh giới giữa nhiều cụm; nên kết hợp thêm đánh giá kinh doanh."
        )

    if cluster == 3:
        reasons.append(
            "Cụm 3 là nhóm VIP: mua gần đây, mua thường xuyên và chi tiêu cao nhất."
        )

    elif cluster in {4, 6}:
        reasons.append(
            f"Cụm {cluster} thuộc nhóm tiềm năng và có khả năng phát triển thành VIP."
        )

    elif cluster == 8:
        reasons.append(
            "Cụm 8 thuộc nhóm tiềm năng: giá trị lịch sử cao nhưng cần ưu tiên tái kích hoạt."
        )

    result.update(
        {
            "segment": segment,
            "cluster": cluster,
            "confidence": confidence,
            "confidence_gap": confidence_gap,
            "probabilities": probabilities,
            "status": status,
            "reasons": reasons,
        }
    )
    return result


def render_prediction_result(result: dict[str, Any]) -> None:
    segment = str(result["segment"])
    card_class = segment_css_class(segment)
    cluster_text = (
        f"Cụm GMM {result['cluster']}"
        if result["cluster"] is not None
        else "Quy tắc/kiểm tra ngoài mô hình"
    )
    confidence_text = (
        f"{result['confidence']:.2%}"
        if result["confidence"] is not None
        else "—"
    )
    average_value_text = (
        format_pound(float(result["average_order_value"]))
        if result["average_order_value"] is not None
        else "—"
    )

    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <div class="result-label">KẾT QUẢ PHÂN NHÓM</div>
            <div class="result-title">{segment}</div>
            <div style="color:#637294; line-height:1.65;">
                {cluster_text} &nbsp;•&nbsp; Mức chắc chắn: <b>{confidence_text}</b>
                &nbsp;•&nbsp; Giá trị trung bình/đơn: <b>{average_value_text}</b>
            </div>
            <div style="margin-top:12px; color:#384b74; line-height:1.65;">
                <b>Gợi ý hành động:</b> {recommendation_for(segment)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result["reasons"]:
        for reason in result["reasons"]:
            if result["status"] in {"warning", "caution"}:
                st.warning(reason)
            else:
                st.info(reason)

    if result["probabilities"] is not None:
        probability_df = pd.DataFrame(
            {
                "Cụm": [f"Cụm {index}" for index in range(len(result["probabilities"]))],
                "Xác suất": result["probabilities"],
            }
        ).set_index("Cụm")
        st.markdown("#### Xác suất thuộc từng cụm")
        st.bar_chart(probability_df, y="Xác suất", height=280)


def sample_recent_customers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [17850, "Olivia", 7, 12, 2180.00],
            [17851, "James", 18, 6, 840.00],
            [17852, "Amelia", 43, 3, 315.50],
            [17853, "George", 96, 2, 162.00],
            [17854, "Isla", 14, 8, 1295.00],
            [17855, "Harry", 165, 1, 84.90],
            [17856, "Mia", 31, 5, 602.00],
            [17857, "Jack", 72, 4, 441.50],
            [17858, "Sophia", 5, 17, 3525.00],
            [17859, "Noah", 120, 1, 1250.00],
        ],
        columns=["CustomerID", "CustomerName", "Recency", "Frequency", "Monetary"],
    )


def classify_customer_table(customer_table: pd.DataFrame) -> pd.DataFrame:
    output_rows: list[dict[str, Any]] = []

    for _, row in customer_table.iterrows():
        prediction = predict_customer(
            row.get("Recency"),
            row.get("Frequency"),
            row.get("Monetary"),
        )

        confidence = prediction["confidence"]
        output_rows.append(
            {
                "CustomerID": row.get("CustomerID", ""),
                "CustomerName": row.get("CustomerName", ""),
                "Recency": row.get("Recency"),
                "Frequency": row.get("Frequency"),
                "Monetary": row.get("Monetary"),
                "AverageOrderValue": prediction.get("average_order_value"),
                "BusinessGroup": prediction["segment"],
                "GMMCluster": prediction["cluster"],
                "Confidence": confidence,
                "Status": prediction["status"],
                "Note": " | ".join(prediction["reasons"]),
            }
        )

    return pd.DataFrame(output_rows)


# =====================================================
# 5. SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("## 📊 RFM + GMM 10 cụm")
    st.caption("Ứng dụng phân nhóm khách hàng thương mại điện tử")

    image_candidates = [
        ASSETS_DIR / "anh_gmm_cong_nghe.png",
        ASSETS_DIR / "anh_tieu_de.png",
        ASSETS_DIR / "ảnh_tiêu_đề.png",
        ASSETS_DIR / "header.png",
    ]
    sidebar_image = next((path for path in image_candidates if path.exists()), None)
    if sidebar_image:
        st.image(str(sidebar_image), use_container_width=True)

    st.markdown("---")
    st.markdown("**Ba chỉ số đầu vào**")
    st.caption("R — Recency: số ngày kể từ lần mua gần nhất")
    st.caption("F — Frequency: tổng số đơn hàng")
    st.caption("M — Monetary: tổng số tiền khách đã chi (£)")

    st.markdown("---")
    st.success("Mô hình đã được tải thành công")
    st.caption(f"Số cụm GMM: {getattr(gmm, 'n_components', '—')}")
    st.caption(f"Ngưỡng OOD: {ood_threshold:.4f}")

    st.markdown("---")
    st.markdown("**Quy tắc gộp cụm mới**")
    st.caption("Cụm 3 → Khách hàng VIP")
    st.caption("Cụm 4, 6, 8 → Khách hàng tiềm năng")
    st.caption("Cụm 0, 1, 2, 5, 7, 9 → Khách hàng bình thường")


# =====================================================
# 6. HERO
# =====================================================

hero_col, image_col = st.columns([0.95, 1.45], gap="large")

with hero_col:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">CUSTOMER ANALYTICS DASHBOARD</div>
            <div class="hero-title">Phân nhóm khách hàng bằng RFM + GMM 10 cụm</div>
            <p class="hero-subtitle">
                Dự đoán nhóm khách hàng, kiểm tra dữ liệu bất thường và phân loại hàng loạt
                ngay trên một giao diện trực quan, dễ sử dụng.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    guide_1, guide_2, guide_3 = st.columns(3)
    with guide_1:
        st.markdown(
            '<div class="mini-guide"><b>🗓️ Recency</b><br><span style="color:#74809a">Càng thấp, khách càng mới quay lại.</span></div>',
            unsafe_allow_html=True,
        )
    with guide_2:
        st.markdown(
            '<div class="mini-guide"><b>🛒 Frequency</b><br><span style="color:#74809a">Số đơn càng cao, mức gắn kết càng lớn.</span></div>',
            unsafe_allow_html=True,
        )
    with guide_3:
        st.markdown(
            '<div class="mini-guide"><b>💷 Monetary</b><br><span style="color:#74809a">Tổng chi tiêu phản ánh giá trị khách hàng.</span></div>',
            unsafe_allow_html=True,
        )

with image_col:
    hero_image = next((path for path in image_candidates if path.exists()), None)
    if hero_image:
        st.image(str(hero_image), use_container_width=True)
    else:
        st.info("Đặt ảnh tiêu đề trong cùng thư mục với app để hiển thị ảnh đại diện.")


# =====================================================
# 7. NỘI DUNG CHÍNH
# =====================================================

single_tab, batch_tab = st.tabs(
    [
        "👤 Phân nhóm một khách hàng",
        "📋 Bảng 10 khách hàng gần đây",
    ]
)


with single_tab:
    st.markdown("### Nhập thông tin RFM")
    st.caption("Điền dữ liệu của một khách hàng rồi nhấn **Phân nhóm khách hàng**.")

    with st.form("single_customer_form"):
        input_col_1, input_col_2, input_col_3 = st.columns(3)

        with input_col_1:
            recency = st.number_input(
                "Recency — Số ngày từ lần mua gần nhất",
                min_value=0,
                value=30,
                step=1,
                help="Ví dụ: mua gần nhất cách đây 30 ngày thì nhập 30.",
            )

        with input_col_2:
            frequency = st.number_input(
                "Frequency — Tổng số đơn hàng",
                min_value=1,
                value=3,
                step=1,
                help="Tổng số hóa đơn/đơn hàng của khách hàng.",
            )

        with input_col_3:
            monetary = st.number_input(
                "Monetary — Tổng chi tiêu (£)",
                min_value=0.01,
                value=300.00,
                step=10.00,
                format="%.2f",
                help="Tổng giá trị mua hàng của khách hàng bằng bảng Anh.",
            )

        submitted = st.form_submit_button("✨ Phân nhóm khách hàng")

    live_aov = monetary / frequency
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Recency", f"{recency:.0f} ngày")
    metric_2.metric("Frequency", f"{frequency:.0f} đơn")
    metric_3.metric("Monetary", format_pound(monetary))
    metric_4.metric("Trung bình/đơn", format_pound(live_aov))

    if submitted:
        prediction = predict_customer(recency, frequency, monetary)
        render_prediction_result(prediction)


with batch_tab:
    st.markdown("### Nhập hoặc chỉnh sửa danh sách khách hàng")
    st.caption(
        "Bảng đã có sẵn 10 khách hàng mẫu. Bạn có thể sửa trực tiếp, thêm dòng mới hoặc xóa dòng trước khi phân nhóm."
    )

    if "customer_editor_data" not in st.session_state:
        st.session_state.customer_editor_data = sample_recent_customers()

    editor_data = st.data_editor(
        st.session_state.customer_editor_data,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="recent_customer_editor",
        column_config={
            "CustomerID": st.column_config.NumberColumn(
                "Mã khách hàng",
                min_value=1,
                step=1,
                format="%d",
            ),
            "CustomerName": st.column_config.TextColumn(
                "Tên khách hàng",
                help="Có thể để trống nếu dữ liệu không có tên.",
            ),
            "Recency": st.column_config.NumberColumn(
                "Recency",
                min_value=0,
                step=1,
                format="%d",
            ),
            "Frequency": st.column_config.NumberColumn(
                "Frequency",
                min_value=1,
                step=1,
                format="%d",
            ),
            "Monetary": st.column_config.NumberColumn(
                "Monetary (£)",
                min_value=0.01,
                step=10.0,
                format="£%.2f",
            ),
        },
    )

    action_col_1, action_col_2 = st.columns([1.1, 1])
    with action_col_1:
        classify_all = st.button(
            "🚀 Phân nhóm toàn bộ danh sách",
            key="classify_all_customers",
        )
    with action_col_2:
        reset_table = st.button(
            "↺ Khôi phục 10 khách hàng mẫu",
            key="reset_customer_table",
        )

    if reset_table:
        st.session_state.customer_editor_data = sample_recent_customers()
        st.session_state.pop("batch_results", None)
        st.rerun()

    if classify_all:
        required_columns = {"CustomerID", "CustomerName", "Recency", "Frequency", "Monetary"}
        missing_columns = required_columns.difference(editor_data.columns)

        if missing_columns:
            st.error("Bảng đang thiếu cột: " + ", ".join(sorted(missing_columns)))
        elif editor_data.empty:
            st.warning("Bảng chưa có khách hàng để phân nhóm.")
        else:
            st.session_state.customer_editor_data = editor_data.copy()
            st.session_state.batch_results = classify_customer_table(editor_data)

    if "batch_results" in st.session_state:
        batch_results = st.session_state.batch_results.copy()

        valid_results = batch_results[
            ~batch_results["BusinessGroup"].isin(
                ["Dữ liệu không hợp lệ", "Không thể dự đoán", "Chưa phân nhóm đáng tin cậy"]
            )
        ]

        total_customers = len(batch_results)
        vip_count = batch_results["BusinessGroup"].str.contains("VIP", case=False, na=False).sum()
        potential_count = batch_results["BusinessGroup"].str.contains(
            "tiềm năng", case=False, na=False
        ).sum()
        warning_count = (batch_results["Status"] == "warning").sum()

        summary_col_1, summary_col_2, summary_col_3, summary_col_4 = st.columns(4)
        summary_col_1.metric("Tổng khách hàng", f"{total_customers}")
        summary_col_2.metric("Khách hàng VIP", f"{vip_count}")
        summary_col_3.metric("Khách hàng tiềm năng", f"{potential_count}")
        summary_col_4.metric("Cần kiểm tra", f"{warning_count}")

        if not valid_results.empty:
            distribution = (
                valid_results["BusinessGroup"]
                .value_counts()
                .rename_axis("Nhóm khách hàng")
                .to_frame("Số lượng")
            )
            st.markdown("#### Phân bố nhóm khách hàng")
            st.bar_chart(distribution, y="Số lượng", height=300)

        display_results = batch_results.copy()
        display_results["Monetary"] = display_results["Monetary"].map(
            lambda value: format_pound(float(value)) if pd.notna(value) else "—"
        )
        display_results["AverageOrderValue"] = display_results["AverageOrderValue"].map(
            lambda value: format_pound(float(value)) if pd.notna(value) else "—"
        )
        display_results["Confidence"] = display_results["Confidence"].map(
            lambda value: f"{float(value):.2%}" if pd.notna(value) else "—"
        )
        display_results["GMMCluster"] = display_results["GMMCluster"].map(
            lambda value: f"Cụm {int(value)}" if pd.notna(value) else "—"
        )

        display_results = display_results.rename(
            columns={
                "CustomerID": "Mã khách hàng",
                "CustomerName": "Tên khách hàng",
                "Recency": "Recency",
                "Frequency": "Frequency",
                "Monetary": "Monetary",
                "AverageOrderValue": "Trung bình/đơn",
                "BusinessGroup": "Nhóm khách hàng",
                "GMMCluster": "Cụm GMM",
                "Confidence": "Mức chắc chắn",
                "Status": "Trạng thái",
                "Note": "Ghi chú",
            }
        )

        st.markdown("#### Kết quả chi tiết")
        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ghi chú": st.column_config.TextColumn(width="large"),
                "Nhóm khách hàng": st.column_config.TextColumn(width="medium"),
            },
        )

        csv_bytes = batch_results.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Tải kết quả phân nhóm CSV",
            data=csv_bytes,
            file_name="ket_qua_phan_nhom_gmm.csv",
            mime="text/csv",
        )


st.markdown(
    '<div class="footer-note">Customer Segmentation Dashboard • GMM 10 cụm → 3 nhóm kinh doanh</div>',
    unsafe_allow_html=True,
)
