from pathlib import Path
import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "rfm_uk_full.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
N_COMPONENTS = 10
FEATURE_COLUMNS = ["Recency", "Frequency", "Monetary"]

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


def prepare_features(rfm: pd.DataFrame) -> pd.DataFrame:
    features = rfm[FEATURE_COLUMNS].copy()
    features["Frequency"] = np.log1p(features["Frequency"])
    features["Monetary"] = np.log1p(features["Monetary"])
    return features


def build_input_limits(rfm: pd.DataFrame) -> dict[str, float]:
    average_order_value = rfm["Monetary"] / rfm["Frequency"]
    return {
        "Recency_min": float(rfm["Recency"].quantile(0.01)),
        "Recency_max": float(rfm["Recency"].quantile(0.99)),
        "Frequency_min": float(rfm["Frequency"].quantile(0.01)),
        "Frequency_max": float(rfm["Frequency"].quantile(0.99)),
        "Monetary_min": float(rfm["Monetary"].quantile(0.01)),
        "Monetary_max": float(rfm["Monetary"].quantile(0.99)),
        "AverageOrderValue_min": float(average_order_value.quantile(0.01)),
        "AverageOrderValue_max": float(average_order_value.quantile(0.99)),
    }


def build_profiles(rfm_segmented: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    technical_profile = (
        rfm_segmented.groupby("GMM_Cluster")
        .agg(
            Customers=("CustomerID", "count"),
            Revenue=("Monetary", "sum"),
            RecencyMean=("Recency", "mean"),
            RecencyMedian=("Recency", "median"),
            FrequencyMean=("Frequency", "mean"),
            FrequencyMedian=("Frequency", "median"),
            MonetaryMean=("Monetary", "mean"),
            MonetaryMedian=("Monetary", "median"),
        )
        .reset_index()
    )
    technical_profile["CustomerPct"] = (
        technical_profile["Customers"] / technical_profile["Customers"].sum() * 100
    )
    technical_profile["RevenuePct"] = (
        technical_profile["Revenue"] / technical_profile["Revenue"].sum() * 100
    )

    business_profile = (
        rfm_segmented.groupby("Business_Group")
        .agg(
            Customers=("CustomerID", "count"),
            Revenue=("Monetary", "sum"),
            RecencyMean=("Recency", "mean"),
            RecencyMedian=("Recency", "median"),
            FrequencyMean=("Frequency", "mean"),
            FrequencyMedian=("Frequency", "median"),
            MonetaryMean=("Monetary", "mean"),
            MonetaryMedian=("Monetary", "median"),
        )
        .reset_index()
    )
    business_profile["CustomerPct"] = (
        business_profile["Customers"] / business_profile["Customers"].sum() * 100
    )
    business_profile["RevenuePct"] = (
        business_profile["Revenue"] / business_profile["Revenue"].sum() * 100
    )

    return technical_profile.round(4), business_profile.round(4)


def build_action_plan(business_profile: pd.DataFrame) -> pd.DataFrame:
    meanings = {
        "Khách hàng bình thường": (
            "Nhóm có quy mô lớn nhất, Frequency và Monetary chưa nổi bật. "
            "Một phần khách đã lâu không quay lại nên cần duy trì tương tác với chi phí hợp lý."
        ),
        "Khách hàng tiềm năng": (
            "Nhóm mua lặp lại tốt hơn nhóm bình thường và có Monetary cao hơn rõ rệt. "
            "Một số cụm trong nhóm này có khả năng phát triển thành VIP nếu được chăm sóc đúng cách."
        ),
        "Khách hàng VIP": (
            "Nhóm nhỏ nhất nhưng mua rất gần đây, mua thường xuyên và chi tiêu cao nhất. "
            "Đây là nhóm cần ưu tiên giữ chân."
        ),
    }
    actions = {
        "Khách hàng bình thường": (
            "Duy trì tương tác bằng ưu đãi nhẹ, email nhắc nhớ, gợi ý sản phẩm liên quan "
            "và chương trình mua kèm để tăng Frequency."
        ),
        "Khách hàng tiềm năng": (
            "Dùng chương trình tích điểm, voucher có thời hạn, bán chéo và ưu đãi nâng hạng. "
            "Các khách từng có giá trị cao nhưng lâu chưa quay lại nên có chiến dịch win-back riêng."
        ),
        "Khách hàng VIP": (
            "Triển khai loyalty program, ưu đãi độc quyền, quyền tiếp cận sớm sản phẩm mới, "
            "quà tặng theo mốc chi tiêu và chăm sóc cá nhân hóa."
        ),
    }

    action_plan = business_profile.copy()
    action_plan["BusinessMeaning"] = action_plan["Business_Group"].map(meanings)
    action_plan["MarketingAction"] = action_plan["Business_Group"].map(actions)
    return action_plan


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    rfm = pd.read_csv(DATA_PATH)
    features = prepare_features(rfm)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    aic_scores: dict[int, float] = {}
    bic_scores: dict[int, float] = {}
    for k in range(2, 11):
        gmm_test = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
        gmm_test.fit(features_scaled)
        aic_scores[k] = float(gmm_test.aic(features_scaled))
        bic_scores[k] = float(gmm_test.bic(features_scaled))

    gmm = GaussianMixture(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    labels = gmm.fit_predict(features_scaled)

    rfm_segmented = rfm.copy()
    rfm_segmented["GMM_Cluster"] = labels
    rfm_segmented["Business_Group"] = rfm_segmented["GMM_Cluster"].map(CLUSTER_MAPPING)

    train_log_likelihood = gmm.score_samples(features_scaled)
    ood_threshold = float(np.percentile(train_log_likelihood, 1))
    input_limits = build_input_limits(rfm)

    technical_profile, business_profile = build_profiles(rfm_segmented)
    action_plan = build_action_plan(business_profile)

    rfm_segmented.to_csv(PROCESSED_DIR / "rfm_gmm_segmented.csv", index=False)
    technical_profile.to_csv(PROCESSED_DIR / "gmm_technical_cluster_profile.csv", index=False)
    business_profile.to_csv(PROCESSED_DIR / "gmm_business_group_profile.csv", index=False)
    action_plan.to_csv(PROCESSED_DIR / "gmm_action_plan.csv", index=False)

    joblib.dump(scaler, MODEL_DIR / "scaler_gmm.pkl")
    joblib.dump(gmm, MODEL_DIR / "gmm.pkl")
    joblib.dump(CLUSTER_MAPPING, MODEL_DIR / "cluster_mapping.pkl")
    joblib.dump(ood_threshold, MODEL_DIR / "gmm_ood_threshold.pkl")
    joblib.dump(input_limits, MODEL_DIR / "gmm_input_limits.pkl")

    metadata = {
        "model_type": "GaussianMixture",
        "country_scope": "United Kingdom",
        "source_file": "data/processed/rfm_uk_full.csv",
        "random_state": RANDOM_STATE,
        "n_components": N_COMPONENTS,
        "feature_columns": FEATURE_COLUMNS,
        "transform_rule": "np.log1p applied to Frequency and Monetary, Recency kept raw, then StandardScaler",
        "selection_reason": (
            "AIC and BIC were evaluated for K=2..10. Both metrics are lowest at K=10. "
            "The 10 technical clusters are mapped into three business groups for reporting and marketing actions."
        ),
        "aic_scores": {str(k): round(v, 4) for k, v in aic_scores.items()},
        "bic_scores": {str(k): round(v, 4) for k, v in bic_scores.items()},
        "ood_threshold_p01_log_likelihood": round(ood_threshold, 6),
        "input_limits_p01_p99": input_limits,
        "cluster_mapping": {str(k): v for k, v in CLUSTER_MAPPING.items()},
    }
    (MODEL_DIR / "gmm_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Saved GMM artifacts")
    print(f"Customers: {len(rfm_segmented)}")
    print(f"Business groups: {business_profile['Business_Group'].nunique()}")


if __name__ == "__main__":
    main()
