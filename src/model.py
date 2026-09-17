"""행정동 x 업종 매출 예측 회귀 모델 — 학습, 예측, 피처 중요도, 지원 우선순위 스코어."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import FEATURE_COLUMNS, MODEL_PARAMS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


@dataclass
class TrainedModel:
    model: RandomForestRegressor
    feature_columns: list
    rmse: float
    r2: float
    feature_table: pd.DataFrame


def train_model(feature_table: pd.DataFrame) -> TrainedModel:
    """RandomForestRegressor로 매출(sales_amount)을 예측하는 모델을 학습한다.

    매출액은 소수의 대형 점포(백화점 등)가 있는 행정동x업종 조합이 나머지보다
    한두 자릿수 더 커서 분포가 심하게 오른쪽으로 치우쳐 있다. 원본 스케일로
    그대로 학습하면 트리 분할이 이 극단값 몇 개에 지배되어 나머지 대다수
    조합에 대한 예측력이 떨어지므로, log1p로 압축한 값을 학습 타깃으로 쓰고
    예측/평가 시에는 expm1로 원래 금액 스케일로 되돌린다.
    """
    X = feature_table[FEATURE_COLUMNS]
    y = feature_table[TARGET_COLUMN]
    y_log = np.log1p(y)

    if len(feature_table) < 10:
        # 표본이 너무 적으면 전체를 학습에 사용하고 평가는 생략(가상데이터 초기 단계 등)
        X_train, X_test, y_train_log, y_test = X, X, y_log, y
    else:
        X_train, X_test, y_train_log, y_test_log = train_test_split(
            X, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        y_test = np.expm1(y_test_log)

    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train_log)

    y_pred = np.expm1(model.predict(X_test))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred)) if len(set(y_test)) > 1 else float("nan")

    return TrainedModel(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        rmse=rmse,
        r2=r2,
        feature_table=feature_table,
    )


def predict_sales(trained: TrainedModel, dong: str, category: str) -> Optional[dict]:
    """특정 (행정동, 업종) 조합의 예상 매출과 근거 피처값을 반환한다."""
    row = trained.feature_table[
        (trained.feature_table["dong"] == dong) & (trained.feature_table["category"] == category)
    ]
    if row.empty:
        return None

    features = row[trained.feature_columns]
    predicted = float(np.expm1(trained.model.predict(features)[0]))

    return {
        "dong": dong,
        "category": category,
        "predicted_sales": predicted,
        "actual_sales": float(row["sales_amount"].iloc[0]),
        "features": features.iloc[0].to_dict(),
    }


def get_feature_importance(trained: TrainedModel) -> pd.DataFrame:
    importances = trained.model.feature_importances_
    df = pd.DataFrame({"feature": trained.feature_columns, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def predict_all(trained: TrainedModel) -> pd.DataFrame:
    """전체 행정동 x 업종 조합에 대한 예측값을 붙인 테이블을 반환한다."""
    table = trained.feature_table.copy()
    table["predicted_sales"] = np.expm1(trained.model.predict(table[trained.feature_columns]))
    return table


def rank_categories_for_dong(trained: TrainedModel, dong: str) -> pd.DataFrame:
    """같은 행정동 내에서 업종별 예상 매출 순위를 매겨 '유리한 업종'을 추천한다."""
    predicted = predict_all(trained)
    subset = predicted[predicted["dong"] == dong].copy()
    return subset.sort_values("predicted_sales", ascending=False).reset_index(drop=True)


def compute_support_priority(trained: TrainedModel) -> pd.DataFrame:
    """행정동 단위 소상공인 지원 우선순위 스코어를 계산한다.

    스코어 설계: 예측 매출 수준이 낮고, 경쟁 강도(경쟁점포수)가 높고,
    유동인구 대비 매출 효율이 낮은 행정동일수록 지원 필요도가 높다고 판단한다.
    0~100 사이로 정규화해 랭킹에 사용한다.
    """
    predicted = predict_all(trained)
    dong_agg = predicted.groupby("dong", as_index=False).agg(
        avg_predicted_sales=("predicted_sales", "mean"),
        total_competitor_count=("competitor_count", "sum"),
        floating_population=("floating_population", "mean"),
        resident_population=("resident_population", "mean"),
        bus_stop_count=("bus_stop_count", "mean"),
    )

    def _normalize(series: pd.Series) -> pd.Series:
        span = series.max() - series.min()
        if span == 0:
            return pd.Series(0.5, index=series.index)
        return (series - series.min()) / span

    sales_score = 1 - _normalize(dong_agg["avg_predicted_sales"])  # 매출 낮을수록 지원 필요 높음
    competition_score = _normalize(dong_agg["total_competitor_count"])  # 경쟁 심할수록 지원 필요 높음
    efficiency = dong_agg["avg_predicted_sales"] / dong_agg["floating_population"].replace(0, np.nan)
    efficiency_score = 1 - _normalize(efficiency.fillna(efficiency.median()))

    dong_agg["priority_score"] = (
        0.45 * sales_score + 0.30 * competition_score + 0.25 * efficiency_score
    ) * 100
    return dong_agg.sort_values("priority_score", ascending=False).reset_index(drop=True)
