"""행정동 x 업종 매출 예측 회귀 모델 — 학습, 예측, 피처 중요도, 지원 우선순위 스코어."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn.inspection import partial_dependence

from src.config import (
    FEATURE_COLUMNS,
    MODEL_PARAMS,
    PRIORITY_WEIGHT_COMPETITION,
    PRIORITY_WEIGHT_EFFICIENCY,
    PRIORITY_WEIGHT_SALES,
    RANDOM_STATE,
    SAFETY_NET_PROGRAMS,
    SIMULATION_FEATURES,
    SIMULATION_GRID_RESOLUTION,
    SIMULATION_MODEL_PARAMS,
    TARGET_COLUMN,
    TEST_SIZE,
)


@dataclass
class TrainedModel:
    model: RandomForestRegressor
    simulation_model: RandomForestRegressor
    feature_columns: list
    rmse: float
    r2: float
    feature_table: pd.DataFrame
    pd_curves: dict


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

    # 시뮬레이터(what-if) 전용 모델 — 학습에 없던 피처 조합으로 재예측할 때
    # 결과가 비상식적으로 튀지 않도록 더 보수적인 하이퍼파라미터로 별도 학습한다.
    simulation_model = RandomForestRegressor(**SIMULATION_MODEL_PARAMS)
    simulation_model.fit(X_train, y_train_log)

    y_pred = np.expm1(model.predict(X_test))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred)) if len(set(y_test)) > 1 else float("nan")

    # 부분의존도(Partial Dependence) 곡선을 미리 계산해둔다: 특정 행 하나의 예측을
    # 재계산하면(predict_with_overrides) 그 행의 트리 분기 잡음에 좌우돼 결과가
    # 비상식적으로 튈 수 있으므로, "이 피처를 바꾸면 전체 데이터 평균적으로 예측이
    # 어떻게 변하는가"라는 안정적인 상대 곡선을 시뮬레이터에 사용한다.
    pd_curves = {}
    X_all = feature_table[FEATURE_COLUMNS].astype(float)
    for feature in SIMULATION_FEATURES:
        idx = FEATURE_COLUMNS.index(feature)
        result = partial_dependence(
            simulation_model,
            X_all,
            features=[idx],
            grid_resolution=SIMULATION_GRID_RESOLUTION,
            percentiles=(0, 1),  # sklearn 기본값(5~95%)이 아니라 실측 최소~최대 전체 범위를 쓴다
            kind="average",
        )
        grid = result["grid_values"][0]
        avg_sales = np.expm1(result["average"][0])
        # 그리드 포인트가 성글고(19~20개) 표본이 작아 곡선 자체에 잔물결(비단조 잡음)이
        # 남아있어, 3포인트 이동평균으로 살짝 다듬는다 — "정류장을 늘렸는데 어느 지점만
        # 콕 집어 확 낮아진다" 같은 부자연스러운 굴곡을 줄이기 위함.
        avg_sales = pd.Series(avg_sales).rolling(window=3, center=True, min_periods=1).mean().to_numpy()
        pd_curves[feature] = (grid, avg_sales)

    return TrainedModel(
        model=model,
        simulation_model=simulation_model,
        feature_columns=FEATURE_COLUMNS,
        rmse=rmse,
        r2=r2,
        feature_table=feature_table,
        pd_curves=pd_curves,
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


def _pd_effect_ratio(trained: TrainedModel, feature: str, from_value: float, to_value: float) -> float:
    """부분의존도 곡선에서 from_value -> to_value로 바뀔 때의 상대 배율을 구한다."""
    if feature not in trained.pd_curves:
        return 1.0
    grid, avg_sales = trained.pd_curves[feature]
    from_effect = float(np.interp(from_value, grid, avg_sales))
    to_effect = float(np.interp(to_value, grid, avg_sales))
    if from_effect <= 0:
        return 1.0
    return to_effect / from_effect


def simulate_scenario(trained: TrainedModel, dong: str, category: str, overrides: Optional[dict] = None) -> Optional[dict]:
    """(행정동, 업종)의 실제 예측치에, 피처를 바꿨을 때의 "평균적 효과"를 곱해 시나리오를 추정한다.

    "버스정류장을 2개 늘리면 매출이 어떻게 될까?" 같은 정책 개입 what-if 시뮬레이션에 쓴다.
    단순히 그 행 하나를 재예측하면(predict_with_overrides처럼) 트리 분기 잡음 때문에
    작은 입력 변화에도 결과가 폭락/폭등하는 문제가 있었다(실측: 정류장 1개 추가만으로
    -80% 급락하는 등). 대신 전체 데이터셋에서 그 피처의 부분의존도(다른 조건은 그대로
    두고 이 피처만 바꿨을 때 평균적으로 예측이 어떻게 변하는지)를 구해, 그 "상대적
    변화율"을 이 행의 실제(정확한) 기준 예측치에 곱하는 방식을 쓴다. 이렇게 하면
    한 행의 우연한 트리 분기 잡음에 휘둘리지 않는, 훨씬 안정적인 시나리오 추정이 된다.
    """
    base = predict_sales(trained, dong, category)
    if base is None:
        return None

    multiplier = 1.0
    new_features = dict(base["features"])
    for feature, new_value in (overrides or {}).items():
        if feature not in trained.feature_columns:
            continue
        old_value = base["features"][feature]
        multiplier *= _pd_effect_ratio(trained, feature, old_value, new_value)
        new_features[feature] = new_value

    return {
        "predicted_sales": base["predicted_sales"] * multiplier,
        "baseline_sales": base["predicted_sales"],
        "features": new_features,
    }


def simulate_dong_scenario(trained: TrainedModel, dong: str, overrides: dict) -> Optional[dict]:
    """행정동 전체(모든 업종 합산) 기준으로 정책 시나리오 효과를 추정한다.

    버스정류장 확충처럼 특정 업종이 아니라 행정동 전체에 영향을 미치는 정책 개입의
    효과를 보려면, 그 동의 모든 업종에 대해 simulate_scenario를 적용해 합산해야 한다.
    """
    categories = trained.feature_table.loc[trained.feature_table["dong"] == dong, "category"].unique()
    if len(categories) == 0:
        return None

    baseline_total = 0.0
    scenario_total = 0.0
    for category in categories:
        result = simulate_scenario(trained, dong, category, overrides)
        if result:
            baseline_total += result["baseline_sales"]
            scenario_total += result["predicted_sales"]

    if baseline_total <= 0:
        return None
    return {"baseline_sales": baseline_total, "predicted_sales": scenario_total}


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

    # 최종 점수뿐 아니라 3개 하위 점수(0~100)도 함께 남겨서, UI에서 "왜 이 점수인지"를
    # 항목별로 분해해 보여줄 수 있게 한다 — 블랙박스처럼 보이지 않도록 하기 위함.
    dong_agg["sales_score"] = sales_score * 100
    dong_agg["competition_score"] = competition_score * 100
    dong_agg["efficiency_score"] = efficiency_score * 100
    dong_agg["priority_score"] = (
        PRIORITY_WEIGHT_SALES * sales_score
        + PRIORITY_WEIGHT_COMPETITION * competition_score
        + PRIORITY_WEIGHT_EFFICIENCY * efficiency_score
    ) * 100
    return dong_agg.sort_values("priority_score", ascending=False).reset_index(drop=True)


# 지배 요인 -> 추천 지원 유형 카탈로그. 어떤 요인이 지배적이든, 정량적으로 시뮬레이션
# 가능한 정책 변수는 버스정류장(대중교통 접근성)뿐이다(모델 피처 중 유일하게 촘촘한
# 실측 범위를 가짐). floating_population은 실데이터에 값이 2개(구 단위)뿐이라 증가
# 시나리오가 관측 범위를 넘어서면 그대로 0% 처리되어 신뢰할 수 없고, competition은
# 이 데이터에서 "경쟁점포가 많을수록 매출도 높게" 나오는 상관관계라 정량화 근거가
# 없다. 그래서 지배 요인별 설명은 정성적 진단으로 두고, 버스정류장 시뮬레이션은
# 지배 요인과 무관하게 항상 "참고 가능한 정량적 근거"로 별도 제공한다 — 없는
# 정밀도를 있는 것처럼 보여주지 않기 위한 의도적 선택.
_SUPPORT_TYPE_CATALOG = {
    "sales": {
        "title": "매출 진작 지원 (마케팅·홍보·지역 행사)",
        "description": "전반적인 매출 수준 자체가 낮은 것이 가장 큰 요인입니다. 유동인구를 끌어들이는 마케팅·이벤트성 지원을 검토해볼 수 있습니다.",
    },
    "competition": {
        "title": "경쟁력 강화 지원 (임대료 지원·컨설팅·특화거리 조성)",
        "description": "경쟁점포 밀집도가 높은 것이 가장 큰 요인입니다. 개별 점포의 차별화·경쟁력을 높이는 지원을 검토해볼 수 있습니다.",
    },
    "efficiency": {
        "title": "접근성 개선 지원 (버스노선·정류장 확충)",
        "description": "유동인구 대비 매출 효율이 낮은 것이 가장 큰 요인입니다. 대중교통 접근성 개선을 우선 검토해볼 수 있습니다.",
    },
}


def recommend_support_type(priority_row: dict) -> dict:
    """지원우선순위 스코어의 3개 하위 점수 중 가장 크게 기여한 요인을 찾아,
    그에 맞는 지원 유형을 추천한다. "왜 필요한지"에서 "그럼 뭘 지원해야 하는지"로
    이어주는 연결고리 역할을 한다.
    """
    contributions = {
        "sales": PRIORITY_WEIGHT_SALES * priority_row["sales_score"],
        "competition": PRIORITY_WEIGHT_COMPETITION * priority_row["competition_score"],
        "efficiency": PRIORITY_WEIGHT_EFFICIENCY * priority_row["efficiency_score"],
    }
    dominant = max(contributions, key=contributions.get)
    result = dict(_SUPPORT_TYPE_CATALOG[dominant])
    result["dominant_factor"] = dominant
    result["contributions"] = contributions
    return result


def match_safety_net(priority_score: float) -> dict:
    """지원우선순위 스코어의 절대 수준(0~100)을 3단계로 나눠 실제 지원제도를 매칭한다.

    recommend_support_type()의 "지배 요인"(왜 필요한지) 축과는 별개다 — 이쪽은
    "얼마나 급한지"에 따라 안양시·국가의 실제 제도명을 매칭하는 안전망 축이다.
    """
    if priority_score >= 66:
        tier = "재기지원"
    elif priority_score >= 33:
        tier = "긴급수혈"
    else:
        tier = "예방"

    result = dict(SAFETY_NET_PROGRAMS[tier])
    result["tier"] = tier
    return result
