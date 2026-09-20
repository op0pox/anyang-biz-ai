"""행정동 x 업종(대분류) 단위로 원본 데이터를 집계해 모델 학습용 피처 테이블을 만든다."""

from typing import Dict

import numpy as np
import pandas as pd

from src.config import (
    ANYANG_DONGS,
    BUS_STOP_RADIUS_M,
    FLOATING_POPULATION_HOUR_WEIGHTS,
)
from src.data_loader import (
    get_dong_gu_map,
    load_bus_stops,
    load_card_sales,
    load_floating_population,
    load_resident_population,
    load_stores,
)

EARTH_RADIUS_M = 6_371_000


def _dong_to_gu_map() -> Dict[str, str]:
    """행정동 -> 소속구 매핑. 실제 상가정보 데이터가 있으면 그 데이터에서 뽑은
    진짜 행정동/구 목록을 쓰고, 없으면(가상 데이터 등) config의 예시 목록으로 폴백한다."""
    real_mapping = get_dong_gu_map()
    if real_mapping:
        return real_mapping

    mapping = {}
    for gu, dongs in ANYANG_DONGS.items():
        for dong in dongs:
            mapping[dong] = gu
    return mapping


def _hour_weight(hour: float) -> float:
    if pd.isna(hour):
        return 1.0
    for hour_range, weight in FLOATING_POPULATION_HOUR_WEIGHTS.items():
        if int(hour) in hour_range:
            return weight
    return 1.0


def _haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def _dong_centroids(stores: pd.DataFrame) -> pd.DataFrame:
    """행정동별 상가 위경도 평균을 그 행정동의 대표 좌표(centroid)로 사용한다."""
    valid = stores.dropna(subset=["lat", "lon", "dong"])
    return valid.groupby("dong", as_index=False)[["lat", "lon"]].mean()


def _bus_stop_counts(centroids: pd.DataFrame, bus_stops: pd.DataFrame) -> pd.DataFrame:
    """행정동 centroid 반경 BUS_STOP_RADIUS_M 안에 있는 버스정류장 수를 센다."""
    if centroids.empty or bus_stops.empty:
        return pd.DataFrame({"dong": centroids.get("dong", pd.Series(dtype=str)), "bus_stop_count": 0})

    counts = []
    for _, row in centroids.iterrows():
        dist = _haversine_m(row["lat"], row["lon"], bus_stops["lat"].values, bus_stops["lon"].values)
        counts.append(int((dist <= BUS_STOP_RADIUS_M).sum()))
    return pd.DataFrame({"dong": centroids["dong"].values, "bus_stop_count": counts})


def get_redevelopment_risk_by_dong() -> Dict[str, dict]:
    """행정동 centroid 반경 REDEVELOPMENT_RADIUS_M 안에 있는 활성(미완료) 정비사업을 찾는다.

    학습된 모델의 피처나 우선순위 스코어 계산식에는 넣지 않는다 — 정확한 이주 시점
    데이터가 부족해 정량 모델에 넣기엔 근거가 약하고, 화면에 "참고 경고"로 보여주는
    것이 더 정직하다(README/CLAUDE.md 스코프 결정 참고). 정비사업이 없는 행정동은
    결과 dict에 아예 키가 없다(빈 리스트가 아니라 부재로 처리해 UI에서 조건부로만 표시).
    """
    from src.config import REDEVELOPMENT_RADIUS_M
    from src.data_loader import load_redevelopment_projects

    stores = load_stores()
    centroids = _dong_centroids(stores)
    projects = load_redevelopment_projects()

    result: Dict[str, dict] = {}
    if centroids.empty or projects.empty:
        return result

    for _, row in centroids.iterrows():
        dist = _haversine_m(row["lat"], row["lon"], projects["lat"].values, projects["lon"].values)
        nearby = projects[dist <= REDEVELOPMENT_RADIUS_M]
        if not nearby.empty:
            result[row["dong"]] = {
                "count": int(len(nearby)),
                "total_households": int(nearby["households"].sum()),
                "projects": nearby[["name", "stage", "status", "households"]].to_dict("records"),
            }
    return result


def compute_closure_rate_table(old_quarter: str = "202403", new_quarter: str = "202606") -> pd.DataFrame:
    """두 분기 상가정보 스냅샷을 상가업소번호로 비교해 행정동×업종별 "폐업 후보 비율"을
    계산한다. 학습된 모델의 피처나 우선순위 스코어에는 넣지 않는다 — 실제 폐업 외
    이전·업종변경·데이터 정정 등도 섞여 있을 수 있는 추정치라, 정량 모델보다는
    화면에 별도로 보여주는 참고 지표로 쓰는 게 더 정직하다(정비사업 리스크와 같은
    스코프 결정). 상가업소번호가 분기 간 안정적인 ID임은 샘플 검증 완료(CLAUDE.md 참고).
    """
    from src.data_loader import load_stores_snapshot

    old = load_stores_snapshot(old_quarter)
    new = load_stores_snapshot(new_quarter)

    # 두 분기 사이 행정동 개편(명칭 변경·통폐합)이 있으면, 이름이 바뀐 동은 "폐업률
    # 100%"처럼 완전히 잘못된 값이 나온다(실제 확인된 사례: 202403의 박달1동+박달2동이
    # 202606엔 박달동으로 통합, 안양8동+안양9동이 명학동/병목안동/호현동으로 개편됨 —
    # 실제 폐업이 아니라 동 이름 자체가 바뀐 것). 두 분기 모두에 존재하는 동만 비교한다.
    common_dongs = set(old["dong"].unique()) & set(new["dong"].unique())
    old = old[old["dong"].isin(common_dongs)]
    new = new[new["dong"].isin(common_dongs)]

    old_ids_by_group = old.groupby(["dong", "category"])["id"].apply(set)
    new_ids_by_group = new.groupby(["dong", "category"])["id"].apply(set)

    rows = []
    for (dong, category), old_ids in old_ids_by_group.items():
        new_ids = new_ids_by_group.get((dong, category), set())
        closed_ids = old_ids - new_ids
        rows.append(
            {
                "dong": dong,
                "category": category,
                "old_count": len(old_ids),
                "closed_count": len(closed_ids),
                "closure_rate": (len(closed_ids) / len(old_ids)) if old_ids else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _floating_population_by_dong() -> pd.DataFrame:
    """구 단위로만 제공되는 유동인구를, 소속 행정동에 동일하게 배분한다.

    한계: 만안구/동안구 2개 구 단위 데이터라 같은 구에 속한 행정동들은
    유동인구 피처값이 동일하게 들어간다(README/모델 상세정보에 명시).
    """
    fp = load_floating_population()
    fp = fp.dropna(subset=["population"])
    fp["weight"] = fp["hour"].apply(_hour_weight) if "hour" in fp.columns else 1.0
    fp["weighted_population"] = fp["population"] * fp["weight"]
    gu_totals = fp.groupby("gu", as_index=False)["weighted_population"].sum()
    gu_totals = gu_totals.rename(columns={"weighted_population": "floating_population"})

    dong_gu = _dong_to_gu_map()
    rows = [{"dong": dong, "gu": gu} for dong, gu in dong_gu.items()]
    dong_gu_df = pd.DataFrame(rows)
    merged = dong_gu_df.merge(gu_totals, on="gu", how="left")
    return merged[["dong", "floating_population"]]


def build_feature_table() -> pd.DataFrame:
    """행정동 x 업종 조합별 target(매출)과 피처를 담은 데이터프레임을 반환한다.

    결측치 처리 방침:
      - sales_amount/sales_count 결측(해당 조합 매출 기록 없음) -> 0
      - competitor_count 결측(해당 조합 점포 없음) -> 0
      - floating_population/resident_population/bus_stop_count 결측
        (좌표 누락 등으로 계산 불가) -> 전체 행정동 중앙값으로 대체
    """
    card_sales = load_card_sales()
    stores = load_stores()
    resident_population = load_resident_population()
    bus_stops = load_bus_stops()

    dongs = sorted(set(card_sales["dong"]) | set(stores["dong"].dropna()))
    categories = sorted(set(card_sales["category"]) | set(stores["category"].dropna()))
    grid = pd.MultiIndex.from_product([dongs, categories], names=["dong", "category"]).to_frame(index=False)

    sales_agg = card_sales.groupby(["dong", "category"], as_index=False).agg(
        sales_amount=("sales_amount", "sum"),
        sales_count=("sales_count", "sum"),
    )
    competitor_agg = (
        stores.dropna(subset=["dong"])
        .groupby(["dong", "category"], as_index=False)
        .size()
        .rename(columns={"size": "competitor_count"})
    )

    table = grid.merge(sales_agg, on=["dong", "category"], how="left")
    table = table.merge(competitor_agg, on=["dong", "category"], how="left")
    table["sales_amount"] = table["sales_amount"].fillna(0)
    table["sales_count"] = table["sales_count"].fillna(0)
    table["competitor_count"] = table["competitor_count"].fillna(0)

    floating = _floating_population_by_dong()
    centroids = _dong_centroids(stores)
    bus_counts = _bus_stop_counts(centroids, bus_stops)

    table = table.merge(floating, on="dong", how="left")
    table = table.merge(resident_population, on="dong", how="left")
    table = table.merge(bus_counts, on="dong", how="left")

    for col in ["floating_population", "resident_population", "bus_stop_count"]:
        median_value = table[col].median()
        table[col] = table[col].fillna(median_value if pd.notna(median_value) else 0)

    return table


def get_available_dongs(feature_table: pd.DataFrame) -> list:
    return sorted(feature_table["dong"].unique().tolist())


def get_available_categories(feature_table: pd.DataFrame) -> list:
    return sorted(feature_table["category"].unique().tolist())


# 업종 키워드 뒤에 흔히 붙는 접미사 — "안경제조"→"안경", "금속가공"→"금속"처럼 어근만
# 남겨야 안양시 상호명("OO안경원", "OO금속" 등 접미사 없이 적힌 경우가 많음)과 매칭될
# 확률이 높아진다. 접미사를 떼도 남는 게 없거나(3토큰 센터명에서 마지막 토큰만 접미사인
# 경우, 예: "보은 식품 제조" → keyword="제조") 그 자체로는 업종을 특정할 수 없는 값은
# 매칭을 포기한다 — 과매칭보다 놓치는 쪽이 안전하다는 판단.
_SUPPORT_CENTER_KEYWORD_SUFFIXES = ["제조", "가공", "봉제"]
_SUPPORT_CENTER_MIN_MATCH_COUNT = 3


def get_other_region_support_center_examples(top_n: int = 5) -> list:
    """안양시엔 없는 업종별 소공인 특화지원센터를, 다른 지역 실제 운영 사례와 함께
    반환한다. "저 지역엔 있는데 안양엔 없는 지원 인프라가 뭔가"를 담당자에게 참고로
    보여주되, 안양시에 해당 업종 자체가 없으면 추천이 무의미하므로 안양시 상가
    상호명에 실제로 관련 업종이 존재하는 경우만 골라 최소한의 근거를 붙인다.
    학습 피처가 아니라 지원우선순위 페이지의 참고 카드 전용이다.
    """
    from src.data_loader import load_specialized_support_centers, load_stores

    centers = load_specialized_support_centers()
    if centers.empty:
        return []

    others = centers[~centers["is_anyang"] & (centers["focus"] != "상생")]
    if others.empty:
        return []

    stores = load_stores()
    store_names = stores["name"].dropna().astype(str) if "name" in stores else pd.Series(dtype=str)
    if store_names.empty:
        return []

    def _root(keyword: str) -> str:
        for suffix in _SUPPORT_CENTER_KEYWORD_SUFFIXES:
            if keyword.endswith(suffix) and len(keyword) > len(suffix):
                return keyword[: -len(suffix)]
        return "" if keyword in _SUPPORT_CENTER_KEYWORD_SUFFIXES else keyword

    best_by_root: Dict[str, dict] = {}
    for _, row in others.iterrows():
        root = _root(row["keyword"])
        if len(root) < 2:
            continue
        match_count = int(store_names.str.contains(root, regex=False, na=False).sum())
        if match_count < _SUPPORT_CENTER_MIN_MATCH_COUNT:
            continue
        if root in best_by_root and best_by_root[root]["anyang_related_store_count"] >= match_count:
            continue
        best_by_root[root] = {
            "name": row["name"],
            "org": row["org"],
            "address": row["address"],
            "focus": row["focus"],
            "anyang_related_store_count": match_count,
        }

    results = sorted(best_by_root.values(), key=lambda r: r["anyang_related_store_count"], reverse=True)
    return results[:top_n]
