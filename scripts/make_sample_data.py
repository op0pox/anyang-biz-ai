"""실제 공공데이터가 도착하기 전, 파이프라인 전체를 검증하기 위한 가상 데이터 생성 스크립트.

안양시 좌표(위도 37.3943, 경도 126.9568) 기준으로 5개 원본 데이터셋을
실제 컬럼 스키마와 최대한 비슷한 형태로 만들어 data/raw/ 에 csv로 저장한다.

실행:
    python scripts/make_sample_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    ANYANG_CENTER_LAT,
    ANYANG_CENTER_LON,
    ANYANG_DONGS,
    BUSINESS_CATEGORIES,
    DATA_RAW_DIR,
)

RNG = np.random.default_rng(42)


def _all_dongs():
    dongs = []
    for gu, dong_list in ANYANG_DONGS.items():
        for dong in dong_list:
            dongs.append((gu, dong))
    return dongs


def _random_point_near(lat, lon, spread_km=6.0):
    """중심 좌표 근방(spread_km 반경)의 임의 위경도를 생성."""
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
    d_lat = RNG.normal(0, spread_km / 2) / km_per_deg_lat
    d_lon = RNG.normal(0, spread_km / 2) / km_per_deg_lon
    return lat + d_lat, lon + d_lon


def make_card_sales() -> pd.DataFrame:
    rows = []
    for gu, dong in _all_dongs():
        for category in BUSINESS_CATEGORIES:
            base = RNG.uniform(3_000_000, 80_000_000)
            noise = RNG.normal(1.0, 0.15)
            amount = max(base * noise, 500_000)
            count = int(amount / RNG.uniform(15_000, 45_000))
            rows.append(
                {
                    "기준년월": "202603",
                    "시군구명": "안양시",
                    "행정동": dong,
                    "업종대분류명": category,
                    "매출금액": round(amount),
                    "매출건수": count,
                }
            )
    return pd.DataFrame(rows)


def make_stores() -> pd.DataFrame:
    rows = []
    store_id = 0
    for gu, dong in _all_dongs():
        n_stores = RNG.integers(15, 60)
        for _ in range(n_stores):
            store_id += 1
            category = RNG.choice(BUSINESS_CATEGORIES)
            lat, lon = _random_point_near(ANYANG_CENTER_LAT, ANYANG_CENTER_LON, spread_km=5.0)
            rows.append(
                {
                    "상호명": f"{dong}_{category}_{store_id}호점",
                    "상권업종대분류명": category,
                    "시도명": "경기도",
                    "시군구명": "안양시",
                    "행정동명": dong,
                    "지번주소": f"경기도 안양시 {gu} {dong} {RNG.integers(1, 999)}번지",
                    "위도": round(lat, 6),
                    "경도": round(lon, 6),
                }
            )
    # 안양시 밖(광명시 "안양천로") 데이터를 섞어 "안양" 키워드만으로 필터링하면
    # 오매칭되는 상황을 재현 — data_loader가 반드시 "안양시"로 필터링해야 함을 검증
    for _ in range(20):
        lat, lon = _random_point_near(37.4780, 126.8650, spread_km=3.0)  # 광명시 인근
        rows.append(
            {
                "상호명": f"광명_매장_{RNG.integers(1000,9999)}",
                "상권업종대분류명": RNG.choice(BUSINESS_CATEGORIES),
                "시도명": "경기도",
                "시군구명": "광명시",
                "행정동명": "철산동",
                "지번주소": f"경기도 광명시 안양천로 {RNG.integers(1, 300)}",
                "위도": round(lat, 6),
                "경도": round(lon, 6),
            }
        )
    return pd.DataFrame(rows)


def make_floating_population() -> pd.DataFrame:
    rows = []
    gus = ["만안구", "동안구"]
    age_groups = ["10대", "20대", "30대", "40대", "50대", "60대이상"]
    for gu in gus:
        for hour in range(24):
            for age in age_groups:
                time_factor = 1.0
                if 11 <= hour < 14 or 18 <= hour < 22:
                    time_factor = 1.4
                elif hour < 6:
                    time_factor = 0.3
                base = RNG.uniform(2000, 9000) * time_factor
                rows.append(
                    {
                        "기준일자": "20260315",
                        "시군명": gu,
                        "시간대구분": f"{hour:02d}시",
                        "연령대구분": age,
                        "유동인구수": round(base),
                    }
                )
    return pd.DataFrame(rows)


def make_resident_population() -> pd.DataFrame:
    rows = []
    for gu, dong in _all_dongs():
        population = int(RNG.uniform(8000, 35000))
        rows.append(
            {
                "기준년월": "202603",
                "시군구명": "안양시",
                "행정동": dong,
                "인구수": population,
                "세대수": int(population / RNG.uniform(1.8, 2.6)),
            }
        )
    return pd.DataFrame(rows)


def make_bus_stops() -> pd.DataFrame:
    rows = []
    stop_id = 0
    for gu, dong in _all_dongs():
        n_stops = RNG.integers(3, 12)
        for _ in range(n_stops):
            stop_id += 1
            lat, lon = _random_point_near(ANYANG_CENTER_LAT, ANYANG_CENTER_LON, spread_km=5.0)
            rows.append(
                {
                    "정류소명": f"{dong}{stop_id}번정류장",
                    "정류소번호": 10000 + stop_id,
                    "소재지지번주소": f"경기도 안양시 {gu} {dong}",
                    "위도": round(lat, 6),
                    "경도": round(lon, 6),
                }
            )
    # 전국 데이터이므로 안양시 밖(서울 관악구 등) 정류장도 섞음 → 필터링 검증용
    for _ in range(15):
        lat, lon = _random_point_near(37.4784, 126.9516, spread_km=3.0)  # 서울 관악구 인근
        rows.append(
            {
                "정류소명": f"관악_정류장_{RNG.integers(1000,9999)}",
                "정류소번호": RNG.integers(90000, 99999),
                "소재지지번주소": "서울특별시 관악구 봉천동",
                "위도": round(lat, 6),
                "경도": round(lon, 6),
            }
        )
    return pd.DataFrame(rows)


def main():
    datasets = {
        "card_sales": make_card_sales(),
        "stores": make_stores(),
        "floating_population": make_floating_population(),
        "resident_population": make_resident_population(),
        "bus_stops": make_bus_stops(),
    }
    for stem, df in datasets.items():
        out_path = DATA_RAW_DIR / f"{stem}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[OK] {stem}: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
