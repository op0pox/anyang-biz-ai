"""5개 원본 데이터셋 로딩 + 안양시 필터링 + 인코딩/확장자 자동 처리.

공공데이터포털/경기데이터드림에서 받는 파일은 인코딩(cp949/euc-kr/utf-8)과
확장자(csv/xlsx/xls)가 제각각이고 컬럼명도 배포 시점마다 조금씩 다르므로,
컬럼명을 하드코딩하지 않고 키워드 부분 매칭으로 탐색한다.

카드소비 데이터(경기데이터드림)처럼 컬럼명이 코드화되어 있어 키워드 매칭이 아예
불가능한 실제 데이터셋도 있어, 그런 경우는 알려진 스키마를 별도로 처리한다.
"""

import functools
import os
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

from src.config import (
    ANYANG_BBOX_RADIUS_KM,
    ANYANG_CENTER_LAT,
    ANYANG_CENTER_LON,
    BUS_API_BASE_URL,
    BUS_API_PAGE_SIZE,
    CITY_FILTER_KEYWORD,
    CITY_NAME,
    COLUMN_HINTS,
    DATA_RAW_DIR,
    GG_OPEN_API_BASE_URL,
    GG_OPEN_API_MAX_PAGES,
    GG_OPEN_API_PAGE_SIZE,
    GG_OPEN_API_SERVICE_NAME,
    RAW_FILE_STEMS,
    STORE_TO_CARD_CATEGORY_MAP,
)


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    import math

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))

_CANDIDATE_EXTENSIONS = [".csv", ".xlsx", ".xls"]
_CANDIDATE_ENCODINGS = ["utf-8-sig", "cp949", "utf-8", "euc-kr"]

# 경기데이터드림 카드소비 데이터 표준 스키마(코드화된 컬럼) — 키워드 매칭이 불가능해
# 별도로 감지해서 처리한다.
_CARD_SALES_CODED_COLUMNS = {
    "dong_code": "admi_cty_no",
    "category": "card_tpbuz_nm_1",
    "amount": "amt",
    "count": "cnt",
}


class DataFileNotFoundError(FileNotFoundError):
    """data/raw/ 안에서 필요한 원본 파일을 찾지 못했을 때 발생시키는 예외."""


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
def _resolve_path(key: str) -> Path:
    """RAW_FILE_STEMS의 stem 이름으로 실제 파일을(확장자 무관) 탐색한다."""
    stem = RAW_FILE_STEMS.get(key, key)
    for ext in _CANDIDATE_EXTENSIONS:
        candidate = DATA_RAW_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise DataFileNotFoundError(
        f"'{stem}' 데이터 파일을 data/raw/ 에서 찾을 수 없습니다. "
        f"({', '.join(_CANDIDATE_EXTENSIONS)} 중 하나로 저장해주세요.) "
        f"먼저 `python scripts/make_sample_data.py`로 가상 데이터를 생성해 파이프라인을 확인할 수 있습니다."
    )


def _resolve_all_paths(key: str) -> List[Path]:
    """stem과 정확히 일치하는 파일이 있으면 그것만, 없으면 stem으로 시작하는
    모든 파일(예: card_sales_202601.csv, card_sales_202602.csv ...)을 찾는다.

    분기 데이터가 여러 개의 월별 파일로 나뉘어 배포되는 경우(카드소비 데이터 등)를
    지원하기 위함이다.
    """
    stem = RAW_FILE_STEMS.get(key, key)
    exact = [DATA_RAW_DIR / f"{stem}{ext}" for ext in _CANDIDATE_EXTENSIONS]
    exact_found = [p for p in exact if p.exists()]
    if exact_found:
        return exact_found

    matched: List[Path] = []
    for ext in _CANDIDATE_EXTENSIONS:
        matched.extend(sorted(DATA_RAW_DIR.glob(f"{stem}*{ext}")))
    if not matched:
        raise DataFileNotFoundError(
            f"'{stem}' 데이터 파일을 data/raw/ 에서 찾을 수 없습니다. "
            f"({', '.join(_CANDIDATE_EXTENSIONS)} 중 하나로 저장해주세요.) "
            f"먼저 `python scripts/make_sample_data.py`로 가상 데이터를 생성해 파이프라인을 확인할 수 있습니다."
        )
    return matched


def _peek_header(path: Path) -> List[str]:
    """전체 파일을 읽지 않고 컬럼명만 확인한다 (대용량 파일 성능 최적화용)."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        try:
            return list(pd.read_excel(path, nrows=0).columns)
        except Exception:  # noqa: BLE001
            return []
    for encoding in _CANDIDATE_ENCODINGS:
        try:
            return list(pd.read_csv(path, nrows=0, encoding=encoding).columns)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:  # noqa: BLE001
            return []
    return []


def _read_table_flex(path: Path, usecols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """csv/excel 겸용, 인코딩 자동 판별로 파일을 읽는다.

    usecols를 지정하면 필요한 컬럼만 읽어 대용량 파일(상가정보 전국 데이터 등)의
    메모리 사용량과 로딩 시간을 크게 줄인다.
    """
    usecols = list(usecols) if usecols else None
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, usecols=usecols)

    last_error: Optional[Exception] = None
    for encoding in _CANDIDATE_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, usecols=usecols)
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
            continue
    raise ValueError(
        f"'{path.name}' 파일의 인코딩을 자동으로 판별하지 못했습니다 "
        f"(시도한 인코딩: {_CANDIDATE_ENCODINGS}). 원본 파일 인코딩을 확인해주세요."
    ) from last_error


def _find_column_name(columns: Iterable[str], keywords) -> Optional[str]:
    """주어진 키워드 목록 중 하나라도 컬럼명에 포함되면 그 컬럼명을 반환."""
    if isinstance(keywords, str):
        keywords = [keywords]
    columns = list(columns)
    for keyword in keywords:
        for col in columns:
            if keyword in str(col):
                return col
    return None


def _find_column(df: pd.DataFrame, keywords) -> Optional[str]:
    return _find_column_name(df.columns, keywords)


def _filter_by_city(df: pd.DataFrame, address_keywords=("주소", "시군구", "소재지", "행정동명", "시도")) -> pd.DataFrame:
    """CITY_FILTER_KEYWORD("안양시")를 포함한 주소/지역 컬럼을 찾아 필터링한다.

    "안양"만으로 필터링하면 다른 지역 도로명(예: 광명시 "안양천로")과 오매칭되므로
    반드시 "안양시" 전체 문자열로 매칭한다. 해당 성격의 컬럼이 없으면 원본을 그대로 반환한다
    (이미 안양시 단위로만 배포되는 데이터셋인 경우).
    """
    address_like_cols = [
        col
        for col in df.columns
        if any(keyword in str(col) for keyword in address_keywords)
    ]
    if not address_like_cols:
        return df

    mask = pd.Series(False, index=df.index)
    for col in address_like_cols:
        mask |= df[col].astype(str).str.contains(CITY_FILTER_KEYWORD, na=False)

    if not mask.any():
        # 어떤 행도 "안양시"를 포함하지 않으면(이미 안양시로만 구성된 데이터일 수 있음)
        # 원본을 그대로 반환해 데이터가 통째로 사라지는 것을 방지한다.
        return df
    return df[mask].reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _load_anyang_dong_reference() -> pd.DataFrame:
    """상가정보 데이터에서 안양시 행정동코드/행정동명/소속구 참조표를 만든다.

    카드소비 데이터의 행정동코드→이름 매핑, 구 단위로만 제공되는 유동인구·
    주민등록인구를 행정동에 배분하는 데 공통으로 쓰인다. 상가정보에 필요한
    컬럼이 없으면 빈 테이블을 반환하고, 이를 쓰는 쪽에서 각자의 폴백으로 대체한다.
    """
    empty = pd.DataFrame(columns=["dong_code", "dong", "gu"])
    try:
        path = _resolve_path("stores")
    except DataFileNotFoundError:
        return empty

    columns = _peek_header(path)
    if not columns:
        return empty

    dong_col = _find_column_name(columns, COLUMN_HINTS["stores"]["dong"])
    city_col = _find_column_name(columns, ["시군구명", "시군구"])
    code_col = _find_column_name(columns, ["행정동코드"])
    if not dong_col or not city_col:
        return empty

    usecols = [c for c in {dong_col, city_col, code_col} if c]
    df = _read_table_flex(path, usecols=usecols)
    df = df[df[city_col].astype(str).str.contains(CITY_FILTER_KEYWORD, na=False)].copy()
    if df.empty:
        return empty

    df["gu"] = df[city_col].astype(str).str.replace(CITY_FILTER_KEYWORD, "", regex=False).str.strip()
    df = df[df["gu"] != ""]
    if df.empty or df["gu"].nunique() < 2:
        # 시군구명이 "안양시"만 있고 구 정보가 없는 경우(가상 데이터 등) — 구 배분이
        # 의미가 없으므로 빈 테이블을 반환해 호출부가 config.ANYANG_DONGS로 폴백하게 한다.
        return empty

    rename_map = {dong_col: "dong"}
    if code_col:
        rename_map[code_col] = "dong_code"
    df = df.rename(columns=rename_map)
    if "dong_code" not in df.columns:
        df["dong_code"] = None

    return df[["dong_code", "dong", "gu"]].drop_duplicates().reset_index(drop=True)


def get_dong_gu_map() -> dict:
    """행정동명 -> 소속구(만안구/동안구) 매핑. 실제 상가정보 데이터가 있으면 그걸 쓰고,
    없으면 빈 dict를 반환해 호출부가 config.ANYANG_DONGS로 폴백하도록 한다."""
    ref = _load_anyang_dong_reference()
    if ref.empty:
        return {}
    return dict(zip(ref["dong"], ref["gu"]))


# ---------------------------------------------------------------------------
# 데이터셋별 로더
# ---------------------------------------------------------------------------
def _load_card_sales_one_file(path: Path) -> pd.DataFrame:
    """카드소비 데이터 파일 하나를 (dong, category, sales_amount, sales_count)로 집계한다."""
    columns = _peek_header(path)
    hints = COLUMN_HINTS["card_sales"]
    dong_col = _find_column_name(columns, hints["dong"])
    category_col = _find_column_name(columns, hints["category"])

    if dong_col and category_col:
        # 표준적인 한글 컬럼명 형태 (스펙에서 원래 가정한 배포 형태)
        amount_col = _find_column_name(columns, hints["amount"])
        count_col = _find_column_name(columns, hints["count"])
        usecols = [c for c in [dong_col, category_col, amount_col, count_col] if c]
        df = _read_table_flex(path, usecols=usecols)
        df = _filter_by_city(df)
        part = pd.DataFrame(
            {
                "dong": df[dong_col].astype(str).str.strip(),
                "category": df[category_col].astype(str).str.strip(),
                "sales_amount": pd.to_numeric(df[amount_col], errors="coerce") if amount_col else 0.0,
                "sales_count": pd.to_numeric(df[count_col], errors="coerce") if count_col else 0.0,
            }
        )
    elif _CARD_SALES_CODED_COLUMNS["dong_code"] in columns and _CARD_SALES_CODED_COLUMNS["category"] in columns:
        # 경기데이터드림 카드소비 데이터 표준 스키마: 행정동은 이름이 아니라 코드(admi_cty_no)로만
        # 제공되므로, 상가정보에서 뽑은 행정동코드->이름 참조표로 조인해야 한다.
        code_col = _CARD_SALES_CODED_COLUMNS["dong_code"]
        cat_col = _CARD_SALES_CODED_COLUMNS["category"]
        amt_col = _CARD_SALES_CODED_COLUMNS["amount"]
        cnt_col = _CARD_SALES_CODED_COLUMNS["count"]
        usecols = [c for c in [code_col, cat_col, amt_col, cnt_col] if c in columns]
        df = _read_table_flex(path, usecols=usecols)

        dong_ref = _load_anyang_dong_reference()
        if dong_ref.empty or dong_ref["dong_code"].isna().all():
            raise ValueError(
                "카드소비 데이터의 행정동코드를 이름으로 변환할 참조표(상가정보 행정동코드)를 만들 수 없습니다. "
                "data/raw/stores 에 행정동코드 컬럼이 있는 상가정보 데이터가 있는지 확인해주세요."
            )
        code_to_dong = dict(
            zip(pd.to_numeric(dong_ref["dong_code"], errors="coerce"), dong_ref["dong"])
        )
        dong_series = pd.to_numeric(df[code_col], errors="coerce").map(code_to_dong)

        part = pd.DataFrame(
            {
                "dong": dong_series,
                "category": df[cat_col].astype(str).str.strip(),
                "sales_amount": pd.to_numeric(df[amt_col], errors="coerce") if amt_col in df else 0.0,
                "sales_count": pd.to_numeric(df[cnt_col], errors="coerce") if cnt_col in df else 0.0,
            }
        )
        part = part.dropna(subset=["dong"])  # 안양시 외 행정동코드(매핑 실패)는 제외
    else:
        raise ValueError(
            f"'{path.name}'에서 card_sales 컬럼을 인식하지 못했습니다. config.COLUMN_HINTS를 확인해주세요."
        )

    return part.groupby(["dong", "category"], as_index=False).agg(
        sales_amount=("sales_amount", "sum"), sales_count=("sales_count", "sum")
    )


def load_card_sales() -> pd.DataFrame:
    """카드소비 데이터를 로드한다. 월별로 여러 파일에 나뉘어 있으면 모두 합산한다."""
    paths = _resolve_all_paths("card_sales")
    parts = [_load_card_sales_one_file(p) for p in paths]
    combined = pd.concat(parts, ignore_index=True)
    return combined.groupby(["dong", "category"], as_index=False).agg(
        sales_amount=("sales_amount", "sum"), sales_count=("sales_count", "sum")
    )


def load_stores() -> pd.DataFrame:
    path = _resolve_path("stores")
    columns = _peek_header(path)
    hints = COLUMN_HINTS["stores"]

    dong_col = _find_column_name(columns, hints["dong"]) if columns else None
    category_col = _find_column_name(columns, hints["category"]) if columns else None
    name_col = _find_column_name(columns, hints["name"]) if columns else None
    lat_col = _find_column_name(columns, hints["lat"]) if columns else None
    lon_col = _find_column_name(columns, hints["lon"]) if columns else None
    address_cols = [
        c for c in columns if any(k in str(c) for k in ("주소", "시군구", "소재지", "행정동명", "시도"))
    ]

    usecols = None
    if columns:
        # 컬럼이 많은 대용량 전국/도 단위 파일(수십 컬럼)일 수 있어, 찾은 컬럼만 읽어
        # 메모리와 로딩 시간을 절약한다. 필요한 컬럼을 하나도 못 찾으면 전체를 읽는다.
        needed = {dong_col, category_col, name_col, lat_col, lon_col, *address_cols}
        needed.discard(None)
        if needed:
            usecols = list(needed)

    df = _read_table_flex(path, usecols=usecols)
    df = _filter_by_city(df)

    out = pd.DataFrame(
        {
            "dong": df[dong_col].astype(str).str.strip() if dong_col else None,
            "category": df[category_col].astype(str).str.strip() if category_col else "미분류",
            "name": df[name_col].astype(str).str.strip() if name_col else None,
            "lat": pd.to_numeric(df[lat_col], errors="coerce") if lat_col else None,
            "lon": pd.to_numeric(df[lon_col], errors="coerce") if lon_col else None,
        }
    )
    # 소상공인시장진흥공단 상가정보(컬럼명 "상권업종대분류명")는 카드소비 데이터와
    # 업종 분류 체계가 달라(대분류명이 서로 다름) 그대로면 같은 업종으로 조인되지 않는다.
    # 이 크로스워크는 그 특정 표준 스키마에서 나온 값에만 적용한다 — 다른 배포 형태나
    # 가상 데이터처럼 이미 카드소비 체계를 쓰는 경우까지 잘못 변형하지 않기 위함이다.
    if category_col == "상권업종대분류명":
        out["category"] = out["category"].map(STORE_TO_CARD_CATEGORY_MAP).fillna(out["category"])
    return out


def load_stores_snapshot(quarter: str) -> pd.DataFrame:
    """특정 분기(예: '202403')의 상가정보 스냅샷을 로드한다. load_stores()와 달리
    상가업소번호(id)를 포함해 반환한다 — 두 분기를 비교해 폐업 흐름을 추정하는 데
    쓰인다(feature_engineering.compute_closure_rate_table 참고). 상가업소번호가
    분기 간 안정적인 ID임은 샘플 검증 완료(CLAUDE.md 참고).
    """
    from src.config import DATA_SOURCES_DIR

    snapshot_dir = DATA_SOURCES_DIR / "stores" / quarter
    if not snapshot_dir.exists():
        raise DataFileNotFoundError(f"'{quarter}' 분기 상가정보를 data/sources/stores/{quarter}/ 에서 찾을 수 없습니다.")
    candidates = list(snapshot_dir.glob("*.csv")) + list(snapshot_dir.glob("*.xlsx")) + list(snapshot_dir.glob("*.xls"))
    if not candidates:
        raise DataFileNotFoundError(f"'{quarter}' 분기 상가정보 파일이 data/sources/stores/{quarter}/ 에 없습니다.")
    path = candidates[0]

    columns = _peek_header(path)
    hints = COLUMN_HINTS["stores"]
    id_col = _find_column_name(columns, ["상가업소번호"])
    dong_col = _find_column_name(columns, hints["dong"])
    category_col = _find_column_name(columns, hints["category"])
    address_cols = [
        c for c in columns if any(k in str(c) for k in ("주소", "시군구", "소재지", "행정동명", "시도"))
    ]

    needed = {id_col, dong_col, category_col, *address_cols}
    needed.discard(None)
    usecols = list(needed) if needed else None

    df = _read_table_flex(path, usecols=usecols)
    df = _filter_by_city(df)

    out = pd.DataFrame(
        {
            "id": df[id_col] if id_col else None,
            "dong": df[dong_col].astype(str).str.strip() if dong_col else None,
            "category": df[category_col].astype(str).str.strip() if category_col else "미분류",
        }
    ).dropna(subset=["id"])

    if category_col == "상권업종대분류명":
        out["category"] = out["category"].map(STORE_TO_CARD_CATEGORY_MAP).fillna(out["category"])
    return out


def _make_fallback_floating_population() -> pd.DataFrame:
    """API 키가 없거나 호출에 실패했을 때 사용할 가상 유동인구 데이터."""
    import numpy as np

    rng = np.random.default_rng(7)
    rows = []
    for gu in ["만안구", "동안구"]:
        for hour in range(24):
            time_factor = 1.4 if (11 <= hour < 14 or 18 <= hour < 22) else (0.3 if hour < 6 else 1.0)
            rows.append(
                {
                    "gu": gu,
                    "hour": hour,
                    "population": round(rng.uniform(2000, 9000) * time_factor),
                }
            )
    return pd.DataFrame(rows)


def _fetch_floating_population_from_api() -> pd.DataFrame:
    """경기데이터드림 OPEN API로 유동인구 데이터를 페이지네이션하며 수집한다.

    1회 호출 최대 1,000건 제한이 있어 pIndex를 늘려가며 반복 호출한다.
    """
    api_key = os.environ.get("GG_OPEN_API_KEY")
    if not api_key:
        print(
            "[안내] GG_OPEN_API_KEY 환경변수가 설정되지 않아 유동인구 데이터를 API로 가져올 수 없습니다. "
            "발급받은 키를 .env 또는 환경변수에 GG_OPEN_API_KEY=... 형태로 등록해주세요. "
            "우선 가상 데이터로 대체합니다."
        )
        return _make_fallback_floating_population()

    url = f"{GG_OPEN_API_BASE_URL}/{GG_OPEN_API_SERVICE_NAME}"
    all_rows = []
    page_index = 1
    try:
        while True:
            params = {
                "KEY": api_key,
                "Type": "json",
                "pIndex": page_index,
                "pSize": GG_OPEN_API_PAGE_SIZE,
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()

            service_block = payload.get(GG_OPEN_API_SERVICE_NAME)
            if not service_block:
                break

            rows = None
            for item in service_block:
                if "row" in item:
                    rows = item["row"]
                    break
            if not rows:
                break

            all_rows.extend(rows)
            if len(rows) < GG_OPEN_API_PAGE_SIZE:
                break
            page_index += 1
            if page_index > GG_OPEN_API_MAX_PAGES:
                # 이 데이터셋은 21만 건이 넘는 다년간 누적 통계라 전체를 받을 필요가
                # 없다 — 만안구/동안구 상대 비교용 대표 표본만 있으면 충분하다.
                break

        if not all_rows:
            raise ValueError("API 응답에 데이터가 없습니다.")

        df = pd.DataFrame(all_rows)
        # 캐시 저장 — 다음 실행부터는 API를 다시 호출하지 않고 캐시를 사용
        cache_path = DATA_RAW_DIR / "floating_population.csv"
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"[안내] 유동인구 OPEN API 호출에 실패했습니다 ({exc}). 가상 데이터로 대체합니다.")
        return _make_fallback_floating_population()


def _parse_gg_floating_population_schema(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """경기데이터드림 유동인구_안양시 API(TBDASANALSGALLERYT214117) 응답 스키마를 파싱한다.

    이 응답은 구(CTY_NM) x 시간대(TIME_CD) x 내/외국인(FORN_GB) x 성별·5세 연령대
    (M_10_CNT ... F_70_CNT) 단위로 잘게 쪼개져 있고 컬럼명이 전부 영문 코드라
    한글 키워드 매칭이 되지 않는다. 성별·연령대·내외국인 구분 없이 시간대별
    합계 유동인구로 뭉쳐서 반환한다(구 단위로만 배분하는 한계는 그대로 유지).
    """
    import re

    count_cols = [c for c in df.columns if re.match(r"^[MF]_\d+_CNT$", str(c))]
    if "CTY_NM" not in df.columns or "TIME_CD" not in df.columns or not count_cols:
        return None

    population = df[count_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    gu = df["CTY_NM"].astype(str).str.replace(CITY_FILTER_KEYWORD, "", regex=False).str.strip()
    hour = pd.to_numeric(df["TIME_CD"], errors="coerce")
    return pd.DataFrame({"gu": gu, "hour": hour, "population": population})


def load_floating_population() -> pd.DataFrame:
    cache_path = DATA_RAW_DIR / "floating_population.csv"
    if cache_path.exists():
        df = _read_table_flex(cache_path)
    else:
        df = _fetch_floating_population_from_api()

    gg_parsed = _parse_gg_floating_population_schema(df)
    if gg_parsed is not None:
        return gg_parsed

    hints = COLUMN_HINTS["floating_population"]
    gu_col = _find_column(df, hints["gu"])
    hour_col = _find_column(df, hints["hour"])
    age_col = _find_column(df, hints["age"])
    pop_col = _find_column(df, hints["population"])

    out = pd.DataFrame(
        {
            "gu": df[gu_col].astype(str).str.strip() if gu_col else df.get("gu", "미상"),
            "hour_raw": df[hour_col] if hour_col else df.get("hour"),
            "age_group": df[age_col].astype(str).str.strip() if age_col else None,
            "population": pd.to_numeric(df[pop_col], errors="coerce") if pop_col else pd.to_numeric(df.get("population"), errors="coerce"),
        }
    )
    out["hour"] = (
        out["hour_raw"].astype(str).str.extract(r"(\d{1,2})")[0].astype(float)
        if "hour_raw" in out
        else None
    )
    return out.drop(columns=["hour_raw"], errors="ignore")


def _load_resident_population_pivot_table(path: Path) -> pd.DataFrame:
    """안양시가 배포하는 "세대 및 인구" 통계표 형식을 파싱한다.

    이 표는 "안양시 / -만안구 / -동안구" 형태의 병합 셀 피벗 테이블이라 행정동 단위
    데이터가 없다. 구 총인구를 그 구에 속한 모든 행정동에 동일하게 배분한다
    (유동인구가 구 단위로만 제공되는 것과 같은 한계 — README에 명시).
    """
    raw = pd.read_excel(path, header=None)
    gu_population = {}
    for _, row in raw.iterrows():
        label = str(row.iloc[0]).strip()
        if label.startswith("-") and not label.startswith("--"):
            gu = label.lstrip("-").strip()
            value = pd.to_numeric(row.iloc[1], errors="coerce")
            if pd.notna(value):
                gu_population[gu] = value

    if not gu_population:
        raise ValueError(
            "resident_population 파일에서 구 단위 인구 정보를 추출하지 못했습니다. "
            "예상 형식: 1열에 '-만안구'/'-동안구', 2열에 총인구수."
        )

    dong_ref = _load_anyang_dong_reference()
    if dong_ref.empty:
        raise ValueError(
            "구 단위 인구를 행정동에 배분할 참조표(상가정보 행정동/구 목록)를 만들 수 없습니다."
        )

    dong_ref = dong_ref.copy()
    dong_ref["resident_population"] = dong_ref["gu"].map(gu_population)
    out = dong_ref.dropna(subset=["resident_population"])[["dong", "resident_population"]]
    if out.empty:
        raise ValueError("행정동에 배분된 인구 데이터가 없습니다. 구 이름 표기를 확인해주세요.")
    return out.groupby("dong", as_index=False)["resident_population"].sum()


def load_resident_population() -> pd.DataFrame:
    path = _resolve_path("resident_population")
    columns = _peek_header(path)
    hints = COLUMN_HINTS["resident_population"]
    dong_col = _find_column_name(columns, hints["dong"]) if columns else None

    if dong_col:
        # 표준적인 행정동 단위 표 형태 (스펙에서 원래 가정한 배포 형태)
        df = _read_table_flex(path)
        df = _filter_by_city(df)
        pop_col = _find_column(df, hints["population"])
        out = pd.DataFrame(
            {
                "dong": df[dong_col].astype(str).str.strip(),
                "resident_population": pd.to_numeric(df[pop_col], errors="coerce") if pop_col else 0.0,
            }
        )
        # 같은 행정동이 여러 행(연령대별 등)으로 나뉘어 있을 수 있으므로 합산
        return out.groupby("dong", as_index=False)["resident_population"].sum()

    if path.suffix.lower() in (".xlsx", ".xls"):
        return _load_resident_population_pivot_table(path)

    raise ValueError("resident_population 데이터에서 행정동 컬럼을 찾을 수 없습니다.")


def _get_anyang_bbox() -> Optional[tuple]:
    """상가정보(stores)의 실제 위경도 범위로 안양시 경계를 근사한다.

    상가정보는 이미 시군구명 "안양시"로 정확히 필터링된 데이터라, 버스정류소 API의
    부정확한 행정구역 코드보다 훨씬 신뢰할 수 있는 경계 기준이 된다.
    """
    try:
        stores = load_stores().dropna(subset=["lat", "lon"])
    except Exception:  # noqa: BLE001
        return None
    if stores.empty:
        return None
    pad = 0.01  # 약 1.1km 여유 — 상가가 없는 외곽 지역의 정류소도 포함시키기 위함
    return (
        stores["lat"].min() - pad,
        stores["lat"].max() + pad,
        stores["lon"].min() - pad,
        stores["lon"].max() + pad,
    )


def _make_fallback_bus_stops() -> pd.DataFrame:
    """API 키가 없거나 호출에 실패했을 때 사용할 가상 버스정류장 데이터."""
    import numpy as np

    rng = np.random.default_rng(11)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(ANYANG_CENTER_LAT))
    rows = []
    for i in range(120):
        d_lat = rng.normal(0, 2.5) / km_per_deg_lat
        d_lon = rng.normal(0, 2.5) / km_per_deg_lon
        rows.append(
            {
                "name": f"가상정류장{i + 1}",
                "lat": ANYANG_CENTER_LAT + d_lat,
                "lon": ANYANG_CENTER_LON + d_lon,
            }
        )
    return pd.DataFrame(rows)


def _build_bus_api_url(path: str, api_key: str) -> str:
    """serviceKey를 URL에 직접 붙인다.

    공공데이터포털은 "일반 인증키(Encoding)"와 "(Decoding)" 두 종류를 발급하는데,
    requests의 params=로 넘기면 Encoding 키는 다시 인코딩되어(이중 인코딩) 인증에
    실패한다. 이미 %가 포함된 Encoding 키는 그대로 URL에 붙이고, 그렇지 않은
    Decoding/원문 키만 직접 인코딩한다.
    """
    from urllib.parse import quote

    key = api_key if "%" in api_key else quote(api_key, safe="")
    return f"{BUS_API_BASE_URL}/{path}?serviceKey={key}"


def _get_city_code(api_key: str) -> Optional[str]:
    """getCtyCodeList를 호출해 CITY_NAME("안양시")의 도시코드를 찾는다."""
    url = _build_bus_api_url("getCtyCodeList", api_key)
    resp = requests.get(url, params={"_type": "json"}, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    items = payload.get("response", {}).get("body", {}).get("items")
    if not items:
        return None
    item_list = items.get("item") if isinstance(items, dict) else items
    if isinstance(item_list, dict):
        item_list = [item_list]
    for item in item_list or []:
        if str(item.get("cityname", "")).strip() == CITY_NAME:
            return str(item.get("citycode"))
    return None


def _fetch_bus_stops_from_api() -> pd.DataFrame:
    """국토교통부 TAGO 버스정류소정보조회 서비스(getSttnNoList)로 안양시 정류소를 수집한다."""
    api_key = os.environ.get("BUS_API_KEY")
    if not api_key:
        print(
            "[안내] BUS_API_KEY 환경변수가 설정되지 않아 버스정류소 데이터를 API로 가져올 수 없습니다. "
            "발급받은 키를 .env 또는 환경변수에 BUS_API_KEY=... 형태로 등록해주세요. "
            "우선 가상 데이터로 대체합니다."
        )
        return _make_fallback_bus_stops()

    try:
        city_code = _get_city_code(api_key)
        if not city_code:
            raise ValueError(f"getCtyCodeList 응답에서 '{CITY_NAME}' 도시코드를 찾지 못했습니다.")

        all_rows = []
        page = 1
        total_count = None
        while True:
            # 페이지 하나가 이상 응답(빈 문자열 items, 예상 밖 타입 등)을 줘도
            # 이미 모은 다른 페이지들을 버리지 않도록, 페이지 단위로 에러를 격리한다.
            try:
                url = _build_bus_api_url("getSttnNoList", api_key)
                resp = requests.get(
                    url,
                    params={
                        "cityCode": city_code,
                        "numOfRows": BUS_API_PAGE_SIZE,
                        "pageNo": page,
                        "_type": "json",
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                payload = resp.json()
                body = payload.get("response", {}).get("body", {})
                if not isinstance(body, dict):
                    break
                items = body.get("items")
                item_list = (items.get("item") if isinstance(items, dict) else items) if items else None
                if not item_list:
                    break
                if isinstance(item_list, dict):
                    item_list = [item_list]
                if not isinstance(item_list, list):
                    break

                all_rows.extend(item_list)
                total_count = int(body.get("totalCount", 0) or 0)
            except Exception as page_exc:  # noqa: BLE001
                print(f"[안내] 버스정류소 API {page}페이지 조회 중 오류가 있어 이후 페이지는 건너뜁니다 ({page_exc}).")
                break

            if (total_count and len(all_rows) >= total_count) or len(item_list) < BUS_API_PAGE_SIZE:
                break
            page += 1
            if page > 50:  # 안전장치: 무한루프 방지
                break

        if not all_rows:
            raise ValueError("API 응답에 정류소 데이터가 없습니다.")

        df = pd.DataFrame(all_rows)

        # TAGO의 cityCode 경계 데이터가 완벽하지 않아, 도시코드로 조회해도
        # 인접 시(광명·군포·의왕·부천 등)의 정류소가 다수 섞여 나온다(실제 확인됨:
        # "광명시보건소", "군포보건소", "의왕시청", "서울구치소" 등이 그대로 포함).
        # 안양시 중심 반경만으로는 걸러지지 않아(안양시는 남북으로 길쭉한 형태),
        # 이미 주소 기반으로 정확히 안양시로 필터링된 상가정보(stores) 데이터의
        # 실제 위경도 범위(bounding box)를 기준으로 걸러낸다 — 훨씬 더 정확하다.
        lat = pd.to_numeric(df.get("gpslati"), errors="coerce")
        lon = pd.to_numeric(df.get("gpslong"), errors="coerce")
        before = len(df)

        bbox = _get_anyang_bbox()
        if bbox:
            lat_min, lat_max, lon_min, lon_max = bbox
            keep = [
                pd.notna(la) and pd.notna(lo) and lat_min <= la <= lat_max and lon_min <= lo <= lon_max
                for la, lo in zip(lat, lon)
            ]
            filter_desc = "상가정보 기준 안양시 경계"
        else:
            # 상가정보를 아직 못 구했으면 중심 반경으로 대략 거른다(정밀도는 떨어짐).
            keep = [
                pd.notna(la) and pd.notna(lo) and _haversine_km(ANYANG_CENTER_LAT, ANYANG_CENTER_LON, la, lo) <= ANYANG_BBOX_RADIUS_KM
                for la, lo in zip(lat, lon)
            ]
            filter_desc = f"중심 반경 {ANYANG_BBOX_RADIUS_KM}km"

        df = df[keep].reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"[안내] 버스정류소 API 응답 중 {filter_desc} 밖 {dropped}건(인접 시 오분류 추정)을 제외했습니다.")

        # 캐시 저장 — 다음 실행부터는 API를 다시 호출하지 않고 캐시를 사용
        cache_path = DATA_RAW_DIR / "bus_stops.csv"
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"[안내] 버스정류소 OPEN API 호출에 실패했습니다 ({exc}). 가상 데이터로 대체합니다.")
        return _make_fallback_bus_stops()


def load_bus_stops() -> pd.DataFrame:
    cache_path = DATA_RAW_DIR / "bus_stops.csv"
    if cache_path.exists():
        df = _read_table_flex(cache_path)
    else:
        df = _fetch_bus_stops_from_api()
        if {"lat", "lon"}.issubset(df.columns):
            # API 실패 시 가상 데이터 폴백은 이미 최종 스키마(name/lat/lon)로 나온다.
            return df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    df = _filter_by_city(df)

    hints = COLUMN_HINTS["bus_stops"]
    lat_col = _find_column(df, hints["lat"])
    lon_col = _find_column(df, hints["lon"])
    name_col = _find_column(df, hints["name"])

    # 국토교통부 TAGO API 응답은 영문 소문자 필드(gpslati/gpslong/nodenm)라
    # 한글 키워드 매칭이 되지 않으므로 별도로 인식한다.
    if lat_col is None and "gpslati" in df.columns:
        lat_col = "gpslati"
    if lon_col is None and "gpslong" in df.columns:
        lon_col = "gpslong"
    if name_col is None and "nodenm" in df.columns:
        name_col = "nodenm"

    if lat_col is None or lon_col is None:
        raise ValueError("bus_stops 데이터에서 위도/경도 컬럼을 찾을 수 없습니다.")

    out = pd.DataFrame(
        {
            "name": df[name_col].astype(str).str.strip() if name_col else None,
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        }
    ).dropna(subset=["lat", "lon"])
    return out


def load_redevelopment_projects() -> pd.DataFrame:
    """정비사업(재건축·리모델링) 추진현황을 로드한다. 안양시가 직접 발행한 데이터.

    이미 완료되어 더 이상 이주 리스크가 없는 사업(사업단계="준공" 또는
    현추진상황="이전고시")은 제외한다 — REDEVELOPMENT_COMPLETED_* (config.py) 참고.
    소량(수십 건) 데이터셋이라 RAW_FILE_STEMS 자동탐색 패턴 대신 data/sources/
    redevelopment/ 폴더를 직접 스캔한다.
    """
    from src.config import DATA_SOURCES_DIR, REDEVELOPMENT_COMPLETED_STAGES, REDEVELOPMENT_COMPLETED_STATUSES

    source_dir = DATA_SOURCES_DIR / "redevelopment"
    empty = pd.DataFrame(columns=["name", "lat", "lon", "households", "stage", "status"])
    if not source_dir.exists():
        return empty

    candidates = list(source_dir.glob("*.csv")) + list(source_dir.glob("*.xlsx")) + list(source_dir.glob("*.xls"))
    if not candidates:
        return empty

    df = _read_table_flex(candidates[0])
    name_col = _find_column(df, ["정비구역명"])
    lat_col = _find_column(df, ["위도"])
    lon_col = _find_column(df, ["경도"])
    households_col = _find_column(df, ["사업시행세대수총계", "세대수"])
    stage_col = _find_column(df, ["사업단계"])
    status_col = _find_column(df, ["현추진상황"])

    if lat_col is None or lon_col is None:
        return empty

    out = pd.DataFrame(
        {
            "name": df[name_col].astype(str).str.strip() if name_col else "정비사업",
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
            "households": pd.to_numeric(df[households_col], errors="coerce").fillna(0) if households_col else 0,
            "stage": df[stage_col].astype(str).str.strip() if stage_col else "",
            "status": df[status_col].astype(str).str.strip() if status_col else "",
        }
    ).dropna(subset=["lat", "lon"])

    out = out[~out["stage"].isin(REDEVELOPMENT_COMPLETED_STAGES)]
    out = out[~out["status"].isin(REDEVELOPMENT_COMPLETED_STATUSES)]
    return out.reset_index(drop=True)
