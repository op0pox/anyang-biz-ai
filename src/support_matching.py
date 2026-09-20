"""개별 소상공인 사장님의 지원사업 매칭 — ③처방 단계를 행정동×업종 집계가 아니라
개인(사업자) 단위로 확장한다. 학습된 매출예측 모델과는 별개 축으로, 사용자가 입력한
프로필(업종/매출/업력/지역/고용인원)과 공고 텍스트를 규칙 기반으로 대조한다.

LLM이 아니라 규칙 기반 매칭을 쓰는 이유: 제외대상을 잘못 읽어 "AI가 신청 가능하다고
해서 냈는데 탈락했다"가 되는 리스크(CLAUDE.md "지원사업 매칭·신청 도우미" 참고)를
줄이려면 매칭 로직 자체를 검증 가능하게 유지하는 게 먼저다. 그 위에 원문 링크 +
"반드시 원문에서 최종 확인" 안내문을 얹는 것으로 리스크를 완화한다.
"""

import pandas as pd

CONFIRM_NOTICE = (
    "이 매칭은 규칙 기반 추정입니다. 실제 신청 가능 여부는 반드시 원문 공고에서 "
    "제외대상·서류요건을 직접 확인한 뒤 진행하세요."
)


def _mentions_category(row: pd.Series, category: str) -> bool:
    """target/exclude 텍스트에 사용자 업종 키워드가 직접 언급되는지 본다 —
    "정렬 우선순위"에만 쓰고 걸러내는 데는 안 쓴다(아래 함수 설명 참고)."""
    if not category:
        return False
    haystack = f"{row.get('target', '')} {row.get('exclude', '')}"
    return category in str(haystack)


def _fits_years(row: pd.Series, years_in_business) -> bool:
    """K-Startup의 years_tag(원본 biz_enyy)는 "예비창업자,3년미만,7년미만"처럼 대상
    업력 구간을 쉼표로 나열한 구조화 필드다(실제 응답으로 확인, 2026-09-21). "~년미만"
    토큰 중 가장 큰 값을 그 공고의 사실상 업력 상한으로 보고, 사용자 업력이 그보다
    작으면 맞는다고 판단한다. "예비창업자" 태그는 업력 0년(아직 창업 전)도 포함한다는
    뜻이라 0년 이하를 포함해서 받아들인다. 기업마당처럼 이 필드가 없는 공고는 판단할
    근거가 없으니 "안 맞다"가 아니라 "모른다"로 처리해 boost만 안 줄 뿐 걸러내지
    않는다(match_support_programs와 같은 원칙)."""
    import re

    tag = str(row.get("years_tag", "") or "")
    if not tag or years_in_business is None:
        return False

    thresholds = [int(n) for n in re.findall(r"(\d+)년미만", tag)]
    if "예비창업자" in tag and years_in_business <= 0:
        return True
    if not thresholds:
        return False
    return years_in_business < max(thresholds)


def match_support_programs(profile: dict, programs: pd.DataFrame) -> list:
    """profile: {"category": str, "years_in_business": float, ...} 등 사용자 입력.

    **업종/업력으로 걸러내지 않고, 맞는 공고를 위로 정렬하는 데만 쓴다.** 기업마당/
    K-Startup 응답의 category 필드(경영/금융/수출/사업화 등)는 "업종"이 아니라
    "지원 유형"을 나타내는 별개의 분류 체계라는 게 실제 데이터로 확인됐다(2026-09-21
    — 이 필드로 9개 업종 전부 매칭을 시도해본 결과 전부 0건이 나와서 발견). 특례보증·
    경영컨설팅 같은 지원사업 대다수는 애초에 업종을 가리지 않으므로, 업종을 자동
    필터링 기준으로 쓰는 것 자체가 데이터를 잘못 해석한 설계였다. 대신 (1) 마감되지
    않은 공고인지만 거르고, (2) 업종 키워드가 target/exclude에 직접 언급되는지,
    (3) K-Startup의 구조화된 업력 태그(years_tag)에 사용자 업력이 맞는지 — 이 둘을
    "정렬 우선순위"로만 반영해 사용자 입력이 실제로 결과 순서에 반영되게 한다.

    매출·고용인원은 공고문마다 조건 표기 형식이 제각각이고(예: "연매출 10억 이하"
    vs "소상공인법 시행령 기준") 구조화된 필드가 없어 자동 반영이 불가능해, 화면에는
    참고 정보로만 노출한다 — 최종 조건 확인은 원문 공고 + CONFIRM_NOTICE로 사용자에게
    넘긴다.
    """
    if programs.empty:
        return []

    today = pd.Timestamp.today().normalize()
    apply_end = pd.to_datetime(programs["apply_end"], errors="coerce")
    matched = programs[apply_end.isna() | (apply_end >= today)].copy()
    if matched.empty:
        return []

    category = profile.get("category", "")
    years_in_business = profile.get("years_in_business")
    matched["_mentions_category"] = matched.apply(lambda row: _mentions_category(row, category), axis=1)
    matched["_fits_years"] = matched.apply(lambda row: _fits_years(row, years_in_business), axis=1)
    matched["_apply_end"] = pd.to_datetime(matched["apply_end"], errors="coerce")
    matched = matched.sort_values(
        ["_mentions_category", "_fits_years", "_apply_end"],
        ascending=[False, False, True],
        na_position="last",
    )

    results = []
    for _, row in matched.iterrows():
        end = row["_apply_end"]
        days_left = int((end - today).days) if pd.notna(end) else None
        results.append(
            {
                "name": row.get("name", ""),
                "agency": row.get("agency", ""),
                "target": row.get("target", ""),
                "apply_end": end.strftime("%Y-%m-%d") if pd.notna(end) else "상시/미상",
                "days_left": days_left,
                "detail_url": row.get("detail_url", "") or "",
                "source": row.get("source", ""),
                "mentions_category": bool(row["_mentions_category"]),
                "fits_years": bool(row["_fits_years"]),
            }
        )
    return results
