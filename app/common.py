"""Streamlit 앱 공통 유틸: CSS 주입, 상단바, 캐시된 데이터/모델 로딩."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DataFileNotFoundError  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    build_feature_table,
    compute_closure_rate_table,
    get_other_region_support_center_examples,
    get_redevelopment_risk_by_dong,
)
from src.model import TrainedModel, train_model  # noqa: E402

SERVICE_NAME = "안양 상권 나침반"
SERVICE_TAGLINE = "안양시 소상공인 지원 우선순위 AI"


def inject_css():
    css_path = PROJECT_ROOT / "app" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_topbar():
    st.markdown(
        f"""
        <div class="anyang-topbar">
            <div class="brand">🧭 {SERVICE_NAME}</div>
            <div class="tagline">{SERVICE_TAGLINE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _cached_feature_table():
    return build_feature_table()


@st.cache_resource(show_spinner=False)
def _cached_trained_model(feature_table_hash: int) -> TrainedModel:
    # feature_table_hash는 캐시 키 용도로만 쓰이고, 실제 학습에는 캐시된 피처 테이블을 사용한다.
    feature_table = _cached_feature_table()
    return train_model(feature_table)


def load_pipeline():
    """피처 테이블과 학습된 모델을 로드한다. 실패 시 (None, None, 에러메시지)를 반환한다."""
    try:
        feature_table = _cached_feature_table()
        if feature_table.empty:
            return None, None, "데이터가 비어 있습니다. data/raw/ 에 원본 데이터 파일이 있는지 확인해주세요."
        trained = _cached_trained_model(len(feature_table))
        return feature_table, trained, None
    except DataFileNotFoundError as exc:
        return None, None, str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, None, f"데이터 처리 중 오류가 발생했습니다: {exc}"


@st.cache_data(show_spinner=False)
def load_redevelopment_risk() -> dict:
    """행정동별 정비사업 인접 리스크 조회 (캐시됨). 데이터가 없거나 오류가 나도
    조용히 빈 dict를 반환한다 — 이 정보는 부가 참고용이라 없어도 서비스는 정상 동작해야 한다."""
    try:
        return get_redevelopment_risk_by_dong()
    except Exception:  # noqa: BLE001
        return {}


@st.cache_data(show_spinner="분기별 상가 데이터를 비교하는 중입니다...")
def load_closure_rate_table():
    """행정동×업종별 폐업 흐름 통계 조회 (캐시됨, 300MB+ 파일 2개 비교라 최초 1회만
    계산). 실패하면 None을 반환해 UI에서 이 스탯만 조용히 생략할 수 있게 한다."""
    try:
        table = compute_closure_rate_table()
        return table if not table.empty else None
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(show_spinner=False)
def load_other_region_support_center_examples() -> list:
    """안양시엔 없는 업종별 소공인 특화지원센터의 타 지역 운영 사례 조회 (캐시됨).
    실패하거나 매칭이 없으면 빈 리스트 — 부가 참고용이라 없어도 서비스는 정상 동작해야 한다."""
    try:
        return get_other_region_support_center_examples()
    except Exception:  # noqa: BLE001
        return []


def show_pipeline_error(message: str):
    st.error(
        "⚠️ 데이터를 불러오지 못했습니다.\n\n"
        f"{message}\n\n"
        "먼저 `python scripts/make_sample_data.py`를 실행해 가상 데이터로 서비스를 확인해보실 수 있습니다."
    )
