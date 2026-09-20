"""지원사업 매칭 도우미 — 사장님 페르소나 전용. ③처방 단계를 행정동×업종 집계가
아니라 개인(사업자) 단위로 확장한 페이지. src/support_matching.py 참고."""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from common import (  # noqa: E402
    SERVICE_NAME,
    inject_css,
    load_pipeline,
    load_support_program_announcements,
    render_topbar,
    show_pipeline_error,
)
from src.support_matching import CONFIRM_NOTICE, match_support_programs  # noqa: E402

st.set_page_config(page_title=f"지원사업 매칭 | {SERVICE_NAME}", page_icon="🧾", layout="wide")
inject_css()
render_topbar()

st.markdown('<div class="anyang-section-title">🧾 지원사업 매칭 도우미</div>', unsafe_allow_html=True)
st.caption("업종·지역 등 몇 가지만 알려주시면, 지금 신청할 수 있는 지원사업을 찾아드립니다.")

feature_table, trained, error = load_pipeline()
if error:
    show_pipeline_error(error)
    st.stop()

categories = sorted(feature_table["category"].unique())

with st.form("match_form", border=True):
    st.markdown("**🔍 내 정보 입력**")
    c1, c2 = st.columns(2)
    with c1:
        selected_category = st.selectbox("업종", categories, key="match_category")
        years_in_business = st.number_input("업력(년)", min_value=0.0, max_value=60.0, value=2.0, step=0.5, key="match_years")
    with c2:
        employee_count = st.number_input(
            "고용 인원(본인 제외) — 참고용", min_value=0, max_value=100, value=0, step=1, key="match_employees",
            help="공고문마다 고용 인원 조건 표기 방식이 제각각이라 아직 결과 순서에는 반영하지 않고, 조건 확인 시 참고할 수 있도록만 함께 보여드립니다.",
        )
        annual_sales = st.number_input(
            "연매출(원, 대략) — 참고용", min_value=0, value=0, step=1_000_000, key="match_sales",
            help="정확하지 않아도 괜찮습니다. 지원사업마다 매출 기준 표기 방식이 달라 아직 결과 순서에는 반영하지 않고, 참고용으로만 보여드립니다.",
        )
    st.caption("※ 업종·업력은 맞는 공고를 위로 정렬하는 데 실제로 쓰입니다. 지원사업 대부분이 안양시 전역/전국 단위라 행정동 입력은 받지 않습니다.")
    submitted = st.form_submit_button("🔍 지원사업 찾아보기", width="stretch")

if submitted:
    st.session_state["match_profile"] = {
        "category": selected_category,
        "years_in_business": years_in_business,
        "employee_count": employee_count,
        "annual_sales": annual_sales,
    }

profile = st.session_state.get("match_profile")

st.markdown('<div class="anyang-section-title">매칭 결과</div>', unsafe_allow_html=True)

if profile is None:
    st.info("정보를 입력하고 '지원사업 찾아보기'를 눌러주세요.")
else:
    programs = load_support_program_announcements()
    matches = match_support_programs(profile, programs)

    is_placeholder_data = not programs.empty and (programs["source"] == "fallback").all()
    if is_placeholder_data:
        st.warning(
            "⚠️ 아직 기업마당·K-Startup 실제 공고 API가 연동되지 않아, 아래는 구조 검증용 예시 "
            "데이터입니다(실제 지원사업이 아닙니다). API 키 발급 후 실데이터로 교체될 예정입니다."
        )

    if not matches:
        st.info("지금 신청 가능한 공고를 찾지 못했습니다. 잠시 후 다시 확인해보세요.")
    else:
        selected_category = profile["category"]
        top_priority = sum(1 for m in matches if m["mentions_category"] or m["fits_years"])
        st.caption(
            f"지금 신청 가능한 공고 {len(matches)}건입니다(마감이 지난 공고는 제외). "
            f"대부분 업종을 가리지 않는 지원사업이라 전체를 보여드리되, '{selected_category}' 업종이나 "
            f"입력하신 업력({profile['years_in_business']:.1f}년)에 맞는 {top_priority}건을 위로 정렬했습니다."
        )
        for m in matches:
            dday = f"D-{m['days_left']}" if m["days_left"] is not None else "상시/미상"
            link_html = (
                f'<a href="{m["detail_url"]}" target="_blank" style="color:#0B5ED7; font-weight:700;">원문 공고 보러가기 →</a>'
                if m["detail_url"]
                else '<span style="color:#9AA5B1;">원문 링크 미제공 — 소관기관에 직접 문의 필요</span>'
            )
            badges = ""
            if m["mentions_category"]:
                badges += f'<span class="anyang-badge" style="background:#17A67322; color:#17A673; margin-left:0.4rem;">"{selected_category}" 언급</span>'
            if m["fits_years"]:
                badges += '<span class="anyang-badge" style="background:#0B5ED722; color:#0B5ED7; margin-left:0.4rem;">업력 조건에 맞음</span>'
            st.markdown(
                f"""
                <div class="anyang-card" style="margin-bottom:0.7rem;">
                    <div style="display:flex; align-items:center; justify-content:space-between;">
                        <div><b style="font-size:1.02rem;">{m['name']}</b>{badges}</div>
                        <span class="anyang-badge">{dday}</span>
                    </div>
                    <div style="font-size:0.82rem; color:#6B7684; margin-top:0.3rem;">{m['agency']} · {m['source']}</div>
                    <p style="font-size:0.9rem; color:#1B2430; margin-top:0.5rem;">{m['target']}</p>
                    <div style="margin-top:0.5rem;">{link_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div class="anyang-card" style="margin-top:1rem; border-left:4px solid #E67E22;">
            ⚠️ <b>{CONFIRM_NOTICE}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
