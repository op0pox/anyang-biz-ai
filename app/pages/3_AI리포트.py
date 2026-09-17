"""AI 정책분석 리포트 생성 페이지."""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from common import (  # noqa: E402
    SERVICE_NAME,
    inject_css,
    load_closure_rate_table,
    load_pipeline,
    load_redevelopment_risk,
    render_topbar,
    show_pipeline_error,
)
from src.ai_report import generate_report  # noqa: E402
from src.model import compute_support_priority, predict_sales  # noqa: E402

st.set_page_config(page_title=f"AI 리포트 | {SERVICE_NAME}", page_icon="🤖", layout="wide")
inject_css()
render_topbar()

st.markdown('<div class="anyang-section-title">🤖 AI 정책분석 리포트</div>', unsafe_allow_html=True)
st.caption(
    "기존 상권분석 서비스는 숫자·그래프만 제공합니다. "
    "이 기능은 AI가 예측 근거를 바탕으로 '왜 우선순위가 높은지'를 설명해드립니다."
)

feature_table, trained, error = load_pipeline()

if error:
    show_pipeline_error(error)
    st.stop()

dongs = sorted(feature_table["dong"].unique())
categories = sorted(feature_table["category"].unique())

with st.sidebar:
    st.subheader("리포트 대상 선택")
    selected_dong = st.selectbox("행정동", dongs, key="report_dong")
    selected_category = st.selectbox("업종", categories, key="report_category")
    generate_clicked = st.button("📝 AI 정책분석 리포트 생성", width="stretch")

if not generate_clicked:
    st.info("왼쪽에서 행정동과 업종을 선택한 뒤 'AI 정책분석 리포트 생성' 버튼을 눌러주세요.")
    st.stop()


def _bullet_card(title: str, icon: str, items: list, numbered: bool = False) -> str:
    if not items:
        items = ["표시할 내용이 없습니다."]
    if numbered:
        rows = "".join(
            f'<li style="margin-bottom:0.4rem;"><b>{i + 1}.</b> {text}</li>' for i, text in enumerate(items)
        )
        list_tag = "ol"
    else:
        rows = "".join(f'<li style="margin-bottom:0.4rem;">{text}</li>' for text in items)
        list_tag = "ul"
    return f"""
    <div class="anyang-card" style="height:100%;">
        <h3 style="margin-bottom:0.6rem;">{icon} {title}</h3>
        <{list_tag} style="margin:0; padding-left:1.2rem; font-size:0.92rem; color:#1B2430; line-height:1.5;">
            {rows}
        </{list_tag}>
    </div>
    """


with st.spinner("AI가 리포트를 작성하고 있습니다..."):
    prediction = predict_sales(trained, selected_dong, selected_category)

    if prediction is None:
        st.warning(f"'{selected_dong}'의 '{selected_category}' 업종에 대한 데이터를 찾을 수 없습니다.")
        st.stop()

    priority_df = compute_support_priority(trained)
    priority_row = None
    matched = priority_df[priority_df["dong"] == selected_dong]
    if not matched.empty:
        priority_row = matched.iloc[0].to_dict()

    try:
        report = generate_report(prediction, priority_row)
    except Exception as exc:  # noqa: BLE001
        report = {
            "diagnosis": [],
            "evidence_commentary": [],
            "recommendations": [],
            "source": "offline",
            "notice": f"리포트 생성 중 오류가 발생했습니다: {exc}",
        }

if report.get("notice"):
    st.info(f"ℹ️ {report['notice']}")

source_badge = "🟢 OpenAI GPT 생성" if report.get("source") == "openai" else "⚪ 오프라인 규칙 기반 생성"
st.markdown(
    f"""
    <div class="anyang-hero" style="padding:1.6rem 1.8rem; text-align:left;">
        <div class="anyang-badge">{source_badge}</div>
        <h1 style="font-size:1.4rem; margin-bottom:0.2rem;">{selected_dong} · {selected_category} 정책분석 리포트</h1>
        <p class="subtitle" style="font-size:0.9rem; margin-bottom:0;">예측 매출과 소상공인 지원 우선순위를 근거로 AI가 해석한 결과입니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2 = st.columns(2)
with m1:
    st.markdown(
        f"""<div class="anyang-metric-box"><div class="value">{prediction['predicted_sales']:,.0f}원</div>
        <div class="label">예상 매출(추정)</div></div>""",
        unsafe_allow_html=True,
    )
with m2:
    if priority_row:
        st.markdown(
            f"""<div class="anyang-metric-box"><div class="value">{priority_row['priority_score']:.1f}점</div>
            <div class="label">소상공인 지원 우선순위</div></div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div class="anyang-metric-box"><div class="value">-</div>
            <div class="label">우선순위 데이터 없음</div></div>""",
            unsafe_allow_html=True,
        )

st.markdown('<div class="anyang-section-title">📌 근거 데이터</div>', unsafe_allow_html=True)
st.caption("AI가 문장을 지어낸 게 아니라, 아래 실제 수치를 바탕으로 해석한 결과입니다.")

features = prediction["features"]
closure_table = load_closure_rate_table()
closure_row = None
if closure_table is not None:
    matched_closure = closure_table[
        (closure_table["dong"] == selected_dong) & (closure_table["category"] == selected_category)
    ]
    if not matched_closure.empty:
        closure_row = matched_closure.iloc[0]

e1, e2, e3, e4, e5 = st.columns(5)
evidence_stats = [
    (e1, "🏪", "경쟁점포수", f"{features['competitor_count']:.0f}개", "소상공인시장진흥공단 상가정보"),
    (e2, "🚶", "유동인구(가중합)", f"{features['floating_population']:,.0f}", "경기데이터드림 유동인구_안양시"),
    (e3, "🏠", "거주인구", f"{features['resident_population']:,.0f}명", "안양시 주민등록인구 통계"),
    (e4, "🚌", "버스정류장(반경 300m)", f"{features['bus_stop_count']:.0f}개", "국토교통부 버스정류소정보(TAGO)"),
    (
        e5,
        "📉",
        "최근 폐업 후보 비율(추정)",
        f"{closure_row['closure_rate']*100:.0f}%" if closure_row is not None else "정보 없음",
        "상가정보 202403→202606 비교 추정치",
    ),
]
for col, icon, label, value, source in evidence_stats:
    with col:
        st.markdown(
            f"""
            <div class="anyang-card" style="text-align:center; padding:1.1rem 0.8rem;">
                <div style="font-size:1.4rem;">{icon}</div>
                <div style="font-size:1.15rem; font-weight:800; color:#0B5ED7; margin:0.3rem 0;">{value}</div>
                <div style="font-size:0.82rem; color:#1B2430; font-weight:600;">{label}</div>
                <div style="font-size:0.68rem; color:#9AA5B1; margin-top:0.3rem;">출처: {source}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if closure_row is not None:
    st.caption(
        "※ 폐업 후보 비율은 2024년 1분기와 2026년 2분기 상가정보에서 사라진 점포 수 "
        "기준 추정치입니다 — 실제 폐업 외 이전·업종변경·데이터 정정도 포함될 수 있습니다."
    )
elif selected_dong in {"명학동", "박달동", "병목안동", "호현동"}:
    st.caption("※ 이 행정동은 비교 기간 중 행정동 개편(명칭 변경·통합)이 있어 폐업 흐름 비교에서 제외했습니다.")

redevelopment_risk = load_redevelopment_risk().get(selected_dong)
if redevelopment_risk:
    project_names = "、".join(p["name"] for p in redevelopment_risk["projects"][:3])
    st.markdown(
        f"""
        <div class="anyang-card" style="border-left:4px solid #E67E22; margin-bottom:0.8rem;">
            <b>⚠️ 정비사업 인접 리스크</b> — {selected_dong} 반경 500m 내 진행 중인 정비사업
            {redevelopment_risk['count']}건({project_names} 등, 총 {redevelopment_risk['total_households']:,}세대).
            향후 이주로 인한 유동인구 변화 가능성이 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("출처: 경기도 안양시_일반 정비사업 추진현황(안양시 AI정책과 발행)")

st.write("")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(_bullet_card("진단", "🔍", report.get("diagnosis", [])), unsafe_allow_html=True)
with c2:
    st.markdown(_bullet_card("근거 해석", "📊", report.get("evidence_commentary", [])), unsafe_allow_html=True)
with c3:
    st.markdown(_bullet_card("정책적 시사점 · 추천", "🎯", report.get("recommendations", []), numbered=True), unsafe_allow_html=True)

_source_line = (
    "카드소비 데이터(경기데이터드림, 2026년 1~3월) · 상가(상권)정보(소상공인시장진흥공단) · "
    "주민등록인구 통계(안양시) · 버스정류소정보(국토교통부 TAGO) · 유동인구_안양시(경기데이터드림)"
)
if redevelopment_risk:
    _source_line += " · 정비사업 추진현황(안양시 AI정책과)"
st.markdown(
    f"""
    <div style="margin-top:1.4rem; padding:0.9rem 1.1rem; background:#F5F8FC; border:1px solid #E3E9F1;
                border-radius:12px; font-size:0.75rem; color:#6B7684;">
        <b>데이터 출처</b> · {_source_line}
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("🔍 리포트에 사용된 원본 예측 데이터"):
    st.json(
        {
            "예상_매출": prediction["predicted_sales"],
            "주요_피처": prediction["features"],
            "우선순위_정보": priority_row,
            "리포트_원본": report,
        }
    )
