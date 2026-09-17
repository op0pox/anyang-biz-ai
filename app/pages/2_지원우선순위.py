"""행정동별 소상공인 지원 우선순위 페이지 — 행정 의사결정 지원 핵심 화면."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from common import (  # noqa: E402
    SERVICE_NAME,
    inject_css,
    load_pipeline,
    load_redevelopment_risk,
    render_topbar,
    show_pipeline_error,
)
from src.config import (  # noqa: E402
    PRIORITY_WEIGHT_COMPETITION,
    PRIORITY_WEIGHT_EFFICIENCY,
    PRIORITY_WEIGHT_SALES,
)
from src.model import (  # noqa: E402
    compute_support_priority,
    match_safety_net,
    rank_categories_for_dong,
    recommend_support_type,
    simulate_dong_scenario,
)

st.set_page_config(page_title=f"지원우선순위 | {SERVICE_NAME}", page_icon="🗺️", layout="wide")
inject_css()
render_topbar()

st.markdown('<div class="anyang-section-title">🗺️ 행정동별 소상공인 지원 우선순위</div>', unsafe_allow_html=True)
st.caption(
    "예측 매출 수준, 경쟁 강도, 유동인구 대비 매출 효율을 종합해 "
    "소상공인 지원 예산·정책을 어느 행정동에 우선 배정할지 판단할 수 있도록 스코어링했습니다."
)

feature_table, trained, error = load_pipeline()

if error:
    show_pipeline_error(error)
    st.stop()

priority_df = compute_support_priority(trained)

if priority_df.empty:
    st.warning("우선순위를 계산할 데이터가 없습니다.")
    st.stop()

top_n = st.slider("표시할 행정동 수", min_value=5, max_value=len(priority_df), value=min(10, len(priority_df)))
display_df = priority_df.head(top_n)

st.markdown('<div class="anyang-section-title">지원 필요도 랭킹</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div style="font-size:0.82rem; color:#6B7684; margin-bottom:0.9rem;">
        점수는 3가지 요소를 <b>가중합</b>해 계산합니다 — 아래 막대에서 각 요소가 점수에
        얼마나 기여했는지 바로 확인할 수 있습니다.
        <span style="color:#0B5ED7; font-weight:700;">■ 매출부진 45%</span>
        <span style="color:#E67E22; font-weight:700; margin-left:0.6rem;">■ 경쟁강도 30%</span>
        <span style="color:#17A673; font-weight:700; margin-left:0.6rem;">■ 매출효율 25%</span>
    </div>
    """,
    unsafe_allow_html=True,
)
for i, row in display_df.iterrows():
    contrib_sales = PRIORITY_WEIGHT_SALES * row["sales_score"]
    contrib_competition = PRIORITY_WEIGHT_COMPETITION * row["competition_score"]
    contrib_efficiency = PRIORITY_WEIGHT_EFFICIENCY * row["efficiency_score"]
    st.markdown(
        f"""
        <div class="anyang-rank-row" style="flex-direction:column; align-items:stretch; gap:0;">
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <div style="display:flex; align-items:center; gap:0.7rem;">
                    <span class="rank-num">{i + 1}</span>
                    <span style="font-weight:700;">{row['dong']}</span>
                    <span style="color:#6B7684; font-size:0.8rem;">
                        평균 예상매출 {row['avg_predicted_sales']:,.0f}원 · 경쟁점포 {row['total_competitor_count']:.0f}개
                    </span>
                </div>
                <span style="font-weight:800; color:#0B5ED7;">{row['priority_score']:.1f}점</span>
            </div>
            <div style="display:flex; width:100%; height:10px; border-radius:6px; overflow:hidden; background:#EEF1F5; margin-top:0.5rem;">
                <div style="width:{contrib_sales}%; background:#0B5ED7;" title="매출부진 기여 {contrib_sales:.1f}점"></div>
                <div style="width:{contrib_competition}%; background:#E67E22;" title="경쟁강도 기여 {contrib_competition:.1f}점"></div>
                <div style="width:{contrib_efficiency}%; background:#17A673;" title="매출효율 기여 {contrib_efficiency:.1f}점"></div>
            </div>
            <div style="display:flex; gap:1.1rem; margin-top:0.35rem; font-size:0.72rem; color:#6B7684;">
                <span>매출부진 {contrib_sales:.1f}점</span>
                <span>경쟁강도 {contrib_competition:.1f}점</span>
                <span>매출효율 {contrib_efficiency:.1f}점</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="anyang-section-title">🎯 이 행정동엔 어떤 지원이 필요할까요?</div>', unsafe_allow_html=True)
st.caption("랭킹에서 행정동을 골라, 어떤 지원이 맞는지와 그 지원을 했을 때 예상되는 효과를 확인하세요.")

FACTOR_LABELS = {"sales": "매출 부진", "competition": "경쟁 강도", "efficiency": "매출 효율"}

detail_dong = st.selectbox("행정동 선택", display_df["dong"].tolist(), key="detail_dong_select")
detail_row = priority_df[priority_df["dong"] == detail_dong].iloc[0].to_dict()
recommendation = recommend_support_type(detail_row)

dc1, dc2 = st.columns(2)
with dc1:
    st.markdown(
        f"""
        <div class="anyang-card" style="height:100%;">
            <div class="anyang-badge">진단: {FACTOR_LABELS[recommendation['dominant_factor']]} 요인이 가장 큼</div>
            <h3 style="margin-top:0.7rem;">{recommendation['title']}</h3>
            <p style="font-size:0.9rem; color:#1B2430; line-height:1.5;">{recommendation['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with dc2:
    current_bus = feature_table.loc[feature_table["dong"] == detail_dong, "bus_stop_count"].iloc[0]
    # 실측 최댓값을 넘어서는 "확충"은 부분의존도 곡선이 평평하게 잘려 신뢰할 수
    # 없으므로, 남은 여유분만큼만 슬라이더로 시험해볼 수 있게 한다.
    observed_bus_max = int(feature_table["bus_stop_count"].max())
    room = max(0, observed_bus_max - int(current_bus))
    if room > 0:
        add_bus = st.slider(
            "🚌 버스정류장을 몇 개 확충하면?", min_value=0, max_value=room, value=min(3, room), key="support_bus_slider"
        )
    else:
        add_bus = 0
        st.write("🚌 버스정류장을 몇 개 확충하면?")
        st.caption(f"※ {detail_dong}은 이미 안양시 관측 최댓값({observed_bus_max}개) 수준이라 늘리는 시나리오를 시험해볼 수 없습니다.")
    scenario = simulate_dong_scenario(trained, detail_dong, {"bus_stop_count": current_bus + add_bus})
    if scenario and scenario["baseline_sales"] > 0:
        diff_pct = 100 * (scenario["predicted_sales"] - scenario["baseline_sales"]) / scenario["baseline_sales"]
        color = "#17A673" if diff_pct >= 0 else "#c0392b"
        arrow = "▲" if diff_pct >= 0 else "▼"
        st.markdown(
            f"""
            <div class="anyang-card" style="height:100%; text-align:center;">
                <div style="font-size:0.8rem; color:#6B7684;">버스정류장 {add_bus}개 확충 시 {detail_dong} 전체 예상 매출 변화</div>
                <div style="font-size:1.5rem; font-weight:800; color:{color}; margin:0.3rem 0;">{arrow} {abs(diff_pct):.1f}%</div>
                <div style="font-size:0.72rem; color:#9AA5B1;">{scenario['baseline_sales']:,.0f}원 → {scenario['predicted_sales']:,.0f}원 (모든 업종 합산 추정)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("이 행정동은 시나리오를 계산할 데이터가 부족합니다.")

st.caption(
    "※ 현재 정량적으로 시뮬레이션 가능한 정책 변수는 버스정류장(대중교통 접근성)뿐입니다 — "
    "이 변수만 실측값이 촘촘해 전체 데이터 평균 효과를 안정적으로 추정할 수 있었습니다. "
    "그 외 지원 유형(마케팅, 임대료 지원 등)은 데이터 근거가 부족해 정성적 권고로만 제공합니다."
)

safety_net = match_safety_net(detail_row["priority_score"])
tier_color = {"예방": "#17A673", "긴급수혈": "#E67E22", "재기지원": "#c0392b"}[safety_net["tier"]]
program_rows = "".join(
    f"""<li style="margin-bottom:0.4rem;"><b>{p['name']}</b> — {p['description']}</li>"""
    for p in safety_net["programs"]
)
st.markdown(
    f"""
    <div class="anyang-card" style="margin-top:1rem; border-left:4px solid {tier_color};">
        <div class="anyang-badge" style="background:{tier_color}22; color:{tier_color};">
            안전망 단계: {safety_net['tier']} (스코어 {safety_net['range']})
        </div>
        <h3 style="margin-top:0.7rem;">{detail_dong}에 매칭되는 지원제도</h3>
        <ul style="margin:0.4rem 0 0; padding-left:1.2rem; font-size:0.9rem; color:#1B2430;">
            {program_rows}
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "※ 안전망 매칭은 지원우선순위 스코어의 절대 수준(얼마나 급한지)에 따른 것으로, "
    "위 지배 요인 진단(왜 필요한지)과는 별개의 기준입니다."
)

redevelopment_risk = load_redevelopment_risk().get(detail_dong)
if redevelopment_risk:
    project_names = "、".join(p["name"] for p in redevelopment_risk["projects"][:3])
    st.markdown(
        f"""
        <div class="anyang-card" style="border-left:4px solid #E67E22; margin-top:0.8rem;">
            <b>⚠️ 정비사업 인접 리스크</b> — {detail_dong} 반경 500m 내 진행 중인 정비사업
            {redevelopment_risk['count']}건({project_names} 등, 총 {redevelopment_risk['total_households']:,}세대).
            향후 이주로 인한 유동인구 변화 가능성이 있어 예산 배정 시 참고할 만합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("출처: 경기도 안양시_일반 정비사업 추진현황(안양시 AI정책과 발행)")

st.write("")
if st.button(f"🤖 {detail_dong} AI 정책분석 리포트 보러가기", key="goto_ai_report", width="stretch"):
    ranking_for_dong = rank_categories_for_dong(trained, detail_dong)
    st.session_state["report_dong"] = detail_dong
    if not ranking_for_dong.empty:
        st.session_state["report_category"] = ranking_for_dong.iloc[0]["category"]
    st.switch_page("pages/3_AI리포트.py")

st.markdown('<div class="anyang-section-title">우선순위 스코어 비교</div>', unsafe_allow_html=True)
chart_df = display_df[["dong", "priority_score"]].set_index("dong")
st.bar_chart(chart_df, width="stretch")

try:
    import folium
    from streamlit_folium import st_folium

    from src.config import ANYANG_CENTER_LAT, ANYANG_CENTER_LON
    from src.data_loader import load_stores

    stores = load_stores()
    centroids = (
        stores.dropna(subset=["lat", "lon", "dong"])
        .groupby("dong", as_index=False)[["lat", "lon"]]
        .mean()
    )
    map_df = priority_df.merge(centroids, on="dong", how="inner")

    if map_df.empty:
        st.info("행정동 좌표 정보가 없어 지도를 표시할 수 없습니다.")
    else:
        st.markdown('<div class="anyang-section-title">지도에서 보기</div>', unsafe_allow_html=True)
        m = folium.Map(location=[ANYANG_CENTER_LAT, ANYANG_CENTER_LON], zoom_start=12, tiles="OpenStreetMap")
        max_score = map_df["priority_score"].max() or 1
        for _, row in map_df.iterrows():
            ratio = row["priority_score"] / max_score
            radius = 8 + ratio * 16
            color = "#c0392b" if ratio > 0.66 else ("#e67e22" if ratio > 0.33 else "#17A673")
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=0.55,
                popup=f"{row['dong']} · 우선순위 {row['priority_score']:.1f}점",
            ).add_to(m)
        st_folium(m, use_container_width=True, height=460, key="priority_map")
except Exception as exc:  # noqa: BLE001
    st.info(f"지도를 표시하는 중 문제가 발생해 생략합니다. ({exc})")

with st.expander("🔍 스코어 산출 방식 상세 — 가중치는 왜 이렇게 정했나요?"):
    st.write(
        """
        - **매출 수준 점수(45%)** — 예측 매출이 낮을수록 지원 필요도가 높다고 판단.
          소상공인 지원 예산의 1차 목적이 "매출이 부진한 곳을 우선 지원"하는 것이므로
          가장 높은 가중치를 부여했습니다.
        - **경쟁 강도 점수(30%)** — 동일 행정동 내 경쟁점포수가 많을수록 지원 필요도가
          높다고 판단. 매출 수준이 비슷해도 점포당 배분 가능한 파이가 작아지는 곳일수록
          지원의 체감 효과가 크다고 보아 두 번째로 반영했습니다.
        - **매출 효율 점수(25%)** — 유동인구 대비 예측 매출(효율)이 낮을수록 지원
          필요도가 높다고 판단. 유동인구는 많은데 매출로 이어지지 못한다면 접근성·
          인지도 등 상권 구조 자체의 약점을 시사하므로 보조 지표로 반영했습니다.

        점수 계산: `priority_score = 45% x 매출부진점수 + 30% x 경쟁강도점수 + 25% x 매출효율점수`
        (각 하위 점수는 전체 행정동 중 상대적 위치를 0~100으로 정규화한 값입니다.)
        """
    )
    st.dataframe(
        priority_df[
            [
                "dong",
                "priority_score",
                "sales_score",
                "competition_score",
                "efficiency_score",
                "avg_predicted_sales",
                "total_competitor_count",
                "floating_population",
                "resident_population",
                "bus_stop_count",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
