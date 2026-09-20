"""매출예측 · 업종추천 페이지."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from common import SERVICE_NAME, inject_css, load_pipeline, render_topbar, show_pipeline_error  # noqa: E402
from src.model import (  # noqa: E402
    compute_support_priority,
    get_feature_importance,
    match_safety_net,
    predict_sales,
    rank_categories_for_dong,
    recommend_support_type,
)

st.set_page_config(page_title=f"매출예측·업종추천 | {SERVICE_NAME}", page_icon="📈", layout="wide")
inject_css()
render_topbar()

st.markdown('<div class="anyang-section-title">📈 매출예측 · 업종추천</div>', unsafe_allow_html=True)
st.caption("행정동을 고르면, 그 동네에서 예상 매출이 높은 업종 순위와 근거 데이터를 함께 보여드립니다.")

feature_table, trained, error = load_pipeline()

if error:
    show_pipeline_error(error)
    st.stop()

dongs = sorted(feature_table["dong"].unique())
categories = sorted(feature_table["category"].unique())

if not dongs or not categories:
    st.warning("조회할 수 있는 행정동/업종 데이터가 없습니다.")
    st.stop()

with st.container(border=True):
    st.markdown("**🔍 조회 조건**")
    search_col1, search_col2 = st.columns(2)
    with search_col1:
        selected_dong = st.selectbox("행정동", dongs, key="dong_select")
    with search_col2:
        selected_category = st.selectbox("업종(선택 시 상세 예측 확인)", categories, key="category_select")

result = predict_sales(trained, selected_dong, selected_category)

if result is None:
    st.warning(f"'{selected_dong}'의 '{selected_category}' 업종에 대한 예측 데이터를 찾을 수 없습니다. 다른 조합을 선택해주세요.")
else:
    _confidence_color = {"높음": "#17A673", "보통": "#0B5ED7", "낮음": "#c0392b"}[result["confidence_level"]]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            f"""
            <div class="anyang-metric-box">
                <div class="value">{result['predicted_sales']:,.0f}원</div>
                <div class="label">{selected_dong} · {selected_category} 예상 매출(월 추정)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="margin-top:0.5rem; display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap;">
                <span style="background:{_confidence_color}; color:white; font-size:0.75rem; font-weight:700;
                             padding:0.2rem 0.6rem; border-radius:999px;">예측 신뢰도 {result['confidence_level']}</span>
                <span style="font-size:0.78rem; color:#6B7684;">예상 범위 {result['predicted_low']:,.0f}원 ~ {result['predicted_high']:,.0f}원</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if result["low_sample_warning"]:
            st.caption(
                f"⚠️ {selected_dong}에는 '{selected_category}' 업종 점포가 관측되지 않아, "
                "다른 지역 데이터를 바탕으로 한 추정치입니다 — 신뢰도가 낮습니다."
            )
        elif result["confidence_level"] == "낮음":
            st.caption(
                f"※ 관측된 경쟁점포 수({result['competitor_count']:.0f}개)가 적어 예측 신뢰도가 낮습니다."
            )
    with c2:
        st.markdown(
            f"""
            <div class="anyang-card">
                <h3>주요 근거 지표</h3>
                <p>경쟁점포수: <b>{result['features']['competitor_count']:.0f}개</b><br/>
                유동인구(가중합): <b>{result['features']['floating_population']:,.0f}</b><br/>
                거주인구: <b>{result['features']['resident_population']:,.0f}명</b><br/>
                버스정류장(반경 300m): <b>{result['features']['bus_stop_count']:.0f}개</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "※ 유동인구는 만안구·동안구 구 단위로만 제공되어 같은 구의 행정동은 값이 "
            "동일합니다. 버스정류장 수는 인접 시(광명·군포·의왕 등) 정류소가 일부 섞여 "
            "있을 수 있습니다."
        )

if result is not None:
    st.markdown('<div class="anyang-section-title">🎯 지금 이 동네에서 받을 수 있는 지원</div>', unsafe_allow_html=True)
    st.caption("우리 가게가 있는 행정동 기준으로, 어떤 지원이 맞는지와 얼마나 급한 단계인지 확인하세요.")

    FACTOR_LABELS = {"sales": "매출 부진", "competition": "경쟁 강도", "efficiency": "매출 효율"}
    priority_df = compute_support_priority(trained)
    matched_priority = priority_df[priority_df["dong"] == selected_dong]

    if matched_priority.empty:
        st.info("이 행정동은 지원 매칭에 필요한 데이터가 부족합니다.")
    else:
        priority_row = matched_priority.iloc[0].to_dict()
        recommendation = recommend_support_type(priority_row)
        safety_net = match_safety_net(priority_row["priority_score"])
        tier_color = {"예방": "#17A673", "긴급수혈": "#E67E22", "재기지원": "#c0392b"}[safety_net["tier"]]

        s1, s2 = st.columns(2)
        with s1:
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
        with s2:
            program_rows = "".join(
                f"""<li style="margin-bottom:0.4rem;"><b>{p['name']}</b> — {p['description']}</li>"""
                for p in safety_net["programs"]
            )
            st.markdown(
                f"""
                <div class="anyang-card" style="height:100%; border-left:4px solid {tier_color};">
                    <div class="anyang-badge" style="background:{tier_color}22; color:{tier_color};">
                        안전망 단계: {safety_net['tier']} (스코어 {safety_net['range']})
                    </div>
                    <h3 style="margin-top:0.7rem;">매칭되는 지원제도</h3>
                    <ul style="margin:0.4rem 0 0; padding-left:1.2rem; font-size:0.9rem; color:#1B2430;">
                        {program_rows}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.caption(
            f"※ {selected_dong} 행정동 전체 기준 지원우선순위 스코어({priority_row['priority_score']:.1f}점)를 "
            "바탕으로 매칭했습니다 — 특정 업종이 아니라 행정동 단위 지표입니다."
        )

st.markdown('<div class="anyang-section-title">🏆 이 동네엔 어떤 업종이 유리할까요?</div>', unsafe_allow_html=True)
st.caption("기존 상권분석 서비스에는 없는 기능입니다 — 업종을 직접 고르지 않아도, 지역 기반으로 유리한 업종을 추천합니다.")

ranking = rank_categories_for_dong(trained, selected_dong)
if ranking.empty:
    st.info("추천할 업종 데이터가 부족합니다.")
else:
    chart_df = ranking[["category", "predicted_sales"]].set_index("category")
    st.bar_chart(chart_df, width="stretch")

    low_confidence_categories = ranking.loc[ranking["confidence_level"] == "낮음", "category"].tolist()
    if low_confidence_categories:
        st.caption(
            "⚠️ 표본(관측 경쟁점포 수)이 적어 예측 신뢰도가 낮은 업종: "
            + ", ".join(low_confidence_categories)
        )

st.markdown('<div class="anyang-section-title">🗺️ 상권 지도</div>', unsafe_allow_html=True)
try:
    import folium
    from streamlit_folium import st_folium

    from src.config import ANYANG_CENTER_LAT, ANYANG_CENTER_LON
    from src.data_loader import load_stores

    stores = load_stores()
    dong_stores = stores[stores["dong"] == selected_dong].dropna(subset=["lat", "lon"])

    if dong_stores.empty:
        st.info("해당 행정동의 상가 위치 데이터가 없어 지도를 표시할 수 없습니다.")
    else:
        center_lat = dong_stores["lat"].mean()
        center_lon = dong_stores["lon"].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")
        for _, row in dong_stores.iterrows():
            color = "#0B5ED7" if row["category"] == selected_category else "#9AA5B1"
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.75,
                popup=f"{row['name']} ({row['category']})",
            ).add_to(m)
        st_folium(m, use_container_width=True, height=420, key="store_map")
except Exception as exc:  # noqa: BLE001
    st.info(f"지도를 표시하는 중 문제가 발생해 생략합니다. ({exc})")

with st.expander("🔍 모델 상세정보 (피처 중요도 등)"):
    st.write(f"모델: RandomForestRegressor · RMSE: {trained.rmse:,.0f} · R²: {trained.r2:.3f}")
    importance_df = get_feature_importance(trained)
    st.dataframe(importance_df, width="stretch", hide_index=True)
    st.caption(
        "R²는 실측 데이터(카드소비·상가정보·인구·버스정류장·유동인구) 기준 값입니다. "
        "소수의 대형 점포가 매출 분포를 크게 왜곡해 log1p 변환 후 학습했으며, "
        "행정동×업종 단위 상권 특성 예측에서 통상적으로 관측되는 수준입니다."
    )
