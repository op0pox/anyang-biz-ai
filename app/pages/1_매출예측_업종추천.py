"""매출예측 · 업종추천 페이지."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from common import SERVICE_NAME, inject_css, load_pipeline, render_topbar, show_pipeline_error  # noqa: E402
from src.model import get_feature_importance, predict_sales, rank_categories_for_dong  # noqa: E402

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

with st.sidebar:
    st.subheader("조회 조건")
    selected_dong = st.selectbox("행정동", dongs, key="dong_select")
    selected_category = st.selectbox("업종(선택 시 상세 예측 확인)", categories, key="category_select")

if not dongs or not categories:
    st.warning("조회할 수 있는 행정동/업종 데이터가 없습니다.")
    st.stop()

result = predict_sales(trained, selected_dong, selected_category)

if result is None:
    st.warning(f"'{selected_dong}'의 '{selected_category}' 업종에 대한 예측 데이터를 찾을 수 없습니다. 다른 조합을 선택해주세요.")
else:
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

st.markdown('<div class="anyang-section-title">🏆 이 동네엔 어떤 업종이 유리할까요?</div>', unsafe_allow_html=True)
st.caption("기존 상권분석 서비스에는 없는 기능입니다 — 업종을 직접 고르지 않아도, 지역 기반으로 유리한 업종을 추천합니다.")

ranking = rank_categories_for_dong(trained, selected_dong)
if ranking.empty:
    st.info("추천할 업종 데이터가 부족합니다.")
else:
    chart_df = ranking[["category", "predicted_sales"]].set_index("category")
    st.bar_chart(chart_df, width="stretch")

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
    st.caption("R²는 가상 데이터 기준 참고값입니다. 실제 데이터로 교체 후 재학습하면 유의미한 값을 얻을 수 있습니다.")
