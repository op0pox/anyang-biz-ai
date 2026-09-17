"""행정동별 소상공인 지원 우선순위 페이지 — 행정 의사결정 지원 핵심 화면."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from common import SERVICE_NAME, inject_css, load_pipeline, render_topbar, show_pipeline_error  # noqa: E402
from src.model import compute_support_priority  # noqa: E402

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
for i, row in display_df.iterrows():
    st.markdown(
        f"""
        <div class="anyang-rank-row">
            <span class="rank-num">{i + 1}</span>
            <span style="flex:1; font-weight:600;">{row['dong']}</span>
            <span style="color:#6B7684; font-size:0.85rem;">
                평균 예상매출 {row['avg_predicted_sales']:,.0f}원 · 경쟁점포 {row['total_competitor_count']:.0f}개
            </span>
            <span style="font-weight:800; color:#0B5ED7; margin-left:1rem;">{row['priority_score']:.1f}점</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

with st.expander("🔍 스코어 산출 방식 상세"):
    st.write(
        """
        - 매출 수준 점수(45%): 예측 매출이 낮을수록 지원 필요도가 높다고 판단
        - 경쟁 강도 점수(30%): 동일 행정동 내 경쟁점포수가 많을수록 지원 필요도가 높다고 판단
        - 매출 효율 점수(25%): 유동인구 대비 예측 매출(효율)이 낮을수록 지원 필요도가 높다고 판단
        """
    )
    st.dataframe(priority_df, width="stretch", hide_index=True)
