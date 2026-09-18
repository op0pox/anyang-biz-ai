"""매출예측 · 업종추천 페이지."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from common import SERVICE_NAME, inject_css, load_pipeline, render_topbar, show_pipeline_error  # noqa: E402
from src.model import get_feature_importance, predict_sales, rank_categories_for_dong, simulate_scenario  # noqa: E402

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

if result is not None:
    st.markdown('<div class="anyang-section-title">🧪 정책 시뮬레이션 — 조건이 바뀌면 매출이 어떻게 될까요?</div>', unsafe_allow_html=True)
    st.caption(
        "실제로 개입 가능한 정책 변수를 조정해 예상 매출 변화를 미리 시험해볼 수 있습니다. "
        "특정 행정동 하나의 우연한 값에 휘둘리지 않도록, 전체 데이터에서 해당 조건 변화가 "
        "평균적으로 미치는 영향을 이 지역의 기준 예측치에 반영한 추정치입니다."
    )

    current_bus = int(result["features"]["bus_stop_count"])
    current_floating = result["features"]["floating_population"]

    # 슬라이더는 실측 데이터 범위를 벗어나지 않게 캡을 씌운다 — 관측 범위 밖으로
    # 나가면 부분의존도 곡선이 평평하게(변화 없음으로) 잘려서, "슬라이더를 움직였는데
    # 왜 아무 변화가 없지?"라는 오해를 부를 수 있기 때문이다.
    observed_bus_max = int(feature_table["bus_stop_count"].max())
    observed_fp_min = feature_table["floating_population"].min()
    observed_fp_max = feature_table["floating_population"].max()
    fp_pct_floor = min(0, int((observed_fp_min / current_floating - 1) * 100)) if current_floating > 0 else 0
    fp_pct_ceiling = max(0, int((observed_fp_max / current_floating - 1) * 100)) if current_floating > 0 else 0

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        bus_stop_slider = st.slider(
            "🚌 버스정류장 개수(반경 300m)",
            min_value=0,
            max_value=max(observed_bus_max, current_bus),
            value=current_bus,
            key="sim_bus_stop",
        )
        if current_bus >= observed_bus_max:
            st.caption(f"※ 이미 안양시 관측 최댓값({observed_bus_max}개) 수준이라 늘리는 방향은 시험해볼 수 없습니다.")
    with sim_col2:
        if fp_pct_ceiling > fp_pct_floor:
            floating_pct_slider = st.slider(
                "🚶 유동인구 증감률",
                min_value=fp_pct_floor,
                max_value=fp_pct_ceiling,
                value=0,
                format="%d%%",
                key="sim_floating_pct",
            )
        else:
            floating_pct_slider = 0
            st.write("🚶 유동인구 증감률")
            st.caption("※ 이 지역은 이미 관측된 유동인구 범위의 최댓값·최솟값이라 시험해볼 여지가 없습니다.")

    scenario = simulate_scenario(
        trained,
        selected_dong,
        selected_category,
        {
            "bus_stop_count": bus_stop_slider,
            "floating_population": current_floating * (1 + floating_pct_slider / 100),
        },
    )

    if scenario:
        diff = scenario["predicted_sales"] - scenario["baseline_sales"]
        pct = (diff / scenario["baseline_sales"] * 100) if scenario["baseline_sales"] else 0
        arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "―")
        color = "#17A673" if diff > 0 else ("#c0392b" if diff < 0 else "#6B7684")

        sr1, sr2, sr3 = st.columns(3)
        with sr1:
            st.markdown(
                f"""<div class="anyang-card" style="text-align:center;">
                <div style="font-size:0.8rem; color:#6B7684;">현재 조건 예상 매출</div>
                <div style="font-size:1.3rem; font-weight:800; color:#1B2430;">{scenario['baseline_sales']:,.0f}원</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with sr2:
            st.markdown(
                f"""<div class="anyang-card" style="text-align:center; background:linear-gradient(135deg,#0B5ED7,#17A673); color:white;">
                <div style="font-size:0.8rem; opacity:0.9;">시나리오 적용 시 예상 매출</div>
                <div style="font-size:1.3rem; font-weight:800;">{scenario['predicted_sales']:,.0f}원</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with sr3:
            st.markdown(
                f"""<div class="anyang-card" style="text-align:center;">
                <div style="font-size:0.8rem; color:#6B7684;">변화</div>
                <div style="font-size:1.3rem; font-weight:800; color:{color};">{arrow} {abs(pct):.1f}%</div>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.info("시나리오를 계산할 수 없습니다.")

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
    st.caption("R²는 가상 데이터 기준 참고값입니다. 실제 데이터로 교체 후 재학습하면 유의미한 값을 얻을 수 있습니다.")
