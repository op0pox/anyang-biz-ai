"""AI 정책분석 리포트 생성 페이지."""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from common import SERVICE_NAME, inject_css, load_pipeline, render_topbar, show_pipeline_error  # noqa: E402
from src.ai_report import generate_report  # noqa: E402
from src.model import compute_support_priority, predict_sales  # noqa: E402

st.set_page_config(page_title=f"AI 리포트 | {SERVICE_NAME}", page_icon="🤖", layout="wide")
inject_css()
render_topbar()

st.markdown('<div class="anyang-section-title">🤖 AI 정책분석 리포트</div>', unsafe_allow_html=True)
st.caption(
    "기존 상권분석 서비스는 숫자·그래프만 제공합니다. "
    "이 기능은 AI가 예측 근거를 바탕으로 '왜 우선순위가 높은지'를 문장으로 설명해드립니다."
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
        report_text = generate_report(prediction, priority_row)
    except Exception as exc:  # noqa: BLE001
        report_text = f"리포트 생성 중 오류가 발생했습니다: {exc}"

st.markdown(
    f"""
    <div class="anyang-card">
        <h3>{selected_dong} · {selected_category} 정책분석 리포트</h3>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write(report_text)

with st.expander("🔍 리포트에 사용된 원본 예측 데이터"):
    st.json(
        {
            "예상_매출": prediction["predicted_sales"],
            "주요_피처": prediction["features"],
            "우선순위_정보": priority_row,
        }
    )
