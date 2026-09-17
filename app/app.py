"""홈(랜딩) 페이지 — 서비스 소개 + 차별화 포인트.

실행: streamlit run app/app.py
"""

import streamlit as st

from common import SERVICE_NAME, inject_css, load_pipeline, render_topbar, show_pipeline_error

st.set_page_config(page_title=SERVICE_NAME, page_icon="🧭", layout="wide")
inject_css()
render_topbar()

st.markdown(
    """
    <div class="anyang-hero">
        <div class="anyang-badge">2026 안양시 공공데이터·AI 활용 대학생 경진대회</div>
        <h1>안양시 소상공인, 데이터로 더 정확하게 지원합니다</h1>
        <p class="subtitle">
            행정동 x 업종 매출 예측 AI로 안양시 소상공인 지원 예산과 정책의 우선순위를 판단하고,
            지역에 유리한 업종을 추천하며, AI가 그 이유를 사람이 읽는 문장으로 설명합니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="anyang-section-title">기존 상권분석 서비스와 무엇이 다른가요?</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        <div class="anyang-card">
            <div class="icon">🏙️</div>
            <h3>① 안양시 전용 데이터 융합</h3>
            <p>안양시가 직접 발행한 주민등록인구 통계 등 안양시 공공데이터를 명시적으로 결합해,
            전국 범용 서비스와 달리 안양시 행정에 바로 활용할 수 있도록 설계했습니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
        <div class="anyang-card">
            <div class="icon">🎯</div>
            <h3>② 업종 역(逆)추천</h3>
            <p>소상공인365·오픈업·나이스비즈맵은 사용자가 업종을 직접 골라야 합니다.
            이 서비스는 지역 데이터를 기반으로 그 동네에 유리한 업종을 먼저 추천합니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """
        <div class="anyang-card">
            <div class="icon">🤖</div>
            <h3>③ AI 정책 리포트 자동생성</h3>
            <p>기존 서비스는 숫자·그래프만 제공합니다. 이 서비스는 AI가 "왜 이 지역/업종의
            우선순위가 높은지"를 정책 담당자가 읽을 수 있는 문장으로 설명합니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="anyang-section-title">기존 서비스 대비 비교</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="anyang-compare">
    <table style="width:100%; border-collapse:collapse;">
        <tr style="border-bottom:1px solid #E3E9F1; text-align:left;">
            <th style="padding:0.5rem;">항목</th>
            <th style="padding:0.5rem;">소상공인365 / 오픈업 / 나이스비즈맵</th>
            <th style="padding:0.5rem; color:#0B5ED7;">안양 상권 나침반</th>
        </tr>
        <tr style="border-bottom:1px solid #E3E9F1;">
            <td style="padding:0.5rem;">타깃</td>
            <td style="padding:0.5rem;">개인 창업자</td>
            <td style="padding:0.5rem;">안양시 정책담당자 + 창업자</td>
        </tr>
        <tr style="border-bottom:1px solid #E3E9F1;">
            <td style="padding:0.5rem;">업종 선택</td>
            <td style="padding:0.5rem;">사용자가 직접 선택</td>
            <td style="padding:0.5rem;">지역 기반 유리 업종 추천</td>
        </tr>
        <tr style="border-bottom:1px solid #E3E9F1;">
            <td style="padding:0.5rem;">결과물</td>
            <td style="padding:0.5rem;">숫자·그래프</td>
            <td style="padding:0.5rem;">숫자·그래프 + AI 서술형 정책 리포트</td>
        </tr>
        <tr>
            <td style="padding:0.5rem;">지역 데이터</td>
            <td style="padding:0.5rem;">전국 범용</td>
            <td style="padding:0.5rem;">안양시 발행 데이터 명시적 융합</td>
        </tr>
    </table>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="anyang-section-title">지금 바로 시작해보세요</div>', unsafe_allow_html=True)

feature_table, trained, error = load_pipeline()

if error:
    show_pipeline_error(error)
else:
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""<div class="anyang-metric-box"><div class="value">{feature_table['dong'].nunique()}</div>
            <div class="label">분석 대상 행정동</div></div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""<div class="anyang-metric-box"><div class="value">{feature_table['category'].nunique()}</div>
            <div class="label">분석 대상 업종</div></div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""<div class="anyang-metric-box"><div class="value">{len(feature_table)}</div>
            <div class="label">행정동x업종 조합</div></div>""",
            unsafe_allow_html=True,
        )

st.markdown('<div class="anyang-section-title">당신은 누구신가요?</div>', unsafe_allow_html=True)
st.caption("데이터와 모델은 동일하지만, 필요한 화면과 정보는 다릅니다 — 맞는 쪽으로 바로 이동하세요.")

p1, p2 = st.columns(2)
with p1:
    st.markdown(
        """
        <div class="anyang-card" style="text-align:center;">
            <div class="icon" style="font-size:2rem;">🏪</div>
            <h3>소상공인 사장님</h3>
            <p>매출이 흔들릴 때, 우리 동네·업종의 예상 매출과 지금 받을 수 있는
            지원제도를 바로 확인하고 싶으신가요?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_매출예측_업종추천.py", label="📈 매출예측·업종추천 보러가기", width="stretch")
with p2:
    st.markdown(
        """
        <div class="anyang-card" style="text-align:center;">
            <div class="icon" style="font-size:2rem;">🏛️</div>
            <h3>안양시 정책담당자</h3>
            <p>관내 소상공인 중 위험군을 조기에 파악해 예산·상담 인력을
            어디에 우선 배정할지 판단하고 싶으신가요?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/2_지원우선순위.py", label="🗺️ 지원 우선순위 보러가기", width="stretch")

st.write("")
with st.expander("또는 원하는 페이지로 바로 이동"):
    b1, b2, b3 = st.columns(3)
    with b1:
        st.page_link("pages/1_매출예측_업종추천.py", label="📈 매출예측·업종추천", width="stretch")
    with b2:
        st.page_link("pages/2_지원우선순위.py", label="🗺️ 지원 우선순위", width="stretch")
    with b3:
        st.page_link("pages/3_AI리포트.py", label="🤖 AI 정책 리포트", width="stretch")
