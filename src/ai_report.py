"""LLM 기반 AI 정책분석 리포트 생성 (+ OPENAI_API_KEY 없을 때 오프라인 규칙 기반 폴백).

리포트를 하나의 긴 문단이 아니라 [진단]/[근거 해석]/[정책적 시사점] 3개 섹션의
짧은 불릿 리스트로 구성된 dict로 반환한다. 경쟁점포수 같은 실제 수치는 이 모듈이
문장으로 다시 서술하지 않고 UI가 prediction 데이터로 직접 표로 렌더링한다 —
숫자를 LLM이 다시 말하게 하면서 생기는 환각 여지(오타·반올림 등)를 원천 차단하기 위함.
"""

import json
import os
from typing import Optional

SYSTEM_PROMPT = """\
당신은 안양시 소상공인정책과 소속 AI 정책분석관입니다.
행정동×업종 매출 예측 모델의 결과를 근거로, 시청 정책담당자가 소상공인 지원 예산·정책의
우선순위를 판단할 수 있도록 리포트를 작성합니다. 동시에 해당 지역에서 실제 창업을 고려하는
소상공인에게도 실질적으로 도움이 되는 조언을 포함합니다.

반드시 지켜야 할 규칙:
- 제공된 JSON 수치 이외의 사실을 지어내지 마세요 (환각 금지). 수치를 문장에 다시 옮겨
  적지 말고(이미 화면에 표로 표시됨), 그 수치가 "의미하는 바"만 해석하세요.
- 각 문장은 한 줄로 스캔 가능한 짧고 자연스러운 완결된 문장(60자 내외, "~습니다"체)으로
  작성하세요. 명사만 나열한 제목처럼 딱딱하게 끊어 쓰지 말고, 문단으로도 쓰지 마세요.
- 반드시 아래 JSON 형식으로만 답하세요 (다른 설명 텍스트 없이 JSON 객체 하나만):
  {
    "diagnosis": ["문장1", "문장2"],
    "evidence_commentary": ["문장1", "문장2"],
    "recommendations": ["문장1", "문장2", "문장3"]
  }
- diagnosis: 이 지역/업종 상황에 대한 핵심 진단 2~3개
- evidence_commentary: 근거 수치가 시사하는 바에 대한 해석 2~3개
- recommendations: 정책적 시사점/추천 조치 2~4개 (행정 지원 관점 + 창업자 관점 섞어서)
- JSON에 "정비사업_인접_리스크" 또는 "최근_폐업_후보_비율" 필드가 있으면, 그 의미를
  diagnosis나 evidence_commentary 중 한 곳에 반드시 반영하세요(안양시가 이 지역에서만
  직접 확인한 특화 신호이므로 누락하지 마세요). 두 필드가 없으면 언급하지 마세요.
"""


def _build_payload(
    prediction: dict,
    priority_row: Optional[dict] = None,
    redevelopment_risk: Optional[dict] = None,
    closure_row=None,
) -> dict:
    payload = {
        "행정동": prediction["dong"],
        "업종": prediction["category"],
        "예상_매출액": round(prediction["predicted_sales"]),
        "실측_매출액_참고": round(prediction.get("actual_sales", 0)),
        "주요_피처": {
            "경쟁점포수": prediction["features"].get("competitor_count"),
            "유동인구_가중합": round(prediction["features"].get("floating_population", 0)),
            "거주인구": prediction["features"].get("resident_population"),
            "버스정류장_접근성": prediction["features"].get("bus_stop_count"),
        },
    }
    if priority_row:
        payload["지원_우선순위"] = {
            "우선순위_스코어_0to100": round(priority_row.get("priority_score", 0), 1),
            "행정동_평균_예상매출": round(priority_row.get("avg_predicted_sales", 0)),
            "행정동_전체_경쟁점포수": priority_row.get("total_competitor_count"),
        }
    if redevelopment_risk:
        payload["정비사업_인접_리스크"] = {
            "반경_500m_내_사업수": redevelopment_risk.get("count"),
            "영향_세대수": redevelopment_risk.get("total_households"),
        }
    if closure_row is not None:
        payload["최근_폐업_후보_비율_퍼센트"] = round(float(closure_row["closure_rate"]) * 100, 1)
    return payload


def _offline_fallback_report(payload: dict) -> dict:
    """OPENAI_API_KEY가 없거나 호출에 실패했을 때 사용하는 규칙 기반 리포트."""
    dong = payload["행정동"]
    category = payload["업종"]
    sales = payload["예상_매출액"]
    features = payload["주요_피처"]

    competitor = features.get("경쟁점포수", 0) or 0
    floating = features.get("유동인구_가중합", 0) or 0
    bus = features.get("버스정류장_접근성", 0) or 0
    priority = payload.get("지원_우선순위")
    redevelopment = payload.get("정비사업_인접_리스크")
    closure_pct = payload.get("최근_폐업_후보_비율_퍼센트")

    diagnosis = [f"{dong} '{category}' 예상 매출액은 약 {sales:,.0f}원입니다."]
    if priority:
        score = priority["우선순위_스코어_0to100"]
        level = "높음" if score >= 66 else ("중간" if score >= 33 else "낮음")
        diagnosis.append(f"소상공인 지원 우선순위 {score}점으로 {level} 구간입니다.")
    else:
        diagnosis.append("행정동 단위 우선순위 비교 데이터는 제공되지 않았습니다.")
    if redevelopment:
        diagnosis.append(
            f"반경 500m 내 정비사업 {redevelopment['반경_500m_내_사업수']}건(총 "
            f"{redevelopment['영향_세대수']:,}세대)이 진행 중이라 향후 유동인구 변화 가능성이 있습니다."
        )

    evidence_commentary = []
    if competitor >= 10:
        evidence_commentary.append("경쟁점포수가 많아 시장 포화도가 높은 편입니다.")
    else:
        evidence_commentary.append("경쟁점포수가 적어 신규 진입 여지가 있는 편입니다.")
    if floating > 0:
        evidence_commentary.append(
            "유동인구 대비 대중교통 접근성이 낮은 편입니다." if bus < 3 else "유동인구와 대중교통 접근성이 모두 양호한 편입니다."
        )
    if closure_pct is not None:
        if closure_pct >= 20:
            evidence_commentary.append(f"최근 폐업 후보 비율이 {closure_pct}%로 높은 편이라 상권 위축 신호로 볼 수 있습니다.")
        else:
            evidence_commentary.append(f"최근 폐업 후보 비율은 {closure_pct}%로 비교적 안정적인 편입니다.")

    recommendations = []
    if competitor >= 10:
        recommendations.append("신규 진입보다 차별화 전략·기존 점포 경쟁력 강화 지원을 검토하세요.")
    else:
        recommendations.append("신규 창업 유인책·초기 정착 지원 프로그램을 검토하세요.")
    if bus < 3:
        recommendations.append("대중교통 접근성 개선(버스 노선 확충 등)이 매출에 도움될 수 있습니다.")
    if priority and priority["우선순위_스코어_0to100"] >= 66:
        recommendations.append("임대료 지원·컨설팅 등 예산 배정을 우선 검토하세요.")
    if redevelopment:
        recommendations.append("정비사업 이주 시점을 모니터링해 유동인구 급변에 선제 대응하세요.")
    if closure_pct is not None and closure_pct >= 20:
        recommendations.append("폐업 위험이 높은 상권이므로 조기 경보 차원의 컨설팅 개입을 검토하세요.")
    if not recommendations:
        recommendations.append("현재 데이터 기준으로는 특별한 우선 조치가 필요하지 않습니다.")

    return {
        "diagnosis": diagnosis,
        "evidence_commentary": evidence_commentary,
        "recommendations": recommendations,
        "source": "offline",
    }


def _validate_llm_report(data: dict) -> dict:
    def _as_str_list(value, limit=5):
        if not isinstance(value, list):
            return []
        return [str(v).strip() for v in value if str(v).strip()][:limit]

    diagnosis = _as_str_list(data.get("diagnosis"))
    evidence_commentary = _as_str_list(data.get("evidence_commentary"))
    recommendations = _as_str_list(data.get("recommendations"))
    if not diagnosis or not recommendations:
        raise ValueError("LLM 응답에 필수 섹션(diagnosis/recommendations)이 비어 있습니다.")
    return {
        "diagnosis": diagnosis,
        "evidence_commentary": evidence_commentary,
        "recommendations": recommendations,
        "source": "openai",
    }


def generate_report(
    prediction: dict,
    priority_row: Optional[dict] = None,
    redevelopment_risk: Optional[dict] = None,
    closure_row=None,
) -> dict:
    """예측 결과를 바탕으로 AI 정책분석 리포트를 생성한다.

    반환값은 {"diagnosis": [...], "evidence_commentary": [...], "recommendations": [...],
    "source": "openai"|"offline", "notice": (선택, 실패 사유)} 형태의 dict.
    OPENAI_API_KEY가 있으면 LLM 호출, 없거나 실패하면 오프라인 규칙 기반 리포트로 폴백한다.
    redevelopment_risk/closure_row는 안양시 특화 신호(정비사업 인접 리스크, 폐업 흐름
    통계)로, 있을 때만 진단/근거 해석에 반영된다(TODO였던 "AI 리포트 구체성 강화").
    """
    payload = _build_payload(prediction, priority_row, redevelopment_risk, closure_row)
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return _offline_fallback_report(payload)

    try:
        from openai import OpenAI

        from src.config import OPENAI_MODEL

        client = OpenAI(api_key=api_key)
        user_content = (
            "아래는 행정동×업종 매출 예측 모델의 결과 JSON입니다. "
            "이 데이터만 근거로 리포트를 작성하세요.\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM 응답이 비어 있습니다.")
        data = json.loads(content)
        return _validate_llm_report(data)
    except Exception as exc:  # noqa: BLE001
        fallback = _offline_fallback_report(payload)
        fallback["notice"] = f"OpenAI API 호출에 실패해 오프라인 리포트로 대체했습니다 ({exc})"
        return fallback
