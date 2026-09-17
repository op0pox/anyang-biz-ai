"""LLM 기반 AI 정책분석 리포트 생성 (+ OPENAI_API_KEY 없을 때 오프라인 규칙 기반 폴백).

모델 예측 결과를 JSON으로 만들어 LLM에 그대로 주입해 환각을 방지하고,
수치에 근거한 리포트를 생성하도록 유도한다.
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
- 제공된 JSON 수치 이외의 사실을 지어내지 마세요 (환각 금지).
- 어조는 객관적이고 행정 보고서에 어울리게 작성하되, 너무 딱딱하지 않게 씁니다.
- 아래 3개 섹션 구조를 반드시 지키세요: [진단] - [예측 근거] - [정책적 시사점/추천]
- 소상공인365, 오픈업 같은 기존 서비스는 숫자·그래프만 제공한다는 점을 의식하고,
  이 리포트는 "왜 그런지"를 설명하는 서술형 해석이라는 차별점을 살려 작성하세요.
"""


def _build_payload(prediction: dict, priority_row: Optional[dict] = None) -> dict:
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
    return payload


def _offline_fallback_report(payload: dict) -> str:
    """OPENAI_API_KEY가 없을 때 사용하는 규칙 기반 리포트."""
    dong = payload["행정동"]
    category = payload["업종"]
    sales = payload["예상_매출액"]
    features = payload["주요_피처"]

    competitor = features.get("경쟁점포수", 0)
    floating = features.get("유동인구_가중합", 0)
    resident = features.get("거주인구", 0)
    bus = features.get("버스정류장_접근성", 0)

    priority = payload.get("지원_우선순위")

    diagnosis_lines = [f"{dong}의 '{category}' 업종 예상 매출액은 약 {sales:,.0f}원으로 추정됩니다."]
    if priority:
        score = priority["우선순위_스코어_0to100"]
        level = "높음" if score >= 66 else ("중간" if score >= 33 else "낮음")
        diagnosis_lines.append(
            f"해당 행정동의 소상공인 지원 우선순위 스코어는 {score}점(100점 만점 중, 지원 필요도 {level} 구간)입니다."
        )

    evidence_lines = [
        f"- 동일 행정동 내 같은 업종 경쟁점포수: {competitor:.0f}개 (경쟁 강도 판단 근거)",
        f"- 시간대 가중 적용 유동인구 합계: {floating:,.0f} (구 단위 데이터를 행정동에 배분한 값)",
        f"- 거주인구: {resident:,.0f}명 (배후 수요 규모)",
        f"- 반경 300m 내 버스정류장 수: {bus:.0f}개 (대중교통 접근성)",
    ]

    recommendation_lines = []
    if competitor >= 10:
        recommendation_lines.append("경쟁점포수가 많은 편으로, 신규 진입보다는 차별화 전략이나 기존 점포 대상 경쟁력 강화 지원이 유효할 수 있습니다.")
    else:
        recommendation_lines.append("경쟁점포수가 상대적으로 적어, 신규 창업 유인책이나 초기 정착 지원 프로그램을 고려할 수 있습니다.")

    if floating and floating > 0:
        recommendation_lines.append("유동인구 수준을 고려할 때 마케팅·홍보보다는 상권 접근성(버스, 주차 등) 개선이 매출에 더 직접적인 영향을 줄 수 있습니다." if bus < 3 else "유동인구와 대중교통 접근성이 양호해 상권 자체의 매력도는 준수한 편입니다.")

    if priority and priority["우선순위_스코어_0to100"] >= 66:
        recommendation_lines.append("해당 행정동은 지원 우선순위가 높게 평가되므로, 임대료 지원·컨설팅 등 소상공인 지원 예산 배정을 우선 검토할 것을 권고합니다.")

    report = (
        "[진단]\n" + " ".join(diagnosis_lines) + "\n\n"
        "[예측 근거]\n" + "\n".join(evidence_lines) + "\n\n"
        "[정책적 시사점/추천]\n" + " ".join(recommendation_lines) + "\n\n"
        "(참고: OPENAI_API_KEY가 설정되지 않아 오프라인 규칙 기반으로 생성된 리포트입니다.)"
    )
    return report


def generate_report(prediction: dict, priority_row: Optional[dict] = None) -> str:
    """예측 결과를 바탕으로 AI 정책분석 리포트를 생성한다.

    OPENAI_API_KEY가 있으면 LLM 호출, 없거나 실패하면 오프라인 규칙 기반 리포트로 폴백한다.
    """
    payload = _build_payload(prediction, priority_row)
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
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM 응답이 비어 있습니다.")
        return content
    except Exception as exc:  # noqa: BLE001
        fallback = _offline_fallback_report(payload)
        return (
            f"(안내: OpenAI API 호출에 실패해 오프라인 리포트로 대체합니다. 사유: {exc})\n\n" + fallback
        )
