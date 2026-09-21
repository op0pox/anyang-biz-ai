"""프로젝트 공통 설정: 경로, 데이터 파일 매핑, 피처 목록, 모델 하이퍼파라미터."""

import shutil
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_SOURCES_DIR = PROJECT_ROOT / "data" / "sources"
DATA_DEPLOY_DIR = PROJECT_ROOT / "data" / "deploy"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# config.py는 다른 모든 모듈이 가장 먼저 임포트하므로, 여기서 .env를 로드해두면
# OPENAI_API_KEY/GG_OPEN_API_KEY/BUS_API_KEY 등이 프로젝트 어디서든 os.environ으로 보인다.
load_dotenv(PROJECT_ROOT / ".env")


def _ensure_deploy_data() -> None:
    """Streamlit Cloud 등 새로 클론한 환경에서 data/raw·data/sources가 비어 있으면
    (로컬 전용 심볼릭 링크는 .gitignore돼 있어 클론 직후엔 없음) data/deploy/의
    안양시만 필터링한 작은 실데이터로 채운다(scripts/prepare_deploy_data.py 참고).

    로컬 개발 환경처럼 대상 파일이 이미 있으면(심볼릭 링크 등) 아무것도 하지 않는다
    — 전체 데이터를 쓰는 로컬 개발 흐름을 건드리지 않기 위함.
    """
    if not DATA_DEPLOY_DIR.exists():
        return

    # card_sales는 로컬 개발 환경에 card_sales_202601.csv 같은 월별 심볼릭 링크
    # 여러 개로 이미 존재할 수 있어(정확한 이름이 card_sales_agg.csv가 아님),
    # "card_sales*.csv" 패턴으로 이미 있는지부터 따로 확인한다 — 그렇지 않으면
    # 로컬 월별 원본과 배포용 집계본이 동시에 존재해 이중 집계될 수 있다.
    if not list(DATA_RAW_DIR.glob("card_sales*.csv")):
        agg_src = DATA_DEPLOY_DIR / "card_sales_agg.csv"
        if agg_src.exists():
            shutil.copyfile(agg_src, DATA_RAW_DIR / "card_sales_agg.csv")

    # data/raw/ 아래 단일 파일들 — RAW_FILE_STEMS가 정확한 파일명으로 찾으므로
    # 정확한 이름으로 이미 있는지(로컬 심볼릭 링크 등)만 확인하면 충분하다.
    single_file_mapping = {
        DATA_DEPLOY_DIR / "stores_202606.csv": DATA_RAW_DIR / "stores.csv",
        DATA_DEPLOY_DIR / "resident_population.xlsx": DATA_RAW_DIR / "resident_population.xlsx",
        DATA_DEPLOY_DIR / "bus_stops.csv": DATA_RAW_DIR / "bus_stops.csv",
        DATA_DEPLOY_DIR / "floating_population.csv": DATA_RAW_DIR / "floating_population.csv",
    }
    for src, dest in single_file_mapping.items():
        if dest.exists() or not src.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)

    # data/sources/ 하위 폴더들(정비사업·폐업통계 스냅샷·특화지원센터)은 로더가
    # "폴더 안 첫 csv"를 그대로 쓰므로(load_redevelopment_projects 등), 정확한
    # 파일명이 아니라 "그 폴더에 파일이 이미 있는지"로 확인해야 한다 — 그렇지
    # 않으면 로컬 원본(다른 파일명)과 배포용 파일이 같은 폴더에 같이 남아
    # 어느 게 쓰일지 불확실해진다.
    folder_mapping = {
        DATA_DEPLOY_DIR / "stores_202403.csv": DATA_SOURCES_DIR / "stores" / "202403" / "stores_202403.csv",
        DATA_DEPLOY_DIR / "redevelopment.csv": DATA_SOURCES_DIR / "redevelopment" / "redevelopment.csv",
        DATA_DEPLOY_DIR / "support_centers.csv": DATA_SOURCES_DIR / "support_centers" / "support_centers.csv",
    }
    for src, dest in folder_mapping.items():
        if not src.exists():
            continue
        existing = list(dest.parent.glob("*.csv")) + list(dest.parent.glob("*.xlsx")) + list(dest.parent.glob("*.xls"))
        if existing:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)


_ensure_deploy_data()

# ---------------------------------------------------------------------------
# 원본 데이터 파일 stem (확장자 없이) — data/raw/ 아래 이 이름 + .csv/.xlsx/.xls
# ---------------------------------------------------------------------------
RAW_FILE_STEMS = {
    "card_sales": "card_sales",
    "stores": "stores",
    "floating_population": "floating_population",
    "resident_population": "resident_population",
    "bus_stops": "bus_stops",
}

# 각 데이터셋에서 탐색할 컬럼 키워드 (부분 문자열 매칭용).
# 실제 파일의 컬럼명이 다르면 여기만 수정하면 됨.
# 주의: 키워드는 더 구체적인(예: "행정동명") 것을 먼저 나열해야 한다.
# "행정동코드" 컬럼도 "행정동"을 부분 문자열로 포함하므로, "행정동"처럼 뭉뚱그린
# 키워드를 먼저 두면 이름 컬럼 대신 코드 컬럼이 잘못 매칭될 수 있다.
COLUMN_HINTS = {
    "card_sales": {
        "dong": ["행정동명", "읍면동명", "행정동", "읍면동"],
        "category": ["업종대분류", "업종분류", "업종"],
        "amount": ["매출금액", "이용금액", "매출액"],
        "count": ["매출건수", "이용건수", "건수"],
    },
    "stores": {
        "dong": ["행정동명", "법정동명", "행정동", "법정동"],
        "category": ["상권업종대분류명", "업종대분류", "업종"],
        "name": ["상호명", "사업장명"],
        "lat": ["위도"],
        "lon": ["경도"],
    },
    "floating_population": {
        "gu": ["시군명", "구", "시군구"],
        "hour": ["시간대", "시간"],
        "age": ["연령대", "연령"],
        "population": ["유동인구", "인구수", "인구"],
    },
    "resident_population": {
        "dong": ["행정동명", "읍면동명", "행정동", "읍면동"],
        "population": ["인구수", "세대인구", "총인구", "인구"],
    },
    "bus_stops": {
        "lat": ["위도"],
        "lon": ["경도"],
        "name": ["정류소명", "정류장명"],
    },
}

# ---------------------------------------------------------------------------
# 안양시 필터링 및 좌표 기준
# ---------------------------------------------------------------------------
CITY_FILTER_KEYWORD = "안양시"
ANYANG_CENTER_LAT = 37.3943
ANYANG_CENTER_LON = 126.9568

# 상가정보 등 시 전역 데이터를 안양시 반경으로 1차 필터링할 때 쓰는 대략적 반경(km)
ANYANG_BBOX_RADIUS_KM = 12

# 안양시 행정동 목록 (만안구/동안구) — 가상데이터 생성 및 검증에 사용
ANYANG_DONGS = {
    "만안구": ["안양1동", "안양2동", "안양3동", "안양5동", "안양6동", "안양9동", "석수1동", "석수2동", "박달1동", "박달2동"],
    "동안구": ["관양1동", "관양2동", "비산1동", "비산2동", "비산3동", "평촌동", "호계1동", "호계2동", "평안동", "귀인동", "부흥동", "달안동", "지구촌동", "범계동"],
}

# 업종 대분류 — 경기데이터드림 카드소비 데이터(card_tpbuz_nm_1)의 9개 분류를
# 시스템 전체의 "정답" 업종 체계로 삼는다 (가상 데이터도 이 체계로 생성해야
# 아래 STORE_TO_CARD_CATEGORY_MAP과 충돌 없이 합쳐진다).
BUSINESS_CATEGORIES = [
    "음식", "소매/유통", "생활서비스", "여가/오락", "학문/교육",
    "의료/건강", "공연/전시", "미디어/통신", "공공/기업/단체",
]

# 카드소비 데이터(card_tpbuz_nm_1)를 업종 대분류의 "정답" 체계로 삼는다.
# 소상공인시장진흥공단 상가정보는 분류 체계(상권업종대분류명)가 달라
# 그대로는 카드소비 데이터와 같은 업종으로 조인되지 않으므로, 아래 매핑으로
# 상가정보 업종명을 카드소비 업종명으로 정렬한다(경쟁점포수 피처 계산용).
# 두 체계가 1:1로 대응하지 않아 다대일 매핑이며, 근사적인 대응임을 감안할 것.
STORE_TO_CARD_CATEGORY_MAP = {
    "음식": "음식",
    "소매": "소매/유통",
    "숙박": "여가/오락",
    "교육": "학문/교육",
    "수리·개인": "생활서비스",
    "과학·기술": "미디어/통신",
    "보건의료": "의료/건강",
    "예술·스포츠": "여가/오락",
    "시설관리·임대": "공공/기업/단체",
    "부동산": "공공/기업/단체",
}

# 유동인구 시간대 가중치 — 상권 매출과 상관이 높은 낮/저녁 시간대에 가중
FLOATING_POPULATION_HOUR_WEIGHTS = {
    range(0, 6): 0.3,
    range(6, 11): 0.8,
    range(11, 14): 1.3,
    range(14, 18): 1.0,
    range(18, 22): 1.3,
    range(22, 24): 0.6,
}

# 버스정류장 접근성 피처를 계산할 반경 (미터)
BUS_STOP_RADIUS_M = 300

# ---------------------------------------------------------------------------
# 경기데이터드림 OPEN API (유동인구)
# ---------------------------------------------------------------------------
GG_OPEN_API_BASE_URL = "https://openapi.gg.go.kr"
GG_OPEN_API_SERVICE_NAME = "TBDASANALSGALLERYT214117"  # 유동인구_안양시 실제 서비스명(확인됨)
GG_OPEN_API_PAGE_SIZE = 1000
# 이 데이터셋은 시간대x성별x5세연령대x내/외국인 단위로 누적된 다년간의 일별 통계라
# 전체가 21만 건이 넘는다. 우리는 절대치가 아니라 만안구/동안구의 상대적 유동인구
# 수준만 필요하므로, 전체를 다 받는 대신 대표 표본만 가져온다.
GG_OPEN_API_MAX_PAGES = 20

# ---------------------------------------------------------------------------
# 국토교통부 TAGO 버스정류소정보조회 서비스 (공공데이터포털 Open API)
# 활용가이드: 오픈API활용가이드_국토교통부(TAGO)_버스정류소정보v1.0
# ---------------------------------------------------------------------------
BUS_API_BASE_URL = "https://apis.data.go.kr/1613000/BusSttnInfoInqireService"
BUS_API_PAGE_SIZE = 100
CITY_NAME = "안양시"  # getCtyCodeList 응답의 cityname과 매칭할 도시명

# ---------------------------------------------------------------------------
# 기업마당(bizinfo.go.kr) 지원사업정보 API + K-Startup 통합공고 API
# 둘 다 실제 키로 호출해 엔드포인트/파라미터/응답 스키마를 확인 완료(2026-09-21).
# ---------------------------------------------------------------------------
BIZINFO_API_BASE_URL = "https://www.bizinfo.go.kr/uss/rss/bizinfoApi.do"
BIZINFO_SEARCH_COUNT = 100
# hashtags 파라미터는 서버 측 키워드 검색 — "안양"만 넣어도 "안양시"가 포함된
# 공고가 그대로 걸린다(실제 호출로 확인됨). 전국 3만여 건을 다 받을 필요 없이
# 안양시 관련 공고만 바로 받을 수 있어 별도 클라이언트 필터링이 필요 없다.
BIZINFO_REGION_HASHTAG = "안양"

KSTARTUP_API_BASE_URL = "https://apis.data.go.kr/B552735/kisedKstartupService01"
KSTARTUP_ANNOUNCEMENT_OPERATION = "getAnnouncementInformation01"
KSTARTUP_PAGE_SIZE = 100
# K-Startup은 기업마당과 달리 지역 검색 파라미터를 찾지 못해(2026-09-21 기준) 최근
# 공고 몇 페이지를 받아 클라이언트에서 "안양 관련" 또는 "전국 대상"만 걸러낸다.
# 전체 3만여 건이라 페이지 수를 제한한다(경기데이터드림 유동인구와 같은 이유).
KSTARTUP_MAX_PAGES = 3

# ---------------------------------------------------------------------------
# 모델 하이퍼파라미터
# ---------------------------------------------------------------------------
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 예상 범위(predicted_low/predicted_high, model.py의 _predicted_range) 산출 전용
# 모델 하이퍼파라미터. 표본이 279개(31개 행정동x9개 업종)로 작아 min_samples_leaf=2인
# 기본 모델은 트리별 예측 분산이 표본 크기와 상관없이 요동쳐(실측: 최대 50배 차이)
# 그대로 신뢰구간처럼 보여주면 오해를 부른다. min_samples_leaf를 높여 리프가 더 많은
# 표본을 평균 내도록 해서 더 안정적인 예상 범위를 만든다. 기본 예측/랭킹에는
# 영향 없음(별도 모델). 2026-09-21 이전엔 정책 시뮬레이션(what-if)에도 같이 썼으나
# 그 기능은 제거됨 — 지금은 예상 범위 산출이 유일한 용도.
SIMULATION_MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 10,
    "random_state": 42,
    "n_jobs": -1,
}

FEATURE_COLUMNS = [
    "competitor_count",
    "floating_population",
    "resident_population",
    "bus_stop_count",
]
TARGET_COLUMN = "sales_amount"

# ---------------------------------------------------------------------------
# 소상공인 지원 우선순위 스코어 가중치
# ---------------------------------------------------------------------------
# 근거: 안양시 소상공인 지원 예산은 "매출이 부진한 곳을 우선 지원한다"는 목적이
# 가장 크므로 매출 수준에 최고 가중치(45%)를 둔다. 경쟁 강도(30%)는 매출이
# 비슷해도 점포당 배분 가능한 파이가 작아지는 곳일수록 지원 체감 효과가 크다고
# 보아 두 번째로 두고, 매출 효율(25%)은 유동인구 대비 매출이 낮다는 건 상권
# 자체의 구조적 약점(접근성·인지도 등)을 시사하므로 보조 지표로 반영한다.
PRIORITY_WEIGHT_SALES = 0.45
PRIORITY_WEIGHT_COMPETITION = 0.30
PRIORITY_WEIGHT_EFFICIENCY = 0.25

# ---------------------------------------------------------------------------
# 매출 예측 신뢰도 배지 — 경쟁점포수(해당 행정동x업종 조합에서 관측된 점포 수)를
# 표본 크기의 대리지표로 삼아 예측 신뢰도를 매긴다. RandomForest 트리별 예측
# 분산(부분의존도 계산에 쓴 것과 같은 방식)도 검토했으나, 매출액 분포가 워낙
# 치우쳐 있어(log1p로도 완전히 못 잡음) 트리 간 분산이 표본 크기와 상관관계가
# 약하고(상관계수 -0.1~-0.3) 이상치 행에서 수백~수천 배로 튀어 사용자에게 그대로
# 보여주면 오히려 신뢰를 떨어뜨린다고 판단해 채택하지 않았다. 대신 "관측된
# 점포 수가 몇 개인가"라는 직접 설명 가능한 기준을 쓴다. 컷은 실측 분포의
# 하위/상위 25% 분위수(약 25개/120개)를 참고해 정함.
# ---------------------------------------------------------------------------
CONFIDENCE_LOW_MAX_COMPETITORS = 25
CONFIDENCE_HIGH_MIN_COMPETITORS = 120

# ---------------------------------------------------------------------------
# 3단계 안전망 매칭 — 지원우선순위 스코어(0~100)의 절대 수준을 3구간으로 나눠
# 실제 지원제도와 매칭한다. recommend_support_type()의 "지배 요인"(왜 필요한지)과는
# 다른 축이다 — 이쪽은 "얼마나 급한지"에 따라 실제 제도명을 매칭한다.
# 구간 컷(33/66)은 기존 코드 곳곳(ai_report.py, UI)에서 이미 쓰던 낮음/중간/높음
# 3단계 분류와 동일하게 맞췄다.
# ---------------------------------------------------------------------------
SAFETY_NET_PROGRAMS = {
    "예방": {
        "range": "0~33점",
        "programs": [
            {"name": "안양시 소상공인 특례보증", "description": "저금리 보증부 대출로 자금 여력 확보"},
            {"name": "안양상권활성화센터 컨설팅", "description": "경영·마케팅 무료 컨설팅으로 사전 리스크 관리"},
        ],
    },
    "긴급수혈": {
        "range": "33~66점",
        "programs": [
            {"name": "중소기업 육성자금", "description": "운영자금 융자로 매출 하락기 버티기 지원"},
            {"name": "안양사랑페이 가맹 프로모션 연계", "description": "지역화폐 프로모션으로 단기 매출 견인"},
        ],
    },
    "재기지원": {
        "range": "66~100점",
        "programs": [
            {"name": "희망리턴패키지", "description": "점포철거비·재창업 컨설팅 등 폐업·재기 국비 지원사업"},
        ],
    },
}

# 정비사업 인접 리스크 반경(미터) — 버스정류장(300m)보다는 넓게 잡되(재건축·리모델링은
# 개별 점포 접근성보다 넓은 범위의 유동인구 구조에 영향을 미치므로), 너무 넓으면
# (예: 800m는 31개 행정동 중 18개, 1000m는 23개가 걸려 "다 위험하다"는 식으로 신호가
# 희석된다) 실측으로 확인 후 500m로 정함 — 11/31개 행정동만 해당돼 선별적인 신호가 된다.
REDEVELOPMENT_RADIUS_M = 500
# 이미 완료되어 더 이상 이주 리스크가 없는 사업 단계/현황 (정비사업 데이터 실제 값 기준)
REDEVELOPMENT_COMPLETED_STAGES = {"준공"}
REDEVELOPMENT_COMPLETED_STATUSES = {"이전고시"}

# ---------------------------------------------------------------------------
# LLM 리포트
# ---------------------------------------------------------------------------
OPENAI_MODEL = "gpt-4o-mini"
