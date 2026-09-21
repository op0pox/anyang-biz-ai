"""배포용(Streamlit Community Cloud)으로 안양시분만 남긴 작은 데이터 파일을 만든다.

원본 data/sources/의 상가정보 2개 분기(경기도 전체, 각 300~350MB)와 카드소비
3개월치(각 90~110MB)는 GitHub 파일당 100MB 제한을 넘고, Streamlit Cloud 무료
티어(RAM 1GB)에서 그대로 읽으면 메모리 초과로 죽을 가능성이 크다.

이 스크립트는 딱 한 번 로컬(실데이터가 이미 있는 이 환경)에서 실행해서:
- 상가정보 2개 분기: 시군구명에 "안양시"가 들어간 행만 남긴다(전체 컬럼 유지,
  스키마는 그대로라 data_loader.py 코드 수정 없이 그대로 읽힌다). 실측 비율
  약 4.2%라 350MB -> 10~15MB 수준으로 줄어든다.
- 카드소비 3개월치: data_loader._load_card_sales_one_file()을 그대로 재사용해
  (dong, category, sales_amount, sales_count)로 이미 집계된 결과만 남긴다
  (원본은 일별x시간대x성별x연령대까지 쪼개진 130만 행짜리라 집계 전엔 못 줄임).
  이 출력 스키마를 data_loader.py가 감지해서 그대로 읽도록 처리해뒀다.
- 나머지(주민등록인구, 정비사업, 소공인특화지원센터, 유동인구/버스정류장 캐시)는
  이미 작아서 그대로 복사만 한다.

실행: ./venv-anyang/bin/python scripts/prepare_deploy_data.py
출력: data/deploy/ (git 추적 대상 — Streamlit Cloud 배포 시 이 폴더에서 부트스트랩됨,
      src/config.py의 ensure_deploy_data() 참고)
"""

import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CITY_FILTER_KEYWORD, DATA_RAW_DIR, DATA_SOURCES_DIR  # noqa: E402
from src.data_loader import _load_card_sales_one_file  # noqa: E402

DEPLOY_DIR = PROJECT_ROOT / "data" / "deploy"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)


def _filter_stores_file(src_path: Path, out_path: Path) -> None:
    print(f"[상가정보] {src_path.name} 필터링 중...")
    chunks = []
    total = 0
    kept = 0
    for chunk in pd.read_csv(src_path, chunksize=200_000, encoding="utf-8"):
        total += len(chunk)
        mask = chunk["시군구명"].astype(str).str.contains(CITY_FILTER_KEYWORD, na=False)
        kept += int(mask.sum())
        if mask.any():
            chunks.append(chunk[mask])
    out = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  {total:,}행 -> {kept:,}행, {out_path.stat().st_size / 1_048_576:.1f}MB -> {out_path}")


def prepare_stores():
    mapping = {
        DATA_SOURCES_DIR / "stores" / "202606" / "소상공인시장진흥공단_상가(상권)정보_경기_202606.csv": DEPLOY_DIR / "stores_202606.csv",
        DATA_SOURCES_DIR / "stores" / "202403" / "소상공인시장진흥공단_상가(상권)정보_경기_202403.csv": DEPLOY_DIR / "stores_202403.csv",
    }
    for src, out in mapping.items():
        if not src.exists():
            print(f"[안내] {src} 없음, 건너뜀")
            continue
        _filter_stores_file(src, out)


def prepare_card_sales():
    print("[카드소비] 3개월치 집계 중 (기존 data_loader 로직 재사용)...")
    src_dir = DATA_SOURCES_DIR / "card_sales"
    paths = sorted(src_dir.glob("*.csv"))
    if not paths:
        print("[안내] card_sales 원본 없음, 건너뜀")
        return
    parts = [_load_card_sales_one_file(p) for p in paths]
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.groupby(["dong", "category"], as_index=False).agg(
        sales_amount=("sales_amount", "sum"), sales_count=("sales_count", "sum")
    )
    out_path = DEPLOY_DIR / "card_sales_agg.csv"
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  {len(combined)}행(행정동x업종 집계) -> {out_path.stat().st_size / 1024:.1f}KB -> {out_path}")


def copy_small_files():
    mapping = {
        DATA_SOURCES_DIR / "resident_population" / "경기도 안양시 주민등록인구 통계 정보.xlsx": DEPLOY_DIR / "resident_population.xlsx",
        DATA_SOURCES_DIR / "redevelopment" / "경기도 안양시_일반 정비사업 추진현황_20250430.csv": DEPLOY_DIR / "redevelopment.csv",
        DATA_SOURCES_DIR / "support_centers" / "소상공인시장진흥공단_전국 소공인 특화지원센터 현황(20260731).csv": DEPLOY_DIR / "support_centers.csv",
        DATA_RAW_DIR / "floating_population.csv": DEPLOY_DIR / "floating_population.csv",
        DATA_RAW_DIR / "bus_stops.csv": DEPLOY_DIR / "bus_stops.csv",
    }
    for src, out in mapping.items():
        if not src.exists():
            print(f"[안내] {src} 없음, 건너뜀")
            continue
        shutil.copyfile(src, out)
        print(f"[복사] {src.name} -> {out} ({out.stat().st_size / 1024:.1f}KB)")


if __name__ == "__main__":
    prepare_stores()
    prepare_card_sales()
    copy_small_files()
    print()
    print("완료. data/deploy/ 내용:")
    total_size = 0
    for f in sorted(DEPLOY_DIR.iterdir()):
        size = f.stat().st_size
        total_size += size
        print(f"  {f.name}: {size / 1024:.1f}KB")
    print(f"합계: {total_size / 1_048_576:.2f}MB")
