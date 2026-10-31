
from pathlib import Path

import pandas as pd
from pyproj import Transformer


# ==========================================
# 1. 환경 설정
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

YEARS = [2023, 2024, 2025]

SOURCE_CRS = "EPSG:5181"
TARGET_CRS = "EPSG:4326"

LOCATION_COLUMNS = [
    "상권_코드",
    "엑스좌표_값",
    "와이좌표_값",
    "경도",
    "위도",
    "자치구_코드_명",
    "행정동_코드_명",
    "영역_면적",
]


# ==========================================
# 2. 공통 함수
# ==========================================

def read_csv_file(file_path):
    """CSV 파일 읽기"""

    encodings = ["utf-8-sig", "cp949", "euc-kr"]

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"[읽기 성공] {file_path.name} ({encoding})")
            return df

        except UnicodeDecodeError:
            continue

    raise ValueError(f"CSV 인코딩 오류: {file_path}")


def save_csv_file(df, file_name):
    """CSV 파일 저장"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / file_name

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"[저장 완료] {output_path}")


def preprocess_data(df):
    """공통 데이터 전처리"""

    df = df.copy()

    # 컬럼명 공백 제거
    df.columns = df.columns.astype(str).str.strip()

    # 문자열 공백 제거
    columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in columns:
        df[column] = df[column].str.strip()

    # 완전히 동일한 중복 행 제거
    df = df.drop_duplicates().copy()

    return df


def validate_columns(df, required_columns):
    """필수 컬럼 존재 여부 검사"""

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")


def normalize_code(df, column="상권_코드"):
    """상권 코드를 문자열로 통일"""

    df = df.copy()

    df[column] = (
        df[column]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    return df


def add_quarter_column(df):
    """기준_년분기_코드에서 분기를 추출하여 분기 컬럼 생성"""

    df = df.copy()

    column = "기준_년분기_코드"

    validate_columns(df, [column])

    code = (
        df[column]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    quarter = code.str.extract(r"^\d{4}([1-4])$", expand=False)

    df["분기"] = pd.to_numeric(
        quarter,
        errors="coerce"
    ).astype("Int64")

    return df


# ==========================================
# 3. 위치 데이터 처리
# ==========================================

def convert_coordinates(df):
    """원본 좌표를 경도, 위도로 변환"""

    df = df.copy()

    x_col = "엑스좌표_값"
    y_col = "와이좌표_값"

    validate_columns(df, [x_col, y_col])

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    valid = df[x_col].notna() & df[y_col].notna()

    transformer = Transformer.from_crs(
        SOURCE_CRS,
        TARGET_CRS,
        always_xy=True
    )

    df["경도"] = float("nan")
    df["위도"] = float("nan")

    longitude, latitude = transformer.transform(
        df.loc[valid, x_col].to_numpy(),
        df.loc[valid, y_col].to_numpy()
    )

    df.loc[valid, "경도"] = longitude
    df.loc[valid, "위도"] = latitude

    return df


def process_location():
    """상권 위치 데이터 처리"""

    file_path = DATA_DIR / "data_location.csv"

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    location = read_csv_file(file_path)
    location = preprocess_data(location)

    validate_columns(
        location,
        ["상권_코드", "엑스좌표_값", "와이좌표_값"]
    )

    location = normalize_code(location)

    location = location.drop_duplicates(
        subset=["상권_코드"],
        keep="first"
    ).copy()

    location = convert_coordinates(location)

    save_csv_file(
        location,
        "data_location_coordinates.csv"
    )

    print(f"[위치 처리 완료] {len(location):,}개 상권")

    return location


# ==========================================
# 4. 매출 데이터 처리
# ==========================================

def merge_location(df, location):
    """상권 코드를 기준으로 위치 데이터 결합"""

    df = df.copy()

    columns = [
        column
        for column in LOCATION_COLUMNS
        if column in location.columns
        and (
            column == "상권_코드"
            or column not in df.columns
        )
    ]

    location_data = location[columns]

    result = df.merge(
        location_data,
        on="상권_코드",
        how="left",
        validate="many_to_one"
    )

    return result


def process_year(year, location):
    """특정 연도의 매출 데이터 처리"""

    file_path = DATA_DIR / f"data_{year}.csv"

    if not file_path.exists():
        print(f"[건너뜀] {file_path.name}")
        return None

    # 1. 읽기 및 기본 전처리
    df = read_csv_file(file_path)
    df = preprocess_data(df)

    # 2. 상권 코드 확인 및 위치 결합
    if "상권_코드" in df.columns:
        df = normalize_code(df)
        df = merge_location(df, location)

    else:
        print(f"[경고] {file_path.name}: 상권 코드 없음")

    # 3. 분석 연도 추가
    df["분석연도"] = year

    # 4. 분기 컬럼 추가
    df = add_quarter_column(df)

    # 5. 결과 저장
    save_csv_file(df, f"processed_{year}.csv")

    print(f"[{year}년 처리 완료] {len(df):,}개 행")

    return df


# ==========================================
# 5. 전체 실행
# ==========================================

def main():
    print("서울시 상권 데이터 전처리 시작")

    location = process_location()

    results = []

    for year in YEARS:
        result = process_year(year, location)

        if result is not None:
            results.append(result)

    if not results:
        print("[종료] 통합할 데이터가 없습니다.")
        return

    combined = pd.concat(
        results,
        ignore_index=True,
        sort=False
    )

    start_year = min(YEARS)
    end_year = max(YEARS)

    save_csv_file(
        combined,
        f"combined_{start_year}_{end_year}.csv"
    )

    print(f"[전체 완료] {len(combined):,}개 행")


if __name__ == "__main__":
    main()
