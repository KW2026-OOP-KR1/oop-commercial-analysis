
from pathlib import Path

import pandas as pd
from pyproj import Transformer


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
# 1. CSV 파일 입출력
# ==========================================

class CsvRepository:
    ENCODINGS = ("utf-8-sig", "cp949", "euc-kr")

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def read(self, file_path):
        file_path = Path(file_path)

        for encoding in self.ENCODINGS:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"[읽기 성공] {file_path.name} ({encoding})")
                return df
            except UnicodeDecodeError:
                continue

        raise ValueError(f"CSV 인코딩 오류: {file_path}")

    def save(self, df, file_name):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / file_name

        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[저장 완료] {output_path}")


# ==========================================
# 2. 데이터 정제
# ==========================================

class DataCleaner:
    def clean(self, df):
        df = df.copy()

        df.columns = df.columns.astype(str).str.strip()

        columns = df.select_dtypes(
            include=["object", "string"]
        ).columns

        for column in columns:
            df[column] = df[column].str.strip()

        return df.drop_duplicates().copy()

    def normalize_code(self, df, column="상권_코드"):
        df = df.copy()

        df[column] = (
            df[column]
            .astype("string")
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )

        return df


# ==========================================
# 3. 필수 컬럼 검증
# ==========================================

class DataValidator:
    def validate_columns(self, df, required_columns):
        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(f"필수 컬럼 누락: {missing}")


# ==========================================
# 4. 좌표계 변환
# ==========================================

class CoordinateConverter:
    def __init__(self, validator, source_crs, target_crs):
        self.validator = validator
        self.source_crs = source_crs
        self.target_crs = target_crs

    def convert(self, df):
        df = df.copy()

        x_col = "엑스좌표_값"
        y_col = "와이좌표_값"

        self.validator.validate_columns(df, [x_col, y_col])

        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

        valid = df[x_col].notna() & df[y_col].notna()

        transformer = Transformer.from_crs(
            self.source_crs,
            self.target_crs,
            always_xy=True,
        )

        df["경도"] = float("nan")
        df["위도"] = float("nan")

        longitude, latitude = transformer.transform(
            df.loc[valid, x_col].to_numpy(),
            df.loc[valid, y_col].to_numpy(),
        )

        df.loc[valid, "경도"] = longitude
        df.loc[valid, "위도"] = latitude

        return df


# ==========================================
# 5. 연도 및 분기 컬럼 처리
# ==========================================

class PeriodColumnProcessor:
    def __init__(self, validator):
        self.validator = validator

    def add_quarter_column(self, df):
        df = df.copy()
        column = "기준_년분기_코드"

        self.validator.validate_columns(df, [column])

        code = (
            df[column]
            .astype("string")
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

        quarter = code.str.extract(r"^\d{4}([1-4])$", expand=False)

        df["분기"] = pd.to_numeric(
            quarter, errors="coerce"
        ).astype("Int64")

        return df

    def add_period_columns(self, df, year):
        df = df.copy()
        df["분석연도"] = year
        df = self.add_quarter_column(df)

        # 분석연도와 분기를 기준_년분기_코드 바로 뒤로 이동
        columns = df.columns.tolist()
        columns.remove("분석연도")
        columns.remove("분기")

        index = columns.index("기준_년분기_코드") + 1
        columns[index:index] = ["분석연도", "분기"]

        return df[columns]


# ==========================================
# 6. 위치 정보 병합
# ==========================================

class LocationMerger:
    def __init__(self, location_columns):
        self.location_columns = location_columns

    def merge(self, df, location):
        df = df.copy()

        columns = [
            column
            for column in self.location_columns
            if column in location.columns
            and (
                column == "상권_코드"
                or column not in df.columns
            )
        ]

        return df.merge(
            location[columns],
            on="상권_코드",
            how="left",
            validate="many_to_one",
        )


# ==========================================
# 7. 상권 위치 데이터 전처리
# ==========================================

class LocationProcessor:
    REQUIRED_COLUMNS = [
        "상권_코드",
        "엑스좌표_값",
        "와이좌표_값",
    ]

    def __init__(self, data_dir, repository, cleaner, validator, converter):
        self.data_dir = Path(data_dir)
        self.repository = repository
        self.cleaner = cleaner
        self.validator = validator
        self.converter = converter

    def process(self):
        file_path = self.data_dir / "data_location.csv"

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        location = self.repository.read(file_path)
        location = self.cleaner.clean(location)

        self.validator.validate_columns(location, self.REQUIRED_COLUMNS)

        location = self.cleaner.normalize_code(location)

        location = location.drop_duplicates(
            subset=["상권_코드"], keep="first"
        ).copy()

        location = self.converter.convert(location)

        self.repository.save(location, "data_location_coordinates.csv")

        print(f"[위치 처리 완료] {len(location):,}개 상권")

        return location


# ==========================================
# 8. 연도별 매출 데이터 전처리
# ==========================================

class SalesDataProcessor:
    def __init__(self, data_dir, repository, cleaner, merger, period_processor):
        self.data_dir = Path(data_dir)
        self.repository = repository
        self.cleaner = cleaner
        self.merger = merger
        self.period_processor = period_processor

    def process(self, year, location):
        file_path = self.data_dir / f"data_{year}.csv"

        if not file_path.exists():
            print(f"[건너뜀] {file_path.name}")
            return None

        df = self.repository.read(file_path)
        df = self.cleaner.clean(df)

        if "상권_코드" in df.columns:
            df = self.cleaner.normalize_code(df)
            df = self.merger.merge(df, location)
        else:
            print(f"[경고] {file_path.name}: 상권 코드 없음")

        df = self.period_processor.add_period_columns(df, year)

        self.repository.save(df, f"processed_{year}.csv")

        print(f"[{year}년 처리 완료] {len(df):,}개 행")

        return df


# ==========================================
# 9. 전체 전처리 실행 순서 관리
# ==========================================

class PreprocessingPipeline:
    def __init__(self, repository, location_processor, sales_processor, years):
        self.repository = repository
        self.location_processor = location_processor
        self.sales_processor = sales_processor
        self.years = years

    def run(self):
        print("서울시 상권 데이터 전처리 시작")

        location = self.location_processor.process()
        results = []

        for year in self.years:
            result = self.sales_processor.process(year, location)

            if result is not None:
                results.append(result)

        if not results:
            print("[종료] 통합할 데이터가 없습니다.")
            return

        combined = pd.concat(results, ignore_index=True, sort=False)

        start_year = min(self.years)
        end_year = max(self.years)

        self.repository.save(
            combined,
            f"combined_{start_year}_{end_year}.csv",
        )

        print(f"[전체 완료] {len(combined):,}개 행")


# ==========================================
# 10. 객체 생성 및 의존성 연결
# ==========================================

def _create_pipeline():
    repository = CsvRepository(OUTPUT_DIR)
    cleaner = DataCleaner()
    validator = DataValidator()

    converter = CoordinateConverter(validator, SOURCE_CRS, TARGET_CRS)
    merger = LocationMerger(LOCATION_COLUMNS)
    period_processor = PeriodColumnProcessor(validator)

    location_processor = LocationProcessor(
        DATA_DIR, repository, cleaner, validator, converter
    )

    sales_processor = SalesDataProcessor(
        DATA_DIR, repository, cleaner, merger, period_processor
    )

    return PreprocessingPipeline(
        repository, location_processor, sales_processor, YEARS
    )


# ==========================================
# 11. 기존 함수 기반 인터페이스
# ==========================================

def read_csv_file(file_path):
    return CsvRepository(OUTPUT_DIR).read(file_path)


def save_csv_file(df, file_name):
    return CsvRepository(OUTPUT_DIR).save(df, file_name)


def preprocess_data(df):
    return DataCleaner().clean(df)


def validate_columns(df, required_columns):
    return DataValidator().validate_columns(df, required_columns)


def normalize_code(df, column="상권_코드"):
    return DataCleaner().normalize_code(df, column)


def add_quarter_column(df):
    return PeriodColumnProcessor(DataValidator()).add_quarter_column(df)


def convert_coordinates(df):
    converter = CoordinateConverter(DataValidator(), SOURCE_CRS, TARGET_CRS)
    return converter.convert(df)


def process_location():
    return _create_pipeline().location_processor.process()


def merge_location(df, location):
    return LocationMerger(LOCATION_COLUMNS).merge(df, location)


def process_year(year, location):
    return _create_pipeline().sales_processor.process(year, location)


def main():
    _create_pipeline().run()


if __name__ == "__main__":
    main()
