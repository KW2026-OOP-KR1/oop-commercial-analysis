
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
# 2. CSV 파일 입출력
# ==========================================

class CsvRepository:
    """CSV 파일의 읽기 및 저장을 담당한다."""

    ENCODINGS = [
        "utf-8-sig",
        "cp949",
        "euc-kr"
    ]

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def read(self, file_path):
        for encoding in self.ENCODINGS:
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding
                )

                print(
                    f"[읽기 성공] "
                    f"{file_path.name} ({encoding})"
                )

                return df

            except UnicodeDecodeError:
                continue

        raise ValueError(
            f"CSV 인코딩 오류: {file_path}"
        )

    def save(self, df, file_name):
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = self.output_dir / file_name

        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"[저장 완료] {output_path}")


# ==========================================
# 3. 데이터 정제
# ==========================================

class DataCleaner:
    """공백, 중복 행 제거 및 데이터 형식 통일을 담당한다."""

    def clean(self, df):
        df = df.copy()

        # 컬럼명 공백 제거
        df.columns = (
            df.columns.astype(str).str.strip()
        )

        # 문자열 데이터 공백 제거
        columns = df.select_dtypes(
            include=["object", "string"]
        ).columns

        for column in columns:
            df[column] = df[column].str.strip()

        # 완전히 동일한 중복 행 제거
        df = df.drop_duplicates().copy()

        return df

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
# 4. 데이터 검증
# ==========================================

class DataValidator:
    """데이터에 필수 컬럼이 존재하는지 검사한다."""

    def validate_columns(self, df, required_columns):
        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"필수 컬럼 누락: {missing}"
            )


# ==========================================
# 5. 좌표계 변환
# ==========================================

class CoordinateConverter:
    """원본 좌표를 경도 및 위도로 변환한다."""

    def __init__(
        self,
        validator,
        source_crs,
        target_crs
    ):
        self.validator = validator
        self.source_crs = source_crs
        self.target_crs = target_crs

    def convert(self, df):
        df = df.copy()

        x_col = "엑스좌표_값"
        y_col = "와이좌표_값"

        self.validator.validate_columns(
            df,
            [x_col, y_col]
        )

        df[x_col] = pd.to_numeric(
            df[x_col],
            errors="coerce"
        )

        df[y_col] = pd.to_numeric(
            df[y_col],
            errors="coerce"
        )

        valid = (
            df[x_col].notna()
            & df[y_col].notna()
        )

        transformer = Transformer.from_crs(
            self.source_crs,
            self.target_crs,
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


# ==========================================
# 6. 상권 위치 데이터 처리
# ==========================================

class LocationProcessor:
    """상권 위치 데이터의 전처리 과정을 담당한다."""

    REQUIRED_COLUMNS = [
        "상권_코드",
        "엑스좌표_값",
        "와이좌표_값"
    ]

    def __init__(
        self,
        data_dir,
        repository,
        cleaner,
        validator,
        converter
    ):
        self.data_dir = Path(data_dir)
        self.repository = repository
        self.cleaner = cleaner
        self.validator = validator
        self.converter = converter

    def process(self):
        file_path = (
            self.data_dir / "data_location.csv"
        )

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        # 위치 데이터 읽기 및 정제
        location = self.repository.read(file_path)
        location = self.cleaner.clean(location)

        # 필수 컬럼 확인
        self.validator.validate_columns(
            location,
            self.REQUIRED_COLUMNS
        )

        # 상권 코드 형식 통일
        location = self.cleaner.normalize_code(location)

        # 동일한 상권 코드 제거
        location = location.drop_duplicates(
            subset=["상권_코드"],
            keep="first"
        ).copy()

        # 좌표계 변환
        location = self.converter.convert(location)

        # 위치 데이터 저장
        self.repository.save(
            location,
            "data_location_coordinates.csv"
        )

        print(
            f"[위치 처리 완료] "
            f"{len(location):,}개 상권"
        )

        return location


# ==========================================
# 7. 연도별 매출 데이터 처리
# ==========================================

class SalesDataProcessor:
    """연도별 매출 데이터의 전처리를 담당한다."""

    def __init__(
        self,
        data_dir,
        repository,
        cleaner,
        validator,
        location_columns
    ):
        self.data_dir = Path(data_dir)
        self.repository = repository
        self.cleaner = cleaner
        self.validator = validator
        self.location_columns = location_columns

    def add_quarter_column(self, df):
        df = df.copy()

        column = "기준_년분기_코드"

        self.validator.validate_columns(
            df,
            [column]
        )

        code = (
            df[column]
            .astype("string")
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

        quarter = code.str.extract(
            r"^\d{4}([1-4])$",
            expand=False
        )

        df["분기"] = pd.to_numeric(
            quarter,
            errors="coerce"
        ).astype("Int64")

        return df

    def merge_location(self, df, location):
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

        location_data = location[columns]

        return df.merge(
            location_data,
            on="상권_코드",
            how="left",
            validate="many_to_one"
        )

    def _reorder_period_columns(self, df):
        """분석연도와 분기 컬럼의 위치를 정리한다."""

        columns = df.columns.tolist()

        columns.remove("분석연도")
        columns.remove("분기")

        index = columns.index(
            "기준_년분기_코드"
        ) + 1

        columns[index:index] = [
            "분석연도",
            "분기"
        ]

        return df[columns]

    def process(self, year, location):
        file_path = (
            self.data_dir / f"data_{year}.csv"
        )

        if not file_path.exists():
            print(f"[건너뜀] {file_path.name}")
            return None

        # 매출 데이터 읽기 및 정제
        df = self.repository.read(file_path)
        df = self.cleaner.clean(df)

        # 상권 코드 통일 및 위치 데이터 병합
        if "상권_코드" in df.columns:
            df = self.cleaner.normalize_code(df)
            df = self.merge_location(df, location)

        else:
            print(
                f"[경고] {file_path.name}: "
                f"상권 코드 없음"
            )

        # 분석 연도 추가
        df["분석연도"] = year

        # 분기 컬럼 추가
        df = self.add_quarter_column(df)

        # 분석연도 및 분기 컬럼 위치 정리
        df = self._reorder_period_columns(df)

        # 연도별 전처리 데이터 저장
        self.repository.save(
            df,
            f"processed_{year}.csv"
        )

        print(
            f"[{year}년 처리 완료] "
            f"{len(df):,}개 행"
        )

        return df


# ==========================================
# 8. 전처리 실행 관리
# ==========================================

class PreprocessingPipeline:
    """전체 전처리 실행 순서 및 결과 통합을 담당한다."""

    def __init__(
        self,
        repository,
        location_processor,
        sales_processor,
        years
    ):
        self.repository = repository
        self.location_processor = location_processor
        self.sales_processor = sales_processor
        self.years = years

    def run(self):
        print("서울시 상권 데이터 전처리 시작")

        # 위치 데이터 전처리
        location = self.location_processor.process()

        results = []

        # 연도별 매출 데이터 전처리
        for year in self.years:
            result = self.sales_processor.process(
                year,
                location
            )

            if result is not None:
                results.append(result)

        if not results:
            print("[종료] 통합할 데이터가 없습니다.")
            return

        # 연도별 데이터 통합
        combined = pd.concat(
            results,
            ignore_index=True,
            sort=False
        )

        start_year = min(self.years)
        end_year = max(self.years)

        # 통합 데이터 저장
        self.repository.save(
            combined,
            f"combined_{start_year}_{end_year}.csv"
        )

        print(
            f"[전체 완료] "
            f"{len(combined):,}개 행"
        )


# ==========================================
# 9. 전처리 객체 생성
# ==========================================

def _create_location_processor():
    repository = CsvRepository(OUTPUT_DIR)
    cleaner = DataCleaner()
    validator = DataValidator()

    converter = CoordinateConverter(
        validator,
        SOURCE_CRS,
        TARGET_CRS
    )

    return LocationProcessor(
        DATA_DIR,
        repository,
        cleaner,
        validator,
        converter
    )


def _create_sales_processor():
    repository = CsvRepository(OUTPUT_DIR)
    cleaner = DataCleaner()
    validator = DataValidator()

    return SalesDataProcessor(
        DATA_DIR,
        repository,
        cleaner,
        validator,
        LOCATION_COLUMNS
    )


def _create_pipeline():
    # 공통 객체 생성
    repository = CsvRepository(OUTPUT_DIR)
    cleaner = DataCleaner()
    validator = DataValidator()

    converter = CoordinateConverter(
        validator,
        SOURCE_CRS,
        TARGET_CRS
    )

    # 공통 객체를 각 전처리 객체에 전달
    location_processor = LocationProcessor(
        DATA_DIR,
        repository,
        cleaner,
        validator,
        converter
    )

    sales_processor = SalesDataProcessor(
        DATA_DIR,
        repository,
        cleaner,
        validator,
        LOCATION_COLUMNS
    )

    return PreprocessingPipeline(
        repository,
        location_processor,
        sales_processor,
        YEARS
    )


# ==========================================
# 10. 기존 함수 인터페이스 유지
# ==========================================

def read_csv_file(file_path):
    return CsvRepository(OUTPUT_DIR).read(file_path)


def save_csv_file(df, file_name):
    return CsvRepository(OUTPUT_DIR).save(
        df,
        file_name
    )


def preprocess_data(df):
    return DataCleaner().clean(df)


def validate_columns(df, required_columns):
    return DataValidator().validate_columns(
        df,
        required_columns
    )


def normalize_code(df, column="상권_코드"):
    return DataCleaner().normalize_code(
        df,
        column
    )


def add_quarter_column(df):
    processor = _create_sales_processor()
    return processor.add_quarter_column(df)


def convert_coordinates(df):
    converter = CoordinateConverter(
        DataValidator(),
        SOURCE_CRS,
        TARGET_CRS
    )

    return converter.convert(df)


def process_location():
    processor = _create_location_processor()
    return processor.process()


def merge_location(df, location):
    processor = _create_sales_processor()
    return processor.merge_location(df, location)


def process_year(year, location):
    processor = _create_sales_processor()
    return processor.process(year, location)


# ==========================================
# 11. 전체 실행
# ==========================================

def main():
    pipeline = _create_pipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
