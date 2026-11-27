
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "output" / "combined_2023_2025.csv"


# ==========================================
# 1. 데이터 로드
# ==========================================

class DataLoader:
    def __init__(self, file_path=None):
        self.base_dir = Path(__file__).resolve().parent
        self.data_path = self.make_path("combined_2023_2025.csv")
        self.file_path = Path(file_path) if file_path is not None else self.data_path
        self.encodings = ("utf-8-sig", "utf-8", "cp949", "euc-kr")

    def make_path(self, file_name):
        return self.base_dir / "output" / file_name

    def load_data(self, file_path=None):
        path = Path(file_path) if file_path is not None else self.file_path

        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

        for encoding in self.encodings:
            try:
                df = pd.read_csv(path, encoding=encoding)
                print(f"[데이터 로드 성공] {path.name}")
                print(f"행 개수: {len(df):,}개")
                return df
            except UnicodeDecodeError:
                continue

        raise ValueError(f"CSV 인코딩을 확인해 주세요: {path}")


# ==========================================
# 2. 데이터 필터링
# ==========================================

class DataFilter:
    def __init__(self, df):
        self.original_df = df.copy()
        self.df = df.copy()

    def reset(self):
        self.df = self.original_df.copy()
        return self

    def by_year(self, year):
        return self._filter_column("분석연도", year)

    def by_quarter(self, quarter):
        return self._filter_column("분기", quarter)

    def by_year_quarter(self, year, quarter):
        return self.by_year(year).by_quarter(quarter)

    def by_district(self, district):
        return self._filter_column("자치구_코드_명", district)

    def by_dong(self, dong):
        return self._filter_column("행정동_코드_명", dong)

    def by_market_type(self, market_type):
        return self._filter_column("상권_구분_코드_명", market_type)

    def by_market(self, market):
        return self._filter_column("상권_코드_명", market)

    def by_business(self, business):
        return self._filter_column("서비스_업종_코드_명", business)

    def _filter_column(self, column, value):
        if column not in self.df.columns:
            return self

        if isinstance(value, (list, tuple, set)):
            self.df = self.df[self.df[column].isin(value)]
        else:
            self.df = self.df[self.df[column] == value]

        return self

    def get(self):
        return self.df.copy()

    def count(self):
        return len(self.df)


# ==========================================
# 3. 공통 집계 연산
# ==========================================

class SalesAggregator:
    SALES_COLUMN = "당월_매출_금액"

    def __init__(self, df):
        self.df = df

    def has_columns(self, columns):
        return all(column in self.df.columns for column in columns)

    def sum_column(self, column):
        if column not in self.df.columns:
            return 0

        return pd.to_numeric(self.df[column], errors="coerce").sum()

    def group_sales(self, group_column):
        required = [group_column, self.SALES_COLUMN]

        if not self.has_columns(required):
            return pd.DataFrame(columns=required)

        return (
            self.df.groupby(group_column)[self.SALES_COLUMN]
            .sum()
            .reset_index()
            .sort_values(self.SALES_COLUMN, ascending=False)
        )

    def column_summary(self, columns):
        result = [
            {"구분": name, "매출금액": self.sum_column(column)}
            for name, column in columns.items()
            if column in self.df.columns
        ]

        return pd.DataFrame(result, columns=["구분", "매출금액"])


# ==========================================
# 4. 기본 매출 지표
# ==========================================

class SalesSummary:
    def __init__(self, aggregator):
        self.aggregator = aggregator

    def get_total_sales(self):
        return self.aggregator.sum_column("당월_매출_금액")

    def get_total_count(self):
        return self.aggregator.sum_column("당월_매출_건수")

    def get_average_spending(self):
        count = self.get_total_count()
        return self.get_total_sales() / count if count != 0 else 0


# ==========================================
# 5. 지역 및 업종별 매출 분석
# ==========================================

class GroupAnalyzer:
    def __init__(self, aggregator):
        self.aggregator = aggregator

    def get_market_sales(self):
        return self.aggregator.group_sales("상권_코드_명")

    def get_business_sales(self):
        return self.aggregator.group_sales("서비스_업종_코드_명")

    def get_district_sales(self):
        return self.aggregator.group_sales("자치구_코드_명")

    def get_dong_sales(self):
        return self.aggregator.group_sales("행정동_코드_명")

    def get_market_type_sales(self):
        return self.aggregator.group_sales("상권_구분_코드_명")

    def get_top_markets(self, n=10):
        return self.get_market_sales().head(n)

    def get_top_businesses(self, n=10):
        return self.get_business_sales().head(n)


# ==========================================
# 6. 시간 관련 매출 분석
# ==========================================

class TemporalAnalyzer:
    WEEKDAY_COLUMNS = {
        "월요일": "월요일_매출_금액",
        "화요일": "화요일_매출_금액",
        "수요일": "수요일_매출_금액",
        "목요일": "목요일_매출_금액",
        "금요일": "금요일_매출_금액",
        "토요일": "토요일_매출_금액",
        "일요일": "일요일_매출_금액",
    }

    TIME_COLUMNS = {
        "00~06": "시간대_00~06_매출_금액",
        "06~11": "시간대_06~11_매출_금액",
        "11~14": "시간대_11~14_매출_금액",
        "14~17": "시간대_14~17_매출_금액",
        "17~21": "시간대_17~21_매출_금액",
        "21~24": "시간대_21~24_매출_금액",
    }

    def __init__(self, aggregator, summary):
        self.aggregator = aggregator
        self.summary = summary

    def get_weekday_sales(self, columns=None):
        return self.aggregator.column_summary(
            self.WEEKDAY_COLUMNS if columns is None else columns
        )

    def get_weekday_weekend_sales(self):
        return {
            "주중": self.aggregator.sum_column("주중_매출_금액"),
            "주말": self.aggregator.sum_column("주말_매출_금액"),
        }

    def get_time_sales(self, columns=None):
        return self.aggregator.column_summary(
            self.TIME_COLUMNS if columns is None else columns
        )

    def get_peak_time(self, columns=None):
        result = self.get_time_sales(columns)

        if result.empty:
            return None

        peak = result.loc[result["매출금액"].idxmax()]
        return peak["구분"], peak["매출금액"]

    def _period_sales(self, column):
        required = [column, "당월_매출_금액"]

        if not self.aggregator.has_columns(required):
            return pd.DataFrame(columns=required)

        return (
            self.aggregator.df.groupby(column)["당월_매출_금액"]
            .sum()
            .reset_index()
            .sort_values(column)
        )

    def get_year_sales(self):
        return self._period_sales("분석연도")

    def get_year_growth(self):
        result = self.get_year_sales()

        if result.empty:
            return result

        result["전년대비_증감액"] = result["당월_매출_금액"].diff()
        result["전년대비_성장률"] = result["당월_매출_금액"].pct_change() * 100

        return result

    def get_quarter_sales(self):
        return self._period_sales("기준_년분기_코드")

    def get_weekend_ratio(self):
        total = self.summary.get_total_sales()
        weekend = self.aggregator.sum_column("주말_매출_금액")

        return weekend / total * 100 if total != 0 else 0


# ==========================================
# 7. 인구통계 매출 분석
# ==========================================

class DemographicAnalyzer:
    AGE_COLUMNS = {
        "10대": "연령대_10_매출_금액",
        "20대": "연령대_20_매출_금액",
        "30대": "연령대_30_매출_금액",
        "40대": "연령대_40_매출_금액",
        "50대": "연령대_50_매출_금액",
        "60대 이상": "연령대_60_이상_매출_금액",
    }

    GENDER_COLUMNS = {
        "남성": "남성_매출_금액",
        "여성": "여성_매출_금액",
    }

    def __init__(self, aggregator):
        self.aggregator = aggregator

    def get_gender_sales(self, columns=None):
        columns = self.GENDER_COLUMNS if columns is None else columns

        return {
            name: self.aggregator.sum_column(column)
            for name, column in columns.items()
        }

    def get_gender_ratio(self, columns=None):
        gender = self.get_gender_sales(columns)
        total = gender["남성"] + gender["여성"]

        if total == 0:
            return {"남성": 0, "여성": 0}

        return {
            "남성": gender["남성"] / total * 100,
            "여성": gender["여성"] / total * 100,
        }

    def get_age_sales(self, columns=None):
        return self.aggregator.column_summary(
            self.AGE_COLUMNS if columns is None else columns
        )

    def get_main_age_group(self, columns=None):
        result = self.get_age_sales(columns)

        if result.empty:
            return None

        main_age = result.loc[result["매출금액"].idxmax()]
        return main_age["구분"], main_age["매출금액"]


# ==========================================
# 8. 기간 비교 분석
# ==========================================

class PeriodComparator:
    REQUIRED_COLUMNS = [
        "분석연도", "분기",
        "당월_매출_금액", "당월_매출_건수",
        "주중_매출_금액", "주말_매출_금액",
    ]

    METRICS = [
        "총매출", "총건수", "평균객단가",
        "주중매출", "주말매출",
    ]

    def __init__(self, aggregator):
        self.aggregator = aggregator

    def _get_period_metrics(self, year, quarter):
        year, quarter = int(year), int(quarter)

        if not 1 <= quarter <= 4:
            raise ValueError("분기는 1부터 4 사이여야 합니다.")

        period_df = (
            DataFilter(self.aggregator.df)
            .by_year_quarter(year, quarter)
            .get()
        )

        if period_df.empty:
            return None

        aggregator = SalesAggregator(period_df)
        summary = SalesSummary(aggregator)
        temporal = TemporalAnalyzer(aggregator, summary)
        weekday_weekend = temporal.get_weekday_weekend_sales()

        return {
            "총매출": summary.get_total_sales(),
            "총건수": summary.get_total_count(),
            "평균객단가": summary.get_average_spending(),
            "주중매출": weekday_weekend["주중"],
            "주말매출": weekday_weekend["주말"],
        }

    def compare_quarters(
        self, base_year, base_quarter, target_year, target_quarter
    ):
        if not self.aggregator.has_columns(self.REQUIRED_COLUMNS):
            return None

        base = self._get_period_metrics(base_year, base_quarter)

        if base is None:
            return None

        target = self._get_period_metrics(target_year, target_quarter)

        if target is None:
            return None

        result = []

        for name in self.METRICS:
            before, after = base[name], target[name]
            difference = after - before
            rate = difference / before * 100 if before != 0 else None

            result.append({
                "항목": name,
                "기준 기간": before,
                "비교 기간": after,
                "증감액": difference,
                "증감률(%)": rate,
            })

        return pd.DataFrame(result)


# ==========================================
# 9. 공간 매출 분석
# ==========================================

class SpatialAnalyzer:
    def __init__(self, aggregator):
        self.aggregator = aggregator

    def get_sales_per_area(self, group="상권_코드_명"):
        required = [group, "당월_매출_금액", "영역_면적"]

        if not self.aggregator.has_columns(required):
            return pd.DataFrame(
                columns=[group, "총매출", "영역면적", "면적당_매출"]
            )

        temp = self.aggregator.df.copy()
        temp["영역_면적"] = pd.to_numeric(temp["영역_면적"], errors="coerce")

        result = (
            temp.groupby(group)
            .agg(
                총매출=("당월_매출_금액", "sum"),
                영역면적=("영역_면적", "first"),
            )
            .reset_index()
        )

        result["면적당_매출"] = result["총매출"] / result["영역면적"]

        return result.sort_values("면적당_매출", ascending=False)

    def get_map_data(self):
        required = ["상권_코드_명", "경도", "위도", "당월_매출_금액"]

        if not self.aggregator.has_columns(required):
            return pd.DataFrame(columns=required)

        return (
            self.aggregator.df
            .groupby(["상권_코드_명", "경도", "위도"], dropna=False)
            ["당월_매출_금액"]
            .sum()
            .reset_index()
        )


# ==========================================
# 10. 기존 외부 인터페이스 유지 (Facade)
# ==========================================

class SalesAnalyzer:
    WEEKDAY_COLUMNS = TemporalAnalyzer.WEEKDAY_COLUMNS
    TIME_COLUMNS = TemporalAnalyzer.TIME_COLUMNS
    AGE_COLUMNS = DemographicAnalyzer.AGE_COLUMNS
    GENDER_COLUMNS = DemographicAnalyzer.GENDER_COLUMNS

    def __init__(self, df):
        self._aggregator = SalesAggregator(df.copy())
        self._summary = SalesSummary(self._aggregator)
        self._group = GroupAnalyzer(self._aggregator)
        self._temporal = TemporalAnalyzer(self._aggregator, self._summary)
        self._demographic = DemographicAnalyzer(self._aggregator)
        self._comparator = PeriodComparator(self._aggregator)
        self._spatial = SpatialAnalyzer(self._aggregator)

    @property
    def df(self):
        return self._aggregator.df

    @df.setter
    def df(self, value):
        self._aggregator.df = value

    def get_total_sales(self):
        return self._summary.get_total_sales()

    def get_total_count(self):
        return self._summary.get_total_count()

    def get_average_spending(self):
        return self._summary.get_average_spending()

    def get_market_sales(self):
        return self._group.get_market_sales()

    def get_business_sales(self):
        return self._group.get_business_sales()

    def get_district_sales(self):
        return self._group.get_district_sales()

    def get_dong_sales(self):
        return self._group.get_dong_sales()

    def get_market_type_sales(self):
        return self._group.get_market_type_sales()

    def get_top_markets(self, n=10):
        return self._group.get_top_markets(n)

    def get_top_businesses(self, n=10):
        return self._group.get_top_businesses(n)

    def get_weekday_sales(self):
        return self._temporal.get_weekday_sales(self.WEEKDAY_COLUMNS)

    def get_weekday_weekend_sales(self):
        return self._temporal.get_weekday_weekend_sales()

    def get_time_sales(self):
        return self._temporal.get_time_sales(self.TIME_COLUMNS)

    def get_peak_time(self):
        return self._temporal.get_peak_time(self.TIME_COLUMNS)

    def get_year_sales(self):
        return self._temporal.get_year_sales()

    def get_year_growth(self):
        return self._temporal.get_year_growth()

    def get_quarter_sales(self):
        return self._temporal.get_quarter_sales()

    def get_weekend_ratio(self):
        return self._temporal.get_weekend_ratio()

    def get_gender_sales(self):
        return self._demographic.get_gender_sales(self.GENDER_COLUMNS)

    def get_gender_ratio(self):
        return self._demographic.get_gender_ratio(self.GENDER_COLUMNS)

    def get_age_sales(self):
        return self._demographic.get_age_sales(self.AGE_COLUMNS)

    def get_main_age_group(self):
        return self._demographic.get_main_age_group(self.AGE_COLUMNS)

    def compare_quarters(
        self, base_year, base_quarter, target_year, target_quarter
    ):
        return self._comparator.compare_quarters(
            base_year, base_quarter, target_year, target_quarter
        )

    def get_sales_per_area(self, group="상권_코드_명"):
        return self._spatial.get_sales_per_area(group)

    def get_map_data(self):
        return self._spatial.get_map_data()

    def _sum(self, column):
        return self._aggregator.sum_column(column)

    def _group_sales(self, group_column):
        return self._aggregator.group_sales(group_column)

    def _column_summary(self, columns):
        return self._aggregator.column_summary(columns)


# ==========================================
# 11. 이전 버전의 함수 기반 인터페이스
# ==========================================

def load_data(file_path=DATA_PATH):
    return DataLoader(file_path).load_data()
