
from pathlib import Path

import pandas as pd


# ==========================================
# 1. 데이터 로더 클래스
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
# 2. 데이터 필터 클래스
# ==========================================

class DataFilter:

    def __init__(self, df):
        self.original_df = df.copy()
        self.df = df.copy()

    # ------------------------------------------
    # 필터 초기화
    # ------------------------------------------

    def reset(self):
        self.df = self.original_df.copy()
        return self

    # ------------------------------------------
    # 분석 연도 필터
    # ------------------------------------------

    def by_year(self, year):
        return self._filter_column("분석연도", year)

    # ------------------------------------------
    # 분기 필터
    # ------------------------------------------

    def by_quarter(self, quarter):
        return self._filter_column("분기", quarter)

    # ------------------------------------------
    # 특정 년분기 필터
    # ------------------------------------------

    def by_year_quarter(self, year, quarter):
        self.by_year(year)
        self.by_quarter(quarter)
        return self

    # ------------------------------------------
    # 자치구 필터
    # ------------------------------------------

    def by_district(self, district):
        return self._filter_column("자치구_코드_명", district)

    # ------------------------------------------
    # 행정동 필터
    # ------------------------------------------

    def by_dong(self, dong):
        return self._filter_column("행정동_코드_명", dong)

    # ------------------------------------------
    # 상권 구분 필터
    # ------------------------------------------

    def by_market_type(self, market_type):
        return self._filter_column("상권_구분_코드_명", market_type)

    # ------------------------------------------
    # 상권명 필터
    # ------------------------------------------

    def by_market(self, market):
        return self._filter_column("상권_코드_명", market)

    # ------------------------------------------
    # 업종 필터
    # ------------------------------------------

    def by_business(self, business):
        return self._filter_column("서비스_업종_코드_명", business)

    # ------------------------------------------
    # 공통 필터 기능
    # ------------------------------------------

    def _filter_column(self, column, value):
        if column not in self.df.columns:
            return self

        if isinstance(value, (list, tuple, set)):
            self.df = self.df[
                self.df[column].isin(value)
            ]
        else:
            self.df = self.df[
                self.df[column] == value
            ]

        return self

    # ------------------------------------------
    # 현재 필터 결과 반환
    # ------------------------------------------

    def get(self):
        return self.df.copy()

    # ------------------------------------------
    # 필터 결과 개수
    # ------------------------------------------

    def count(self):
        return len(self.df)


# ==========================================
# 3. 매출 분석 클래스
# ==========================================

class SalesAnalyzer:

    def __init__(self, df):
        self.df = df.copy()

    # ------------------------------------------
    # 전체 매출 금액
    # ------------------------------------------

    def get_total_sales(self):
        return self._sum("당월_매출_금액")

    # ------------------------------------------
    # 전체 매출 건수
    # ------------------------------------------

    def get_total_count(self):
        return self._sum("당월_매출_건수")

    # ------------------------------------------
    # 평균 객단가
    # ------------------------------------------

    def get_average_spending(self):
        sales = self.get_total_sales()
        count = self.get_total_count()

        if count == 0:
            return 0

        return sales / count

    # ------------------------------------------
    # 상권별 매출
    # ------------------------------------------

    def get_market_sales(self):
        return self._group_sales("상권_코드_명")

    # ------------------------------------------
    # 업종별 매출
    # ------------------------------------------

    def get_business_sales(self):
        return self._group_sales("서비스_업종_코드_명")

    # ------------------------------------------
    # 자치구별 매출
    # ------------------------------------------

    def get_district_sales(self):
        return self._group_sales("자치구_코드_명")

    # ------------------------------------------
    # 행정동별 매출
    # ------------------------------------------

    def get_dong_sales(self):
        return self._group_sales("행정동_코드_명")

    # ------------------------------------------
    # 상권 유형별 매출
    # ------------------------------------------

    def get_market_type_sales(self):
        return self._group_sales("상권_구분_코드_명")

    # ------------------------------------------
    # TOP N 상권
    # ------------------------------------------

    def get_top_markets(self, n=10):
        result = self.get_market_sales()
        return result.head(n)

    # ------------------------------------------
    # TOP N 업종
    # ------------------------------------------

    def get_top_businesses(self, n=10):
        result = self.get_business_sales()
        return result.head(n)

    # ------------------------------------------
    # 요일별 매출
    # ------------------------------------------

    def get_weekday_sales(self):
        columns = {
            "월요일": "월요일_매출_금액",
            "화요일": "화요일_매출_금액",
            "수요일": "수요일_매출_금액",
            "목요일": "목요일_매출_금액",
            "금요일": "금요일_매출_금액",
            "토요일": "토요일_매출_금액",
            "일요일": "일요일_매출_금액",
        }

        return self._column_summary(columns)

    # ------------------------------------------
    # 주중 / 주말 매출
    # ------------------------------------------

    def get_weekday_weekend_sales(self):
        return {
            "주중": self._sum("주중_매출_금액"),
            "주말": self._sum("주말_매출_금액"),
        }

    # ------------------------------------------
    # 시간대별 매출
    # ------------------------------------------

    def get_time_sales(self):
        columns = {
            "00~06": "시간대_00~06_매출_금액",
            "06~11": "시간대_06~11_매출_금액",
            "11~14": "시간대_11~14_매출_금액",
            "14~17": "시간대_14~17_매출_금액",
            "17~21": "시간대_17~21_매출_금액",
            "21~24": "시간대_21~24_매출_금액",
        }

        return self._column_summary(columns)

    # ------------------------------------------
    # 피크 시간대
    # ------------------------------------------

    def get_peak_time(self):
        result = self.get_time_sales()

        if len(result) == 0:
            return None

        peak = result.loc[
            result["매출금액"].idxmax()
        ]

        return (
            peak["구분"],
            peak["매출금액"]
        )

    # ------------------------------------------
    # 성별 매출
    # ------------------------------------------

    def get_gender_sales(self):
        return {
            "남성": self._sum("남성_매출_금액"),
            "여성": self._sum("여성_매출_금액"),
        }

    # ------------------------------------------
    # 성별 매출 비율
    # ------------------------------------------

    def get_gender_ratio(self):
        gender = self.get_gender_sales()

        total = gender["남성"] + gender["여성"]

        if total == 0:
            return {
                "남성": 0,
                "여성": 0
            }

        return {
            "남성": gender["남성"] / total * 100,
            "여성": gender["여성"] / total * 100,
        }

    # ------------------------------------------
    # 연령대별 매출
    # ------------------------------------------

    def get_age_sales(self):
        columns = {
            "10대": "연령대_10_매출_금액",
            "20대": "연령대_20_매출_금액",
            "30대": "연령대_30_매출_금액",
            "40대": "연령대_40_매출_금액",
            "50대": "연령대_50_매출_금액",
            "60대 이상": "연령대_60_이상_매출_금액",
        }

        return self._column_summary(columns)

    # ------------------------------------------
    # 주요 소비 연령층
    # ------------------------------------------

    def get_main_age_group(self):
        result = self.get_age_sales()

        if len(result) == 0:
            return None

        main_age = result.loc[
            result["매출금액"].idxmax()
        ]

        return (
            main_age["구분"],
            main_age["매출금액"]
        )

    # ------------------------------------------
    # 연도별 매출
    # ------------------------------------------

    def get_year_sales(self):
        if "분석연도" not in self.df.columns:
            return pd.DataFrame()

        result = (
            self.df
            .groupby("분석연도")["당월_매출_금액"]
            .sum()
            .reset_index()
            .sort_values("분석연도")
        )

        return result

    # ------------------------------------------
    # 연도별 성장률
    # ------------------------------------------

    def get_year_growth(self):
        result = self.get_year_sales()

        if len(result) == 0:
            return result

        result["전년대비_증감액"] = (
            result["당월_매출_금액"].diff()
        )

        result["전년대비_성장률"] = (
            result["당월_매출_금액"].pct_change() * 100
        )

        return result

    # ------------------------------------------
    # 분기별 매출
    # ------------------------------------------

    def get_quarter_sales(self):
        column = "기준_년분기_코드"

        if column not in self.df.columns:
            return pd.DataFrame()

        result = (
            self.df
            .groupby(column)["당월_매출_금액"]
            .sum()
            .reset_index()
            .sort_values(column)
        )

        return result

    # ------------------------------------------
    # 두 기간 비교
    # ------------------------------------------

    def compare_quarters(
            self,
            base_year,
            base_quarter,
            target_year,
            target_quarter
    ):
        required = [
            "분석연도",
            "분기",
            "당월_매출_금액",
            "당월_매출_건수",
            "주중_매출_금액",
            "주말_매출_금액"
        ]

        if any(column not in self.df.columns for column in required):
            return None

        periods = []

        for year, quarter in [
            (base_year, base_quarter),
            (target_year, target_quarter)
        ]:
            year = int(year)
            quarter = int(quarter)

            if quarter < 1 or quarter > 4:
                raise ValueError("분기는 1부터 4 사이여야 합니다.")

            period_df = (
                DataFilter(self.df)
                .by_year_quarter(year, quarter)
                .get()
            )

            if period_df.empty:
                return None

            analyzer = SalesAnalyzer(period_df)
            weekday_weekend = analyzer.get_weekday_weekend_sales()

            periods.append({
                "총매출": analyzer.get_total_sales(),
                "총건수": analyzer.get_total_count(),
                "평균객단가": analyzer.get_average_spending(),
                "주중매출": weekday_weekend["주중"],
                "주말매출": weekday_weekend["주말"]
            })

        base, target = periods
        result = []

        for name in [
            "총매출",
            "총건수",
            "평균객단가",
            "주중매출",
            "주말매출"
        ]:
            before = base[name]
            after = target[name]

            difference = after - before

            rate = (
                difference / before * 100
                if before != 0
                else None
            )

            result.append({
                "항목": name,
                "기준 기간": before,
                "비교 기간": after,
                "증감액": difference,
                "증감률(%)": rate
            })

        return pd.DataFrame(result)

    # ------------------------------------------
    # 면적당 매출
    # ------------------------------------------

    def get_sales_per_area(self, group="상권_코드_명"):
        required = [
            group,
            "당월_매출_금액",
            "영역_면적",
        ]

        for column in required:
            if column not in self.df.columns:
                return pd.DataFrame()

        temp = self.df.copy()

        temp["영역_면적"] = pd.to_numeric(
            temp["영역_면적"],
            errors="coerce"
        )

        result = (
            temp
            .groupby(group)
            .agg(
                총매출=("당월_매출_금액", "sum"),
                영역면적=("영역_면적", "first")
            )
            .reset_index()
        )

        result["면적당_매출"] = (
            result["총매출"] / result["영역면적"]
        )

        return result.sort_values(
            "면적당_매출",
            ascending=False
        )

    # ------------------------------------------
    # 주말 매출 비율
    # ------------------------------------------

    def get_weekend_ratio(self):
        total = self.get_total_sales()
        weekend = self._sum("주말_매출_금액")

        if total == 0:
            return 0

        return weekend / total * 100

    # ------------------------------------------
    # 지도용 데이터
    # ------------------------------------------

    def get_map_data(self):
        required = [
            "상권_코드_명",
            "당월_매출_금액",
            "경도",
            "위도",
        ]

        for column in required:
            if column not in self.df.columns:
                return pd.DataFrame()

        result = (
            self.df
            .groupby(
                ["상권_코드_명", "경도", "위도"],
                dropna=False
            )["당월_매출_금액"]
            .sum()
            .reset_index()
        )

        return result

    # ==========================================
    # 내부 공통 함수
    # ==========================================

    def _sum(self, column):
        if column not in self.df.columns:
            return 0

        data = pd.to_numeric(
            self.df[column],
            errors="coerce"
        )

        return data.sum()

    def _group_sales(self, group_column):
        if group_column not in self.df.columns:
            return pd.DataFrame()

        if "당월_매출_금액" not in self.df.columns:
            return pd.DataFrame()

        result = (
            self.df
            .groupby(group_column)["당월_매출_금액"]
            .sum()
            .reset_index()
            .sort_values(
                "당월_매출_금액",
                ascending=False
            )
        )

        return result

    def _column_summary(self, columns):
        result = []

        for name, column in columns.items():
            if column not in self.df.columns:
                continue

            result.append({
                "구분": name,
                "매출금액": self._sum(column)
            })

        return pd.DataFrame(result)
