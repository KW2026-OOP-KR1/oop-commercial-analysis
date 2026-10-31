from pathlib import Path

import pandas as pd


# ==========================================
# 1. 파일 경로 설정
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "output" / "combined_2023_2025.csv"


# ==========================================
# 2. 데이터 불러오기
# ==========================================

def load_data(file_path=DATA_PATH):
    """
    전처리 완료된 통합 CSV 파일을 불러옵니다.

    기본 사용:
        df = load_data()

    다른 파일 사용:
        df = load_data("output/processed_2025.csv")
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {file_path}"
        )

    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"[데이터 로드 성공] {file_path.name}")
            print(f"행 개수: {len(df):,}개")
            return df

        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"CSV 인코딩을 확인해 주세요: {file_path}"
    )


# ==========================================
# 3. 데이터 필터 클래스
# ==========================================

class DataFilter:

    def __init__(self, df):
        self.original_df = df.copy()
        self.df = df.copy()


    # ------------------------------------------
    # 필터 초기화
    # ------------------------------------------

    def reset(self):
        """
        적용된 모든 필터를 초기화합니다.

        사용법:
            filter.reset()
        """

        self.df = self.original_df.copy()

        return self


    # ------------------------------------------
    # 분석 연도 필터
    # ------------------------------------------

    def by_year(self, year):
        """
        분석연도를 기준으로 데이터를 필터링합니다.

        사용법:
            filter.by_year(2025)

        여러 연도:
            filter.by_year([2024, 2025])
        """

        return self._filter_column("분석연도", year)



    # ------------------------------------------
    # 분기 필터
    # ------------------------------------------

    def by_quarter(self, quarter):
        """
        분기를 기준으로 데이터를 필터링합니다.

        기준_년분기_코드 예시:
            20231 -> 2023년 1분기
            20232 -> 2023년 2분기
            20241 -> 2024년 1분기

        사용법:
            filter.by_quarter(1)

        여러 분기:
            filter.by_quarter([1, 2])
        """

        column = "기준_년분기_코드"

        if column not in self.df.columns:
            return self

        quarter_data = (
            self.df[column]
            .astype(str)
            .str[-1]
        )

        if isinstance(quarter, (list, tuple, set)):

            quarters = [
                str(value)
                for value in quarter
            ]

            self.df = self.df[
                quarter_data.isin(quarters)
            ]

        else:

            self.df = self.df[
                quarter_data == str(quarter)
            ]

        return self


    # ------------------------------------------
    # 특정 년분기 필터
    # ------------------------------------------

    def by_year_quarter(self, year, quarter):
        """
        특정 연도 + 특정 분기를 필터링합니다.

        사용법:
            filter.by_year_quarter(2025, 2)

        의미:
            2025년 2분기 데이터만 선택
        """

        self.by_year(year)
        self.by_quarter(quarter)

        return self


    # ------------------------------------------
    # 자치구 필터
    # ------------------------------------------

    def by_district(self, district):
        """
        자치구를 기준으로 필터링합니다.

        사용법:
            filter.by_district("강남구")

        여러 자치구:
            filter.by_district(["강남구", "서초구"])
        """

        return self._filter_column(
            "자치구_코드_명",
            district
        )


    # ------------------------------------------
    # 행정동 필터
    # ------------------------------------------

    def by_dong(self, dong):
        """
        행정동을 기준으로 필터링합니다.

        사용법:
            filter.by_dong("역삼1동")

        여러 행정동:
            filter.by_dong(["역삼1동", "역삼2동"])
        """

        return self._filter_column(
            "행정동_코드_명",
            dong
        )


    # ------------------------------------------
    # 상권 구분 필터
    # ------------------------------------------

    def by_market_type(self, market_type):
        """
        상권 유형을 기준으로 필터링합니다.

        예:
            골목상권
            발달상권
            전통시장
            관광특구

        사용법:
            filter.by_market_type("골목상권")
        """

        return self._filter_column(
            "상권_구분_코드_명",
            market_type
        )


    # ------------------------------------------
    # 상권명 필터
    # ------------------------------------------

    def by_market(self, market):
        """
        특정 상권을 선택합니다.

        사용법:
            filter.by_market("강남역")

        여러 상권:
            filter.by_market(["강남역", "홍대입구역"])
        """

        return self._filter_column(
            "상권_코드_명",
            market
        )


    # ------------------------------------------
    # 업종 필터
    # ------------------------------------------

    def by_business(self, business):
        """
        서비스 업종을 기준으로 필터링합니다.

        사용법:
            filter.by_business("한식음식점")

        여러 업종:
            filter.by_business([
                "한식음식점",
                "커피-음료"
            ])
        """

        return self._filter_column(
            "서비스_업종_코드_명",
            business
        )


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
        """
        현재 필터링된 DataFrame을 반환합니다.

        사용법:
            result = filter.get()
        """

        return self.df.copy()


    # ------------------------------------------
    # 필터 결과 개수
    # ------------------------------------------

    def count(self):
        """
        현재 필터 결과의 행 개수를 반환합니다.

        사용법:
            print(filter.count())
        """

        return len(self.df)


# ==========================================
# 4. 매출 분석 클래스
# ==========================================

class SalesAnalyzer:

    def __init__(self, df):
        self.df = df.copy()


    # ------------------------------------------
    # 전체 매출 금액
    # ------------------------------------------

    def get_total_sales(self):
        """
        현재 데이터의 전체 매출 금액을 반환합니다.

        사용법:
            analyzer.get_total_sales()
        """

        return self._sum(
            "당월_매출_금액"
        )


    # ------------------------------------------
    # 전체 매출 건수
    # ------------------------------------------

    def get_total_count(self):
        """
        전체 거래 건수를 반환합니다.

        사용법:
            analyzer.get_total_count()
        """

        return self._sum(
            "당월_매출_건수"
        )


    # ------------------------------------------
    # 평균 객단가
    # ------------------------------------------

    def get_average_spending(self):
        """
        평균 객단가를 계산합니다.

        객단가 =
            전체 매출 금액 / 전체 매출 건수

        사용법:
            analyzer.get_average_spending()
        """

        sales = self.get_total_sales()
        count = self.get_total_count()

        if count == 0:
            return 0

        return sales / count


    # ------------------------------------------
    # 상권별 매출
    # ------------------------------------------

    def get_market_sales(self):
        """
        상권별 전체 매출을 반환합니다.

        사용법:
            analyzer.get_market_sales()
        """

        return self._group_sales(
            "상권_코드_명"
        )


    # ------------------------------------------
    # 업종별 매출
    # ------------------------------------------

    def get_business_sales(self):
        """
        업종별 전체 매출을 반환합니다.

        사용법:
            analyzer.get_business_sales()
        """

        return self._group_sales(
            "서비스_업종_코드_명"
        )


    # ------------------------------------------
    # 자치구별 매출
    # ------------------------------------------

    def get_district_sales(self):
        """
        자치구별 전체 매출을 반환합니다.

        사용법:
            analyzer.get_district_sales()
        """

        return self._group_sales(
            "자치구_코드_명"
        )


    # ------------------------------------------
    # 행정동별 매출
    # ------------------------------------------

    def get_dong_sales(self):
        """
        행정동별 매출을 반환합니다.
        """

        return self._group_sales(
            "행정동_코드_명"
        )


    # ------------------------------------------
    # 상권 유형별 매출
    # ------------------------------------------

    def get_market_type_sales(self):
        """
        골목상권, 발달상권 등
        상권 유형별 매출을 반환합니다.
        """

        return self._group_sales(
            "상권_구분_코드_명"
        )


    # ------------------------------------------
    # TOP N 상권
    # ------------------------------------------

    def get_top_markets(self, n=10):
        """
        매출이 높은 상권 TOP N을 반환합니다.

        사용법:
            analyzer.get_top_markets()

            analyzer.get_top_markets(5)
        """

        result = self.get_market_sales()

        return result.head(n)


    # ------------------------------------------
    # TOP N 업종
    # ------------------------------------------

    def get_top_businesses(self, n=10):
        """
        매출이 높은 업종 TOP N을 반환합니다.

        사용법:
            analyzer.get_top_businesses(10)
        """

        result = self.get_business_sales()

        return result.head(n)


    # ------------------------------------------
    # 요일별 매출
    # ------------------------------------------

    def get_weekday_sales(self):
        """
        월요일부터 일요일까지의
        매출 금액을 반환합니다.

        사용법:
            analyzer.get_weekday_sales()
        """

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
        """
        주중과 주말 매출을 비교합니다.

        사용법:
            analyzer.get_weekday_weekend_sales()
        """

        return {
            "주중": self._sum("주중_매출_금액"),
            "주말": self._sum("주말_매출_금액"),
        }


    # ------------------------------------------
    # 시간대별 매출
    # ------------------------------------------

    def get_time_sales(self):
        """
        시간대별 매출을 반환합니다.

        사용법:
            analyzer.get_time_sales()
        """

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
        """
        매출이 가장 높은 시간대를 반환합니다.

        사용법:
            analyzer.get_peak_time()

        반환 예:
            ("17~21", 300000000)
        """

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
        """
        남성 / 여성 매출을 반환합니다.

        사용법:
            analyzer.get_gender_sales()
        """

        return {
            "남성": self._sum(
                "남성_매출_금액"
            ),

            "여성": self._sum(
                "여성_매출_금액"
            ),
        }


    # ------------------------------------------
    # 성별 매출 비율
    # ------------------------------------------

    def get_gender_ratio(self):
        """
        남녀 매출 비율을 계산합니다.

        사용법:
            analyzer.get_gender_ratio()
        """

        gender = self.get_gender_sales()

        total = (
            gender["남성"]
            + gender["여성"]
        )

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
        """
        연령대별 매출을 반환합니다.

        사용법:
            analyzer.get_age_sales()
        """

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
        """
        가장 매출이 높은 연령대를 반환합니다.

        사용법:
            analyzer.get_main_age_group()
        """

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
        """
        연도별 전체 매출을 반환합니다.

        사용법
            analyzer.get_year_sales()
        """

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
        """
        전년 대비 매출 성장률을 계산합니다.

        성장률 =
            (현재연도 - 이전연도)
            / 이전연도 * 100

        사용법:
            analyzer.get_year_growth()
        """

        result = self.get_year_sales()

        if len(result) == 0:
            return result

        result["전년대비_증감액"] = (
            result["당월_매출_금액"]
            .diff()
        )

        result["전년대비_성장률"] = (
            result["당월_매출_금액"]
            .pct_change()
            * 100
        )

        return result


    # ------------------------------------------
    # 분기별 매출
    # ------------------------------------------

    def get_quarter_sales(self):
        """
        기준_년분기_코드 기준으로
        분기별 매출을 반환합니다.

        사용법:
            analyzer.get_quarter_sales()
        """

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
    # 두 기간 매출 비교
    # ------------------------------------------

    def compare_quarters(self, base_year, base_quarter, target_year, target_quarter):
        """
        서로 다른 두 연도 및 분기의 매출을 비교합니다.

        사용법:
            analyzer.compare_quarters(2024, 1, 2025, 3)

        반환값:
            기준 및 비교 기간의 매출, 건수, 객단가,
            각 지표의 증감액과 증감률
        """

        required = ["분석연도", "기준_년분기_코드", "당월_매출_금액", "당월_매출_건수"]

        # 비교 분석에 필요한 컬럼 검사
        if any(column not in self.df.columns for column in required):
            return None

        periods = []

        for year, quarter in [(base_year, base_quarter), (target_year, target_quarter)]:
            year = int(year)
            quarter = int(quarter)

            if quarter < 1 or quarter > 4:
                raise ValueError("분기는 1부터 4 사이여야 합니다.")

            # DataFilter에 연도 및 분기 필터링을 위임
            data_filter = DataFilter(self.df)
            period_df = data_filter.by_year_quarter(year, quarter).get()

            if period_df.empty:
                return None

            # 추출된 기간의 매출 분석은 SalesAnalyzer가 담당
            analyzer = SalesAnalyzer(period_df)

            periods.append({
                "총매출": analyzer.get_total_sales(),
                "총건수": analyzer.get_total_count(),
                "평균객단가": analyzer.get_average_spending()
            })

        base, target = periods
        result = []

        # 두 기간의 매출 지표 비교
        for name in ["총매출", "총건수", "평균객단가"]:
            before = base[name]
            after = target[name]

            difference = after - before
            rate = difference / before * 100 if before != 0 else None

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
        """
        상권 면적 대비 매출을 계산합니다.

        기본:
            analyzer.get_sales_per_area()

        자치구 기준:
            analyzer.get_sales_per_area("자치구_코드_명")

        주의:
            동일 상권의 영역_면적이
            여러 행에 반복되어 있으므로
            면적을 단순 합산하지 않고
            첫 번째 값을 사용합니다.
        """

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
                총매출=(
                    "당월_매출_금액",
                    "sum"
                ),

                영역면적=(
                    "영역_면적",
                    "first"
                )
            )
            .reset_index()
        )

        result["면적당_매출"] = (
            result["총매출"]
            / result["영역면적"]
        )

        return result.sort_values(
            "면적당_매출",
            ascending=False
        )


    # ------------------------------------------
    # 주말 매출 비율
    # ------------------------------------------

    def get_weekend_ratio(self):
        """
        전체 매출 중 주말 매출 비율을 계산합니다.

        사용법:
            analyzer.get_weekend_ratio()

        반환값:
            퍼센트 값
        """

        total = self.get_total_sales()

        weekend = self._sum(
            "주말_매출_금액"
        )

        if total == 0:
            return 0

        return weekend / total * 100


    # ------------------------------------------
    # 지도용 데이터
    # ------------------------------------------

    def get_map_data(self):
        """
        지도 시각화에 사용할 데이터를 반환합니다.

        주요 컬럼:
            상권명
            매출
            경도
            위도

        사용법:
            analyzer.get_map_data()
        """

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
                [
                    "상권_코드_명",
                    "경도",
                    "위도",
                ],
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
