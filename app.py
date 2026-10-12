from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from filter_analyzer import DATA_PATH, BASE_DIR, DataFilter, SalesAnalyzer, load_data
from viz_plotly import PlotlyVisualizer
from viz_pydeck import PydeckVisualizer

ALL = "전체"
PROCESSED_FILES = [BASE_DIR / "output" / f"processed_{year}.csv" for year in (2023, 2024, 2025)]

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
    "00~06시": "시간대_00~06_매출_금액",
    "06~11시": "시간대_06~11_매출_금액",
    "11~14시": "시간대_11~14_매출_금액",
    "14~17시": "시간대_14~17_매출_금액",
    "17~21시": "시간대_17~21_매출_금액",
    "21~24시": "시간대_21~24_매출_금액",
}
GENDERS = [ALL, "남성", "여성"]
AGE_GROUPS = [ALL, "10대", "20대", "30대", "40대", "50대", "60대 이상"]


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_dashboard_data() -> pd.DataFrame:
    """통합 CSV가 있으면 사용하고, 없으면 연도별 전처리 CSV를 합쳐서 사용한다."""
    if DATA_PATH.exists():
        return load_data()
    return pd.concat([load_data(path) for path in PROCESSED_FILES], ignore_index=True)


def make_exclusive(changed_key: str, other_key: str) -> None:
    """상호 배타 필터: 방금 바꾼 값이 '전체'가 아니면 반대쪽 필터를 '전체'로 되돌린다."""
    if st.session_state[changed_key] != ALL:
        st.session_state[other_key] = ALL


@dataclass
class FilterSelection:
    """사이드바에서 사용자가 선택한 조건 묶음"""
    year: int | str = ALL
    quarter: int | str = ALL
    district: str = ALL
    market_type: str = ALL
    market: str = ALL
    businesses: list[str] = field(default_factory=list)
    gender: str = ALL
    age_group: str = ALL
    compare_enabled: bool = False
    base_year: int | None = None
    base_quarter: int | None = None

    def has_single_period(self) -> bool:
        """연도와 분기가 모두 하나로 정해졌는지 여부 (기간 비교 가능 조건)"""
        return self.year != ALL and self.quarter != ALL


class SidebarFilterPanel:
    """사이드바 필터 UI를 그리고 선택값을 FilterSelection으로 반환하는 클래스"""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def render(self) -> FilterSelection:
        """사이드바 위젯을 그리고 사용자의 선택을 반환한다."""
        st.sidebar.header("필터 조건")
        selection = FilterSelection()

        self._render_period(selection)
        self._render_region(selection)
        self._render_business(selection)
        self._render_customer(selection)
        return selection

    def _render_period(self, selection: FilterSelection) -> None:
        st.sidebar.subheader("기간")
        years = sorted(self.df["분석연도"].dropna().unique().tolist())
        selection.year = st.sidebar.selectbox("연도", [ALL] + years, index=len(years))
        selection.quarter = st.sidebar.selectbox("분기", [ALL, 1, 2, 3, 4])

        selection.compare_enabled = st.sidebar.checkbox(
            "다른 기간과 비교", disabled=not selection.has_single_period(),
            help="연도와 분기를 하나씩 선택하면 사용할 수 있습니다.",
        )
        if selection.compare_enabled:
            selection.base_year = st.sidebar.selectbox("기준 연도", years)
            selection.base_quarter = st.sidebar.selectbox("기준 분기", [1, 2, 3, 4])

    def _render_region(self, selection: FilterSelection) -> None:
        st.sidebar.subheader("지역")
        districts = sorted(self.df["자치구_코드_명"].dropna().unique().tolist())
        selection.district = st.sidebar.selectbox("자치구", [ALL] + districts)

        types = sorted(self.df["상권_구분_코드_명"].dropna().unique().tolist())
        selection.market_type = st.sidebar.selectbox("상권 유형", [ALL] + types)

        # 상권 목록은 앞에서 고른 자치구·상권 유형에 속한 것만 보여준다.
        region_filter = DataFilter(self.df[["자치구_코드_명", "상권_구분_코드_명", "상권_코드_명"]])
        if selection.district != ALL:
            region_filter.by_district(selection.district)
        if selection.market_type != ALL:
            region_filter.by_market_type(selection.market_type)
        markets = sorted(region_filter.get()["상권_코드_명"].dropna().unique().tolist())
        selection.market = st.sidebar.selectbox("상권", [ALL] + markets)

    def _render_business(self, selection: FilterSelection) -> None:
        st.sidebar.subheader("업종")
        businesses = sorted(self.df["서비스_업종_코드_명"].dropna().unique().tolist())
        selection.businesses = st.sidebar.multiselect("서비스 업종 (미선택 시 전체)", businesses)

    def _render_customer(self, selection: FilterSelection) -> None:
        st.sidebar.subheader("고객층")
        st.sidebar.caption("성별과 연령대는 동시에 선택할 수 없습니다.")
        selection.gender = st.sidebar.selectbox(
            "성별", GENDERS, key="gender",
            on_change=make_exclusive, args=("gender", "age_group"),
        )
        selection.age_group = st.sidebar.selectbox(
            "연령대", AGE_GROUPS, key="age_group",
            on_change=make_exclusive, args=("age_group", "gender"),
        )


class DashboardApp:
    """데이터 로드 → 필터 → 분석 → 시각화를 연결하는 대시보드 통합 클래스"""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self) -> None:
        """대시보드 전체를 그린다."""
        st.title("서울시 상권 분석 3D 인터랙티브 대시보드")

        selection = SidebarFilterPanel(self.df).render()
        region_df = self._apply_region_filters(selection)
        current_df = self._apply_period_filter(region_df, selection.year, selection.quarter)

        if current_df.empty:
            st.warning("선택한 조건에 해당하는 데이터가 없습니다. 필터를 변경해 주세요.")
            return

        self._render_metrics(current_df, region_df, selection)
        st.divider()

        tab_sales, tab_customer, tab_compare, tab_map, tab_data = st.tabs(
            ["매출 분석", "고객층 분석", "기간 비교", "3D 지도", "데이터 상세"]
        )
        visualizer = PlotlyVisualizer(current_df)
        with tab_sales:
            self._render_sales_charts(current_df, visualizer)
        with tab_customer:
            self._render_customer_charts(current_df, visualizer, selection)
        with tab_compare:
            self._render_comparison(region_df, selection)
        with tab_map:
            self._render_map(current_df)
        with tab_data:
            st.caption(f"필터링된 데이터 {len(current_df):,}행 (상위 1,000행 표시)")
            st.dataframe(current_df.head(1000), width="stretch")

    # ------------------------------------------
    # 필터 적용
    # ------------------------------------------

    def _apply_region_filters(self, selection: FilterSelection) -> pd.DataFrame:
        """기간을 제외한 필터(지역·업종)를 적용한다. 기간 비교 시 두 기간에 같은 조건을 쓰기 위해 분리했다."""
        data_filter = DataFilter(self.df)
        if selection.district != ALL:
            data_filter.by_district(selection.district)
        if selection.market_type != ALL:
            data_filter.by_market_type(selection.market_type)
        if selection.market != ALL:
            data_filter.by_market(selection.market)
        if selection.businesses:
            data_filter.by_business(selection.businesses)
        return data_filter.get()

    @staticmethod
    def _apply_period_filter(df: pd.DataFrame, year: int | str, quarter: int | str) -> pd.DataFrame:
        """연도·분기 필터를 적용한다. '전체'는 건너뛴다."""
        data_filter = DataFilter(df)
        if year != ALL:
            data_filter.by_year(year)
        if quarter != ALL:
            data_filter.by_quarter(quarter)
        return data_filter.get()

    # ------------------------------------------
    # 핵심 지표
    # ------------------------------------------

    def _render_metrics(self, current_df: pd.DataFrame, region_df: pd.DataFrame,
                        selection: FilterSelection) -> None:
        """총매출·총건수·평균 객단가·상권 수 카드. 기간 비교 시 기준 기간 대비 증감을 함께 표시한다."""
        current = SalesAnalyzer(current_df)
        base = None
        if selection.compare_enabled:
            base_df = self._apply_period_filter(region_df, selection.base_year, selection.base_quarter)
            base = SalesAnalyzer(base_df)
            st.caption(
                f"증감 기준: {selection.base_year}년 {selection.base_quarter}분기 → "
                f"{selection.year}년 {selection.quarter}분기"
            )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("총매출", f"{current.get_total_sales():,.0f} 원",
                    self._delta(current.get_total_sales(), base.get_total_sales() if base else None))
        col2.metric("총매출건수", f"{current.get_total_count():,.0f} 건",
                    self._delta(current.get_total_count(), base.get_total_count() if base else None))
        col3.metric("평균 객단가", f"{current.get_average_spending():,.0f} 원",
                    self._delta(current.get_average_spending(), base.get_average_spending() if base else None))
        col4.metric("분석 대상 상권 수", f"{current_df['상권_코드'].nunique():,} 개")

    @staticmethod
    def _delta(current_value: float, base_value: float | None) -> str | None:
        """기준값 대비 증감률 문자열. 기준값이 없거나 0이면 증감을 표시하지 않는다."""
        if base_value is None:
            return None
        if base_value == 0:
            return "기준값 0 (비교 불가)"
        return f"{(current_value - base_value) / base_value * 100:+.1f}%"

    # ------------------------------------------
    # 차트
    # ------------------------------------------

    @staticmethod
    def _render_sales_charts(current_df: pd.DataFrame, visualizer: PlotlyVisualizer) -> None:
        """상권 TOP 10, 요일별(월~일), 주중/주말, 시간대별 매출"""
        st.subheader("매출 상위 상권 TOP 10")
        top_markets = SalesAnalyzer(current_df).get_top_markets(10)
        st.bar_chart(top_markets, x="상권_코드_명", y="당월_매출_금액",
                     x_label="상권", y_label="매출 금액 (원)", horizontal=True)

        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(visualizer.create_weekday_sales(), width="stretch")
        with right:
            st.plotly_chart(visualizer.create_weekday_weekend_sales(), width="stretch")
        st.plotly_chart(visualizer.create_time_sales(), width="stretch")

    @staticmethod
    def _render_customer_charts(current_df: pd.DataFrame, visualizer: PlotlyVisualizer,
                                selection: FilterSelection) -> None:
        """성별·연령대 매출. 선택한 고객층의 매출과 비중을 강조한다."""
        analyzer = SalesAnalyzer(current_df)
        total = analyzer.get_total_sales()

        if selection.gender != ALL:
            value = analyzer.get_gender_sales()[selection.gender]
            st.metric(f"{selection.gender} 매출", f"{value:,.0f} 원",
                      f"전체 매출의 {value / total * 100:.1f}%" if total else None, delta_color="off")
        elif selection.age_group != ALL:
            age_df = analyzer.get_age_sales()
            value = age_df.loc[age_df["구분"] == selection.age_group, "매출금액"].sum()
            st.metric(f"{selection.age_group} 매출", f"{value:,.0f} 원",
                      f"전체 매출의 {value / total * 100:.1f}%" if total else None, delta_color="off")

        left, right = st.columns(2)
        with left:
            st.plotly_chart(visualizer.create_gender_sales(), width="stretch")
        with right:
            st.plotly_chart(visualizer.create_age_sales(), width="stretch")

    @staticmethod
    def _render_comparison(region_df: pd.DataFrame, selection: FilterSelection) -> None:
        """기준 기간 대비 비교 기간의 지표 증감 (워터폴 차트 + 표)"""
        if not selection.compare_enabled:
            st.info("사이드바에서 연도·분기를 하나씩 선택한 뒤 '다른 기간과 비교'를 켜 주세요.")
            return
        waterfall, table = PlotlyVisualizer(region_df).create_quarter_comparison(
            selection.base_year, selection.base_quarter, selection.year, selection.quarter
        )
        st.plotly_chart(waterfall, width="stretch")
        st.plotly_chart(table, width="stretch")

    @staticmethod
    def _render_map(current_df: pd.DataFrame) -> None:
        """요일 또는 시간대를 드롭다운으로 골라 3D 지도를 그린다. 요일과 시간대는 동시에 적용하지 않는다."""
        st.caption("요일과 시간대는 동시에 선택할 수 없습니다. 기둥 높이 단위: 억 원")
        col1, col2 = st.columns(2)
        weekday = col1.selectbox(
            "요일", [ALL] + list(WEEKDAY_COLUMNS), key="map_weekday",
            on_change=make_exclusive, args=("map_weekday", "map_time"),
        )
        time_slot = col2.selectbox(
            "시간대", [ALL] + list(TIME_COLUMNS), key="map_time",
            on_change=make_exclusive, args=("map_time", "map_weekday"),
        )

        value_column = "당월_매출_금액"
        if weekday != ALL:
            value_column = WEEKDAY_COLUMNS[weekday]
        elif time_slot != ALL:
            value_column = TIME_COLUMNS[time_slot]

        # 지도용 상권별 합산은 SalesAnalyzer.get_map_data()를 그대로 쓰고,
        # 선택한 요일/시간대 컬럼을 합산 대상으로 넘긴다.
        source = current_df.assign(당월_매출_금액=current_df[value_column])
        map_df = SalesAnalyzer(source).get_map_data().dropna(subset=["경도", "위도"])
        if map_df.empty:
            st.info("지도에 표시할 위치 데이터가 없습니다.")
            return
        # PydeckVisualizer는 '매출금액' 컬럼을 높이로 사용한다.
        map_df["매출금액"] = map_df["당월_매출_금액"] / 1e8
        st.pydeck_chart(PydeckVisualizer(map_df).create_column_map())


st.set_page_config(page_title="서울시 상권 분석 대시보드", layout="wide")
DashboardApp(load_dashboard_data()).run()
