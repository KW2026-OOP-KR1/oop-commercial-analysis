from dataclasses import dataclass, field

import pandas as pd
import plotly.graph_objects as go
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
CHART_HEIGHT = 430
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

    @staticmethod
    def _box(title: str):
        """필터 그룹을 테두리 있는 네모 박스로 묶는다."""
        box = st.sidebar.container(border=True)
        box.markdown(f"**{title}**")
        return box

    def _render_period(self, selection: FilterSelection) -> None:
        box = self._box("📅 기간")
        years = sorted(self.df["분석연도"].dropna().unique().tolist())
        col1, col2 = box.columns(2)
        selection.year = col1.selectbox("연도", [ALL] + years, index=len(years))
        selection.quarter = col2.selectbox("분기", [ALL, 1, 2, 3, 4])

        selection.compare_enabled = box.checkbox(
            "다른 기간과 비교", disabled=not selection.has_single_period(),
            help="연도와 분기를 하나씩 선택하면 사용할 수 있습니다.",
        )
        if selection.compare_enabled:
            col1, col2 = box.columns(2)
            selection.base_year = col1.selectbox("기준 연도", years)
            selection.base_quarter = col2.selectbox("기준 분기", [1, 2, 3, 4])

    def _render_region(self, selection: FilterSelection) -> None:
        box = self._box("📍 지역")
        districts = sorted(self.df["자치구_코드_명"].dropna().unique().tolist())
        selection.district = box.selectbox("자치구", [ALL] + districts)

        types = sorted(self.df["상권_구분_코드_명"].dropna().unique().tolist())
        selection.market_type = box.selectbox("상권 유형", [ALL] + types)

        # 상권 목록은 앞에서 고른 자치구·상권 유형에 속한 것만 보여준다.
        region_filter = DataFilter(self.df[["자치구_코드_명", "상권_구분_코드_명", "상권_코드_명"]])
        if selection.district != ALL:
            region_filter.by_district(selection.district)
        if selection.market_type != ALL:
            region_filter.by_market_type(selection.market_type)
        markets = sorted(region_filter.get()["상권_코드_명"].dropna().unique().tolist())
        selection.market = box.selectbox("상권", [ALL] + markets)

    def _render_business(self, selection: FilterSelection) -> None:
        box = self._box("🏪 업종")
        businesses = sorted(self.df["서비스_업종_코드_명"].dropna().unique().tolist())
        selection.businesses = box.multiselect("서비스 업종 (미선택 시 전체)", businesses)

    def _render_customer(self, selection: FilterSelection) -> None:
        box = self._box("👥 고객층")
        box.caption("성별과 연령대는 동시에 선택할 수 없습니다.")
        selection.gender = box.selectbox(
            "성별", GENDERS, key="gender",
            on_change=make_exclusive, args=("gender", "age_group"),
        )
        selection.age_group = box.selectbox(
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
            st.caption(self._describe_selection(selection))
            st.warning("선택한 조건에 해당하는 데이터가 없습니다. 필터를 변경해 주세요.")
            return

        visualizer = PlotlyVisualizer(current_df)

        # 메인 화면: 핵심 지표(제목 바로 아래) → 3D 지도(크게) → 동향 그래프 | 막대그래프(나란히)
        self._render_metrics(current_df, region_df, selection)
        st.caption(self._describe_selection(selection))

        with st.container(border=True):
            self._render_map(current_df)

        left, right = st.columns(2)
        with left.container(border=True):
            self._render_trend_chart(visualizer)
        with right.container(border=True):
            self._render_bar_chart(visualizer, selection)
        st.divider()

        st.subheader("상세 분석")
        tab_market, tab_compare, tab_data = st.tabs(["상권 TOP 10", "기간 비교", "데이터 상세"])
        with tab_market:
            self._render_top_markets(current_df)
        with tab_compare:
            self._render_comparison(region_df, selection)
        with tab_data:
            st.caption(f"필터링된 데이터 {len(current_df):,}행 (상위 1,000행 표시)")
            st.dataframe(current_df.head(1000), width="stretch")

    @staticmethod
    def _describe_selection(selection: FilterSelection) -> str:
        """현재 적용 중인 필터를 한 줄 요약으로 만든다."""
        year = "전체 연도" if selection.year == ALL else f"{selection.year}년"
        quarter = "전체 분기" if selection.quarter == ALL else f"{selection.quarter}분기"
        parts = [f"{year} {quarter}"]
        parts += [value for value in (selection.district, selection.market_type, selection.market) if value != ALL]
        parts.append(", ".join(selection.businesses) if selection.businesses else "전체 업종")
        if selection.gender != ALL:
            parts.append(selection.gender)
        if selection.age_group != ALL:
            parts.append(selection.age_group)
        return "📌 현재 조건: " + " · ".join(parts)

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

        customer = self._customer_sales(current, selection)
        cols = st.columns(5 if customer else 4)
        cols[0].metric("총매출", f"{current.get_total_sales():,.0f} 원",
                       self._delta(current.get_total_sales(), base.get_total_sales() if base else None))
        cols[1].metric("총매출건수", f"{current.get_total_count():,.0f} 건",
                       self._delta(current.get_total_count(), base.get_total_count() if base else None))
        cols[2].metric("평균 객단가", f"{current.get_average_spending():,.0f} 원",
                       self._delta(current.get_average_spending(), base.get_average_spending() if base else None))
        cols[3].metric("분석 대상 상권 수", f"{current_df['상권_코드'].nunique():,} 개")
        if customer:
            label, value = customer
            total = current.get_total_sales()
            cols[4].metric(f"{label} 매출", f"{value:,.0f} 원",
                           f"전체 매출의 {value / total * 100:.1f}%" if total else None, delta_color="off")

    @staticmethod
    def _customer_sales(analyzer: SalesAnalyzer, selection: FilterSelection) -> tuple[str, float] | None:
        """사이드바에서 고른 성별 또는 연령대의 매출. 선택하지 않았으면 None"""
        if selection.gender != ALL:
            return selection.gender, analyzer.get_gender_sales()[selection.gender]
        if selection.age_group != ALL:
            age_df = analyzer.get_age_sales()
            return selection.age_group, age_df.loc[age_df["구분"] == selection.age_group, "매출금액"].sum()
        return None

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
    def _show_chart(fig, key: str) -> None:
        """나란히 놓인 두 차트의 높이를 맞춰서 그린다."""
        fig.update_layout(height=CHART_HEIGHT, margin=dict(t=60, b=40, l=40, r=20))
        st.plotly_chart(fig, width="stretch", key=key)

    def _render_trend_chart(self, visualizer: PlotlyVisualizer) -> None:
        """동향 그래프(꺾은선): 시간대별 또는 요일별(월~일), 최고·최저 지점 강조"""
        st.markdown("**📈 매출 동향**")
        trend = st.segmented_control("기준", ["시간대별", "요일별"], default="시간대별",
                                     key="trend_type", label_visibility="collapsed") or "시간대별"
        if trend == "시간대별":
            fig = visualizer.create_time_sales()
        else:
            fig = self._weekday_trend_figure(visualizer)
        self._show_chart(fig, "main_trend_chart")

    # 차트는 동향(꺾은선)·막대 두 종류만 쓴다. PlotlyVisualizer의 요일(레이더)·주중/주말(도넛)
    # 차트 대신, 같은 클래스의 렌더러와 SalesAnalyzer 집계 결과로 꺾은선·막대를 그린다.

    @staticmethod
    def _weekday_trend_figure(visualizer: PlotlyVisualizer) -> go.Figure:
        """요일별(월~일) 매출 꺾은선. 최고·최저 요일(요일 편중도)을 굵은 점으로 표시"""
        weekday_df = visualizer.analyzer.get_weekday_sales()
        if weekday_df.empty:
            return go.Figure().update_layout(title="요일별 매출 데이터가 없습니다.", template="plotly_white")

        labels = weekday_df["구분"].tolist()
        values = weekday_df["매출금액"].tolist()
        fig = visualizer.line_renderer.render(
            labels, values, "요일별(월~일) 매출 추이 및 최고/최저 요일",
            x_title="요일", y_title="매출 금액 (원)",
            hovertemplate="%{x}<br>매출: %{y:,.0f}원<extra></extra>",
        )
        for index, name, color, position in (
            (values.index(max(values)), "최고 요일", "red", "top center"),
            (values.index(min(values)), "최저 요일", "blue", "bottom center"),
        ):
            fig.add_trace(go.Scatter(
                x=[labels[index]], y=[values[index]], mode="markers+text", name=name,
                text=[name], textposition=position,
                marker=dict(size=14, color=color, symbol="diamond"),
                hovertemplate=name + ": %{x}<br>매출: %{y:,.0f}원<extra></extra>",
            ))
        return fig

    @staticmethod
    def _weekday_weekend_bar_figure(visualizer: PlotlyVisualizer) -> go.Figure:
        """주중/주말 매출 막대. 마우스를 올리면 매출액과 비중 표시"""
        sales = visualizer.analyzer.get_weekday_weekend_sales()
        total = sum(sales.values())
        ratios = [value / total * 100 if total else 0 for value in sales.values()]
        return visualizer.bar_renderer.render(
            list(sales.keys()), list(sales.values()), "주중 / 주말 매출 및 비중 비교",
            x_title="구분", y_title="매출 금액 (원)",
            texttemplate="%{customdata:.1f}%", customdata=ratios,
            hovertemplate="%{x}<br>매출: %{y:,.0f}원<br>비중: %{customdata:.1f}%<extra></extra>",
        )

    def _render_bar_chart(self, visualizer: PlotlyVisualizer, selection: FilterSelection) -> None:
        """비교 막대그래프: 성별 / 연령대 / 주중·주말. 마우스를 올리면 수치 표시"""
        st.markdown("**📊 매출 비교**")
        # 사이드바에서 연령대를 골랐으면 연령대 그래프를 먼저 보여 준다.
        options = ["성별", "연령대", "주중/주말"]
        default = "연령대" if selection.age_group != ALL else "성별"
        compare = st.segmented_control("기준", options, default=default,
                                       key="bar_type", label_visibility="collapsed") or default
        if compare == "성별":
            fig = visualizer.create_gender_sales()
        elif compare == "연령대":
            fig = visualizer.create_age_sales()
        else:
            fig = self._weekday_weekend_bar_figure(visualizer)
        self._show_chart(fig, "main_bar_chart")

    @staticmethod
    def _render_top_markets(current_df: pd.DataFrame) -> None:
        """매출 상위 상권 TOP 10"""
        top_markets = SalesAnalyzer(current_df).get_top_markets(10)
        st.bar_chart(top_markets, x="상권_코드_명", y="당월_매출_금액",
                     x_label="상권", y_label="매출 금액 (원)", horizontal=True)

    @staticmethod
    def _render_comparison(region_df: pd.DataFrame, selection: FilterSelection) -> None:
        """기준 기간 대비 비교 기간의 지표 증감 (워터폴 차트 + 표)"""
        if not selection.compare_enabled:
            st.info("사이드바에서 연도·분기를 하나씩 선택한 뒤 '다른 기간과 비교'를 켜 주세요.")
            return
        waterfall, table = PlotlyVisualizer(region_df).create_quarter_comparison(
            selection.base_year, selection.base_quarter, selection.year, selection.quarter
        )
        st.plotly_chart(waterfall, width="stretch", key="compare_waterfall")
        st.plotly_chart(table, width="stretch", key="compare_table")

    @staticmethod
    def _render_map(current_df: pd.DataFrame) -> None:
        """지도 아래 슬라이더로 요일 또는 시간대를 골라 3D 지도를 그린다. 요일과 시간대는 동시에 적용하지 않는다."""
        st.markdown("**🗺️ 3D 상권 매출 지도**")
        # 슬라이더는 지도 아래에 두지만 지도는 슬라이더 값으로 그려야 하므로 자리만 먼저 잡아 둔다.
        map_slot = st.empty()

        st.caption(
            "슬라이더를 좌우로 드래그해 선택하세요. 요일과 시간대는 동시에 선택할 수 없습니다.  \n"
            "기둥 높이는 상권 간 매출 비교용입니다(실제 금액은 기둥에 마우스를 올리면 표시). "
            "산맥은 지역 매출 집중도를 나타내는 추정값입니다."
        )
        col1, col2 = st.columns(2)
        weekday = col1.select_slider(
            "요일", [ALL] + list(WEEKDAY_COLUMNS), key="map_weekday",
            on_change=make_exclusive, args=("map_weekday", "map_time"),
        )
        time_slot = col2.select_slider(
            "시간대", [ALL] + list(TIME_COLUMNS), key="map_time",
            on_change=make_exclusive, args=("map_time", "map_weekday"),
        )

        value_column, value_name = "당월_매출_금액", "매출"
        if weekday != ALL:
            value_column, value_name = WEEKDAY_COLUMNS[weekday], f"{weekday} 매출"
        elif time_slot != ALL:
            value_column, value_name = TIME_COLUMNS[time_slot], f"{time_slot} 매출"

        # 지도용 상권별 합산은 SalesAnalyzer.get_map_data()를 그대로 쓰고,
        # 선택한 요일/시간대 컬럼을 합산 대상으로 넘긴다.
        source = current_df.assign(당월_매출_금액=current_df[value_column])
        map_df = SalesAnalyzer(source).get_map_data().dropna(subset=["경도", "위도"])
        if map_df.empty:
            map_slot.info("지도에 표시할 위치 데이터가 없습니다.")
            return
        deck = PydeckVisualizer(map_df, value_name=value_name).create_combined_map()
        map_slot.pydeck_chart(deck, height=550)


st.set_page_config(page_title="서울시 상권 분석 대시보드", layout="wide")
DashboardApp(load_dashboard_data()).run()
