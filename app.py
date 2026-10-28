import html
from dataclasses import dataclass, field

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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
CHART_HEIGHT = 430
MAP_HEIGHT = 550
MAP_START_PITCH_LIMIT = 60  # 바탕 지도(MapLibre)가 처음부터 그려지는 최대 시작 기울기
COLOR_DEFAULT = "#9DB2F5"
COLOR_SELECTED = "#EF553B"
DONUT_COLORS = ["#636EFA", "#EF553B"]
MANWON = 10_000  # 화면에 보이는 매출 금액 단위: 만 원
# TOP 10 목록 표 스타일 (라이트·다크 테마 모두에서 보이도록 반투명 회색 선 사용)
TOP_LIST_STYLE = """
<style>
.top-list { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 0.85rem; }
.top-list th { font-weight: 600; font-size: 0.75rem; opacity: 0.65; text-align: left; white-space: nowrap;
               padding: 6px 4px; border-bottom: 2px solid rgba(128, 128, 128, 0.45); }
.top-list td { padding: 7px 4px; border-bottom: 1px solid rgba(128, 128, 128, 0.25); }
.top-list .rank { width: 2.8em; text-align: center; font-weight: 600; }
.top-list .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.top-list .value { width: 6.5em; text-align: right; font-variant-numeric: tabular-nums; }
.top-list tbody tr:nth-child(-n+3) .rank { color: #EF553B; }
</style>
"""
# 핵심 지표 카드의 숫자 크기 (큰 금액도 카드 안에 다 보이도록 기본값보다 작게)
METRIC_STYLE = """
<style>
[data-testid="stMetricValue"] { font-size: 1.7rem; }
</style>
"""
# 기간 비교 항목: compare_quarters()의 '항목' 값 → (화면 이름, 단위, 나눌 값)
COMPARE_ITEMS = {
    "총매출": ("총매출", "만원", MANWON),
    "총건수": ("총매출건수", "건", 1),
    "평균객단가": ("평균 객단가", "원", 1),
}


def format_manwon(won: float) -> str:
    """원 단위 금액을 만원 단위 숫자 '1,234' 형태로 (단위 '만원'은 카드 라벨에 표시)"""
    return f"{won / MANWON:,.0f}"


@dataclass(frozen=True)
class SalesDimension:
    """지도 슬라이더와 동향·막대 그래프가 함께 쓰는 분석 기준 (요일·시간대·연령대)"""
    name: str                 # 화면에 보이는 기준 이름
    state_key: str            # 지도 아래 슬라이더의 session_state 키
    columns: dict[str, str]   # 항목 이름 → 매출 금액 컬럼
    analyzer_method: str      # 항목별 매출을 돌려주는 SalesAnalyzer 메서드 이름
    label_suffix: str = ""    # SalesAnalyzer 결과의 '구분' 값에 붙여 화면 이름과 맞추는 접미사

    def summary(self, analyzer: SalesAnalyzer) -> pd.DataFrame:
        """항목별 매출(구분, 매출금액). 집계는 SalesAnalyzer가 하고 여기서는 이름만 맞춘다."""
        result = getattr(analyzer, self.analyzer_method)()
        if result.empty:
            return result
        return result.assign(구분=result["구분"] + self.label_suffix)


DIMENSIONS = {
    "map_weekday": SalesDimension("요일", "map_weekday", WEEKDAY_COLUMNS, "get_weekday_sales"),
    "map_time": SalesDimension("시간대", "map_time", TIME_COLUMNS, "get_time_sales", "시"),
    "map_age": SalesDimension("연령대", "map_age", AGE_COLUMNS, "get_age_sales"),
}
DEFAULT_DIMENSION_KEY = "map_weekday"
# 요일·시간대·연령대·성별 매출은 서로 독립적으로 집계되어 있어 한 번에 하나만 적용한다.
EXCLUSIVE_KEYS = list(DIMENSIONS) + ["gender"]


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_dashboard_data() -> pd.DataFrame:
    """통합 CSV가 있으면 사용하고, 없으면 연도별 전처리 CSV를 합쳐서 사용한다."""
    if DATA_PATH.exists():
        return load_data()
    return pd.concat([load_data(path) for path in PROCESSED_FILES], ignore_index=True)


def make_exclusive(changed_key: str) -> None:
    """상호 배타 선택: 방금 바꾼 값이 '전체'가 아니면 나머지(요일·시간대·연령대·성별)를 '전체'로 되돌린다.
    슬라이더를 움직이면 그 기준을 그래프 기준으로 삼는다."""
    if st.session_state[changed_key] != ALL:
        for key in EXCLUSIVE_KEYS:
            if key != changed_key:
                st.session_state[key] = ALL
    if changed_key in DIMENSIONS:
        st.session_state["active_dimension"] = changed_key


@dataclass
class FilterSelection:
    """사용자가 선택한 조건 묶음 (사이드바 + 지도 아래 슬라이더)"""
    year: int | str = ALL
    quarter: int | str = ALL
    district: str = ALL
    market_type: str = ALL
    market: str = ALL
    businesses: list[str] = field(default_factory=list)
    gender: str = ALL
    compare_enabled: bool = False
    base_year: int | None = None
    base_quarter: int | None = None
    dimension: SalesDimension = DIMENSIONS[DEFAULT_DIMENSION_KEY]
    dimension_item: str = ALL

    def has_single_period(self) -> bool:
        """연도와 분기가 모두 하나로 정해졌는지 여부 (기간 비교 가능 조건)"""
        return self.year != ALL and self.quarter != ALL

    def load_dimension_state(self) -> None:
        """지도 아래 슬라이더에서 고른 기준과 항목을 읽어 온다."""
        self.dimension = DIMENSIONS[st.session_state.get("active_dimension", DEFAULT_DIMENSION_KEY)]
        self.dimension_item = st.session_state.get(self.dimension.state_key, ALL)

    def value_column(self) -> tuple[str, str]:
        """지도 높이와 선택 항목 카드에 쓸 (매출 컬럼, 표시 이름)"""
        if self.gender != ALL:
            return GENDER_COLUMNS[self.gender], self.gender
        if self.dimension_item != ALL:
            return self.dimension.columns[self.dimension_item], self.dimension_item
        return "당월_매출_금액", ALL


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
        selection.load_dimension_state()
        return selection

    @staticmethod
    def _box(title: str):
        """필터 그룹을 테두리 있는 네모 박스로 묶는다."""
        box = st.sidebar.container(border=True)
        box.markdown(f"**{title}**")
        return box

    def _render_period(self, selection: FilterSelection) -> None:
        box = self._box("기간")
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
        box = self._box("지역")
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
        box = self._box("업종")
        businesses = sorted(self.df["서비스_업종_코드_명"].dropna().unique().tolist())
        selection.businesses = box.multiselect("서비스 업종 (미선택 시 전체)", businesses)

    def _render_customer(self, selection: FilterSelection) -> None:
        box = self._box("성별")
        box.caption("성별을 고르면 지도 아래 슬라이더가 숨겨지고 남/녀 매출 비중 도넛이 표시됩니다.")
        selection.gender = box.selectbox(
            "성별", [ALL] + list(GENDER_COLUMNS), key="gender",
            on_change=make_exclusive, args=("gender",),
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

        # 메인 화면: 핵심 지표(제목 바로 아래) → 3D 지도 + 슬라이더 | TOP 10 목록 → 선택 기준에 맞는 그래프
        self._render_metrics(current_df, region_df, selection)
        st.caption(self._describe_selection(selection))

        with st.container(border=True):
            self._render_map_section(current_df, selection)

        self._render_selection_charts(visualizer, selection)
        st.divider()

        st.subheader("상세 분석")
        tab_compare, tab_data = st.tabs(["기간 비교", "데이터 상세"])
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
        _, item_name = selection.value_column()
        if item_name != ALL:
            parts.append(item_name)
        return "현재 조건: " + " · ".join(parts)

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

    @staticmethod
    def _value_source(current_df: pd.DataFrame, selection: FilterSelection) -> pd.DataFrame:
        """선택한 항목(요일·시간대·연령대·성별)의 매출을 '당월_매출_금액' 자리에 넣은 DataFrame.
        지도와 TOP 10이 SalesAnalyzer의 상권별 합산을 그대로 쓰도록 하기 위함이다."""
        value_column, _ = selection.value_column()
        return current_df.assign(당월_매출_금액=current_df[value_column])

    # ------------------------------------------
    # 핵심 지표
    # ------------------------------------------

    def _render_metrics(self, current_df: pd.DataFrame, region_df: pd.DataFrame,
                        selection: FilterSelection) -> None:
        """총매출·총건수·평균 객단가·상권 수 카드. 기간 비교 시 증감, 항목 선택 시 해당 항목 매출 카드를 추가한다."""
        current = SalesAnalyzer(current_df)
        base = None
        if selection.compare_enabled:
            base_df = self._apply_period_filter(region_df, selection.base_year, selection.base_quarter)
            base = SalesAnalyzer(base_df)
            st.caption(
                f"증감 기준: {selection.base_year}년 {selection.base_quarter}분기 → "
                f"{selection.year}년 {selection.quarter}분기"
            )

        item = self._selected_item_sales(current, selection)
        # 금액이 커서 카드 폭을 넘으면 '…'로 잘리므로 단위는 라벨에 적고 숫자 글자 크기를 줄인다.
        st.markdown(METRIC_STYLE, unsafe_allow_html=True)
        cols = st.columns(5 if item else 4)
        cols[0].metric("총매출 (만원)", format_manwon(current.get_total_sales()),
                       self._delta(current.get_total_sales(), base.get_total_sales() if base else None))
        cols[1].metric("총매출건수 (건)", f"{current.get_total_count():,.0f}",
                       self._delta(current.get_total_count(), base.get_total_count() if base else None))
        cols[2].metric("평균 객단가 (원)", f"{current.get_average_spending():,.0f}",
                       self._delta(current.get_average_spending(), base.get_average_spending() if base else None))
        cols[3].metric("분석 대상 상권 수 (개)", f"{current_df['상권_코드'].nunique():,}")
        if item:
            label, value = item
            total = current.get_total_sales()
            cols[4].metric(f"{label} 매출 (만원)", format_manwon(value),
                           f"전체 매출의 {value / total * 100:.1f}%" if total else None, delta_color="off")

    @staticmethod
    def _selected_item_sales(analyzer: SalesAnalyzer, selection: FilterSelection) -> tuple[str, float] | None:
        """성별 또는 슬라이더에서 고른 항목의 매출. 선택하지 않았으면 None"""
        if selection.gender != ALL:
            return selection.gender, analyzer.get_gender_sales()[selection.gender]
        if selection.dimension_item != ALL:
            summary = selection.dimension.summary(analyzer)
            value = summary.loc[summary["구분"] == selection.dimension_item, "매출금액"].sum()
            return selection.dimension_item, value
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
    # 3D 지도 + 슬라이더 + TOP 10 목록
    # ------------------------------------------

    def _render_map_section(self, current_df: pd.DataFrame, selection: FilterSelection) -> None:
        """왼쪽: 3D 지도와 그 아래 슬라이더 / 오른쪽: 매출 상위 상권 TOP 10 목록"""
        source = self._value_source(current_df, selection)
        _, item_name = selection.value_column()
        value_name = "매출" if item_name == ALL else f"{item_name} 매출"

        map_col, top_col = st.columns([4, 1])
        with map_col:
            self._render_map(source, selection, value_name)
        with top_col:
            self._render_top_list(source, value_name)

    @staticmethod
    def _render_map(source: pd.DataFrame, selection: FilterSelection, value_name: str) -> None:
        """3D 지도와 그 아래 요일·시간대·연령대 슬라이더. 성별을 고르면 슬라이더는 숨긴다."""
        st.markdown("**3D 상권 매출 지도**")
        # 슬라이더는 지도 아래에 두지만 지도는 슬라이더 값으로 그려야 하므로 자리만 먼저 잡아 둔다.
        map_slot = st.empty()

        legend = ("기둥 높이는 상권 간 매출 비교용입니다(실제 금액은 기둥에 마우스를 올리면 표시). "
                  "산맥은 지역 매출 집중도를 나타내는 추정값입니다.")
        if selection.gender != ALL:
            st.caption(f"성별({selection.gender}) 매출 기준입니다. 요일·시간대·연령대를 보려면 사이드바의 성별을 '전체'로 바꿔 주세요.  \n" + legend)
        else:
            st.caption("슬라이더를 좌우로 드래그하면 지도와 아래 그래프가 함께 바뀝니다. "
                       "요일·시간대·연령대는 하나만 선택됩니다.  \n" + legend)
            for col, dimension in zip(st.columns(len(DIMENSIONS)), DIMENSIONS.values()):
                col.select_slider(
                    dimension.name, [ALL] + list(dimension.columns), key=dimension.state_key,
                    on_change=make_exclusive, args=(dimension.state_key,),
                )

        map_df = SalesAnalyzer(source).get_map_data().dropna(subset=["경도", "위도"])
        if map_df.empty:
            map_slot.info("지도에 표시할 위치 데이터가 없습니다.")
            return
        deck = PydeckVisualizer(map_df, value_name=value_name).create_combined_map()
        # 처음 기울기가 60°를 넘으면 Streamlit에서 바탕 지도가 그려지지 않아(확대·축소를 해야 보임) 시작 기울기만 낮춘다.
        # 마우스로 더 기울이는 것(max_pitch)은 그대로 허용한다.
        deck.initial_view_state.pitch = min(deck.initial_view_state.pitch, MAP_START_PITCH_LIMIT)
        map_slot.pydeck_chart(deck, height=MAP_HEIGHT)

    @staticmethod
    def _render_top_list(source: pd.DataFrame, value_name: str) -> None:
        """지도 오른쪽에 작게 세로로 보이는 매출 상위 상권 TOP 10 (지도와 같은 매출 기준)"""
        st.markdown(f"**TOP 10**  \n<small>{value_name} 기준</small>", unsafe_allow_html=True)
        top = SalesAnalyzer(source).get_top_markets(10).reset_index(drop=True)
        if top.empty:
            st.caption("데이터가 없습니다.")
            return
        # 표 높이를 고정하지 않고 결과 행 수만큼만 그리기 위해 HTML 표로 표시한다.
        rows = "".join(
            f"<tr><td class='rank'>{rank}</td>"
            f"<td class='name' title='{html.escape(name)}'>{html.escape(name)}</td>"
            f"<td class='value'>{value / MANWON:,.0f}</td></tr>"
            for rank, (name, value) in enumerate(zip(top["상권_코드_명"], top["당월_매출_금액"]), start=1)
        )
        st.markdown(TOP_LIST_STYLE + (
            "<table class='top-list'><thead><tr><th class='rank'>순위</th><th>상권</th>"
            "<th class='value'>만원</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        ), unsafe_allow_html=True)

    # ------------------------------------------
    # 지도 아래 그래프 (선택 기준과 연동)
    # ------------------------------------------

    def _render_selection_charts(self, visualizer: PlotlyVisualizer, selection: FilterSelection) -> None:
        """선택한 기준에 맞는 그래프를 보여 준다.
        - 성별: 남/녀 도넛만
        - 요일: 동향 + 막대 + 주중/주말 도넛
        - 시간대·연령대: 동향 + 막대"""
        if selection.gender != ALL:
            _, center, _ = st.columns([1, 2, 1])
            with center.container(border=True):
                st.markdown("**남/녀 매출 비중**")
                sales = visualizer.analyzer.get_gender_sales()
                self._show_chart(self._donut_figure(sales, "남/녀 매출 비중", selection.gender), "gender_donut")
            return

        dimension, item = selection.dimension, selection.dimension_item
        summary = dimension.summary(visualizer.analyzer)
        show_weekend = dimension.state_key == "map_weekday"

        cols = st.columns(3 if show_weekend else 2)
        with cols[0].container(border=True):
            st.markdown(f"**{dimension.name}별 매출 동향**")
            self._show_chart(self._trend_figure(visualizer, summary, dimension.name, item), "main_trend_chart")
        with cols[1].container(border=True):
            st.markdown(f"**{dimension.name}별 매출 비교**")
            self._show_chart(self._bar_figure(visualizer, summary, dimension.name, item), "main_bar_chart")
        if show_weekend:
            with cols[2].container(border=True):
                st.markdown("**주중/주말 매출 비중**")
                sales = visualizer.analyzer.get_weekday_weekend_sales()
                self._show_chart(self._donut_figure(sales, "주중/주말 매출 비중"), "weekend_donut")

    @staticmethod
    def _show_chart(fig: go.Figure, key: str) -> None:
        """나란히 놓인 차트의 높이를 맞춰서 그린다."""
        fig.update_layout(height=CHART_HEIGHT, margin=dict(t=60, b=40, l=40, r=20))
        st.plotly_chart(fig, width="stretch", key=key)

    @staticmethod
    def _empty_figure(message: str) -> go.Figure:
        return go.Figure().update_layout(title=message, template="plotly_white")

    def _trend_figure(self, visualizer: PlotlyVisualizer, summary: pd.DataFrame,
                      name: str, item: str) -> go.Figure:
        """항목별 매출 꺾은선(만원). 최고·최저 항목은 굵은 점, 슬라이더로 고른 항목은 별로 표시"""
        if summary.empty:
            return self._empty_figure(f"{name}별 매출 데이터가 없습니다.")
        labels = summary["구분"].tolist()
        values = (summary["매출금액"] / MANWON).tolist()
        fig = visualizer.line_renderer.render(
            labels, values, f"{name}별 매출 추이 및 최고/최저",
            x_title=name, y_title="매출 금액 (만 원)",
            hovertemplate="%{x}<br>매출: %{y:,.0f}만원<extra></extra>",
        )
        marks = [
            (values.index(max(values)), "최고", "red", "diamond", "top center"),
            (values.index(min(values)), "최저", "blue", "diamond", "bottom center"),
        ]
        if item in labels:
            marks.append((labels.index(item), "선택", "gold", "star", "middle right"))
        for index, mark, color, symbol, position in marks:
            fig.add_trace(go.Scatter(
                x=[labels[index]], y=[values[index]], mode="markers+text", name=f"{mark} {name}",
                text=[mark], textposition=position,
                marker=dict(size=16 if mark == "선택" else 14, color=color, symbol=symbol,
                            line=dict(width=1, color="black")),
                hovertemplate=f"{mark}: %{{x}}<br>매출: %{{y:,.0f}}만원<extra></extra>",
            ))
        return fig

    def _bar_figure(self, visualizer: PlotlyVisualizer, summary: pd.DataFrame,
                    name: str, item: str) -> go.Figure:
        """항목별 매출 막대(만원). 마우스를 올리면 매출액과 비중, 슬라이더로 고른 항목은 다른 색으로 강조"""
        if summary.empty:
            return self._empty_figure(f"{name}별 매출 데이터가 없습니다.")
        labels = summary["구분"].tolist()
        values = (summary["매출금액"] / MANWON).tolist()
        total = sum(values)
        ratios = [value / total * 100 if total else 0 for value in values]
        fig = visualizer.bar_renderer.render(
            labels, values, f"{name}별 매출 및 비중 비교",
            x_title=name, y_title="매출 금액 (만 원)",
            texttemplate="%{customdata:.1f}%", customdata=ratios,
            hovertemplate="%{x}<br>매출: %{y:,.0f}만원<br>비중: %{customdata:.1f}%<extra></extra>",
        )
        fig.update_traces(marker_color=[COLOR_SELECTED if label == item else COLOR_DEFAULT for label in labels])
        return fig

    def _donut_figure(self, sales: dict[str, float], title: str, highlight: str = ALL) -> go.Figure:
        """항목 2개(주중/주말, 남/녀)의 매출 비중 도넛(만원). 고른 항목은 조각을 살짝 떼어 강조"""
        if not sales or sum(sales.values()) == 0:
            return self._empty_figure(f"{title} 데이터가 없습니다.")
        labels = list(sales)
        fig = go.Figure(go.Pie(
            labels=labels, values=[value / MANWON for value in sales.values()], hole=0.45,
            textinfo="label+percent", sort=False, marker=dict(colors=DONUT_COLORS),
            pull=[0.08 if label == highlight else 0 for label in labels],
            hovertemplate="%{label}<br>매출: %{value:,.0f}만원<br>비중: %{percent}<extra></extra>",
        ))
        fig.update_layout(title=title, template="plotly_white", showlegend=False)
        return fig

    # ------------------------------------------
    # 상세 분석
    # ------------------------------------------

    def _render_comparison(self, region_df: pd.DataFrame, selection: FilterSelection) -> None:
        """기준 기간 대비 비교 기간의 지표 증감 (워터폴 차트 + 표).
        집계는 SalesAnalyzer.compare_quarters()가 하고, 여기서는 단위(만원·건·원)만 맞춰 그린다."""
        if not selection.compare_enabled:
            st.info("사이드바에서 연도·분기를 하나씩 선택한 뒤 '다른 기간과 비교'를 켜 주세요.")
            return
        base_label = f"{selection.base_year}년 {selection.base_quarter}분기"
        target_label = f"{selection.year}년 {selection.quarter}분기"
        result = SalesAnalyzer(region_df).compare_quarters(
            selection.base_year, selection.base_quarter, selection.year, selection.quarter
        )
        if result is None or result.empty:
            st.info(f"{base_label} 또는 {target_label}에 해당하는 데이터가 없어 비교할 수 없습니다.")
            return

        st.caption(f"기준 기간 {base_label} → 비교 기간 {target_label} "
                   "(자치구·상권·업종 조건은 두 기간에 똑같이 적용)")
        st.plotly_chart(self._comparison_figure(result, base_label, target_label),
                        width="stretch", key="compare_waterfall")
        st.dataframe(self._comparison_table(result), hide_index=True, width="stretch")

    @staticmethod
    def _comparison_figure(result: pd.DataFrame, base_label: str, target_label: str) -> go.Figure:
        """지표별 워터폴(기준 → 증감 → 비교). 총매출은 만원, 건수는 건, 객단가는 원"""
        names = [COMPARE_ITEMS[item] for item in result["항목"]]
        fig = make_subplots(rows=1, cols=len(result), horizontal_spacing=0.08,
                            subplot_titles=[f"{label} ({unit})" for label, unit, _ in names])
        for col, ((_, row), (label, unit, divisor)) in enumerate(zip(result.iterrows(), names), start=1):
            fig.add_trace(go.Waterfall(
                name=label, orientation="v", measure=["absolute", "relative", "total"],
                x=[base_label, "증감", target_label],
                y=[row["기준 기간"] / divisor, row["증감액"] / divisor, row["비교 기간"] / divisor],
                texttemplate="%{y:,.0f}", textposition="outside",
                hovertemplate=f"%{{x}}<br>%{{y:,.0f}} {unit}<extra></extra>",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#EF553B"}},
                decreasing={"marker": {"color": "#636EFA"}},
            ), row=1, col=col)
        fig.update_layout(title=f"{base_label} 대비 {target_label} 지표별 증감",
                          template="plotly_white", showlegend=False, height=480, margin=dict(t=90, b=40))
        return fig

    @staticmethod
    def _comparison_table(result: pd.DataFrame) -> pd.DataFrame:
        """증감 표 (화면 표시용 문자열). 기준값이 0이면 증감률 대신 안내 문구"""
        rows = []
        for _, row in result.iterrows():
            label, unit, divisor = COMPARE_ITEMS[row["항목"]]
            rate = row["증감률(%)"]
            rows.append({
                "항목": label,
                "기준 기간": f"{row['기준 기간'] / divisor:,.0f} {unit}",
                "비교 기간": f"{row['비교 기간'] / divisor:,.0f} {unit}",
                "증감액": f"{row['증감액'] / divisor:+,.0f} {unit}",
                "증감률": "기준값 0 (계산 불가)" if rate is None or pd.isna(rate) else f"{rate:+.1f}%",
            })
        return pd.DataFrame(rows)


st.set_page_config(page_title="서울시 상권 분석 대시보드", layout="wide")
DashboardApp(load_dashboard_data()).run()
