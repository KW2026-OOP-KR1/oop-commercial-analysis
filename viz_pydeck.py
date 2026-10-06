import numpy as np
import pydeck as pdk
import pandas as pd


class PydeckVisualizer:
    """서울 상권 3D 기둥 지도 생성 클래스"""

    # 통합 데이터 컬럼 이름 (1번과 확정되면 여기만 수정)
    COL_CODE = "상권_코드"
    COL_NAME = "상권_코드_명"
    COL_LAT = "위도"
    COL_LON = "경도"
    COL_GU = "자치구_코드_명"           # 없으면 tooltip에 "-"
    COL_KIND = "상권_구분_코드_명"      # 없으면 tooltip에 "-"
    COL_PERIOD = "기준_년분기_코드"     # 예: 20251

    SEOUL_CENTER = (37.5665, 126.9780)  # (위도, 경도)
    MAX_HEIGHT_M = 3000                 # 매출 1위 상권의 기둥 높이(m)
    RADIUS_M = 60                       # 기둥 반경(m)
    COLOR_LOW = (255, 235, 150)         # 매출 낮음
    COLOR_HIGH = (200, 30, 30)          # 매출 높음

    SCALES = ("linear", "sqrt", "log")  # 높이 스케일: 선형 / 제곱근 / 로그

    TOOLTIP_HTML = (
        "<b>{name}</b><br/>"
        "{gu} · {kind}<br/>"
        "기간: {period}<br/>"
        "{value_name}: <b>{value_text}</b>"
    )

    def __init__(self, df: pd.DataFrame, value_col: str = "당월_매출_금액",
                 value_name: str = "매출", scale: str = "sqrt"):
        """
        df: 필터가 적용된 통합 데이터 (상권 × 업종 행 단위)
        value_col: 기둥 높이로 쓸 매출 컬럼 (예: "토요일_매출_금액")
        value_name: tooltip에 보일 분석 기준 이름 (예: "토요일 매출")
        scale: 높이 스케일 "linear" / "sqrt" / "log"
        """
        if scale not in self.SCALES:
            raise ValueError(f"scale은 {self.SCALES} 중 하나여야 함: {scale}")
        self.df = df
        self.value_col = value_col
        self.value_name = value_name
        self.scale = scale

    def aggregate_by_district(self) -> pd.DataFrame:
        """상권 × 업종 행을 상권 단위로 합산 (담당 미정: 2번 쪽으로 옮겨질 수 있음)"""
        aggs = {
            "name": (self.COL_NAME, "first"),
            "lat": (self.COL_LAT, "first"),
            "lon": (self.COL_LON, "first"),
            "value": (self.value_col, "sum"),
        }
        if self.COL_GU in self.df.columns:
            aggs["gu"] = (self.COL_GU, "first")
        if self.COL_KIND in self.df.columns:
            aggs["kind"] = (self.COL_KIND, "first")

        agg = (
            self.df.dropna(subset=[self.COL_LAT, self.COL_LON])
            .groupby(self.COL_CODE, as_index=False)
            .agg(**aggs)
        )
        for col in ("gu", "kind"):
            if col not in agg.columns:
                agg[col] = "-"
        return agg

    def _add_height_and_color(self, agg: pd.DataFrame) -> pd.DataFrame:
        """매출을 스케일 변환 후 최대값 기준 비율(0~1)로 바꿔 기둥 높이와 색을 정함"""
        values = agg["value"].clip(lower=0)
        if self.scale == "sqrt":
            values = np.sqrt(values)
        elif self.scale == "log":
            values = np.log1p(values)
        max_value = values.max()
        ratio = values / max_value if max_value > 0 else values * 0
        agg["height"] = ratio * self.MAX_HEIGHT_M
        agg["color"] = [
            [int(lo + (hi - lo) * r) for lo, hi in zip(self.COLOR_LOW, self.COLOR_HIGH)] + [220]
            for r in ratio
        ]
        return agg

    def _add_tooltip_text(self, agg: pd.DataFrame) -> pd.DataFrame:
        """tooltip은 숫자 서식을 못 바꾸므로 미리 문자열 컬럼으로 만듦"""
        agg["value_text"] = agg["value"].map(self.format_won)
        agg["value_name"] = self.value_name
        agg["period"] = self._period_label()
        return agg

    @staticmethod
    def format_won(value: float) -> str:
        """금액을 '1.04조 원' / '12.3억 원' / '4,560만 원' / '7,800원' 형태로"""
        if value >= 1e12:
            return f"{value / 1e12:,.2f}조 원"
        if value >= 1e8:
            return f"{value / 1e8:,.1f}억 원"
        if value >= 1e4:
            return f"{value / 1e4:,.0f}만 원"
        return f"{value:,.0f}원"

    def _period_label(self) -> str:
        """기준_년분기_코드(예: 20251) → '2025년 1분기', 여러 분기면 '2025년 1분기 ~ 2025년 4분기 합계'"""
        if self.COL_PERIOD not in self.df.columns:
            return "-"
        codes = sorted(self.df[self.COL_PERIOD].dropna().astype(int).unique())
        if not codes:
            return "-"
        to_text = lambda c: f"{c // 10}년 {c % 10}분기"
        if len(codes) == 1:
            return to_text(codes[0])
        return f"{to_text(codes[0])} ~ {to_text(codes[-1])} 합계"

    def create_column_map(self) -> pdk.Deck:
        """Pydeck 3D 기둥 지도 생성"""
        view_state = pdk.ViewState(
            latitude=self.SEOUL_CENTER[0],
            longitude=self.SEOUL_CENTER[1],
            zoom=10.5,
            pitch=45,
        )
        if self.df.empty:
            return pdk.Deck(layers=[], initial_view_state=view_state)

        agg = self.aggregate_by_district()
        agg = self._add_height_and_color(agg)
        agg = self._add_tooltip_text(agg)
        layer = pdk.Layer(
            "ColumnLayer",
            agg,
            get_position=["lon", "lat"],
            get_elevation="height",
            elevation_scale=1,
            radius=self.RADIUS_M,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
        )
        tooltip = {
            "html": self.TOOLTIP_HTML,
            "style": {"backgroundColor": "rgba(30, 30, 30, 0.9)", "color": "white", "fontSize": "13px"},
        }
        return pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
