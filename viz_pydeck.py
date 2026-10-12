import base64
import io
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pydeck as pdk
from PIL import Image


class MapLayer(ABC):
    """지도 레이어 공통 부모 클래스

    입력 df는 상권 단위로 집계된 지도용 데이터 (SalesAnalyzer.get_map_data() 결과)이다.
    필수 컬럼 검증, 유효하지 않은 좌표 제외, 높이 스케일 계산을 공통으로 처리한다.
    """

    # 지도용 데이터 컬럼 이름
    COL_CODE = "상권_코드"        # 있으면 중복 확인 기준으로 사용
    COL_NAME = "상권_코드_명"
    COL_LAT = "위도"
    COL_LON = "경도"

    SCALES = ("linear", "sqrt", "log")  # 높이 스케일: 선형 / 제곱근 / 로그

    def __init__(self, df: pd.DataFrame, value_col: str = "당월_매출_금액", scale: str = "sqrt"):
        if scale not in self.SCALES:
            raise ValueError(f"scale은 {self.SCALES} 중 하나여야 합니다: {scale}")
        self.value_col = value_col
        self.scale = scale
        self.excluded_count = 0  # 좌표·값이 유효하지 않아 지도에서 제외된 상권 수
        self._check_required_columns(df)
        self.df = self._select_valid_rows(df)

    def _check_required_columns(self, df: pd.DataFrame) -> None:
        """필수 컬럼이 없으면 어떤 컬럼이 없는지 알려주는 오류를 냄"""
        required = [self.COL_NAME, self.COL_LAT, self.COL_LON, self.value_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"지도에 필요한 컬럼이 없습니다: {missing} "
                "(SalesAnalyzer.get_map_data() 결과를 넘겨주세요)"
            )

    def _select_valid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """좌표·값이 유효한 상권만 남기고, 상권이 중복되지 않았는지 확인"""
        data = df.copy()
        data[self.value_col] = pd.to_numeric(data[self.value_col], errors="coerce")
        valid = (
            data[self.COL_LAT].between(-90, 90)
            & data[self.COL_LON].between(-180, 180)
            & data[self.value_col].notna()
        )
        self.excluded_count = int((~valid).sum())
        data = data[valid]

        key = self.COL_CODE if self.COL_CODE in data.columns else self.COL_NAME
        duplicated = data[key].duplicated()
        if duplicated.any():
            raise ValueError(
                f"같은 상권이 여러 행에 있습니다 ({key} 중복 {int(duplicated.sum())}건). "
                "상권 단위로 집계된 데이터(SalesAnalyzer.get_map_data())를 넘겨주세요"
            )
        return data.reset_index(drop=True)

    def _scale_ratio(self, values: np.ndarray) -> np.ndarray:
        """값을 스케일 변환한 뒤 최대값 기준 0~1 비율로 바꿈"""
        values = np.clip(np.asarray(values, dtype=float), 0, None)
        if self.scale == "sqrt":
            values = np.sqrt(values)
        elif self.scale == "log":
            values = np.log1p(values)
        max_value = values.max() if values.size else 0
        if max_value <= 0:
            return np.zeros_like(values)
        return values / max_value

    @staticmethod
    def _blend_color(ratio: np.ndarray, low: tuple, high: tuple, alpha: int) -> list[list[int]]:
        """비율(0~1)에 따라 low 색에서 high 색으로 바뀌는 RGBA 목록"""
        return [
            [int(lo + (hi - lo) * r) for lo, hi in zip(low, high)] + [alpha]
            for r in ratio
        ]

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

    @abstractmethod
    def build(self) -> pdk.Layer:
        """Pydeck 레이어 생성 (자식 클래스에서 구현)"""


class SalesColumnLayer(MapLayer):
    """상권별 매출 3D 기둥 레이어 (실제 관측값)"""

    COL_GU = "자치구_코드_명"           # 있으면 tooltip에 표시
    COL_KIND = "상권_구분_코드_명"      # 있으면 tooltip에 표시

    MAX_HEIGHT_M = 3000                 # 매출 1위 상권의 기둥 높이(m)
    RADIUS_M = 60                       # 기둥 반경(m)
    COLOR_LOW = (255, 235, 150)         # 매출 낮음
    COLOR_HIGH = (200, 30, 30)          # 매출 높음

    def __init__(self, df: pd.DataFrame, value_col: str = "당월_매출_금액",
                 value_name: str = "매출", scale: str = "sqrt"):
        super().__init__(df, value_col, scale)
        self.value_name = value_name

    def build(self) -> pdk.Layer:
        ratio = self._scale_ratio(self.df[self.value_col].to_numpy())
        data = pd.DataFrame({
            "lon": self.df[self.COL_LON],
            "lat": self.df[self.COL_LAT],
            "height": ratio * self.MAX_HEIGHT_M,
            "color": self._blend_color(ratio, self.COLOR_LOW, self.COLOR_HIGH, 220),
            "tooltip_title": self.df[self.COL_NAME],
            "tooltip_body": self._tooltip_body(),
        })
        return pdk.Layer(
            "ColumnLayer",
            data,
            get_position=["lon", "lat"],
            get_elevation="height",
            elevation_scale=1,
            radius=self.RADIUS_M,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
        )

    def _tooltip_body(self) -> list[str]:
        """'자치구 · 상권구분<br/>매출: 12.3억 원' 형태 (없는 컬럼은 생략)"""
        bodies = []
        for _, row in self.df.iterrows():
            area = " · ".join(
                str(row[col]) for col in (self.COL_GU, self.COL_KIND) if col in self.df.columns
            )
            value_line = f"{self.value_name}: <b>{self.format_won(row[self.value_col])}</b>"
            bodies.append(f"{area}<br/>{value_line}" if area else value_line)
        return bodies


class GaussianSurfaceLayer(MapLayer):
    """지역 매출 집중도를 나타내는 가우스 산맥 레이어 (보간된 추정값)

    각 상권 위치에 매출만큼의 가우스 언덕을 놓고 모두 더한 값을 격자로 계산한다.
    상권 사이의 값은 실제 관측값이 아니라 계산으로 채운 추정값이다.
    격자를 높이 이미지(elevation_data)와 색 이미지(texture)로 만들어
    Pydeck TerrainLayer의 매끈한 곡면으로 그린다.
    """

    SIGMA_M = 1000                      # 언덕 폭(표준편차, m)
    CELL_SIZE_M = 200                   # 격자 한 칸 크기(m) = 높이 이미지 1픽셀
    MAX_HEIGHT_M = 1500                 # 산맥 최고점 기본 높이(m)
    HEIGHT_EXAGGERATION = 3.0           # 높이 과장 배율 (최고점 = MAX_HEIGHT_M × 배율)
    MIN_RATIO = 0.02                    # 최고점 대비 이 비율 미만인 곳은 투명 처리
    OPACITY = 0.6                       # 산맥 불투명도 (0=투명, 1=불투명)

    M_PER_DEG_LAT = 110_540             # 위도 1도의 길이(m)
    M_PER_DEG_LON_EQUATOR = 111_320     # 적도에서 경도 1도의 길이(m)

    # turbo 컬러맵 다항식 근사 계수 (파랑 → 청록 → 초록 → 노랑 → 빨강)
    TURBO_COEFFS = {
        "r": (0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943),
        "g": (0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604),
        "b": (0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973),
    }

    def __init__(self, df: pd.DataFrame, value_col: str = "당월_매출_금액", scale: str = "sqrt",
                 sigma_m: float | None = None, cell_size_m: float | None = None):
        super().__init__(df, value_col, scale)
        # 매개변수를 안 주면 클래스 기본값 사용 (미리보기 슬라이더로 바꿀 수 있게 실행 시점에 읽음)
        self.sigma_m = sigma_m or self.SIGMA_M
        self.cell_size_m = cell_size_m or self.CELL_SIZE_M

    def _density_matrix(self) -> tuple[np.ndarray, list[float]]:
        """격자 칸별 가우스 매출 밀도 2차원 배열과 경계 [서, 남, 동, 북] 계산

        배열의 행은 남→북, 열은 서→동 순서다.
        """
        lat0 = self.df[self.COL_LAT].mean()
        lon0 = self.df[self.COL_LON].mean()
        m_per_deg_lon = self.M_PER_DEG_LON_EQUATOR * np.cos(np.radians(lat0))

        # 위도·경도 → 중심 기준 미터 좌표
        xs = (self.df[self.COL_LON].to_numpy() - lon0) * m_per_deg_lon
        ys = (self.df[self.COL_LAT].to_numpy() - lat0) * self.M_PER_DEG_LAT
        weights = self.df[self.value_col].to_numpy(dtype=float)

        margin = 3 * self.sigma_m
        grid_x = np.arange(xs.min() - margin, xs.max() + margin, self.cell_size_m)
        grid_y = np.arange(ys.min() - margin, ys.max() + margin, self.cell_size_m)
        density = np.zeros((grid_y.size, grid_x.size))

        # 상권마다 주변 ±4σ 범위에만 언덕을 더함 (멀리는 거의 0이라 생략)
        reach = 4 * self.sigma_m
        for x, y, w in zip(xs, ys, weights):
            ix = slice(np.searchsorted(grid_x, x - reach), np.searchsorted(grid_x, x + reach))
            iy = slice(np.searchsorted(grid_y, y - reach), np.searchsorted(grid_y, y + reach))
            dx = grid_x[ix][None, :] - x
            dy = grid_y[iy][:, None] - y
            density[iy, ix] += w * np.exp(-(dx ** 2 + dy ** 2) / (2 * self.sigma_m ** 2))

        bounds = [
            grid_x[0] / m_per_deg_lon + lon0,
            grid_y[0] / self.M_PER_DEG_LAT + lat0,
            grid_x[-1] / m_per_deg_lon + lon0,
            grid_y[-1] / self.M_PER_DEG_LAT + lat0,
        ]
        return density, bounds

    def compute_density_grid(self) -> pd.DataFrame:
        """격자 칸별 가우스 매출 밀도 → DataFrame(lon, lat, density) (검증·분석용)"""
        if self.df.empty:
            return pd.DataFrame(columns=["lon", "lat", "density"])
        density, (west, south, east, north) = self._density_matrix()
        lons = np.linspace(west, east, density.shape[1])
        lats = np.linspace(south, north, density.shape[0])
        gx, gy = np.meshgrid(lons, lats)
        return pd.DataFrame({"lon": gx.ravel(), "lat": gy.ravel(), "density": density.ravel()})

    @property
    def peak_height_m(self) -> float:
        """과장 배율을 적용한 산맥 최고점 높이(m)"""
        return self.MAX_HEIGHT_M * self.HEIGHT_EXAGGERATION

    def _turbo_rgb(self, ratio: np.ndarray) -> np.ndarray:
        """0~1 비율 → turbo 컬러맵 RGB (0~255 정수)"""
        channels = []
        for key in ("r", "g", "b"):
            value = np.zeros_like(ratio)
            for coeff in reversed(self.TURBO_COEFFS[key]):
                value = value * ratio + coeff
            channels.append(np.clip(value, 0, 1))
        return (np.stack(channels, axis=-1) * 255).astype(np.uint8)

    @staticmethod
    def _to_png_data_uri(pixels: np.ndarray) -> str:
        """numpy 이미지 배열 → 'data:image/png;base64,...' 문자열"""
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def _elevation_image(self, ratio: np.ndarray) -> tuple[str, dict]:
        """높이 비율 → 높이 이미지와 해독 규칙

        높이(m) = (R × 256 + G) × step 으로 16비트에 담는다.
        """
        step = self.peak_height_m / 65535
        code = np.round(ratio * 65535).astype(np.uint32)
        pixels = np.zeros(ratio.shape + (3,), dtype=np.uint8)
        pixels[..., 0] = code // 256
        pixels[..., 1] = code % 256
        decoder = {"rScaler": 256 * step, "gScaler": step, "bScaler": 0, "offset": 0}
        return self._to_png_data_uri(pixels), decoder

    def _texture_image(self, ratio: np.ndarray) -> str:
        """높이 비율 → 색 이미지 (turbo 색, 낮은 곳은 투명)"""
        alpha = np.where(ratio >= self.MIN_RATIO, 255, 0).astype(np.uint8)
        pixels = np.dstack([self._turbo_rgb(ratio), alpha])
        return self._to_png_data_uri(pixels)

    def build(self) -> pdk.Layer:
        density, bounds = self._density_matrix()
        ratio = self._scale_ratio(density)
        ratio = np.where(ratio >= self.MIN_RATIO, ratio, 0)
        ratio = np.flipud(ratio)  # 이미지는 위쪽이 북쪽이므로 남→북 배열을 뒤집음

        elevation_data, decoder = self._elevation_image(ratio)
        # pydeck은 문자열을 계산식(@@=)으로 바꾸므로, 이미지 주소는 String으로 감싸 그대로 전달
        return pdk.Layer(
            "TerrainLayer",
            elevation_data=pdk.types.String(elevation_data),
            texture=pdk.types.String(self._texture_image(ratio)),
            elevation_decoder=decoder,
            bounds=bounds,
            opacity=self.OPACITY,
            pickable=False,  # 곡면은 계산된 추정값이라 상권 tooltip을 띄우지 않음
        )


class PydeckVisualizer:
    """서울 상권 3D 지도 생성 클래스 (기둥·가우스 산맥 레이어를 조합)"""

    SCALES = MapLayer.SCALES
    SEOUL_CENTER = (37.5665, 126.9780)  # (위도, 경도)
    ZOOM = 10.5
    MAX_PITCH = 85                      # 마우스로 옆에서 볼 수 있을 만큼 기울이기 허용

    # 시야: 2D = 위에서 내려다보기, 3D = 비스듬히 옆에서 보기
    VIEW_MODES = {
        "2D": {"pitch": 0, "bearing": 0},
        "3D": {"pitch": 70, "bearing": -20},
    }

    TOOLTIP = {
        "html": "<b>{tooltip_title}</b><br/>{tooltip_body}",
        "style": {"backgroundColor": "rgba(30, 30, 30, 0.9)", "color": "white", "fontSize": "13px"},
    }

    def __init__(self, df: pd.DataFrame, value_col: str = "당월_매출_금액",
                 value_name: str = "매출", scale: str = "sqrt", view_mode: str = "3D"):
        """
        df: 상권 단위 지도용 데이터 (SalesAnalyzer.get_map_data() 결과)
        value_col: 높이로 쓸 매출 컬럼
        value_name: tooltip에 보일 분석 기준 이름 (예: "토요일 매출")
        scale: 높이 스케일 "linear" / "sqrt" / "log"
        view_mode: 시작 시야 "2D" / "3D" (어느 쪽이든 마우스로 회전·기울이기 가능)
        """
        if view_mode not in self.VIEW_MODES:
            raise ValueError(f"view_mode는 {tuple(self.VIEW_MODES)} 중 하나여야 합니다: {view_mode}")
        self.df = df
        self.value_col = value_col
        self.value_name = value_name
        self.scale = scale
        self.view_mode = view_mode

    def create_column_map(self) -> pdk.Deck:
        """상권별 3D 기둥 지도"""
        return self._make_deck([self._column_layer])

    def create_gaussian_map(self) -> pdk.Deck:
        """가우스 매출 산맥 지도"""
        return self._make_deck([self._gaussian_layer])

    def create_combined_map(self) -> pdk.Deck:
        """가우스 산맥과 상권 기둥을 겹친 지도

        반투명 산맥 안쪽의 기둥이 비쳐 보이도록 기둥을 먼저 그리고 산맥을 나중에 그린다.
        """
        return self._make_deck([self._column_layer, self._gaussian_layer])

    def _column_layer(self) -> MapLayer:
        return SalesColumnLayer(self.df, self.value_col, self.value_name, self.scale)

    def _gaussian_layer(self) -> MapLayer:
        return GaussianSurfaceLayer(self.df, self.value_col, self.scale)

    def _make_deck(self, layer_factories: list) -> pdk.Deck:
        """레이어들을 만들어 하나의 Deck으로 조립 (데이터가 없으면 빈 서울 지도)"""
        view_state = pdk.ViewState(
            latitude=self.SEOUL_CENTER[0],
            longitude=self.SEOUL_CENTER[1],
            zoom=self.ZOOM,
            max_pitch=self.MAX_PITCH,
            **self.VIEW_MODES[self.view_mode],
        )
        if self.df.empty:
            return pdk.Deck(layers=[], initial_view_state=view_state)
        layers = [make_layer().build() for make_layer in layer_factories]
        return pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=self.TOOLTIP)
