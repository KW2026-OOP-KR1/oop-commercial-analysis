from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from filter_analyzer import SalesAnalyzer, DataFilter


# ==========================================
# 1. 렌더러 추상 클래스 
# ==========================================
class ChartRenderer(ABC):
    @abstractmethod
    def render(self, labels, values, title, x_title="구분", y_title="금액", **kwargs):
        pass


# ==========================================
# 2. 구체적인 렌더러 클래스들 
# ==========================================
class BarChartRenderer(ChartRenderer):
    """막대 그래프 전담 렌더러"""
    def render(self, labels, values, title, x_title="구분", y_title="금액", **kwargs):
        text = kwargs.get("text", None)
        texttemplate = kwargs.get("texttemplate", None)
        customdata = kwargs.get("customdata", None)
        hovertemplate = kwargs.get("hovertemplate", f"%{{x}}<br>{y_title}: %{{y:,.0f}}<extra></extra>")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=values,
            text=text if text is not None else (values if texttemplate else None),
            texttemplate=texttemplate,
            textposition="outside" if (texttemplate or text is not None) else None,
            customdata=customdata,
            hovertemplate=hovertemplate
        ))
        fig.update_layout(
            title=title, xaxis_title=x_title, yaxis_title=y_title, 
            template="plotly_white", hovermode="x unified"
        )
        return fig


class LineChartRenderer(ChartRenderer):
    """꺾은선 그래프 전담 렌더러 (시간대별 추이 등에 활용)"""
    def render(self, labels, values, title, x_title="구분", y_title="금액", **kwargs):
        text = kwargs.get("text", None)
        customdata = kwargs.get("customdata", None)
        hovertemplate = kwargs.get("hovertemplate", f"%{{x}}<br>{y_title}: %{{y:,.0f}}<extra></extra>")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=labels, y=values,
            mode="lines+markers+text" if text is not None else "lines+markers",
            text=text,
            textposition="top center" if text is not None else None,
            customdata=customdata,
            hovertemplate=hovertemplate,
            line=dict(width=3, color="#636EFA"),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title=title, xaxis_title=x_title, yaxis_title=y_title, 
            template="plotly_white", hovermode="x unified"
        )
        return fig


class DonutChartRenderer(ChartRenderer):
    """도넛 차트 전담 렌더러 (주중/주말 비중 등에 활용)"""
    def render(self, labels, values, title, x_title="구분", y_title="금액", **kwargs):
        hole = kwargs.get("hole", 0.4)
        hovertemplate = kwargs.get("hovertemplate", "%{label}<br>매출: %{value:,.0f}원<br>비중: %{percent}<extra></extra>")

        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=hole, 
            textinfo="label+percent",
            hovertemplate=hovertemplate
        )])
        fig.update_layout(
            title=title,
            template="plotly_white"
        )
        return fig


class RadarChartRenderer(ChartRenderer):
    """레이더(폴라) 차트 전담 렌더러 (요일별 편중도 등에 활용)"""
    def render(self, labels, values, title, x_title="구분", y_title="금액", **kwargs):
        # 레이더 차트 특성상 시작점과 끝점을 연결하기 위해 첫 번째 값을 마지막에 추가
        radar_labels = labels + [labels[0]]
        radar_values = values + [values[0]]
        line_color = kwargs.get("line_color", "orange")
        hovertemplate = kwargs.get("hovertemplate", "%{theta}<br>매출: %{r:,.0f}원<extra></extra>")

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=radar_labels,
            fill='toself',
            name=title,
            line=dict(color=line_color, width=2),
            hovertemplate=hovertemplate
        ))

        max_val = max(values) * 1.1 if values else 1
        fig.update_layout(
            title=title,
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max_val])
            ),
            template="plotly_white",
            showlegend=False
        )
        return fig


# ==========================================
# 3. 시각화 메인 클래스
# ==========================================
class PlotlyVisualizer:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.analyzer = SalesAnalyzer(df)
        self.bar_renderer = BarChartRenderer()
        self.line_renderer = LineChartRenderer()
        self.donut_renderer = DonutChartRenderer()
        self.radar_renderer = RadarChartRenderer()

    def _empty_figure(self, message):
        fig = go.Figure()
        fig.update_layout(title=message, template="plotly_white")
        return fig

    # ==========================================
    # 2. 연도 + 분기 비교 (워터폴 차트)
    # ==========================================
    def create_quarter_comparison(self, year_a, quarter_a, year_b, quarter_b):
        comparison_df = self.analyzer.compare_quarters(year_a, quarter_a, year_b, quarter_b)

        if comparison_df is None or comparison_df.empty:
            missing = []
            return self._empty_figure(f"데이터가 없거나 비교에 필요한 컬럼이 없습니다 ({year_a}년 {quarter_a}분기 vs {year_b}년 {quarter_b}분기)."), go.Figure()

        labels = comparison_df["항목"].tolist()  #
        
        labels = ["총매출", "총건수", "평균 객단가"]
        
        values_a = []
        values_b = []
        changes = []

        for _, row in comparison_df.iterrows():
            item_name = row["항목"]
            if item_name == "평균객단가":
                item_name = "평균 객단가"
                
            a = row["기준 기간"]
            b = row["비교 기간"]
            diff = row["증감액"]
            rate = row["증감률(%)"]
            
            values_a.append(a)
            values_b.append(b)
            
            changes.append({
                "항목": item_name, 
                "기준 기간": a, 
                "비교 기간": b, 
                "증감액": diff,
                "증감률(%)": rate
            })

        fig = make_subplots(rows=1, cols=3, subplot_titles=labels, horizontal_spacing=0.1)

        for i, label in enumerate(labels, start=1):
            a, b = values_a[i-1], values_b[i-1]
            diff = b - a
            fig.add_trace(
                go.Waterfall(
                    name=label, orientation="v",
                    measure=["absolute", "relative", "total"],
                    x=[f"{year_a}년 {quarter_a}분기", "증감", f"{year_b}년 {quarter_b}분기"],
                    y=[a, diff, b],
                    texttemplate="%{y:,.0f}", textposition="outside",
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": "#EF553B"}},
                    decreasing={"marker": {"color": "#636EFA"}}
                ),
                row=1, col=i
            )

        fig.update_layout(
            title=f"<b>{year_a}년 {quarter_a}분기 vs {year_b}년 {quarter_b}분기 지표별 증감 비교</b>",
            template="plotly_white", showlegend=False, height=500, margin=dict(t=80, b=40)
        )

        # 4. 결과 테이블 생성
        table = go.Figure(data=[go.Table(
            header=dict(
                values=["항목", "기준 기간", "비교 기간", "증감액", "증감률(%)"],
                fill_color="#2b313e", font=dict(color="white", size=13), align="center", height=35
            ),
            cells=dict(
                values=[
                    [row["항목"] for row in changes],
                    [f'{row["기준 기간"]:,.0f}' for row in changes],
                    [f'{row["비교 기간"]:,.0f}' for row in changes],
                    [f'{row["증감액"]:+,.0f}' for row in changes],
                    [f'{row["증감률(%)"]:+.2f}%' if row["증감률(%)"] is not None else "기준값 0" for row in changes]
                ],
                fill_color="#1e222b", font=dict(color="white", size=12), align="center", height=30
            )
        )])
        table.update_layout(title="분기별 증감 상세 분석", template="plotly_white", height=220, margin=dict(t=40, b=10))

        return fig, table

    # ==========================================
    # 3. 시간대별 매출 => 꺾은선 렌더러 적용
    # ==========================================
    def create_time_sales(self):
        time_df = self.analyzer.get_time_sales()
        if time_df.empty:
            return self._empty_figure("시간대별 매출 데이터가 없습니다.")

        labels = time_df["구분"].tolist()
        values = time_df["매출금액"].tolist()

        peak_idx = int(np.argmax(values))
        low_idx = int(np.argmin(values))

        fig = self.line_renderer.render(
            labels, values, 
            "시간대별 매출 추이 및 피크/최저 시간대", 
            x_title="시간대", 
            y_title="매출 금액 (원)",
            hovertemplate="%{x}시<br>매출: %{y:,.0f}원<extra></extra>"
        )

        fig.add_trace(go.Scatter(
            x=[labels[peak_idx]], y=[values[peak_idx]],
            mode="markers+text", name="피크 시간대",
            text=["🔥 피크"], textposition="top center",
            marker=dict(size=14, color="red", symbol="diamond"),
            hovertemplate="피크 시간대: %{x}시<br>매출: %{y:,.0f}원<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=[labels[low_idx]], y=[values[low_idx]],
            mode="markers+text", name="최저 시간대",
            text=["❄️ 최저"], textposition="bottom center",
            marker=dict(size=14, color="blue", symbol="diamond"),
            hovertemplate="최저 시간대: %{x}시<br>매출: %{y:,.0f}원<extra></extra>"
        ))
        return fig

    # ==========================================
    # 4. 성별 매출 => 막대 렌더러 적용
    # ==========================================
    def create_gender_sales(self):
        gender_data = self.analyzer.get_gender_sales()
        ratio_data = self.analyzer.get_gender_ratio()

        labels = list(gender_data.keys())
        values = list(gender_data.values())
        ratios = list(ratio_data.values())

        fig = self.bar_renderer.render(
            labels, values, 
            title="거래 성별 매출 및 비중 비교", 
            x_title="성별", 
            y_title="매출 금액 (원)", 
            texttemplate="%{customdata:.1f}%",
            customdata=ratios,
            hovertemplate="성별: %{x}<br>매출: %{y:,.0f}원<br>비중: %{customdata:.1f}%<extra></extra>"
        )
        return fig

    # ==========================================
    # 5. 주중 / 주말 매출 => 도넛 렌더러 적용
    # ==========================================
    def create_weekday_weekend_sales(self):
        ww_data = self.analyzer.get_weekday_weekend_sales()
        if not ww_data or sum(ww_data.values()) == 0:
            return self._empty_figure("주중/주말 매출 데이터가 없습니다.")
        
        labels = list(ww_data.keys())
        values = list(ww_data.values())

        return self.donut_renderer.render(
            labels, values,
            title="주중 / 주말 매출 비중 비교"
        )

    # ==========================================
    # 6. 요일별(월~일) 매출 => 레이더 렌더러 적용
    # ==========================================
    def create_weekday_sales(self):
        weekday_df = self.analyzer.get_weekday_sales()
        if weekday_df.empty:
            return self._empty_figure("요일별 매출 데이터가 없습니다.")

        labels = weekday_df["구분"].tolist()
        values = weekday_df["매출금액"].tolist() if "매출금액" in weekday_df.columns else []

        return self.radar_renderer.render(
            labels, values,
            title="요일별(월~일) 매출 편중도 분석 (레이더 차트)"
        )

    # ==========================================
    # 7. 연령대별 매출 => 막대 렌더러 적용
    # ==========================================
    def create_age_sales(self):
        age_df = self.analyzer.get_age_sales()
        if age_df.empty:
            return self._empty_figure("연령대별 매출 데이터가 없습니다.")
        labels = age_df["구분"].tolist()
        values = age_df["매출금액"].tolist()
        
        return self.bar_renderer.render(
            labels, values,
            title="연령대별 매출 및 소비 비중",
            x_title="연령대",
            y_title="매출 금액 (원)"
        )

    @staticmethod
    def normalize_demographic_selection(age_group="전체", gender="전체"):
        if age_group != "전체":
            gender = "전체"
        elif gender != "전체":
            age_group = "전체"
        return {"age_group": age_group, "gender": gender}