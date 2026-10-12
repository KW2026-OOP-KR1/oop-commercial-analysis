import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from filter_analyzer import SalesAnalyzer, DataFilter

class PlotlyVisualizer:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.analyzer = SalesAnalyzer(df)

    def _empty_figure(self, message):
        fig = go.Figure()
        fig.update_layout(title=message, template="plotly_white")
        return fig

    def _bar_figure(self, labels, values, title, x_title="구분", y_title="금액", texttemplate=None):
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=values,
            text=values if texttemplate else None,
            texttemplate=texttemplate,
            textposition="outside" if texttemplate else None,
            hovertemplate="%{x}<br>" + y_title + ": %{y:,.0f}<extra></extra>"
        ))
        fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, template="plotly_white", hovermode="x unified")
        return fig

    # ==========================================
    # 2. 연도 + 분기 비교 (워터폴 차트)
    # ==========================================
    def create_quarter_comparison(self, year_a, quarter_a, year_b, quarter_b):
        quarter_col = "기준_년분기_코드"
        year_col = "분석연도"
        sales_col = "당월_매출_금액"
        count_col = "당월_매출_건수"

        required = [quarter_col, sales_col, count_col]
        if not all(col in self.df.columns for col in required):
            return self._empty_figure("분기 비교에 필요한 컬럼이 없습니다."), go.Figure()

        df = self.df.copy()
        quarter_codes = pd.to_numeric(df[quarter_col], errors="coerce")

        if year_col in df.columns:
            years = pd.to_numeric(df[year_col], errors="coerce").fillna(quarter_codes // 10)
        else:
            years = quarter_codes // 10

        df["_분석연도"] = years
        df["_분석분기"] = quarter_codes % 10
        df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce").fillna(0)
        df[count_col] = pd.to_numeric(df[count_col], errors="coerce").fillna(0)

        def get_period(year, quarter):
            period_df = df[(df["_분석연도"] == int(year)) & (df["_분석분기"] == int(quarter))]
            if period_df.empty:
                return None
            sales = period_df[sales_col].sum()
            count = period_df[count_col].sum()
            return {
                "총매출": sales,
                "총건수": count,
                "평균 객단가": sales / count if count > 0 else 0
            }

        period_a = get_period(year_a, quarter_a)
        period_b = get_period(year_b, quarter_b)

        if period_a is None or period_b is None:
            missing = []
            if period_a is None: missing.append(f"{year_a}년 {quarter_a}분기")
            if period_b is None: missing.append(f"{year_b}년 {quarter_b}분기")
            return self._empty_figure("데이터가 없는 기간: " + ", ".join(missing)), go.Figure()

        labels = ["총매출", "총건수", "평균 객단가"]
        values_a = [period_a["총매출"], period_a["총건수"], period_a["평균 객단가"]]
        values_b = [period_b["총매출"], period_b["총건수"], period_b["평균 객단가"]]

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

        changes = []
        for label, a, b in zip(labels, values_a, values_b):
            changes.append({
                "항목": label, "기준 기간": a, "비교 기간": b, "증감액": b - a,
                "증감률(%)": ((b - a) / abs(a) * 100) if a != 0 else None
            })

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
    # 3. 시간대별 매출 및 피크/최저 시간대 표시
    # ==========================================
    def create_time_sales(self):
        time_df = self.analyzer.get_time_sales()
        if time_df.empty:
            return self._empty_figure("시간대별 매출 데이터가 없습니다.")

        labels = time_df["구분"].tolist()
        values = time_df["매출금액"].tolist()

        peak_idx = int(np.argmax(values))
        low_idx = int(np.argmin(values))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=values, name="시간대별 매출",
            hovertemplate="%{x}시<br>매출: %{y:,.0f}원<extra></extra>"
        ))

        # 피크(최고) 표시
        fig.add_trace(go.Scatter(
            x=[labels[peak_idx]], y=[values[peak_idx]],
            mode="markers+text", name="피크 시간대",
            text=["🔥 피크"], textposition="top center",
            marker=dict(size=14, color="red", symbol="diamond"),
            hovertemplate="피크 시간대: %{x}시<br>매출: %{y:,.0f}원<extra></extra>"
        ))

        # 최저 표시
        fig.add_trace(go.Scatter(
            x=[labels[low_idx]], y=[values[low_idx]],
            mode="markers+text", name="최저 시간대",
            text=["❄️ 최저"], textposition="bottom center",
            marker=dict(size=14, color="blue", symbol="diamond"),
            hovertemplate="최저 시간대: %{x}시<br>매출: %{y:,.0f}원<extra></extra>"
        ))

        fig.update_layout(
            title="시간대별 매출 및 피크/최저 시간대 분석",
            xaxis_title="시간대", yaxis_title="매출 금액 (원)",
            template="plotly_white", hovermode="x unified"
        )
        return fig

    # ==========================================
    # 4. 성별 매출 및 비중
    # ==========================================
    def create_gender_sales(self):
        gender_data = self.analyzer.get_gender_sales()
        ratio_data = self.analyzer.get_gender_ratio()

        labels = list(gender_data.keys())
        values = list(gender_data.values())
        ratios = list(ratio_data.values())

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=values,
            text=[f"{ratio:.1f}%" for ratio in ratios],
            textposition="outside", name="성별 매출 비중",
            customdata=ratios,
            hovertemplate="성별: %{x}<br>매출: %{y:,.0f}원<br>비중: %{customdata:.1f}%<extra></extra>"
        ))
        fig.update_layout(
            title="거래 성별 매출 및 비중 비교",
            xaxis_title="성별", yaxis_title="매출 금액 (원)",
            template="plotly_white"
        )
        return fig

    # ==========================================
    # 5. 주중 / 주말 매출 비교 (별도 막대)
    # ==========================================
    def create_weekday_weekend_sales(self):
        ww_data = self.analyzer.get_weekday_weekend_sales()
        total = sum(ww_data.values())
        labels = list(ww_data.keys())
        values = list(ww_data.values())
        ratios = [value / total * 100 if total else 0 for value in values]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=values,
            text=[f"{ratio:.1f}%" for ratio in ratios],
            textposition="outside", customdata=ratios,
            hovertemplate="%{x}<br>매출: %{y:,.0f}원<br>비중: %{customdata:.1f}%<extra></extra>"
        ))
        fig.update_layout(
            title="주중 / 주말 매출 비교",
            xaxis_title="구분", yaxis_title="매출 금액 (원)",
            template="plotly_white"
        )
        return fig

    # ==========================================
    # 6. 요일별(월~일) 매출 및 최고/최저 요일 편중도 표시
    # ==========================================
    def create_weekday_sales(self):
        weekday_df = self.analyzer.get_weekday_sales()
        if weekday_df.empty:
            return self._empty_figure("요일별 매출 데이터가 없습니다.")

        labels = weekday_df["구분"].tolist()
        values = weekday_df["매출금액"].tolist() if "매출금액" in weekday_df.columns else []

        peak_idx = int(np.argmax(values))
        low_idx = int(np.argmin(values))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=values, name="요일별 매출",
            hovertemplate="%{x}요일<br>매출: %{y:,.0f}원<extra></extra>"
        ))

        # 요일 편중도 (최고 요일)
        fig.add_trace(go.Scatter(
            x=[labels[peak_idx]], y=[values[peak_idx]],
            mode="markers+text", name="최고 매출 요일",
            text=["⭐ 최고"], textposition="top center",
            marker=dict(size=14, color="orange", symbol="star")
        ))

        # 요일 편중도 (최저 요일)
        fig.add_trace(go.Scatter(
            x=[labels[low_idx]], y=[values[low_idx]],
            mode="markers+text", name="최저 매출 요일",
            text=["📉 최저"], textposition="bottom center",
            marker=dict(size=14, color="gray", symbol="star")
        ))

        fig.update_layout(
            title="요일별(월~일) 매출 편중도 분석",
            xaxis_title="요일", yaxis_title="매출 금액 (원)",
            template="plotly_white"
        )
        return fig

    # ==========================================
    # 7. 연령대별 매출
    # ==========================================
    def create_age_sales(self):
        age_df = self.analyzer.get_age_sales()
        if age_df.empty:
            return self._empty_figure("연령대별 매출 데이터가 없습니다.")
        labels = age_df["구분"].tolist()
        values = age_df["매출금액"].tolist()
        return self._bar_figure(labels, values, "연령대별 매출 및 소비 비중", x_title="연령대", y_title="매출 금액 (원)")

    # ==========================================
    # 8. 연령대 / 성별 상호 배타 규칙 함수
    # ==========================================
    @staticmethod
    def normalize_demographic_selection(age_group="전체", gender="전체"):
        if age_group != "전체":
            gender = "전체"
        elif gender != "전체":
            age_group = "전체"
        return {"age_group": age_group, "gender": gender}