"""
줄서기 싫어요 (Zero Waiting) - 대전 오월드 방문자 수 예측 UI
팀: 티익스프레스 / UI 담당

[역할 경계]
- 다른 팀에서 받는 값: 기상 정보, 예측 이용자 수, 예측 혼잡도, 이용자 Score, 날씨 Score
  → UI는 계산하지 않고 그대로 표시한다. (수신 창구: data_provider.fetch_forecast)
- UI가 계산하는 값: 최종 추천 점수, 최종 추천 날짜
  → 이용자 Score + 날씨 Score + 사용자 선호도(슬라이더) 세 변수로 산출한다.

  최종 점수 = 날씨 Score x 날씨 가중치 + 이용자 Score x 이용자 가중치
  (이용자 Score 는 높을수록 한산하다는 뜻이므로, 최고점 날짜를 추천하면 된다)

실행 방법:
    pip install -r requirements.txt
    streamlit run app.py

※ width="stretch" 문법을 쓰므로 Streamlit 1.49 이상이 필요합니다.
   팀원 환경이 더 낮으면 width="stretch" → use_container_width=True 로 바꾸면 됩니다.
"""

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_provider as dp
from theme import (
    BEST_COLOR,
    FONT_FACE_CSS,
    FONT_OPTIONS,
    LEVEL_COLOR,
    LEVEL_COLOR_FALLBACK,
    humidity_color,
    rain_color,
    temp_color,
)

# ---------------------------------------------------------------------------
# 0. 기본 설정
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="줄서기 싫어요 | 대전 오월드 방문자 예측",
    page_icon="🎡",
    layout="wide",
)

N_DAYS = 10  # 당일 포함 예측 일수 (기상 웹서치로 확보 가능한 일수에 맞춰 조정)
SLIDER_DEFAULT = 5  # 선호도 기본값 (날씨 50% / 혼잡도 50%)

# 날씨 Score / 이용자 Score / 최종 추천 점수는 추천 순서를 정하기 위한 내부 계산값이므로
# 사용자 화면에는 노출하지 않는다. 팀 내부 검증이 필요할 때만 True 로 바꿔 쓴다.
SHOW_SCORES = False

# True: 추천은 사이드바, 메인에는 예측 추이와 선택일 기상 (현재 기본).
# False: 예전 기본 레이아웃 — 메인에 추천 카드, 사이드바에 선택일 날씨.
# 예전 레이아웃은 화면에서 쓰지 않지만, 비교·복원용으로 코드는 남겨 둔다.
# 다시 보려면 아래 값을 False 로 바꾸면 된다.
ALT_LAYOUT = True


# ---------------------------------------------------------------------------
# 1. 데이터 수신 (캐시 필수)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60, show_spinner="예측 데이터를 불러오는 중이에요...")
def load_forecast(start_date: dt.date, n_days: int) -> pd.DataFrame:
    """
    슬라이더를 움직일 때마다 스크립트 전체가 재실행되므로 반드시 캐시를 건다.
    캐시가 없으면 선호도 조작 1회당 기상 웹서치가 n_days 번 다시 호출된다.
    """
    return pd.DataFrame(dp.fetch_forecast(start_date, n_days))


today = dt.date.today()
df = load_forecast(today, N_DAYS)
df["label"] = df.apply(lambda r: f"{r['date'].month}/{r['date'].day}({r['weekday']})", axis=1)


# ---------------------------------------------------------------------------
# 2. UI 담당 계산 - 최종 추천 점수 / 최종 추천 날짜
# ---------------------------------------------------------------------------
def calc_final_scores(frame: pd.DataFrame, w_weather: float, w_visitor: float) -> pd.Series:
    """선호도 가중치로 두 Score 를 합산한 0~100 점수."""
    return (frame["weather_score"] * w_weather + frame["visitor_score"] * w_visitor).round(1)


def recommend_reason(row, w_weather: float) -> str:
    """점수 대신 사용자가 읽을 수 있는 말로 추천 이유를 설명한다."""
    crowd = {
        "여유": "사람이 적어 여유롭고",
        "보통": "지나치게 붐비지 않고",
        "혼잡": "사람은 조금 많지만",
        "매우혼잡": "사람은 많은 편이지만",
    }.get(row["crowd_level"], "이 기간 중에서는")

    if row["rain_prob"] >= 60:
        weather = "비 소식이 있어요"
    elif row["temp"] >= 30:
        weather = "조금 더운 편이에요"
    elif row["temp"] <= 5:
        weather = "많이 추운 편이에요"
    elif row["rain_prob"] <= 30 and 15 <= row["temp"] <= 27:
        weather = "나들이하기 좋은 날씨예요"
    else:
        weather = "무난한 날씨예요"

    if w_weather >= 0.7:
        pref = "좋은 날씨를 우선해서 찾았어요."
    elif w_weather <= 0.3:
        pref = "한산한 날을 우선해서 찾았어요."
    else:
        pref = "날씨와 혼잡도를 함께 보고 골랐어요."

    return f"{crowd} {weather}. {pref}"


def outfit_tips(row) -> list[str]:
    """받은 기상값으로 복장·준비물을 안내한다. (기획서 구현 목표 '다')"""
    tips = []
    t, p, h = row["temp"], row["rain_prob"], row["humidity"]
    if p >= 60:
        tips.append("☔ 우산은 꼭 챙기세요")
    elif p >= 30:
        tips.append("🌂 접이식 우산이 있으면 안심이에요")
    if t >= 28:
        tips.append("💧 물과 모자, 그늘에서 쉴 시간을 넉넉히")
    elif t >= 20:
        tips.append("👕 반팔에 얇은 겉옷 하나면 충분해요")
    elif t >= 10:
        tips.append("🧥 아침저녁으로 쌀쌀하니 겉옷을 챙기세요")
    else:
        tips.append("🧣 두꺼운 외투와 장갑이 필요해요")
    if h >= 80 and t >= 25:
        tips.append("🥵 습도가 높아 체감이 더 덥습니다")
    return tips


def weather_metric(label: str, value: str, color: str) -> None:
    st.markdown(
        f"""
        <div style="margin-bottom:0.6rem;">
            <div style="font-size:0.8rem; color:gray;">{label}</div>
            <div style="font-size:1.75rem; font-weight:600; color:{color}; line-height:1.3;">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sel_stat(label: str, value_html: str) -> None:
    """선택 날짜 수치 한 칸 (입장객/혼잡도/기상 공통 구조)."""
    st.markdown(
        f"""
        <div style='font-size:0.8rem; color:gray;'>{label}</div>
        <div style='font-size:1.8rem; font-weight:700; line-height:1.3;'>
            {value_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def hex_rgba(hex_color: str, alpha: float) -> str:
    """막대 선택 강조용 rgba 문자열."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def selection_bar_index(selection, labels: list) -> int | None:
    """Plotly on_select 결과에서 클릭된 막대 인덱스를 읽는다. 선택이 없으면 None."""
    try:
        points = selection["points"]
        if not points:
            return None
        p = points[0]
        for key in ("point_index", "point_number"):
            if p.get(key) is not None:
                idx = int(p[key])
                if 0 <= idx < len(labels):
                    return idx
        x = p.get("x")
        if x in labels:
            return labels.index(x)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return None


def recommend_metrics_html(row) -> str:
    """추천 카드용 혼잡도·기온·강수확률·습도. 레이블로 항목을 구분한다."""
    crowd_c = LEVEL_COLOR.get(row["crowd_level"], LEVEL_COLOR_FALLBACK)
    items = [
        (
            "혼잡도",
            f"<span style='color:{crowd_c};'>●</span> {row['crowd_level']}",
        ),
        (
            "기온",
            f"<span style='color:{temp_color(row['temp'])};'>{row['temp']}°C</span>",
        ),
        (
            "강수확률",
            f"<span style='color:{rain_color(row['rain_prob'])};'>{row['rain_prob']:.0f}%</span>",
        ),
        (
            "습도",
            f"<span style='color:{humidity_color(row['humidity'])};'>{row['humidity']}%</span>",
        ),
    ]
    chips = "".join(
        f"""<span style="display:inline-flex; align-items:baseline; gap:5px; white-space:nowrap;">
                <span style="font-size:0.75rem; color:gray;">{label}</span>
                <span>{value}</span>
            </span>"""
        for label, value in items
    )
    return (
        f'<div style="display:flex; flex-wrap:wrap; align-items:baseline; '
        f'gap:6px 14px; margin-top:8px; font-size:0.95rem; color:#444;">{chips}</div>'
    )


# ---------------------------------------------------------------------------
# 3. 사이드바 (1) - 선호도
# ---------------------------------------------------------------------------
alt_layout = ALT_LAYOUT
st.session_state.setdefault("chart_token", 0)
st.session_state.setdefault("pending_recommend", False)
st.session_state.setdefault("chart_sel_label", None)


def on_weather_pref_change() -> None:
    """슬라이더가 움직이면 예측 추이 선택을 추천일로 바꾼다."""
    st.session_state.pending_recommend = True
    chart_key = f"trend_chart_{st.session_state.chart_token}"
    try:
        points = st.session_state[chart_key]["selection"]["points"]
        if not points:
            st.session_state.ignored_point_index = None
            return
        idx = points[0].get("point_index", points[0].get("point_number"))
        st.session_state.ignored_point_index = int(idx) if idx is not None else None
    except (KeyError, IndexError, TypeError, ValueError):
        st.session_state.ignored_point_index = None


with st.sidebar:
    st.markdown("### 🎯 어떤 날을 찾고 계신가요?")
    weather_pref = st.slider(
        "날씨 중요도",
        min_value=0, max_value=10, value=SLIDER_DEFAULT,
        key="weather_pref",
        on_change=on_weather_pref_change,
        label_visibility="collapsed",
    )
    ends_l, ends_r = st.columns(2)
    ends_l.caption("← 한산한 날")
    ends_r.markdown(
        "<div style='text-align:right; color:gray; font-size:0.8rem;'>좋은 날씨 →</div>",
        unsafe_allow_html=True,
    )

    # 선호도 = float 2개 (합 1.0) - 파이프라인 정의와 동일
    w_weather = weather_pref / 10
    w_visitor = 1 - w_weather
    st.caption(f"지금 기준 · 날씨 {w_weather:.0%} / 한산함 {w_visitor:.0%}")

# 선호도가 정해져야 최종 점수와 추천 날짜가 나온다
df["final_score"] = calc_final_scores(df, w_weather, w_visitor)
ranked = df.sort_values(["final_score", "date"], ascending=[False, True]).reset_index()
best_idx = int(ranked.loc[0, "index"])
best = df.loc[best_idx]


# ---------------------------------------------------------------------------
# 4. 그래프에서 선택한 날짜 판별
#    첫 진입 = 오늘, 슬라이더 변경 후 = 추천일, 막대 클릭 = 그 날짜
# ---------------------------------------------------------------------------
chart_key = f"trend_chart_{st.session_state.chart_token}"
labels = list(df["label"])
today_idx = next((i for i, d in enumerate(df["date"]) if d == today), 0)

if st.session_state.pending_recommend:
    st.session_state.chart_sel_label = labels[best_idx]
    st.session_state.pending_recommend = False

click_idx = None
try:
    click_idx = selection_bar_index(st.session_state[chart_key]["selection"], labels)
except (KeyError, TypeError):
    click_idx = None

if click_idx is not None and click_idx != st.session_state.get("ignored_point_index"):
    st.session_state.chart_sel_label = labels[click_idx]
    st.session_state.ignored_point_index = None

if st.session_state.chart_sel_label not in labels:
    st.session_state.chart_sel_label = labels[today_idx]
sel_idx = labels.index(st.session_state.chart_sel_label)

sel = df.loc[sel_idx]
sel_date = sel["date"]
sel_level_color = LEVEL_COLOR.get(sel["crowd_level"], LEVEL_COLOR_FALLBACK)


def render_recommend(*, compact: bool) -> None:
    """최종 추천 섹션. compact=True 이면 사이드바용 좁은 레이아웃."""
    if compact:
        st.markdown("### 🏆 이 날 추천해요")
        st.markdown(
            f"""
            <div style="border:1px solid rgba(0,0,0,0.08); border-left:6px solid {BEST_COLOR};
                        border-radius:12px; padding:14px 16px;">
                <div style="font-size:0.8rem; color:gray;">최종 추천 날짜</div>
                <div style="font-size:1.45rem; font-weight:700; line-height:1.3;">
                    {best['date'].strftime('%m월 %d일')} ({best['weekday']})
                </div>
                {recommend_metrics_html(best)}
                <div style="margin-top:10px; padding-top:10px; border-top:1px solid rgba(0,0,0,0.06);">
                    <div style="font-size:0.8rem; color:gray;">예상 입장객 수</div>
                    <div style="font-size:1.45rem; font-weight:700;">
                        {best['pred_visitors']:,}<span style="font-size:1.1rem; font-weight:600;">명</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(recommend_reason(best, w_weather))
        if SHOW_SCORES:
            st.caption(
                f"[내부] 날씨 {best['weather_score']} × {w_weather:.0%} + "
                f"한산함 {best['visitor_score']} × {w_visitor:.0%} = {best['final_score']:.1f}"
            )
        tips = outfit_tips(best)
        if tips:
            st.caption("　".join(tips))

        st.markdown(
            "<div style='font-size:0.85rem; color:gray; margin-top:0.6rem;'>다음 후보</div>",
            unsafe_allow_html=True,
        )
        for i in range(1, min(4, len(ranked))):
            row = df.loc[int(ranked.loc[i, "index"])]
            st.markdown(
                f"**{i + 1}위** {row['date'].strftime('%m/%d')}({row['weekday']}) · "
                f"<span style='color:{LEVEL_COLOR.get(row['crowd_level'], LEVEL_COLOR_FALLBACK)};'>●</span> "
                f"{row['crowd_level']} · {row['temp']}°C"
                + (f" · {row['final_score']:.1f}" if SHOW_SCORES else ""),
                unsafe_allow_html=True,
            )
        return

    st.markdown("### 🏆 이 날 가시는 걸 추천해요")
    rec_l, rec_r = st.columns([1.15, 1])

    with rec_l:
        st.markdown(
            f"""
            <div style="border:1px solid rgba(0,0,0,0.08); border-left:6px solid {BEST_COLOR};
                        border-radius:12px; padding:18px 40px 18px 22px;
                        display:flex; justify-content:space-between; align-items:center; gap:16px;">
                <div style="min-width:0; flex:1;">
                    <div style="font-size:0.85rem; color:gray;">최종 추천 날짜</div>
                    <div style="font-size:2rem; font-weight:700; line-height:1.3;">
                        {best['date'].strftime('%m월 %d일')} ({best['weekday']})
                    </div>
                    {recommend_metrics_html(best)}
                </div>
                <div style="text-align:right; flex-shrink:0;">
                    <div style="font-size:0.85rem; color:gray;">예상 입장객 수</div>
                    <div style="font-size:2rem; font-weight:700; line-height:1.3;">
                        {best['pred_visitors']:,}<span style="font-size:2.0rem; font-weight:600;">명</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(recommend_reason(best, w_weather))
        if SHOW_SCORES:
            st.caption(
                f"[내부] 날씨 {best['weather_score']} × {w_weather:.0%} + "
                f"한산함 {best['visitor_score']} × {w_visitor:.0%} = {best['final_score']:.1f}"
            )
        tips = outfit_tips(best)
        if tips:
            st.markdown("　".join(tips))

    with rec_r:
        st.markdown("<div style='font-size:0.85rem; color:gray;'>다음 후보</div>", unsafe_allow_html=True)
        for i in range(1, min(4, len(ranked))):
            row = df.loc[int(ranked.loc[i, "index"])]
            c1, c2, c3 = st.columns([1.1, 1, 1])
            c1.markdown(f"**{i + 1}위** {row['date'].strftime('%m/%d')}({row['weekday']})")
            c2.markdown(
                f"<span style='color:{LEVEL_COLOR.get(row['crowd_level'], LEVEL_COLOR_FALLBACK)};'>●</span> "
                f"{row['crowd_level']}",
                unsafe_allow_html=True,
            )
            c3.markdown(
                f"{row['temp']}°C · 💧{row['rain_prob']:.0f}%"
                + (f" · {row['final_score']:.1f}" if SHOW_SCORES else "")
            )


def render_trend(*, show_weather: bool) -> None:
    """향후 N일 예측 추이 섹션."""
    st.markdown(f"### 📈 향후 {N_DAYS}일 예측 추이")
    st.caption("궁금한 날짜의 막대를 클릭하면 아래 수치가 그 날 기준으로 바뀌어요.")

    _suffix = ", 오늘" if sel_date == today else (", 추천일" if sel_idx == best_idx else "")
    st.markdown(
        f"""
        <div style="font-size:1.15rem; font-weight:600; color:#1F2A37; margin:0.35rem 0 0.6rem;">
            {sel_date.strftime('%m월 %d일')} ({sel['weekday']}{_suffix}) 기준
        </div>
        """,
        unsafe_allow_html=True,
    )

    if show_weather:
        r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
        with r1c1:
            sel_stat("예상 입장객 수", f"{sel['pred_visitors']:,}명")
        with r1c2:
            sel_stat(
                "혼잡도",
                f"<span style='color:{sel_level_color};'>●</span> {sel['crowd_level']}",
            )
        with r1c3:
            sel_stat(
                "평균기온",
                f"<span style='color:{temp_color(sel['temp'])};'>{sel['temp']}°C</span>",
            )
        with r1c4:
            sel_stat(
                "강수확률",
                f"<span style='color:{rain_color(sel['rain_prob'])};'>{sel['rain_prob']:.0f}%</span>",
            )
        with r1c5:
            sel_stat(
                "평균습도",
                f"<span style='color:{humidity_color(sel['humidity'])};'>{sel['humidity']}%</span>",
            )
    else:
        m1, m2 = st.columns(2)
        with m1:
            sel_stat("예상 입장객 수", f"{sel['pred_visitors']:,}명")
        with m2:
            sel_stat(
                "혼잡도",
                f"<span style='color:{sel_level_color};'>●</span> {sel['crowd_level']}",
            )

    y_values = df["pred_visitors"]
    y_title = "예상 입장객 수(명)"
    bar_text = [f"{v:,}" for v in df["pred_visitors"]]
    hover = "%{x}<br>예상 입장객: %{y:,}명<br>혼잡도: %{customdata}<extra></extra>"

    bar_colors = [
        hex_rgba(LEVEL_COLOR.get(lv, LEVEL_COLOR_FALLBACK), 1.0 if i == sel_idx else 0.55)
        for i, lv in enumerate(df["crowd_level"])
    ]
    line_colors = [BEST_COLOR if i == best_idx else "rgba(0,0,0,0)" for i in range(len(df))]
    line_widths = [3 if i == best_idx else 0 for i in range(len(df))]

    fig = go.Figure()
    fig.add_bar(
        x=df["label"],
        y=y_values,
        marker_color=bar_colors,
        marker_line_color=line_colors,
        marker_line_width=line_widths,
        text=bar_text,
        textposition="outside",
        textfont=dict(color="#555555"),
        customdata=list(df["crowd_level"]),
        hovertemplate=hover,
    )
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title=y_title,
        xaxis_title=None,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#555555"),
    )
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.06)", zeroline=False)
    fig.update_xaxes(
        showgrid=False,
        tickvals=list(df["label"]),
        ticktext=[f"<b>{l}</b>" for l in df["label"]],
        tickfont=dict(size=13, color="#1F2A37"),
    )
    st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key=chart_key,
        config={"displayModeBar": False},
    )

    st.markdown(
        " &nbsp;·&nbsp; ".join(
            f"<span style='color:{LEVEL_COLOR.get(lv, LEVEL_COLOR_FALLBACK)}; font-size:1.1rem;'>●</span> {lv}"
            for lv in dp.CROWD_LEVELS
        )
        + f" &nbsp;·&nbsp; <span style='color:{BEST_COLOR};'>▢</span> 추천일",
        unsafe_allow_html=True,
    )

    with st.expander(f"📋 {N_DAYS}일치 표로 보기"):
        cols = ["label", "temp", "rain_prob", "humidity", "pred_visitors", "crowd_level"]
        names = {
            "label": "날짜", "temp": "기온(°C)", "rain_prob": "강수확률(%)", "humidity": "습도(%)",
            "pred_visitors": "예상 입장객", "crowd_level": "혼잡도",
        }
        if SHOW_SCORES:  # 내부 검증용 컬럼
            cols += ["visitor_score", "weather_score", "final_score"]
            names |= {"visitor_score": "이용자 Score", "weather_score": "날씨 Score",
                      "final_score": "최종 점수"}
        table = df[cols].rename(columns=names)
        st.dataframe(table, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# 5. 사이드바 (2) - 날씨 / 추천 / 화면 설정
# ---------------------------------------------------------------------------
with st.sidebar:
    st.divider()

    if alt_layout:
        render_recommend(compact=True)
    else:
        # 미사용(예전 기본 UI). ALT_LAYOUT=False 일 때만 사이드바에 선택일 날씨를 보여 준다.
        _label = "오늘" if sel_date == today else sel_date.strftime("%m월 %d일")
        st.markdown(f"### 🌤️ {_label} 날씨")
        st.caption("그래프에서 선택한 날짜의 날씨예요.")
        weather_metric("평균기온", f"{sel['temp']}°C", temp_color(sel["temp"]))
        weather_metric("강수확률", f"{sel['rain_prob']:.0f}%", rain_color(sel["rain_prob"]))
        weather_metric("평균습도", f"{sel['humidity']}%", humidity_color(sel["humidity"]))

    st.divider()
    with st.expander("🔤 화면 설정"):
        selected_font = st.selectbox("사이트 폰트", options=list(FONT_OPTIONS.keys()), index=0)
        if st.button("날짜 선택 초기화", width="stretch"):
            st.session_state.chart_token += 1
            st.session_state.pending_recommend = False
            st.session_state.chart_sel_label = None
            st.session_state.ignored_point_index = None
            st.rerun()

    st.markdown("## 🎡 줄서기 싫어요")
    st.caption("대전 오월드 방문자 수 예측 · 티익스프레스")

st.markdown(
    f"""
    <style>
    {FONT_FACE_CSS}
    html, body, .stApp, .stApp *,
    [class*="css"], [class^="st-"], [class*=" st-"],
    button, input, select, textarea, label,
    [data-baseweb="select"], [data-baseweb="select"] *,
    [data-baseweb="popover"], [data-baseweb="popover"] *,
    [data-baseweb="menu"], [data-baseweb="menu"] *,
    [role="listbox"], [role="listbox"] *,
    [role="option"] {{
        font-family: '{FONT_OPTIONS[selected_font]}', sans-serif !important;
    }}
    /* Material Icons 는 커스텀 폰트 덮어쓰기를 되돌린다 (arrow_down 등 글자 노출 방지) */
    [data-testid="stIconMaterial"],
    [data-testid="stSpinnerIcon"],
    [data-testid="stImageIcon"] {{
        font-family: "Material Symbols Rounded" !important;
        font-feature-settings: "liga" !important;
        -webkit-font-feature-settings: "liga" !important;
    }}
    /* 호버 툴팁이 막대 클릭을 가로채지 않게 한다 */
    .stPlotlyChart .hoverlayer, .js-plotly-plot .hoverlayer {{
        pointer-events: none !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 6. 메인
# ---------------------------------------------------------------------------
st.title("대전 오월드 방문자 수 예측")
st.caption(f"오늘부터 {N_DAYS}일간의 예상 입장객과 혼잡도를 보고, 취향에 맞는 방문일을 찾아보세요.")

if alt_layout:
    render_trend(show_weather=True)
else:
    # 미사용(예전 기본 UI). ALT_LAYOUT=False 일 때만 메인에 추천 카드가 먼저 온다.
    render_recommend(compact=False)
    st.divider()
    render_trend(show_weather=False)

# --- 푸터 ---
st.divider()
st.caption(
    "프로토타입 화면입니다. 예측·혼잡도 값은 아직 임시 데이터이며, "
    "data_provider.py 의 USE_MOCK 을 끄면 실제 모델·API 값으로 바뀝니다. | 티익스프레스"
)
