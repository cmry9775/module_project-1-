"""
줄서기 싫어요 (Zero Waiting) - 대전 오월드 방문자 수 예측 UI
팀: 티익스프레스 / UI 담당

[역할 경계]
- 다른 팀에서 받는 값: 기상 정보, 예측 이용자 수, 예측 혼잡도, 이용자 Score, 날씨 Score
  → UI는 계산하지 않고 그대로 표시한다.
    (수신 창구: data_provider.fetch_forecast / fetch_notice)
- UI가 계산하는 값: 최종 추천 점수, 최종 추천 날짜
  → 이용자 Score + 날씨 Score + 사용자 선호도(슬라이더) 세 변수로 산출한다.

  최종 점수 = 날씨 Score x 날씨 가중치 + 이용자 Score x 이용자 가중치
  (이용자 Score 는 높을수록 한산하다는 뜻이므로, 최고점 날짜를 추천하면 된다)

실행 방법:
    pip install -r requirements.txt
    streamlit run app.py

※ 사이드바 초기 폭(initial_sidebar_state=정수)과 width="stretch" 문법을 쓰므로
   Streamlit 1.61 이상이 필요합니다.
"""

import hashlib
import pathlib
from zoneinfo import ZoneInfo

import folium
from streamlit_folium import st_folium

# 뉴스 블록 보류 (사이드바 '대전 오월드 최신 뉴스' 참고)
# import os
# from openai import OpenAI
# api_key = os.getenv("OPENAI_API_KEY")

import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_provider as dp
from theme import (
    BEST_COLOR,
    DATE_LABEL_COLOR,
    FONT_FACE_CSS,
    FONT_OPTIONS,
    HOLIDAY_LABEL_COLOR,
    LEVEL_COLOR,
    LEVEL_COLOR_FALLBACK,
    NOTICE_COLOR,
    NOTICE_COLOR_FALLBACK,
    NOTICE_STATE_COLOR,
    NOTICE_STATE_COLOR_FALLBACK,
    SATURDAY_LABEL_COLOR,
    humidity_color,
    rain_color,
    score_color,
    temp_color,
)

# ---------------------------------------------------------------------------
# 0. 기본 설정
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="줄서기 싫어요 | 대전 오월드 방문자 예측",
    page_icon="🎡",
    layout="wide",
    # 추천 카드와 후보 목록이 접히지 않도록 사이드바를 넓게 연다.
    # 정수는 초기 폭(px)이고 허용 범위는 200~600 이다.
    initial_sidebar_state=600,
)

N_DAYS = 10  # 당일 포함 예측 일수 (기상 웹서치로 확보 가능한 일수에 맞춰 조정)

# 파이프라인이 Asia/Seoul 기준으로 오늘부터 N_DAYS 를 주므로 '오늘'을 같은 기준으로 잡는다.
# 서버 시간대가 다르면 첫 칸이 오늘이 아니게 되어 '오늘' 표시와 첫 선택이 어긋난다.
SEOUL = ZoneInfo("Asia/Seoul")
SLIDER_DEFAULT = 5  # 선호도 기본값 (날씨 50% / 혼잡도 50%)

# 세 점수의 원시값(계산식)까지 보여 줄지 여부. 화면에는 '추천 점수 자세히 보기'로
# 가공한 형태만 내보내고, 계산 과정은 팀 내부 검증이 필요할 때만 True 로 바꿔 쓴다.
SHOW_SCORES = False


# ---------------------------------------------------------------------------
# 1. 데이터 수신 (캐시 필수)
# ---------------------------------------------------------------------------
# st.cache_data 는 로더 본문만 해시하므로 data_provider.py 를 고쳐도
# 캐시가 그대로 남는다. 수신 스키마가 바뀌면 예전 결과에 없는 컬럼을 읽다가
# KeyError 로 죽으므로, 파일 지문을 캐시 키에 넣어 자동으로 무효화한다.
PROVIDER_FINGERPRINT = hashlib.md5(
    pathlib.Path(dp.__file__).read_bytes(), usedforsecurity=False
).hexdigest()


# 첫 로딩은 기상 조회와 모델 추론이 함께 돌아 몇 초 걸린다. 캐시 기본 스피너는
# 화면 왼쪽 위에 작게 떠서 멈춘 화면처럼 보이므로, 화면을 덮는 로딩 화면을 쓴다.
LOADING_SPLASH_HTML = f"""
<style>
{FONT_FACE_CSS}
@keyframes zw-spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes zw-pulse {{ 0%, 100% {{ opacity: 0.45; }} 50% {{ opacity: 1; }} }}
.zw-splash {{
    position: fixed; inset: 0; z-index: 9999;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 20px;
    background: rgba(250, 251, 253, 0.94);
    backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
    font-family: 'Pretendard', sans-serif;
}}
.zw-splash-ring {{
    width: 66px; height: 66px; border-radius: 50%;
    border: 5px solid rgba(123, 167, 212, 0.22);
    border-top-color: {BEST_COLOR};
    animation: zw-spin 0.9s linear infinite;
}}
.zw-splash-title {{ font-size: 1.2rem; font-weight: 700; color: #1F2A37; }}
.zw-splash-sub {{
    font-size: 0.9rem; color: #6B7684;
    animation: zw-pulse 1.6s ease-in-out infinite;
}}
</style>
<div class="zw-splash">
    <div class="zw-splash-ring"></div>
    <div class="zw-splash-title">🎡 예측 데이터를 불러오는 중이에요</div>
    <div class="zw-splash-sub">기상 정보를 모아 10일간 방문자 수를 계산하고 있어요</div>
</div>
"""


def cache_slot(hours: int) -> str:
    """hours 시간마다 값이 바뀌는 문자열. 캐시 신선도를 키로 만든다.

    persist="disk" 는 ttl 을 무시하므로 ttl 로는 갱신 주기를 잡을 수 없다.
    대신 이 값을 인자로 넘겨, 시간대가 바뀌면 새 키가 되어 그때 한 번만
    다시 받게 한다.
    """
    now = dt.datetime.now(SEOUL)
    return f"{now:%Y-%m-%d}-{now.hour // hours}"


# persist="disk" 를 쓰는 이유: 기본 캐시는 메모리에만 있어서 앱을 재시작할 때마다
# 기상 조회와 OpenAI 호출이 처음부터 다시 돈다. 디스크에 두면 재시작해도 남는다.
@st.cache_data(persist="disk", max_entries=48, show_spinner=False)
def load_forecast(
    start_date: dt.date, n_days: int, slot: str, fingerprint: str
) -> pd.DataFrame:
    """
    슬라이더를 움직일 때마다 스크립트 전체가 재실행되므로 반드시 캐시를 건다.
    캐시가 없으면 선호도 조작 1회당 기상 조회가 다시 호출된다.

    slot 과 fingerprint 는 캐시 키 용도로만 받는다. 함수 안에서는 쓰지 않는다.
    """
    # 수신 건수가 0이거나 일부 키가 빠져도 컬럼 구성은 항상 같게 만든다.
    return pd.DataFrame(dp.fetch_forecast(start_date, n_days), columns=dp.ROW_COLUMNS)


@st.cache_data(persist="disk", max_entries=16, show_spinner=False)
def load_notice(start_date: dt.date, slot: str, fingerprint: str) -> dict:
    """공지는 예측과 따로 받는다.

    공지는 게시판 크롤링과 AI 요약을 거쳐 예측보다 훨씬 오래 걸린다. 한 번에
    받으면 공지가 끝날 때까지 그래프와 추천이 화면에 나오지 않으므로, 예측만
    먼저 받아 메인을 그린 뒤 스크립트 맨 끝에서 이 함수를 호출한다.

    OpenAI 호출이라 예측보다 비싸고 공지는 자주 바뀌지 않으므로, 갱신 주기를
    예측(1시간)보다 길게 잡는다.
    """
    return dp.fetch_notice(start_date)


today = dt.datetime.now(SEOUL).date()

# 캐시가 비어 있을 때만 실제로 기다리게 된다. 캐시가 있으면 곧바로 지워진다.
splash = st.empty()
splash.markdown(LOADING_SPLASH_HTML, unsafe_allow_html=True)
df = load_forecast(today, N_DAYS, cache_slot(1), PROVIDER_FINGERPRINT)
splash.empty()

if df.empty or not df["ok"].any():
    st.title("대전 오월드 방문자 수 예측")
    st.error("예측 데이터를 받지 못했어요. 잠시 후 다시 시도해 주세요.")
    st.stop()

df["label"] = df.apply(
    lambda r: f"{r['date'].month}/{r['date'].day}({r['weekday']})", axis=1
)


# ---------------------------------------------------------------------------
# 2. UI 담당 계산 - 최종 추천 점수 / 최종 추천 날짜
# ---------------------------------------------------------------------------
def calc_final_scores(
    frame: pd.DataFrame, w_weather: float, w_visitor: float
) -> pd.Series:
    """선호도 가중치로 두 Score 를 합산한 0~100 점수."""
    return (
        frame["weather_score"] * w_weather + frame["visitor_score"] * w_visitor
    ).round(1)


def day_tag(row) -> str:
    """요일 뒤에 붙일 휴일/주말 표시. 평일이면 빈 문자열."""
    if row["is_holiday"]:
        return row["holiday_name"] or "공휴일"
    if row["is_weekend"]:
        return "주말"
    return "휴일" if row["is_dayoff"] else ""


def date_label_color(row) -> str:
    """달력 표기를 따라 토요일은 파랑, 일요일과 공휴일은 빨강."""
    if row["is_holiday"] or row["date"].weekday() == 6:
        return HOLIDAY_LABEL_COLOR
    if row["date"].weekday() == 5:
        return SATURDAY_LABEL_COLOR
    return DATE_LABEL_COLOR


def date_title(row) -> str:
    """카드 제목용 '08월 15일 (토 · 광복절)' 형식."""
    tag = day_tag(row)
    suffix = f" · {tag}" if tag else ""
    return f"{row['date'].strftime('%m월 %d일')} ({row['weekday']}{suffix})"


def recommend_reason(row, w_weather: float) -> str:
    """점수 대신 사용자가 읽을 수 있는 말로 추천 이유를 설명한다."""
    crowd = {
        "여유": "사람이 적어 여유롭고",
        "보통": "지나치게 붐비지 않고",
        "혼잡": "사람은 조금 많지만",
    }.get(row["crowd_level"], "이 기간 중에서는")

    summary = row["weather_summary"]
    if row["rain_prob"] >= 60:
        weather = f"{summary or '비'} 예보라 우산이 필요해요"
    elif row["temp"] >= 30:
        weather = "조금 더운 편이에요"
    elif row["temp"] <= 5:
        weather = "많이 추운 편이에요"
    elif row["rain_prob"] <= 30 and 15 <= row["temp"] <= 27:
        weather = f"{summary or '맑은 날씨'} 예보라 나들이하기 좋아요"
    else:
        weather = f"{summary or '무난한 날씨'} 예보예요"

    if w_weather >= 0.7:
        pref = "좋은 날씨를 우선해서 찾았어요."
    elif w_weather <= 0.3:
        pref = "한산한 날을 우선해서 찾았어요."
    else:
        pref = "날씨와 혼잡도를 함께 보고 골랐어요."

    if row["is_next_dayoff"]:
        pref += " 다음 날도 쉬는 날이라 평소보다 붐빌 수 있어요."

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


def chip_row(items) -> str:
    """(레이블, 값) 목록을 한 줄 칩으로 그린다."""
    return "".join(
        f"""<span style="display:inline-flex; align-items:baseline; gap:5px; white-space:nowrap;">
                <span style="font-size:0.75rem; color:gray;">{label}</span>
                <span>{value}</span>
            </span>"""
        for label, value in items
    )


def weather_chip_items(row) -> list[tuple[str, str]]:
    """날씨·기온·강수확률·습도. 추천 카드와 예측 추이가 같은 형식을 쓴다."""
    return [
        (
            "날씨",
            f"{row['weather_icon']} {row['weather_summary']}".strip() or "-",
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


def crowd_chip_value(row) -> str:
    crowd_c = LEVEL_COLOR.get(row["crowd_level"], LEVEL_COLOR_FALLBACK)
    return f"<span style='color:{crowd_c};'>●</span> {row['crowd_level']}"


def recommend_metric_chips(row) -> str:
    """추천 카드용. 날씨 칩 사이에 혼잡도를 끼워 넣는다."""
    items = weather_chip_items(row)
    items.insert(1, ("혼잡도", crowd_chip_value(row)))
    return chip_row(items)


def recommend_metrics_html(row) -> str:
    """추천 카드용 날씨·혼잡도 줄. 레이블로 항목을 구분한다."""
    return (
        f'<div style="display:flex; flex-wrap:wrap; align-items:baseline; '
        f'gap:6px 14px; margin-top:8px; font-size:0.95rem; color:#444;">'
        f"{recommend_metric_chips(row)}</div>"
    )


def rank_candidate_html(rank: int, row) -> str:
    """다음 후보 한 줄. 사이드바가 좁아 이모지로 항목을 구분한다.

    순위 · 날짜 · 날씨 · 기온 · 혼잡도 · 예상 입장객
    """
    values = [
        f"<span style='font-weight:700;'>"
        f"{rank}위 {row['date'].strftime('%m/%d')}({row['weekday']})</span>",
        f"{row['weather_icon']} {row['weather_summary']}".strip() or "-",
        f"🌡️ <span style='color:{temp_color(row['temp'])};'>{row['temp']}°C</span>",
        crowd_chip_value(row),
        f"예상 입장객 {row['pred_visitors']:,}명",
    ]
    if SHOW_SCORES:
        values.append(
            f"<span style='color:gray;'>{row['final_score']:.1f}</span>"
        )
    cells = "<span style='color:#C9CED6; margin:0 6px;'>·</span>".join(
        f"<span style='white-space:nowrap;'>{v}</span>" for v in values
    )
    return (
        f'<div style="margin:6px 0; font-size:0.95rem; color:#444; '
        f'line-height:1.7;">{cells}</div>'
    )


# ---------------------------------------------------------------------------
# 3. 사이드바 (1) - 선호도
# ---------------------------------------------------------------------------
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
        min_value=0,
        max_value=10,
        value=SLIDER_DEFAULT,
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
# 예보를 못 받은 날(status != success)은 점수가 신뢰할 수 없으므로 추천 후보에서 뺀다.
# 전부 실패한 경우는 위에서 걸러 냈으므로 여기서는 최소 한 건이 남는다.
ranked = df[df["ok"]].sort_values(
    ["final_score", "date"], ascending=[False, True]
).reset_index()
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


def render_score_breakdown(row) -> None:
    """추천에 쓰인 세 점수를 접이식으로 보여 준다."""
    weather = round(float(row["weather_score"]))
    visitor = round(float(row["visitor_score"]))
    total = round(float(row["final_score"]))

    with st.expander("추천 점수 자세히 보기"):
        st.caption(f"현재 반영 비율 => 날씨 {w_weather:.0%} / 한산함 {w_visitor:.0%}")
        cols = st.columns(3)
        for col, label, value in zip(
            cols,
            ("날씨 점수", "한산함 점수", "종합 추천 점수"),
            (weather, visitor, total),
        ):
            color = score_color(value)
            with col:
                st.markdown(
                    f"""
                    <div style='font-size:0.8rem; color:gray;'>{label}</div>
                    <div style='font-size:1.45rem; font-weight:700; line-height:1.3; color:{color}; margin-bottom:0.4rem;'>
                        {value}점
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_recommend() -> None:
    """최종 추천 섹션. 사이드바 폭에 맞춘 좁은 레이아웃이다."""
    st.markdown("### 🏆 이 날 추천해요")
    st.markdown(
        f"""
            <div style="border:1px solid rgba(0,0,0,0.08); border-left:6px solid {BEST_COLOR};
                        border-radius:12px; padding:14px 16px; margin-bottom:8px;">
                <div style="font-size:0.8rem; color:gray;">최종 추천 날짜</div>
                <div style="font-size:1.45rem; font-weight:700; line-height:1.3;">
                    {date_title(best)}
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
    # 두 문장을 한 캡션으로 묶는다. 블록을 나누면 Streamlit 기본 간격이 끼어든다.
    lines = [recommend_reason(best, w_weather)]
    tips = outfit_tips(best)
    if tips:
        lines.append("　".join(tips))
    st.caption("<br>".join(lines), unsafe_allow_html=True)
    render_outlier_note(best)
    render_score_breakdown(best)
    if SHOW_SCORES:
        st.caption(
            f"[내부] 날씨 {best['weather_score']} × {w_weather:.0%} + "
            f"한산함 {best['visitor_score']} × {w_visitor:.0%} = {best['final_score']:.1f}"
        )

    st.markdown(
        "<div style='font-size:0.85rem; color:gray; margin-top:0.6rem;'>다음 후보</div>",
        unsafe_allow_html=True,
    )
    cards = []
    for i in range(1, min(4, len(ranked))):
        row = df.loc[int(ranked.loc[i, "index"])]
        cards.append(rank_candidate_html(i + 1, row))
    st.markdown("".join(cards), unsafe_allow_html=True)


def render_outlier_note(row) -> None:
    """학습에서 이상치로 다루는 날이면 예측값을 곧이곧대로 믿지 않도록 안내한다."""
    note = dp.outlier_note(row["date"])
    if note:
        st.info(note, icon="🎈")


def render_trend() -> None:
    """향후 N일 예측 추이 섹션."""
    st.markdown(f"### 📈 향후 {N_DAYS}일 예측 추이")
    st.caption("궁금한 날짜의 막대를 클릭하면 아래 수치가 그 날 기준으로 바뀌어요.")

    _tag = day_tag(sel)
    _marks = [sel["weekday"]]
    if _tag:
        _marks.append(_tag)
    if sel_date == today:
        _marks.append("오늘")
    elif sel_idx == best_idx:
        _marks.append("추천일")
    st.markdown(
        f"""
        <div style="font-size:1.15rem; font-weight:600; color:#1F2A37; margin:0.35rem 0 0.6rem;">
            {sel_date.strftime('%m월 %d일')} ({' · '.join(_marks)}) 기준
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_outlier_note(sel)

    if not sel["ok"]:
        st.warning("이 날은 예측 데이터를 받지 못했어요. 다른 날짜를 선택해 주세요.")
    else:
        # 날씨 칸만 글자가 길어 폭을 조금 더 준다.
        c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1.3, 1, 1, 1])
        with c1:
            sel_stat("예상 입장객 수", f"{sel['pred_visitors']:,}명")
        with c2:
            sel_stat(
                "혼잡도",
                f"<span style='color:{sel_level_color};'>●</span> {sel['crowd_level']}",
            )
        with c3:
            sel_stat(
                "날씨",
                f"{sel['weather_icon']} {sel['weather_summary']}".strip() or "-",
            )
        with c4:
            sel_stat(
                "평균기온",
                f"<span style='color:{temp_color(sel['temp'])};'>{sel['temp']}°C</span>",
            )
        with c5:
            sel_stat(
                "강수확률",
                f"<span style='color:{rain_color(sel['rain_prob'])};'>{sel['rain_prob']:.0f}%</span>",
            )
        with c6:
            sel_stat(
                "평균습도",
                f"<span style='color:{humidity_color(sel['humidity'])};'>{sel['humidity']}%</span>",
            )

    y_values = df["pred_visitors"]
    y_title = "예상 입장객 수(명)"
    bar_text = [
        f"{v:,}" if ok else "예보 없음" for v, ok in zip(df["pred_visitors"], df["ok"])
    ]
    hover = (
        "%{x}<br>예상 입장객: %{y:,}명<br>혼잡도: %{customdata[0]}"
        "<br>날씨: %{customdata[1]}<extra></extra>"
    )

    bar_colors = [
        hex_rgba(
            LEVEL_COLOR.get(lv, LEVEL_COLOR_FALLBACK) if ok else LEVEL_COLOR_FALLBACK,
            (1.0 if i == sel_idx else 0.55) if ok else 0.25,
        )
        for i, (lv, ok) in enumerate(zip(df["crowd_level"], df["ok"]))
    ]
    line_colors = [
        BEST_COLOR if i == best_idx else "rgba(0,0,0,0)" for i in range(len(df))
    ]
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
        customdata=[
            [lv, f"{icon} {summary}".strip() or "-"]
            for lv, icon, summary in zip(
                df["crowd_level"], df["weather_icon"], df["weather_summary"]
            )
        ],
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
        ticktext=[
            f"<b><span style='color:{date_label_color(r)}'>{r['label']}</span></b>"
            for _, r in df.iterrows()
        ],
        tickfont=dict(size=13, color=DATE_LABEL_COLOR),
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

    failed = list(df.loc[~df["ok"], "label"])
    if failed:
        st.warning(
            f"{', '.join(failed)} 은(는) 예보를 받지 못해 추천 후보에서 제외했어요."
        )

    with st.expander(f"📋 {N_DAYS}일치 표로 보기"):
        source = df.assign(
            sky=(df["weather_icon"] + " " + df["weather_summary"]).str.strip(),
        )
        cols = [
            "label",
            "sky",
            "temp",
            "rain_prob",
            "humidity",
            "pred_visitors",
            "crowd_level",
        ]
        names = {
            "label": "날짜",
            "sky": "날씨",
            "temp": "기온(°C)",
            "rain_prob": "강수확률(%)",
            "humidity": "습도(%)",
            "pred_visitors": "예상 입장객",
            "crowd_level": "혼잡도",
        }
        if SHOW_SCORES:  # 내부 검증용 컬럼
            cols += ["visitor_score", "weather_score", "final_score"]
            names |= {
                "visitor_score": "이용자 Score",
                "weather_score": "날씨 Score",
                "final_score": "최종 점수",
            }
        table = source[cols].rename(columns=names)
        st.dataframe(table, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# 4-2. 축제·행사 공지
# ---------------------------------------------------------------------------
def notice_color(item) -> str:
    return NOTICE_COLOR.get(item["category"], NOTICE_COLOR_FALLBACK)


def render_notice_summary(notice: dict) -> None:
    """공지 전체를 훑은 한 줄 브리핑. 공지팀이 요약해 준 문장을 그대로 쓴다."""
    if notice["summary"]:
        st.info(notice["summary"], icon="📝")


def render_notices(notice: dict, limit: int = 5) -> None:
    """공지 목록. 줄 전체가 링크라 어디를 눌러도 공지 원문이 새 탭에서 열린다."""
    items = notice["items"][:limit]
    if not items:
        st.caption("등록된 공지가 없어요.")
        return

    rows = []
    for n in items:
        color = notice_color(n)
        state_color = NOTICE_STATE_COLOR.get(n["state"], NOTICE_STATE_COLOR_FALLBACK)
        rows.append(
            f'<a href="{n["url"]}" target="_blank" rel="noopener noreferrer"'
            f' style="display:block; text-decoration:none; color:inherit; padding:9px 4px;'
            f' border-bottom:1px solid rgba(0,0,0,0.06);">'
            f'<div style="display:flex; align-items:baseline; gap:7px; flex-wrap:wrap;">'
            f'<span style="color:{color}; font-size:1.05rem;">●</span>'
            f'<span style="font-size:0.72rem; color:{color}; font-weight:700;">'
            f'{n["category"]}</span>'
            f'<span style="font-weight:600; font-size:0.95rem;">{n["title"]}</span>'
            f'<span style="color:#2F6FD0; font-size:0.82rem; white-space:nowrap;">바로가기 ↗</span>'
            f"</div>"
            f'<div style="font-size:0.78rem; color:gray; margin-top:3px; margin-left:20px;">'
            f'{n["period_text"]}<span style="color:#C9CED6; margin:0 6px;">·</span>'
            f'<span style="color:{state_color}; font-weight:600;">{n["state_label"]}</span></div>'
            f"</a>"
        )
    rows.append(
        '<div style="margin-top:10px; font-size:0.85rem;">'
        f'<a href="{notice["board_url"]}" target="_blank" rel="noopener noreferrer"'
        ' style="color:#2F6FD0; text-decoration:none; font-weight:600;">'
        "공지사항 전체 보기 →</a></div>"
    )
    # 바깥 <div> 로 감싸는 것이 중요하다. 문자열이 <a> 로 시작하면 마크다운이 전체를
    # 문단(<p>)으로 감싸고, <p> 안에 블록 요소인 <div> 가 오면 브라우저가 <p> 를 강제로
    # 닫으면서 첫 줄의 flex 레이아웃(gap)이 통째로 날아간다.
    st.markdown(f'<div>{"".join(rows)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 5. 사이드바 (2) - 추천 / 공지 / 화면 설정
# ---------------------------------------------------------------------------
with st.sidebar:
    st.divider()
    render_recommend()

    st.divider()
    st.markdown("### 📢 축제 · 행사 공지")
    # 공지는 예측보다 느리다. 자리만 먼저 잡아 두고 메인을 다 그린 뒤 채운다.
    notice_slot = st.container()

    # 뉴스 블록 보류: OpenAI 호출부(response 생성)가 아직 없어 키가 있으면 NameError 가 난다.
    # 웹서치 응답 코드를 채운 뒤 아래를 되살리면 된다.
    # st.divider()
    # st.markdown("### 📰 대전 오월드 최신 뉴스")
    #
    # if not api_key:
    #     st.info("최신 뉴스를 보려면 키 등록 필요")
    # else:
    #     if "oworld_news" not in st.session_state:
    #         with st.spinner("대전 오월드 관련 뉴스를 검색하고 있습니다."):
    #             st.session_state.oworld_news = response.output_text
    #
    #     st.markdown(st.session_state.oworld_news)

    st.divider()
    with st.expander("📍 대전오월드 위치"):
        st.markdown("### 📍 대전오월드 위치")
        st.info('주소: 대전 중구 사정공원로 70')

        latitude = 36.2886167
        longitude = 127.3969124

        oworld_map = folium.Map(
            location=[latitude, longitude],
            zoom_start=15,
        )

        folium.Marker(
            location=[latitude, longitude],
            popup="대전 오월드",
            tooltip="대전 오월드",
            icon=folium.Icon(color="red", icon="star"),
        ).add_to(oworld_map)

        st_folium(
            oworld_map,
            width=700,
            height=350,
            key="oworld_map",
        )

        st.divider()
    with st.expander("🔤 화면 설정"):
        selected_font = st.selectbox(
            "사이트 폰트", options=list(FONT_OPTIONS.keys()), index=0
        )
        if st.button("날짜 선택 초기화", width="stretch"):
            st.session_state.chart_token += 1
            st.session_state.pending_recommend = False
            st.session_state.chart_sel_label = None
            st.session_state.ignored_point_index = None
            st.rerun()

    st.divider()
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
    /* 메인 상단 기본 여백(6rem)이 과해서 제목이 너무 내려온다 */
    .stMainBlockContainer, [data-testid="stMainBlockContainer"] {{
        padding-top: 2.5rem !important;
    }}
    .stMainBlockContainer h1, [data-testid="stMainBlockContainer"] h1 {{
        padding-top: 0 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 6. 메인
# ---------------------------------------------------------------------------
st.title("대전 오월드 방문자 수 예측")
st.caption(
    f"오늘부터 {N_DAYS}일간의 예상 입장객과 혼잡도를 보고, 취향에 맞는 방문일을 찾아보세요."
)

render_trend()


# ---------------------------------------------------------------------------
# 7. 공지 (가장 느린 작업이라 맨 마지막)
# ---------------------------------------------------------------------------
# 여기까지 오면 그래프와 추천은 이미 화면에 그려져 있다. 공지는 사이드바에
# 잡아 둔 자리에서 로딩 표시를 띄운 채 받아서 채운다.
with notice_slot:
    with st.spinner("공지를 불러오는 중이에요...", show_time=True):
        notice = load_notice(today, cache_slot(6), PROVIDER_FINGERPRINT)
    render_notice_summary(notice)
    render_notices(notice)


# --- 푸터 ---
st.divider()
st.caption(
    ("프로토타입 화면입니다. 예측·혼잡도 값은 아직 임시 데이터이며, "
     "data_provider.py 의 USE_MOCK 을 끄면 실제 모델·API 값으로 바뀝니다."
     if dp.USE_MOCK
     else "예측·혼잡도 값은 모델 결과를 그대로 표시합니다.")
    + " | 티익스프레스"
)