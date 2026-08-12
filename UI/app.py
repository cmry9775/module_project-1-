"""
줄서기 싫어요 (Zero Waiting) - 대전 오월드 방문자 수 예측 UI 프로토타입
팀: 티익스프레스

[안내]
- 이 파일은 UI(프론트) 프로토타입입니다. 실제 머신러닝 모델/외부 API는 연결돼 있지 않으며,
  예측/날씨 값은 회의에서 확인한 상관관계(주말 > 기온 > 계절 > 강수량)를 반영한
  임시 로직(mock)입니다. 모델/API가 준비되면 "TODO" 표시 함수만 교체하면 됩니다.

[화면 구성]
- 사이드바: 선호(날씨 중요도) 슬라이더 · 선택한 날짜의 날씨 · 사이트 폰트 선택
- 메인: 제목 · 오늘 날짜 · 예상 입장객 수/혼잡도/날씨 쾌적도 · 향후 7일 예측 추이 그래프
- 공휴일/이상치(예: 어린이날) 안내 문구 출력

실행 방법:
    pip install -r requirements.txt
    streamlit run app.py
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# 0. 페이지 기본 설정
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="줄서기 싫어요 | 대전 오월드 방문자 예측",
    page_icon="🎡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 1. 상수 / 참고 데이터
# ---------------------------------------------------------------------------
# 2025~2026 주요 공휴일 (프로토타입용 최소 집합, 실제 전처리 데이터로 교체 예정)
HOLIDAYS = {
    dt.date(2025, 1, 1): "신정",
    dt.date(2025, 3, 1): "삼일절",
    dt.date(2025, 5, 5): "어린이날",
    dt.date(2025, 6, 6): "현충일",
    dt.date(2025, 8, 15): "광복절",
    dt.date(2025, 10, 3): "개천절",
    dt.date(2025, 10, 9): "한글날",
    dt.date(2025, 12, 25): "성탄절",
    dt.date(2026, 1, 1): "신정",
    dt.date(2026, 3, 1): "삼일절",
    dt.date(2026, 5, 5): "어린이날",
    dt.date(2026, 6, 6): "현충일",
    dt.date(2026, 8, 15): "광복절",
    dt.date(2026, 10, 3): "개천절",
    dt.date(2026, 10, 9): "한글날",
    dt.date(2026, 12, 25): "성탄절",
}

# 어린이날처럼 학습에서 이상치로 다루기로 논의된 날 (하드코딩 안내용)
OUTLIER_DAYS = {
    (5, 5): "어린이날은 방문객이 평소 예측을 크게 웃도는 특별한 날이라, 예측값은 참고용으로만 봐주세요.",
}

WEEKDAY_KR = ["월", "화", "수", "목", "금", "토", "일"]

# 사이트 폰트 옵션 (표시명 -> CSS font-family 이름)
FONT_OPTIONS = {
    "Pretendard": "Pretendard",
    "에스코어 드림": "Escoredream",
    "G마켓 산스": "GMarketSans",
    "원스토어 모바일": "OneStoreMobileGothicBody",
}

# 웹폰트 @font-face 정의 (noonnu CDN, 직접 삽입)
FONT_FACE_CSS = """
@font-face {
    font-family: 'Pretendard';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/pretendard@1.0/Pretendard-Light.woff2') format('woff2');
    font-weight: 300; font-display: swap;
}
@font-face {
    font-family: 'Pretendard';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/pretendard@1.0/Pretendard-Regular.woff2') format('woff2');
    font-weight: 400; font-display: swap;
}
@font-face {
    font-family: 'Pretendard';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/pretendard@1.0/Pretendard-Medium.woff2') format('woff2');
    font-weight: 500; font-display: swap;
}
@font-face {
    font-family: 'Pretendard';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/pretendard@1.0/Pretendard-SemiBold.woff2') format('woff2');
    font-weight: 600; font-display: swap;
}
@font-face {
    font-family: 'Pretendard';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/pretendard@1.0/Pretendard-Bold.woff2') format('woff2');
    font-weight: 700; font-display: swap;
}
@font-face {
    font-family: 'Escoredream';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_six@1.2/S-CoreDream-3Light.woff') format('woff');
    font-weight: 300; font-display: swap;
}
@font-face {
    font-family: 'Escoredream';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_six@1.2/S-CoreDream-4Regular.woff') format('woff');
    font-weight: 400; font-display: swap;
}
@font-face {
    font-family: 'Escoredream';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_six@1.2/S-CoreDream-5Medium.woff') format('woff');
    font-weight: 500; font-display: swap;
}
@font-face {
    font-family: 'Escoredream';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_six@1.2/S-CoreDream-6Bold.woff') format('woff');
    font-weight: 600; font-display: swap;
}
@font-face {
    font-family: 'Escoredream';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_six@1.2/S-CoreDream-7ExtraBold.woff') format('woff');
    font-weight: 700; font-display: swap;
}
@font-face {
    font-family: 'GMarketSans';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2001@1.1/GmarketSansLight.woff') format('woff');
    font-weight: 300; font-display: swap;
}
@font-face {
    font-family: 'GMarketSans';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2001@1.1/GmarketSansMedium.woff') format('woff');
    font-weight: 500; font-display: swap;
}
@font-face {
    font-family: 'GMarketSans';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2001@1.1/GmarketSansBold.woff') format('woff');
    font-weight: 700; font-display: swap;
}
@font-face {
    font-family: 'OneStoreMobileGothicBody';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2105_2@1.0/ONE-Mobile-Regular.woff') format('woff');
    font-weight: normal; font-display: swap;
}
"""


# ---------------------------------------------------------------------------
# 2. 피처 엔지니어링 헬퍼 (전처리 담당 로직과 동일한 규칙으로 맞출 부분)
# ---------------------------------------------------------------------------
def get_season(date: dt.date) -> str:
    m = date.month
    if m in (3, 4, 5):
        return "봄"
    if m in (6, 7, 8):
        return "여름"
    if m in (9, 10, 11):
        return "가을"
    return "겨울"


def is_holiday(date: dt.date) -> bool:
    return date in HOLIDAYS


def is_weekend(date: dt.date) -> bool:
    return date.weekday() >= 5


def holiday_name(date: dt.date) -> str:
    return HOLIDAYS.get(date, "")


# ---------------------------------------------------------------------------
# 3. Mock 함수 (실제 모델/API 연결 시 이 부분만 교체)
# ---------------------------------------------------------------------------
def get_weather(date: dt.date) -> dict:
    """
    TODO(OpenAI 담당): 기상청 웹서치 API로 해당 날짜의
    평균기온 / 일강수량 / 평균상대습도를 받아오도록 교체.
    지금은 계절 기반으로 그럴듯한 값을 결정적으로 생성.
    """
    rng = np.random.default_rng(date.toordinal())
    season = get_season(date)
    base_temp = {"봄": 17, "여름": 28, "가을": 16, "겨울": 2}[season]
    temp = round(base_temp + rng.normal(0, 3), 1)
    rain_prob = {"봄": 0.25, "여름": 0.45, "가을": 0.2, "겨울": 0.2}[season]
    rainfall = round(float(rng.gamma(2, 6)), 1) if rng.random() < rain_prob else 0.0
    humidity = int(np.clip(55 + rng.normal(0, 12), 30, 95))
    return {"temp": temp, "rainfall": rainfall, "humidity": humidity, "season": season}


def predict_visitors(date: dt.date, weather: dict) -> int:
    """
    TODO(모델 담당): 학습된 회귀 모델(Random Forest 등)의 predict()로 교체.
    지금은 회의에서 확인한 상관관계를 반영한 임시 규칙 기반 예측.
        주말(0.5) > 기온 > 계절 > 강수량(≈0, 약한 음의 상관)
    """
    base = 2200.0

    # 주말 / 공휴일
    if is_weekend(date):
        base *= 2.3
    if is_holiday(date):
        base *= 2.6

    # 계절 (가을 최고, 겨울 최저 - 회의 시각화 결과 반영)
    base *= {"봄": 1.25, "여름": 0.85, "가을": 1.35, "겨울": 0.70}[weather["season"]]

    # 기온 (쾌적 구간 18~25도, 폭염/한파 감소)
    t = weather["temp"]
    if t < 0:
        base *= 0.50
    elif t < 10:
        base *= 0.75
    elif t <= 25:
        base *= 1.10
    elif t <= 30:
        base *= 0.90
    else:
        base *= 0.65

    # 강수량 (상관 약하지만 비 오면 감소)
    r = weather["rainfall"]
    if r > 20:
        base *= 0.40
    elif r > 5:
        base *= 0.70
    elif r > 0:
        base *= 0.90

    # 날짜별 결정적 노이즈 (재실행해도 값 고정)
    rng = np.random.default_rng(date.toordinal() + 7)
    base *= 1 + rng.normal(0, 0.05)

    return int(max(base, 0))


def crowd_level(pred: int) -> str:
    """예상 입장객 수를 혼잡/보통/여유 3단계로 구분."""
    if pred > 9000:
        return "혼잡"
    if pred > 4500:
        return "보통"
    return "여유"


def weather_score(weather: dict) -> float:
    """날씨 쾌적도 0~100 (추천 가중치 계산용)."""
    t = weather["temp"]
    temp_score = max(0, 100 - abs(t - 21) * 5)  # 21도 최적
    rain_penalty = min(weather["rainfall"] * 3, 60)
    humid_penalty = max(0, weather["humidity"] - 70) * 1.2
    return float(np.clip(temp_score - rain_penalty - humid_penalty, 0, 100))


# ---------------------------------------------------------------------------
# 4. 데이터 준비 & 선택 날짜 판별
# ---------------------------------------------------------------------------
today = dt.date.today()

# 색상 팔레트 (혼잡도 단계별 파스텔 톤)
LEVEL_COLOR = {"혼잡": "#E7A9A0", "보통": "#E8C79A", "여유": "#A8CBB0"}

# 오늘 기준 7일치 예측 미리 계산
days = [today + dt.timedelta(days=i) for i in range(7)]
weathers = [get_weather(d) for d in days]
preds = [predict_visitors(d, w) for d, w in zip(days, weathers)]
labels = [f"{d.month}/{d.day}({WEEKDAY_KR[d.weekday()]})" for d in days]
weekend_flags = [is_weekend(d) or is_holiday(d) for d in days]
levels = [crowd_level(p) for p in preds]
df = pd.DataFrame({
    "날짜": days,
    "라벨": labels,
    "예상 입장객": preds,
    "기온": [w["temp"] for w in weathers],
    "강수량": [w["rainfall"] for w in weathers],
    "주말/공휴일": weekend_flags,
    "혼잡도": levels,
})

# 막대그래프에서 클릭한 날짜 판별 (기본값: 오늘 = index 0)
# on_select="rerun" 이므로 이전 클릭 결과는 session_state["trend_chart"]에 남아 있어,
# 사이드바(아래에서 먼저 렌더)에서도 선택 날짜의 날씨를 표시할 수 있음.
sel_idx = 0
try:
    _points = st.session_state["trend_chart"]["selection"]["points"]
    if _points:
        _clicked_label = _points[0].get("x")
        if _clicked_label in labels:
            sel_idx = labels.index(_clicked_label)
except (KeyError, IndexError, TypeError):
    sel_idx = 0

sel_date = days[sel_idx]
sel_weather = weathers[sel_idx]
sel_pred = preds[sel_idx]
sel_level = crowd_level(sel_pred)


# ---------------------------------------------------------------------------
# 5. 사이드바 (선택한 날짜의 날씨 · 폰트)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎡 줄서기 싫어요")
    st.caption("대전 오월드 방문자 수 예측 · 티익스프레스")
    st.divider()

    # --- 선호 설정 슬라이더 (사이트 제목 바로 아래) ---
    st.markdown("### 🎯 어떤 날을 찾고 계신가요?")
    weather_weight = st.slider(
        "날씨 중요도",
        min_value=0, max_value=10, value=5,
        label_visibility="collapsed",
    )
    ends_l, ends_r = st.columns(2)
    ends_l.caption("← 한산한 날")
    ends_r.markdown(
        "<div style='text-align:right; color:gray; font-size:0.8rem;'>"
        "좋은 날씨 →</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "왼쪽에 둘수록 사람이 적어 여유로운 날을, "
        "오른쪽에 둘수록 날씨가 좋은 날을 먼저 추천해 드려요."
    )

    st.divider()
    _sel_label = "오늘" if sel_idx == 0 else sel_date.strftime("%m월 %d일")
    st.markdown(f"### 🌤️ {_sel_label} 날씨")
    st.caption("아래 그래프에서 선택한 날짜의 날씨예요.")
    st.metric("평균기온", f"{sel_weather['temp']}°C")
    st.metric("일강수량", f"{sel_weather['rainfall']}mm")
    st.metric("습도", f"{sel_weather['humidity']}%")
    st.caption(
        f"{WEEKDAY_KR[sel_date.weekday()]}요일 · {sel_weather['season']}"
        + (f" · {holiday_name(sel_date)}" if is_holiday(sel_date) else "")
    )

    st.divider()
    st.markdown("### 🔤 폰트")
    selected_font = st.selectbox(
        "사이트 폰트",
        options=list(FONT_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
    )

# 선택한 폰트를 사이트 전체에 적용
_font_family = FONT_OPTIONS[selected_font]
st.markdown(
    f"""
    <style>
    {FONT_FACE_CSS}
    html, body, .stApp, .stApp *,
    [class*="css"], [class^="st-"], [class*=" st-"],
    button, input, select, textarea, label {{
        font-family: '{_font_family}', sans-serif !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 6. 메인 화면
# ---------------------------------------------------------------------------
st.title("대전 오월드 방문자 수 예측")
st.caption("오늘부터 일주일간 예상 입장객 수와 혼잡도 흐름을 한눈에 확인해 보세요.")

# --- 6-1. 예측 결과 요약 (그래프 클릭 결과를 반영하므로 자리만 먼저 확보) ---
summary_area = st.container()

st.divider()

# --- 6-2. 향후 7일 예측 추이 ---
st.markdown("### 📈 향후 7일 예측 추이")
st.caption("궁금한 날짜의 막대를 클릭하면 위쪽 요약이 그 날 기준으로 바뀌어요.")

bar_colors = [LEVEL_COLOR[lv] for lv in df["혼잡도"]]
fig = go.Figure()
fig.add_bar(
    x=df["라벨"],
    y=df["예상 입장객"],
    marker_color=bar_colors,
    marker_line_width=0,
    text=[f"{v:,}" for v in df["예상 입장객"]],
    textposition="outside",
    textfont=dict(color="#555555"),
    customdata=list(df["혼잡도"]),
    hovertemplate="%{x}<br>예상 입장객: %{y:,}명<br>혼잡도: %{customdata}<extra></extra>",
)
fig.update_layout(
    height=380,
    margin=dict(l=10, r=10, t=30, b=10),
    yaxis_title="예상 입장객 수(명)",
    xaxis_title=None,
    showlegend=False,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#555555"),
)
fig.update_yaxes(gridcolor="rgba(0,0,0,0.06)", zeroline=False)
# 날짜 레이블: 더 크고 진하게
fig.update_xaxes(
    showgrid=False,
    tickvals=list(df["라벨"]),
    ticktext=[f"<b>{l}</b>" for l in df["라벨"]],
    tickfont=dict(size=16, color="#1F2A37"),
)
st.plotly_chart(
    fig, use_container_width=True, on_select="rerun", key="trend_chart"
)
st.markdown(
    f"<span style='color:{LEVEL_COLOR['혼잡']}; font-size:1.1rem;'>●</span> 혼잡 &nbsp;·&nbsp; "
    f"<span style='color:{LEVEL_COLOR['보통']}; font-size:1.1rem;'>●</span> 보통 &nbsp;·&nbsp; "
    f"<span style='color:{LEVEL_COLOR['여유']}; font-size:1.1rem;'>●</span> 여유",
    unsafe_allow_html=True,
)
# 선택 날짜(sel_idx 등)는 상단 데이터 준비 단계에서 session_state로 이미 계산됨

# 확보해 둔 상단 요약 자리를 선택 날짜 기준으로 채움
with summary_area:
    _weekday_label = WEEKDAY_KR[sel_date.weekday()] + (", 오늘" if sel_idx == 0 else "")
    st.markdown(
        f"#### 📊 {sel_date.strftime('%Y년 %m월 %d일')} ({_weekday_label}) 예측"
    )

    if (sel_date.month, sel_date.day) in OUTLIER_DAYS:
        st.warning("⚠️ " + OUTLIER_DAYS[(sel_date.month, sel_date.day)])

    m1, m2, m3 = st.columns(3)
    m1.metric("예상 입장객 수", f"{sel_pred:,}명")
    with m2:
        st.markdown(
            f"""
            <div style='font-size:0.8rem; color:gray;'>혼잡도</div>
            <div style='font-size:1.8rem; font-weight:600;'>
                <span style='color:{LEVEL_COLOR[sel_level]};'>●</span> {sel_level}
            </div>
            """,
            unsafe_allow_html=True,
        )
    m3.metric("날씨 쾌적도", f"{weather_score(sel_weather):.0f} / 100")

    if is_holiday(sel_date):
        st.info(
            f"📌 이날은 '{holiday_name(sel_date)}' 공휴일이라 "
            "예상보다 더 많은 분들이 찾을 수 있어요."
        )

# --- 푸터 ---
st.divider()
st.caption(
    "프로토타입 화면입니다. 지금 보이는 예측·날씨 값은 임시 데이터이며, "
    "실제 머신러닝 모델과 기상청 API를 연결하면 실제 값으로 바뀝니다. | 티익스프레스"
)
