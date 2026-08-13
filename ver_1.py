"""
날씨를 아래 추가한 버전

줄서기 싫어요 (Zero Waiting) - 대전 오월드 방문자 수 예측 UI 프로토타입
팀: 티익스프레스

[안내]
- 이 파일은 UI(프론트) 프로토타입입니다. 실제 머신러닝 모델/외부 API는 연결돼 있지 않으며,
  예측/날씨 값은 회의에서 확인한 상관관계(주말 > 기온 > 계절 > 강수량)를 반영한
  임시 로직(mock)입니다. 모델/API가 준비되면 "TODO" 표시 함수만 교체하면 됩니다.

[화면 구성]
- 사이드바: 오늘 날씨 · 선호(날씨 중요도) 슬라이더 · 사이트 폰트 선택
- 메인: 오늘 날짜 · 예상 입장객 수/혼잡도/날씨 쾌적도 · 오늘 기준 향후 7일 예측 추이 그래프
- 공휴일/이상치(예: 어린이날) 안내 문구 출력

실행 방법:
    pip install -r requirements.txt
    streamlit run app.py

[참고]
- 같은 경로에 OpenAI api 키가 들어간 .env 파일이 없다면
  최신 뉴스 파트는 실행되지 않을 수 있습니다.
"""

import datetime as dt
import os

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# 0. 페이지 기본 설정
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="줄서기 싫어요 | 대전 오월드 방문자 예측",
    page_icon="🎡",
    layout="wide",
)

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    client = OpenAI(api_key=api_key)

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
    (5, 5): "어린이날은 방문객이 예측치를 크게 벗어나는 이상치라, 별도 안내로 처리합니다.",
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


def is_vacation(date: dt.date) -> bool:
    # 대략적인 방학 구간 (여름/겨울), 실제 학사 일정 데이터로 교체 예정
    m, d = date.month, date.day
    summer = (m == 7) or (m == 8 and d <= 20)
    winter = (m == 12 and d >= 24) or (m in (1, 2))
    return summer or winter


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

    # 방학
    if is_vacation(date):
        base *= 1.15

    # 날짜별 결정적 노이즈 (재실행해도 값 고정)
    rng = np.random.default_rng(date.toordinal() + 7)
    base *= 1 + rng.normal(0, 0.05)

    return int(max(base, 0))


def weather_score(weather: dict) -> float:
    """날씨 쾌적도 0~100 (추천 가중치 계산용)."""
    t = weather["temp"]
    temp_score = max(0, 100 - abs(t - 21) * 5)  # 21도 최적
    rain_penalty = min(weather["rainfall"] * 3, 60)
    humid_penalty = max(0, weather["humidity"] - 70) * 1.2
    return float(np.clip(temp_score - rain_penalty - humid_penalty, 0, 100))


# ---------------------------------------------------------------------------
# 4. 사이드바 (오늘 날씨 정보)
# ---------------------------------------------------------------------------
today = dt.date.today()
today_weather = get_weather(today)

with st.sidebar:
    st.markdown("## 🎡 줄서기 싫어요")
    st.caption("대전 오월드 방문자 수 예측 · 티익스프레스")
    user_role = st.selectbox(
        "페이지 선택",
        ["메인", "추가 정보"],
    )
    st.divider()

    st.markdown("### 🌤️ 오늘 날씨")
    st.metric("평균기온", f"{today_weather['temp']}°C")
    st.metric("일강수량", f"{today_weather['rainfall']}mm")
    st.metric("습도", f"{today_weather['humidity']}%")
    st.caption(
        f"{WEEKDAY_KR[today.weekday()]}요일 · {today_weather['season']}"
        + (f" · {holiday_name(today)}" if is_holiday(today) else "")
    )

    st.divider()
    st.markdown("### 🎯 선호 설정")
    weather_weight = st.slider(
        "날씨 중요도",
        min_value=0, max_value=10, value=5,
    )
    ends_l, ends_r = st.columns(2)
    ends_l.caption("← 혼잡도 우선")
    ends_r.markdown(
        "<div style='text-align:right; color:gray; font-size:0.8rem;'>날씨 우선 →</div>",
        unsafe_allow_html=True,
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
    # 5. 메인 화면
    # ---------------------------------------------------------------------------

if user_role == "메인":
    st.title("대전 오월드 방문자 수 예측")
    st.caption("오늘 기준 예상 입장객 수와 앞으로 7일간의 예측 추이를 확인하세요.")

    # 오늘 날짜 표시
    st.markdown(
        f"## 📅 오늘 날짜: {today.strftime('%Y년 %m월 %d일')} "
        f"({WEEKDAY_KR[today.weekday()]})"
    )

    # 색상 팔레트 (부드러운 파스텔 톤)
    LEVEL_COLOR = {"혼잡": "#E7A9A0", "보통": "#E8C79A", "여유": "#A8CBB0"}
    COLOR_WEEKDAY = "#8FB9D9"   # 차분한 소프트 블루 (평일)
    COLOR_WEEKEND = "#E7A9A0"   # 부드러운 코랄 (주말/공휴일)

    # 오늘 기준 7일치 예측 미리 계산
    days = [today + dt.timedelta(days=i) for i in range(7)]
    weathers = [get_weather(d) for d in days]
    preds = [predict_visitors(d, w) for d, w in zip(days, weathers)]
    labels = [f"{d.month}/{d.day}({WEEKDAY_KR[d.weekday()]})" for d in days]
    weekend_flags = [is_weekend(d) or is_holiday(d) for d in days]
    df = pd.DataFrame({
        "날짜": days,
        "라벨": labels,
        "예상 입장객": preds,
        "기온": [w["temp"] for w in weathers],
        "강수량": [w["rainfall"] for w in weathers],
        "주말/공휴일": weekend_flags,
    })

    # --- 5-1. 예측 결과 요약 (그래프 클릭 결과를 반영하므로 자리만 먼저 확보) ---
    summary_area = st.container()

    st.divider()

    # --- 5-2. 향후 7일 예측 추이 ---
    st.markdown("### 📈 오늘 기준 향후 7일 예측 추이")
    st.caption("막대를 클릭하면 위 요약이 해당 날짜 기준으로 바뀝니다.")

    bar_colors = [COLOR_WEEKEND if flag else COLOR_WEEKDAY for flag in df["주말/공휴일"]]
    fig = go.Figure()
    fig.add_bar(
        x=df["라벨"],
        y=df["예상 입장객"],
        marker_color=bar_colors,
        marker_line_width=0,
        text=[f"{v:,}" for v in df["예상 입장객"]],
        textposition="outside",
        textfont=dict(color="#555555"),
        hovertemplate="%{x}<br>예상 입장객: %{y:,}명<extra></extra>",
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
    event = st.plotly_chart(
        fig, use_container_width=True, on_select="rerun", key="trend_chart"
    )
    st.markdown(
        f"<span style='color:{COLOR_WEEKDAY}; font-size:1.1rem;'>●</span> 평일 &nbsp;·&nbsp; "
        f"<span style='color:{COLOR_WEEKEND}; font-size:1.1rem;'>●</span> 주말/공휴일 (혼잡 예상)",
        unsafe_allow_html=True,
    )

    # 클릭된 막대 판별 (기본값: 오늘 = index 0)
    sel_idx = 0
    try:
        points = event["selection"]["points"]
        if points:
            clicked_label = points[0].get("x")
            if clicked_label in labels:
                sel_idx = labels.index(clicked_label)
    except (TypeError, KeyError, IndexError):
        sel_idx = 0

    sel_date = days[sel_idx]
    sel_weather = weathers[sel_idx]
    sel_pred = preds[sel_idx]
    sel_level = "혼잡" if sel_pred > 9000 else ("보통" if sel_pred > 4500 else "여유")

    # 확보해 둔 상단 요약 자리를 선택 날짜 기준으로 채움
    with summary_area:
        if sel_idx == 0:
            st.markdown(f"#### 📊 오늘 · {sel_date.strftime('%m월 %d일')} ({WEEKDAY_KR[sel_date.weekday()]}) 예측")
        else:
            st.markdown(f"#### 📊 {sel_date.strftime('%m월 %d일')} ({WEEKDAY_KR[sel_date.weekday()]}) 예측")

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
                f"📌 이 날은 '{holiday_name(sel_date)}' 공휴일이라, "
                "예측보다 사람이 더 몰릴 수 있습니다."
            )
        # 선택한 날짜의 날씨 정보


    # 추가한 부분
    st.divider()
    st.markdown("### 🌤️ 선택한 날짜의 날씨")

    st.write(
        f"{sel_date.strftime('%Y년 %m월 %d일')} "
        f"({WEEKDAY_KR[sel_date.weekday()]})"
    )


    weather_col1, weather_col2, weather_col3, weather_col4 = st.columns(4)

    weather_col1.metric(
        "평균기온",
        f"{sel_weather['temp']}도"
    )

    weather_col2.metric(
        "일강수량",
        f"{sel_weather['rainfall']}mm"
    )

    weather_col3.metric(
        "습도",
        f"{sel_weather['humidity']}%"
    )

    weather_col4.metric(
        "날씨 쾌적도",
        f"{weather_score(sel_weather):.0f} / 100"
    )

    if sel_weather["rainfall"] > 20:
        st.warning(
            "비가 많이 올 것으로 예상됩니다. "
            "야외 놀이기구 이용이 제한될 수 있습니다."
        )

    elif sel_weather["rainfall"] > 0:
        st.info(
            "비가 예상됩니다. 우산이나 우비를 준비하세요."
        )

    elif sel_weather["temp"] >= 30:
        st.warning(
            "기온이 높습니다. 물을 충분히 마시고 "
            "온열질환에 주의하세요."
        )

    elif sel_weather["temp"] <= 5:
        st.warning(
            "기온이 낮습니다. 따뜻한 옷을 준비하세요."
        )

    else:
        st.success(
            "야외 활동에 비교적 적합한 날씨입니다."
        )

    # ---------------------------------------------------------------------------
    # 6. 대전 오월드 위치
    # ---------------------------------------------------------------------------
elif user_role == "추가 정보":
    st.divider()
    st.markdown("### 📍 대전 오월드 위치")

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


    # ---------------------------------------------------------------------------
    # 7. 대전 오월드 최신 뉴스
    # ---------------------------------------------------------------------------
    st.divider()
    st.markdown("### 📰 대전 오월드 최신 뉴스")

    if not api_key:
        st.info("최신 뉴스를 보려면 .env 파일에 OPENAI_API_KEY를 등록하세요.")
    else:
        if "oworld_news" not in st.session_state:
            with st.spinner("대전 오월드 관련 뉴스를 검색하고 있습니다."):
                response = client.responses.create(
                    model="gpt-5.5",
                    instructions="""
                    대전 오월드와 직접 관련된 최신 뉴스만 검색하세요.
                    다른 지역이나 일반 테마파크 뉴스는 제외하세요.
                    최신순으로 최대 5개를 출력하세요.

                    출력 형식:
                    - [기사 제목](기사 URL) — 기사 날짜

                    반드시 클릭 가능한 실제 기사 링크를 포함하세요.
                    """,
                    input="대전 오월드 최신 뉴스를 검색하세요.",
                    tools=[{"type": "web_search"}],
                )

            st.session_state.oworld_news = response.output_text

        st.markdown(st.session_state.oworld_news)


# --- 푸터 ---
st.divider()
st.caption(
    "프로토타입 · 예측/날씨 값은 임시(mock) 로직입니다. "
    "실제 머신러닝 모델과 기상청 웹서치 API 연결 후 교체 예정입니다. | 티익스프레스"
)
