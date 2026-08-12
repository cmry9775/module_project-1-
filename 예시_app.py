from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 💡 분리해둔 날씨 함수 가져오기
from weather import get_forecast_weather

# ==========================================
# 1. 페이지 및 상태 설정
# ==========================================
st.set_page_config(
    page_title="대전 오월드 방문자 수 예측", page_icon="🎢", layout="wide"
)

today_str = datetime.now().strftime("%Y-%m-%d")

# 현재 선택된 날짜를 저장하는 Session State
if "selected_date" not in st.session_state:
    st.session_state.selected_date = today_str

# ==========================================
# 2. 사이드바 UI (선택된 날짜 날씨 출력)
# ==========================================
with st.sidebar:
    st.header("🌤️ 선택일 날씨 정보")
    st.caption("기준: 대전 오월드 (낮 12시 예보)")

    # weather.py의 함수 실행
    weather = get_forecast_weather(st.session_state.selected_date)

    st.subheader(f"📅 {st.session_state.selected_date}")
    st.divider()

    if weather and weather.get("weather") != "데이터 없음":
        # 상태
        st.metric(
            label="날씨 상태", value=f"{weather['icon']} {weather['weather']}"
        )

        # 기온 & 습도
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="기온", value=f"{weather['temp']} ℃")
        with col2:
            st.metric(label="습도", value=f"{weather['humidity']} %")

        # 강수확률
        st.metric(label="강수확률", value=f"{weather['pop']} %")
    else:
        st.warning("해당 날짜의 날씨 정보를 불러올 수 없습니다.")

    st.divider()
    st.info("💡 메인 화면의 그래프 막대를 클릭하면 날씨 정보가 업데이트됩니다.")

# ==========================================
# 3. 메인 화면 UI
# ==========================================
st.title("🎢 대전 오월드 방문자 수 예측 대시보드")
st.divider()

# ① 날짜 선택 달력
col_date, _ = st.columns([1, 2])
with col_date:
    picked_date = st.date_input(
        "📅 조회할 날짜 선택",
        value=datetime.strptime(st.session_state.selected_date, "%Y-%m-%d"),
        min_value=datetime.now(),
        max_value=datetime.now() + timedelta(days=13),
    )
    picked_date_str = picked_date.strftime("%Y-%m-%d")

    # 달력 날짜 변경 시 즉시 반영
    if picked_date_str != st.session_state.selected_date:
        st.session_state.selected_date = picked_date_str
        st.rerun()

# ② 14일치 방문자 수 가상 데이터 생성
date_list = [
    (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(14)
]
visitor_counts = [
    2023, 1850, 2100, 4500, 8256, 7900, 3100, 
    2050, 1900, 2200, 4800, 8500, 7600, 2900
]
df = pd.DataFrame({"date": date_list, "visitors": visitor_counts})

# ③ Plotly Graph Objects(go) 막대 그래프
st.subheader("📈 향후 14일간 방문자 수 예측 추이")

fig = go.Figure(
    data=[
        go.Bar(
            x=df["date"],
            y=df["visitors"],
            marker=dict(
                color=df["visitors"],
                colorscale="Reds",  # 값에 따른 그라데이션
                showscale=True,
            ),
            hovertemplate="날짜: %{x}<br>예상 방문자: %{y:,}명<extra></extra>",
        )
    ]
)

fig.update_layout(
    xaxis_title="날짜",
    yaxis_title="예상 입장객 수 (명)",
    height=400,
    margin=dict(l=20, r=20, t=30, b=20),
)

# 💡 막대 클릭 이벤트 등록
chart_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

# 💡 클릭 이벤트 감지 시 날짜 업데이트 및 화면 새로고침
if chart_event and "selection" in chart_event:
    points = chart_event["selection"].get("points", [])
    if points:
        clicked_date = points[0]["x"]
        if clicked_date != st.session_state.selected_date:
            st.session_state.selected_date = clicked_date
            st.rerun()