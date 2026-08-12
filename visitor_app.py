# 글로벌 설정
from datetime import date, timedelta
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

image_path = "C:/study/SK-Shilders/python/Python_project/" "과제/Team_project/map.png"

# CSV 파일 불러오기
df = pd.read_csv("Input.csv")

# 날짜 형식으로 변경
df["일자"] = pd.to_datetime(df["일자"])

# 월과 요일 번호(월~금 => 0~6) 생성
df["월"] = df["일자"].dt.month
df["요일번호"] = df["일자"].dt.dayofweek


# 혼잡도 기준값
low_limit = df["전체건수"].quantile(0.33)
high_limit = df["전체건수"].quantile(0.66)

# 페이지 설정
st.set_page_config(page_title="줄서기싫어요", page_icon="🎡", layout="wide")


# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# 사이드바 설정
with st.sidebar:
    st.header("조건 설정")
    user_role = st.selectbox("장소 선택", ["메인", "추가 정보"])

    preference = st.slider("관광객 수 ←→ 날씨", min_value=0, max_value=100, value=50)

    st.write(f"관광객 수 비중: {100 - preference}%")
    st.write(f"날씨 비중: {preference}%")

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# 메인 페이지 설정

if user_role == "메인":
    st.title("🎡2조 [줄서기 싫어요]")
    st.caption("기간을 선택하면 날짜별 예상 방문객 수를 확인할 수 있습니다.")

    left, right = st.columns(2, gap="large")

    with left:
        # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ선택 가능한 기간 설정ㅡㅡㅡㅡㅡㅡㅡㅡ
        today = date.today()
        last_day = today + timedelta(days=6)

        # 방문 기간 선택
        selected_period = st.date_input(
            "방문 기간을 선택하세요.",
            value=(today, today + timedelta(days=2)),
            min_value=today,
            max_value=last_day,
        )

        # 시작일과 종료일을 모두 선택한 경우
        if len(selected_period) == 2:
            start_date = selected_period[0]
            end_date = selected_period[1]

            # 결과 입력 받기 => 출력
            result_list = []
            current_date = start_date

            # 선택 기간의 날짜별 예상 방문객 계산
            while current_date <= end_date:
                same_days = df[
                    (df["월"] == current_date.month)
                    & (df["요일번호"] == current_date.weekday())
                ]

                expected_visitors = round(same_days["전체건수"].mean())

                if expected_visitors <= low_limit:
                    crowd_level = "여유"
                elif expected_visitors <= high_limit:
                    crowd_level = "보통"
                else:
                    crowd_level = "혼잡"

                result_list.append(
                    {
                        "날짜": current_date.strftime("%Y-%m-%d"),
                        "예상 방문객 수": expected_visitors,
                        "혼잡도": crowd_level,
                    }
                )

                current_date = current_date + timedelta(days=1)

            result = pd.DataFrame(result_list)

            # 그래프 출력
            st.subheader("날짜별 예상 방문객 수")

            chart_data = result.set_index("날짜")[["예상 방문객 수"]]
            st.bar_chart(chart_data)    # 그래프에 마우스 올리면 정보 확인 가능

            st.dataframe(result, hide_index=True, use_container_width=True)
        else:
            st.info("캘린더에서 시작일과 종료일을 모두 선택하세요.")

    with right:
        st.subheader("추가 가능한 기능")
        st.write("대전 오월드 위치")

        latitude = 36.2886167
        longitude = 127.3969124

        m = folium.Map(location=[latitude, longitude], zoom_start=15)

        folium.Marker(
            location=[latitude, longitude],
            popup="대전 오월드",
            tooltip="대전 오월드",
            icon=folium.Icon(color="red", icon="star"),
        ).add_to(m)

        st_folium(m, width=500, height=300, key="oworld_map")


elif user_role == "추가 정보":
    st.title("🔮 테마파크 추가 정보")
    st.write("임시 페이지입니다.")
