import requests

def get_forecast_weather(target_date):
    """
    [위치 기반 14일 기상 데이터 수집]
    Open-Meteo API를 활용하여, 대전 오월드 좌표를 기준으로 14일간의 시간별 예보 데이터를 실시간 수집
    """
    # 대전 오월드 위도와 경도
    lat = 36.2875
    lon = 127.3985
    
    # 조회 일자 14일 (forecast_days=14)
    url = (
        # https://open-meteo.com/ 참초
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"hourly=temperature_2m,relative_humidity_2m,precipitation_probability,weather_code&"
        f"timezone=Asia%2FSeoul&forecast_days=14"
    )

    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        
        # 날씨 코드(WMO)에 맞춰 [한글 설명, 이모지 아이콘]을 매핑하는 함수
        def get_weather_info(code):
            if code == 0:
                return "맑음", "☀️"
            elif code in [1, 2]:
                return "구름 조금", "🌤️"
            elif code == 3:
                return "흐림", "☁️"
            elif code in [45, 48]:
                return "안개", "🌫️"
            elif code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
                return "비", "🌧️"
            elif code in [71, 73, 75, 85, 86]:
                return "눈", "❄️"
            elif code in [95, 96, 99]:
                return "뇌우", "🌩️"
            return "알 수 없음", "🌡️"

        # 전체 시간별 데이터 중 타겟 날짜 추출
        times = data["hourly"]["time"]

        # 방문객이 가장 활동하기 좋은 '낮 12시(T12:00)'를 기준 데이터로 타겟팅하여 인덱싱
        target_time_str = f"{target_date}T12:00"
        
        if target_time_str in times:
            idx = times.index(target_time_str)
            
            # 인덱스를 기반으로 해당 시간의 날씨 코드와 온도, 습도, 강수확률 파싱
            weather_desc, weather_icon = get_weather_info(data["hourly"]["weather_code"][idx])

            # Dictionary로 구조화하여 반환
            return {
                "weather": weather_desc,
                "icon": weather_icon,
                "temp": data["hourly"]["temperature_2m"][idx],
                "humidity": data["hourly"]["relative_humidity_2m"][idx],
                "pop": data["hourly"]["precipitation_probability"][idx]
            }
        else:
            # 타겟 날짜가 14일 범위를 벗어난 경우의 예외 처리
            return {"weather": "데이터 없음", "icon": "❓", "temperature": "-", "humidity": "-", "rain": "-"}

    except Exception as e:
        print(f"오류: {e}")
        return None

# 테스트 실행
if __name__ == "__main__":
    selected_date = "2026-08-25" 
    weather_data = get_forecast_weather(selected_date)
    
    print(f"['{selected_date}'의 날씨]")
    print(weather_data)


# 분리해둔 날씨 함수 가져와서 사용
# from weather import get_forecast_weather