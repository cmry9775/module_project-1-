import requests

def get_forecast_weather(target_date):
    """
    대전 오월드 위치를 기준으로 특정 날짜의 '낮 12시' 날씨를 반환합니다.
    (최대 14일 이내 조회 가능)
    """
    # 대전 오월드 위도/경도
    lat = 36.2875
    lon = 127.3985
    
    # 💡 조회 일자 14일 (forecast_days=14)
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

        times = data["hourly"]["time"]
        target_time_str = f"{target_date}T12:00"
        
        if target_time_str in times:
            idx = times.index(target_time_str)
            
            # 한글 설명과 아이콘 추출
            weather_desc, weather_icon = get_weather_info(data["hourly"]["weather_code"][idx])
            
            return {
                "weather": weather_desc,
                "icon": weather_icon,
                "temp": data["hourly"]["temperature_2m"][idx],
                "humidity": data["hourly"]["relative_humidity_2m"][idx],
                "pop": data["hourly"]["precipitation_probability"][idx]
            }
        else:
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