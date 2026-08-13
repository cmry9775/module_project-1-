 # 함수 호출 사용 예시
  from oworld_openai_pipeline import (
    build_ui_payload,
    get_selected_day
)

payload = build_ui_payload(days=10)
  
  
  # 테스트용 3일 출력 결과
  "days": [
    {
      "date": "2026-08-13",
      "status": "success",
      "data": {
        "holiday": false,
        "weekend": false,
        "weekday": {
          "mon": false,
          "tue": false,
          "wed": false,
          "thur": true,
          "fri": false,
          "sat": false,
          "sun": false
        },
        "temperature": 28.1,
        "rain": 10.0,
        "humidity": 68.0,
        "predicted_visitors": 256,
        "congestion": 0,
        "user_score": 48,
        "weather_score": 85
      },
      "meta": {
        "weather_summary": "구름 조금",
        "weather_icon": "🌤️",
        "weather_source": "Open-Meteo",
        "forecast_time": "12:00",
        "congestion_dayoff": false,
        "congestion_nextdayoff": false,
        "congestion_month": "08"
      }
    },
    {
      "date": "2026-08-14",
      "status": "success",
      "data": {
        "holiday": false,
        "weekend": false,
        "weekday": {
          "mon": false,
          "tue": false,
          "wed": false,
          "thur": false,
          "fri": true,
          "sat": false,
          "sun": false
        },
        "temperature": 28.1,
        "rain": 51.0,
        "humidity": 75.0,
        "predicted_visitors": 586,
        "congestion": 0,
        "user_score": 39,
        "weather_score": 71
      },
      "meta": {
        "weather_summary": "비",
        "weather_icon": "🌧️",
        "weather_source": "Open-Meteo",
        "forecast_time": "12:00",
        "congestion_dayoff": false,
        "congestion_nextdayoff": true,
        "congestion_month": "08"
      }
    },
    {
      "date": "2026-08-15",
      "status": "success",
      "data": {
        "holiday": true,
        "weekend": true,
        "weekday": {
          "mon": false,
          "tue": false,
          "wed": false,
          "thur": false,
          "fri": false,
          "sat": true,
          "sun": false
        },
        "temperature": 25.4,
        "rain": 16.0,
        "humidity": 88.0,
        "predicted_visitors": 3956,
        "congestion": 1,
        "user_score": 18,
        "weather_score": 79
      },
      "meta": {
        "weather_summary": "구름 조금",
        "weather_icon": "🌤️",
        "weather_source": "Open-Meteo",
        "forecast_time": "12:00",
        "congestion_dayoff": true,
        "congestion_nextdayoff": true,
        "congestion_month": "08"
      }
    }
  ]