"""대전 오월드 날씨 조회 모듈.

Open-Meteo에서 대전 오월드 좌표의 시간별 예보를 한 번 조회한 뒤,
선택 날짜의 낮 12시 기온·습도·강수확률을 반환합니다.
파이프라인은 이 파일의 함수만 호출하며, 날씨 조회 구현은 분리합니다.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo

import requests

SEOUL = ZoneInfo("Asia/Seoul")
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OWORLD_LATITUDE = 36.2875
OWORLD_LONGITUDE = 127.3985
MAX_FORECAST_DAYS = 14
PROJECT_MAX_DAYS = 10


def _get_weather_info(code: int | None) -> tuple[str, str]:
    """WMO 날씨 코드를 한글 설명과 아이콘으로 변환합니다."""
    if code == 0:
        return "맑음", "☀️"
    if code in (1, 2):
        return "구름 조금", "🌤️"
    if code == 3:
        return "흐림", "☁️"
    if code in (45, 48):
        return "안개", "🌫️"
    if code in (51, 53, 55, 61, 63, 65, 80, 81, 82):
        return "비", "🌧️"
    if code in (71, 73, 75, 85, 86):
        return "눈", "❄️"
    if code in (95, 96, 99):
        return "뇌우", "🌩️"
    return "알 수 없음", "🌡️"


@lru_cache(maxsize=1)
def _fetch_hourly_forecast() -> dict[str, Any]:
    """시간별 예보를 한 번 받아 메모리에 저장합니다."""
    params = {
        "latitude": OWORLD_LATITUDE,
        "longitude": OWORLD_LONGITUDE,
        "hourly": (
            "temperature_2m,relative_humidity_2m,"
            "precipitation_probability,weather_code"
        ),
        "timezone": "Asia/Seoul",
        "forecast_days": MAX_FORECAST_DAYS,
    }

    response = requests.get(OPEN_METEO_URL, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()

    hourly = payload.get("hourly")
    required = [
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation_probability",
        "weather_code",
    ]
    if not isinstance(hourly, dict) or any(name not in hourly for name in required):
        raise RuntimeError("Open-Meteo 응답에 필요한 시간별 날씨 항목이 없습니다.")

    return hourly


def clear_weather_cache() -> None:
    """새 예보를 강제로 다시 받을 때 사용합니다."""
    _fetch_hourly_forecast.cache_clear()


def get_forecast_weather(target_date: str) -> dict[str, Any]:
    """특정 날짜의 낮 12시 날씨를 모델 입력 형태로 반환합니다.

    반환 성공 예시:
    {
        "target_date": "2026-08-15",
        "status": "success",
        "weather": "맑음",
        "icon": "☀️",
        "temperature": 27.0,
        "humidity": 65.0,
        "rain": 20.0,
        "source": "Open-Meteo",
        "forecast_time": "12:00"
    }
    """
    try:
        # 날짜 형식 검증
        datetime.strptime(target_date, "%Y-%m-%d")

        hourly = _fetch_hourly_forecast()
        target_time = f"{target_date}T12:00"
        times = hourly["time"]

        if target_time not in times:
            return {
                "target_date": target_date,
                "status": "unavailable",
                "message": "해당 날짜의 낮 12시 예보가 없습니다.",
                "data": None,
            }

        index = times.index(target_time)
        temperature = hourly["temperature_2m"][index]
        humidity = hourly["relative_humidity_2m"][index]
        rain = hourly["precipitation_probability"][index]
        weather_code = hourly["weather_code"][index]

        if temperature is None or humidity is None or rain is None:
            return {
                "target_date": target_date,
                "status": "unavailable",
                "message": "모델 입력에 필요한 날씨값이 누락되었습니다.",
                "data": None,
            }

        weather_text, icon = _get_weather_info(int(weather_code))

        return {
            "target_date": target_date,
            "status": "success",
            "weather": weather_text,
            "icon": icon,
            "temperature": round(float(temperature), 1),
            "humidity": round(float(humidity), 1),
            "rain": round(float(rain), 1),
            "source": "Open-Meteo",
            "forecast_time": "12:00",
        }

    except ValueError:
        return {
            "target_date": target_date,
            "status": "error",
            "message": "날짜는 YYYY-MM-DD 형식이어야 합니다.",
            "data": None,
        }
    except requests.RequestException as exc:
        return {
            "target_date": target_date,
            "status": "error",
            "message": f"날씨 서버 요청 실패: {exc}",
            "data": None,
        }
    except (KeyError, IndexError, TypeError, RuntimeError) as exc:
        return {
            "target_date": target_date,
            "status": "error",
            "message": f"날씨 데이터 처리 실패: {exc}",
            "data": None,
        }


def get_forecast_weather_range(days: int = 10) -> list[dict[str, Any]]:
    """오늘 포함 최대 10일의 날씨를 반환합니다.

    내부 API 응답은 캐시되므로 날짜마다 외부 요청을 반복하지 않습니다.
    """
    if not 1 <= days <= PROJECT_MAX_DAYS:
        raise ValueError("days는 1~10이어야 합니다.")

    today = datetime.now(SEOUL).date()
    return [
        get_forecast_weather((today + timedelta(days=offset)).isoformat())
        for offset in range(days)
    ]


if __name__ == "__main__":
    import json

    print(
        json.dumps(
            get_forecast_weather_range(3),
            ensure_ascii=False,
            indent=2,
        )
    )
