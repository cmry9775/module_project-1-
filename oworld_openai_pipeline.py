from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any, Callable
from zoneinfo import ZoneInfo

import holidays
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PIPELINE_VERSION = "5.0-11-features"
WEATHER_URL = "https://weather.com/ko-KR/kr/city/daejeon/tenday"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
SEOUL = ZoneInfo("Asia/Seoul")


# 학습된 컬럼과 매치
MODEL_FEATURES = [
    "holiday",
    "mon",
    "tue",
    "wed",
    "thur",
    "fri",
    "sat",
    "sun",
    "temperature",
    "rain",
    "humidity",
]
WEEKDAYS = ["mon", "tue", "wed", "thur", "fri", "sat", "sun"]

WEATHER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "days": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target_date": {"type": "string"},
                    "forecast_found": {"type": "boolean"},
                    "max_temp": {"type": ["number", "null"]},
                    "min_temp": {"type": ["number", "null"]},
                    "day_humidity": {"type": ["number", "null"]},
                    "night_humidity": {"type": ["number", "null"]},
                    "day_rain_probability": {"type": ["number", "null"]},
                    "night_rain_probability": {"type": ["number", "null"]},
                    "weather_summary": {"type": ["string", "null"]},
                },
                "required": [
                    "target_date",
                    "forecast_found",
                    "max_temp",
                    "min_temp",
                    "day_humidity",
                    "night_humidity",
                    "day_rain_probability",
                    "night_rain_probability",
                    "weather_summary",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["days"],
    "additionalProperties": False,
}


def get_client():
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(".env에 OPENAI_API_KEY를 설정하세요.")

    return OpenAI(
        api_key=api_key,
        timeout=90.0,
        max_retries=1,
    )


def get_dates(days: int = 10) -> list[str]:
    if not 1 <= days <= 10:
        raise ValueError("days는 1~10이어야 합니다.")

    today = datetime.now(SEOUL).date()
    return [(today + timedelta(days=i)).isoformat() for i in range(days)]


# Web Search
@lru_cache(maxsize=10)
def search_weather(days: int = 10) -> list[dict[str, Any]]:
    dates = get_dates(days)
    client = get_client()

    search_prompt = f"""
아래 Weather.com 대전광역시 10일 예보 페이지를 우선 확인하세요.
{WEATHER_URL}

조회 날짜: {', '.join(dates)}

각 날짜별로 다음 값을 찾으세요.
- 최고기온, 최저기온
- 낮 습도, 밤 습도
- 낮 강수확률, 밤 강수확률
- 날씨 요약

규칙:
- 대전광역시 자료만 사용하세요.
- 월간 평균값이나 과거 평년값은 사용하지 마세요.
- 확인할 수 없는 값은 확인 불가라고 표시하세요.
- 온도는 섭씨, 습도와 강수확률은 퍼센트 기준입니다.
"""

    search_response = client.responses.create(
        model=OPENAI_MODEL,
        input=search_prompt,
        tools=[
            {
                "type": "web_search",
                "filters": {"allowed_domains": ["weather.com"]},
                "user_location": {
                    "type": "approximate",
                    "country": "KR",
                    "city": "Daejeon",
                    "region": "Daejeon",
                    "timezone": "Asia/Seoul",
                },
            }
        ],
        tool_choice="required",
        store=False,
    )

    searched_text = (search_response.output_text or "").strip()
    if not searched_text:
        raise RuntimeError("Weather.com 검색 결과가 비어 있습니다.")

    parse_prompt = f"""
아래 웹 검색 결과를 요청 날짜별 날씨 JSON으로 구조화하세요.

요청 날짜: {', '.join(dates)}

검색 결과:
{searched_text}

규칙:
- 요청 날짜마다 객체를 정확히 하나씩 만드세요.
- 실제 예보를 확인했으면 forecast_found=true로 작성하세요.
- 찾지 못한 값은 null로 작성하고 추측하지 마세요.
- 숫자에는 단위를 넣지 마세요.
"""

    parse_response = client.responses.create(
        model=OPENAI_MODEL,
        input=parse_prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "daejeon_weather_forecast",
                "strict": True,
                "schema": WEATHER_RESPONSE_SCHEMA,
            }
        },
        store=False,
    )

    try:
        parsed = json.loads(parse_response.output_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("날씨 검색 결과를 JSON으로 변환하지 못했습니다.") from exc

    by_date = {
        item["target_date"]: item
        for item in parsed.get("days", [])
        if item.get("target_date") in dates
    }

    empty = {
        "forecast_found": False,
        "max_temp": None,
        "min_temp": None,
        "day_humidity": None,
        "night_humidity": None,
        "day_rain_probability": None,
        "night_rain_probability": None,
        "weather_summary": None,
    }

    return [
        by_date.get(day, {"target_date": day, **empty})
        for day in dates
    ]

#공휴일과 요일 One-Hot 생성
def make_date_features(target_date: str) -> dict[str, int]:
    target = date.fromisoformat(target_date)
    weekday_index = target.weekday()
    kr_holidays = holidays.KR(years=[target.year])

    result = {"holiday": int(target in kr_holidays)}
    for index, name in enumerate(WEEKDAYS):
        result[name] = int(index == weekday_index)

    return result

#UI 표시용 주말 여부
def is_weekend(target_date: str) -> bool:
    return date.fromisoformat(target_date).weekday() >= 5

#검색 날씨를 temperature, rain, humidity로 변환
def normalize_weather(raw: dict[str, Any]) -> dict[str, Any] | None:
    required = [
        "max_temp",
        "min_temp",
        "day_humidity",
        "night_humidity",
        "day_rain_probability",
    ]

    if not raw.get("forecast_found"):
        return None
    if any(raw.get(name) is None for name in required):
        return None

    temperature = (float(raw["max_temp"]) + float(raw["min_temp"])) / 2
    humidity = (
        float(raw["day_humidity"]) + float(raw["night_humidity"])
    ) / 2

    # 오월드 이용 시간대를 기준으로 낮 강수확률을 사용합니다.
    rain = float(raw["day_rain_probability"])

    if not -40 <= temperature <= 50:
        return None
    if not 0 <= rain <= 100 or not 0 <= humidity <= 100:
        return None

    return {
        "temperature": round(temperature, 1),
        "rain": round(rain, 1),
        "humidity": round(humidity, 1),
        "weather_summary": raw.get("weather_summary"),
    }

#모델 input Feature DataFrame 제작
def make_model_input(target_date: str, weather: dict[str, Any]) -> pd.DataFrame:
    row = make_date_features(target_date)
    row.update(
        {
            "temperature": weather["temperature"],
            "rain": weather["rain"],
            "humidity": weather["humidity"],
        }
    )
    return pd.DataFrame([row], columns=MODEL_FEATURES)

#ML 연결 함수에서 방문객·혼잡도·두 점수를 받아오기
def call_ml(
    model_input: pd.DataFrame,
    predictor: Callable[[pd.DataFrame], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if predictor is None:
        try:
            from ml_service import predict_all
        except ImportError as exc:
            raise RuntimeError("ml_service.py를 같은 폴더에 두세요.") from exc
        predictor = predict_all

    result = predictor(model_input.copy())
    required = [
        "predicted_visitors",
        "congestion",
        "user_score",
        "weather_score",
    ]

    if not isinstance(result, dict):
        raise TypeError("ML 결과는 dict여야 합니다.")
    missing = [key for key in required if key not in result]
    if missing:
        raise KeyError(f"ML 결과에서 누락된 값: {missing}")

    congestion = int(result["congestion"])
    if congestion not in (0, 1, 2):
        raise ValueError("congestion은 0, 1, 2 중 하나여야 합니다.")

    return {
        "predicted_visitors": max(0, round(float(result["predicted_visitors"]))),
        "congestion": congestion,
        "user_score": round(float(result["user_score"])),
        "weather_score": round(float(result["weather_score"])),
    }

#날짜 한 개의 UI용 결과
def make_day_output(
    raw: dict[str, Any],
    predictor: Callable[[pd.DataFrame], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    target_date = raw["target_date"]
    weather = normalize_weather(raw)

    if weather is None:
        return {
            "date": target_date,
            "status": "weather_unavailable",
            "data": None,
        }

    model_input = make_model_input(target_date, weather)

    try:
        prediction = call_ml(model_input, predictor)
    except Exception as exc:
        return {
            "date": target_date,
            "status": "model_error",
            "message": str(exc),
            "model_input": model_input.iloc[0].to_dict(),
            "data": None,
        }

    row = model_input.iloc[0]
    return {
        "date": target_date,
        "status": "success",
        "data": {
            "holiday": bool(row["holiday"]),
            # 주말은 UI 표시용으로만 유지합니다.
            "weekend": is_weekend(target_date),
            "weekday": {name: bool(row[name]) for name in WEEKDAYS},
            "temperature": weather["temperature"],
            "rain": weather["rain"],
            "humidity": weather["humidity"],
            "predicted_visitors": prediction["predicted_visitors"],
            "congestion": prediction["congestion"],
            "user_score": prediction["user_score"],
            "weather_score": prediction["weather_score"],
        },
        "meta": {
            "weather_summary": weather["weather_summary"],
            "weather_source": WEATHER_URL,
        },
    }

#UI용 오늘 포함 최대 10일치 데이터
def build_ui_payload(
    days: int = 10,
    predictor: Callable[[pd.DataFrame], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    results = [
        make_day_output(raw, predictor)
        for raw in search_weather(days)
    ]

    return {
        "generated_at": datetime.now(SEOUL).isoformat(timespec="seconds"),
        "weather_source": WEATHER_URL,
        "feature_columns": MODEL_FEATURES,
        "days": results,
    }


def get_selected_day(payload: dict[str, Any], selected_date: str):
    """UI에서 선택한 날짜의 이미 계산된 결과를 반환합니다."""
    return next(
        (item for item in payload["days"] if item["date"] == selected_date),
        None,
    )


if __name__ == "__main__":
    print(json.dumps(build_ui_payload(10), ensure_ascii=False, indent=2))
