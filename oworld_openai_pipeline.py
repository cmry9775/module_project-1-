import json
import os
from datetime import date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import holidays
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

WEATHER_URL = "https://weather.com/ko-KR/kr/city/daejeon/tenday"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
SEOUL = ZoneInfo("Asia/Seoul")

FEATURES = [
    "holiday", "weekend",
    "mon", "tue", "wed", "thur", "fri", "sat", "sun",
    "temperature", "rain", "humidity",
]
WEEKDAYS = ["mon", "tue", "wed", "thur", "fri", "sat", "sun"]


def get_client():
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(".env에 OPENAI_API_KEY를 설정하세요.")
    return OpenAI(api_key=key)


def get_dates(days=10):
    if days < 1 or days > 10:
        raise ValueError("days는 1~10이어야 합니다.")

    today = datetime.now(SEOUL).date()
    return [(today + timedelta(days=i)).isoformat() for i in range(days)]


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

#Weather.com 검색 후, 별도 요청에서 결과를 JSON으로 정리
@lru_cache(maxsize=1)
def search_weather(days=10):
    dates = get_dates(days)
    client = get_client()

    # 1차 요청: Web Search만 실행합니다.
    # Web Search와 JSON mode를 같은 요청에서 사용하면 400 오류가 발생합니다.
    search_prompt = f"""
반드시 아래 Weather.com 대전광역시 10일 예보 페이지를 우선 확인하세요.
{WEATHER_URL}

조회 날짜: {', '.join(dates)}

각 날짜별로 다음 값을 찾아 정리하세요.
- 최고기온
- 최저기온
- 낮 습도
- 밤 습도
- 낮 강수확률
- 밤 강수확률
- 날씨 요약

규칙:
- 대전광역시 자료만 사용하세요.
- 월간 평균값이나 과거 평년값은 사용하지 마세요.
- 페이지에서 확인할 수 없는 값은 '확인 불가'라고 표시하세요.
- 요청 날짜가 10일 예보에 없으면 그 사실을 표시하세요.
- 온도는 섭씨, 습도와 강수확률은 퍼센트 기준으로 정리하세요.
"""

    search_response = client.responses.create(
        model=OPENAI_MODEL,
        input=search_prompt,
        tools=[{
            "type": "web_search",
            "filters": {"allowed_domains": ["weather.com"]},
            "user_location": {
                "type": "approximate",
                "country": "KR",
                "city": "Daejeon",
                "region": "Daejeon",
                "timezone": "Asia/Seoul",
            },
        }],
        tool_choice="required",
        store=False,
    )

    searched_text = (search_response.output_text or "").strip()
    if not searched_text:
        raise RuntimeError("Weather.com 검색 결과가 비어 있습니다.")

    # 2차 요청: Web Search 없이 검색 결과를 JSON Schema에 맞춰 변환
    parse_prompt = f"""
아래는 Weather.com 대전 10일 예보를 웹 검색한 결과입니다.
요청 날짜 목록에 맞춰 기상 데이터를 구조화하세요.

요청 날짜:
{', '.join(dates)}

검색 결과:
{searched_text}

변환 규칙:
- 요청 날짜마다 객체를 정확히 하나씩 만드세요.
- 실제 예보를 확인했으면 forecast_found=true로 작성하세요.
- 날짜가 없거나 예보를 확인하지 못했으면 forecast_found=false로 작성하고,
  확인하지 못한 숫자와 문장은 null로 작성하세요.
- 기온, 습도, 강수확률에서 단위를 제거하고 숫자만 작성하세요.
- 검색 결과에 없는 값을 추측하지 마세요.
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
        result = json.loads(parse_response.output_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "날씨 검색 결과를 JSON으로 변환하지 못했습니다.\n"
            + str(parse_response.output_text)
        ) from exc

    requested_dates = set(dates)
    by_date = {
        item["target_date"]: item
        for item in result.get("days", [])
        if item.get("target_date") in requested_dates
    }

    return [
        by_date.get(
            day,
            {
                "target_date": day,
                "forecast_found": False,
                "max_temp": None,
                "min_temp": None,
                "day_humidity": None,
                "night_humidity": None,
                "day_rain_probability": None,
                "night_rain_probability": None,
                "weather_summary": None,
            },
        )
        for day in dates
    ]

#holiday, weekend, mon~sun 값 생성
def make_date_features(target_date):
    day = date.fromisoformat(target_date)
    weekday = day.weekday()
    kr_holidays = holidays.KR(years=[day.year])

    result = {
        "holiday": int(day in kr_holidays),
        "weekend": int(weekday >= 5),
    }

    for index, name in enumerate(WEEKDAYS):
        result[name] = int(index == weekday)

    return result

# 검색값을 temperature, rain, humidity로 변환
def normalize_weather(raw):
    needed = [
        "max_temp", "min_temp",
        "day_humidity", "night_humidity",
        "day_rain_probability",
    ]

    if not raw.get("forecast_found"):
        return None
    if any(raw.get(name) is None for name in needed):
        return None

    temperature = (
        float(raw["max_temp"]) + float(raw["min_temp"])
    ) / 2
    humidity = (
        float(raw["day_humidity"]) + float(raw["night_humidity"])
    ) / 2

    # 오월드 운영 시간대를 고려하여 낮 강수확률을 사용
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

#ML Input
def make_model_input(target_date, weather):
    row = make_date_features(target_date)
    row["temperature"] = weather["temperature"]
    row["rain"] = weather["rain"]
    row["humidity"] = weather["humidity"]

    return pd.DataFrame([row], columns=FEATURES)

#ML에서 받아오는 4가지 데이터
def call_ml(model_input, predictor=None):
    if predictor is None:
        try:
            from ml_service import predict_all
        except ImportError as exc:
            raise RuntimeError(
                "ml_service.py에 ML 팀의 predict_all()을 연결하세요."
            ) from exc
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
    if any(key not in result for key in required):
        raise KeyError(f"ML 결과에는 {required}가 필요합니다.")

    congestion = int(result["congestion"])
    if congestion not in [0, 1, 2]:
        raise ValueError("congestion은 0, 1, 2여야 합니다.")

    return {
        "predicted_visitors": max(
            0, round(float(result["predicted_visitors"]))
        ),
        "congestion": congestion,
        "user_score": round(float(result["user_score"])),
        "weather_score": round(float(result["weather_score"])),
    }

# ui OutPut
def make_day_output(raw, predictor=None):
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
            "weekend": bool(row["weekend"]),
            "weekday": {
                name: bool(row[name]) for name in WEEKDAYS
            },
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

#10일치 데이터 생성
def build_ui_payload(days=10, predictor=None):
    results = [
        make_day_output(raw, predictor)
        for raw in search_weather(days)
    ]

    return {
        "generated_at": datetime.now(SEOUL).isoformat(timespec="seconds"),
        "weather_source": WEATHER_URL,
        "feature_columns": FEATURES,
        "days": results,
    }

#선택된 날짜의 데이터 출력
def get_selected_day(payload, selected_date):
    return next(
        (item for item in payload["days"] if item["date"] == selected_date),
        None,
    )


if __name__ == "__main__":
    print(json.dumps(build_ui_payload(10), ensure_ascii=False, indent=2))
