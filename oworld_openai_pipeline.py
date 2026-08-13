"""오월드 10일 예측 데이터 파이프라인.

역할
- weather.py에서 날짜별 실제 날씨를 받음
- 방문객 회귀 모델용 11개 Feature를 생성
- 혼잡도 분류 모델용 17개 Feature를 별도로 생성
- ml_service.py에 두 입력을 전달
- Streamlit UI가 사용할 날짜별 결과를 반환
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

import holidays
import pandas as pd

from weather import get_forecast_weather_range

PIPELINE_VERSION = "9.0-dual-model-input"
SEOUL = ZoneInfo("Asia/Seoul")
WEATHER_SOURCE = "Open-Meteo / 대전 오월드 좌표 / 낮 12시 예보"

# 방문객 수 회귀 모델의 실제 Feature 11개
VISITOR_FEATURES = [
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

# 혼잡도 분류 모델의 실제 Feature 17개
CONGESTION_FEATURES = [
    "dayoff",
    "nextdayoff",
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
    "temperature",
    "rain",
    "humidity",
]

WEEKDAYS = ["mon", "tue", "wed", "thur", "fri", "sat", "sun"]
MONTHS = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]


def _holiday_calendar(*dates: date):
    """입력 날짜들에 필요한 연도의 대한민국 공휴일 달력을 만듭니다."""
    years = sorted({value.year for value in dates})
    return holidays.KR(years=years)


def _is_dayoff(target: date) -> bool:
    """주말 또는 대한민국 법정공휴일이면 True를 반환합니다.

    현재 혼잡도 모델 Feature 이름을 기준으로 dayoff를 이렇게 해석했습니다.
    ML 학습 코드의 dayoff 정의가 다르면 이 함수만 같은 기준으로 수정해야 합니다.
    """
    calendar = _holiday_calendar(target)
    return target.weekday() >= 5 or target in calendar


def make_visitor_date_features(target_date: str) -> dict[str, int]:
    """방문객 회귀 모델용 공휴일·요일 One-Hot Feature를 만듭니다."""
    target = date.fromisoformat(target_date)
    calendar = _holiday_calendar(target)

    result = {"holiday": int(target in calendar)}
    weekday_index = target.weekday()

    for index, name in enumerate(WEEKDAYS):
        result[name] = int(index == weekday_index)

    return result


def make_congestion_date_features(target_date: str) -> dict[str, int]:
    """혼잡도 분류 모델용 휴일·다음날 휴일·월 One-Hot을 만듭니다."""
    target = date.fromisoformat(target_date)
    next_date = target + timedelta(days=1)

    result = {
        "dayoff": int(_is_dayoff(target)),
        "nextdayoff": int(_is_dayoff(next_date)),
    }

    for month_number, name in enumerate(MONTHS, start=1):
        result[name] = int(target.month == month_number)

    return result


def is_weekend(target_date: str) -> bool:
    """UI 표시용 주말 여부입니다."""
    return date.fromisoformat(target_date).weekday() >= 5


def _validate_weather(weather: dict[str, Any]) -> dict[str, float]:
    """두 모델이 공통으로 쓰는 날씨값을 검증하고 숫자로 변환합니다."""
    required = ["temperature", "rain", "humidity"]
    missing = [name for name in required if weather.get(name) is None]

    if missing:
        raise ValueError(f"날씨 데이터에서 누락된 값: {missing}")

    temperature = float(weather["temperature"])
    rain = float(weather["rain"])
    humidity = float(weather["humidity"])

    if not -40 <= temperature <= 50:
        raise ValueError(f"기온 범위가 잘못되었습니다: {temperature}")
    if not 0 <= rain <= 100:
        raise ValueError(f"강수확률 범위가 잘못되었습니다: {rain}")
    if not 0 <= humidity <= 100:
        raise ValueError(f"습도 범위가 잘못되었습니다: {humidity}")

    return {
        "temperature": temperature,
        "rain": rain,
        "humidity": humidity,
    }


def make_visitor_model_input(
    target_date: str,
    weather: dict[str, Any],
) -> pd.DataFrame:
    """방문객 회귀 모델용 11개 Feature DataFrame을 만듭니다."""
    weather_values = _validate_weather(weather)
    row = make_visitor_date_features(target_date)
    row.update(weather_values)
    return pd.DataFrame([row], columns=VISITOR_FEATURES).astype(float)


def make_congestion_model_input(
    target_date: str,
    weather: dict[str, Any],
) -> pd.DataFrame:
    """혼잡도 분류 모델용 17개 Feature DataFrame을 만듭니다."""
    weather_values = _validate_weather(weather)
    row = make_congestion_date_features(target_date)
    row.update(weather_values)
    return pd.DataFrame([row], columns=CONGESTION_FEATURES).astype(float)


def make_model_input(
    target_date: str,
    weather: dict[str, Any],
) -> pd.DataFrame:
    """이전 코드와의 호환용: 방문객 모델 입력만 반환합니다."""
    return make_visitor_model_input(target_date, weather)


def call_ml(
    visitor_input: pd.DataFrame,
    congestion_input: pd.DataFrame,
    predictor: Callable[[pd.DataFrame, pd.DataFrame], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """ML 서비스에서 방문객·혼잡도·두 점수를 받아 검증합니다."""
    if predictor is None:
        try:
            from ml_service import predict_all
        except ImportError as exc:
            raise RuntimeError("ml_service.py를 같은 폴더에 두세요.") from exc
        predictor = predict_all

    result = predictor(visitor_input.copy(), congestion_input.copy())

    required = [
        "predicted_visitors",
        "congestion",
        "user_score",
        "weather_score",
    ]

    if not isinstance(result, dict):
        raise TypeError("ML 결과는 dict여야 합니다.")

    missing = [name for name in required if name not in result]
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


def make_day_output(
    weather: dict[str, Any],
    predictor: Callable[[pd.DataFrame, pd.DataFrame], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """날짜 한 개의 UI용 결과를 만듭니다."""
    target_date = weather.get("target_date")

    if not target_date:
        return {
            "date": None,
            "status": "weather_error",
            "message": "날씨 결과에 target_date가 없습니다.",
            "data": None,
        }

    if weather.get("status") != "success":
        return {
            "date": target_date,
            "status": "weather_unavailable",
            "message": weather.get("message", "날씨 정보를 사용할 수 없습니다."),
            "data": None,
        }

    try:
        visitor_input = make_visitor_model_input(target_date, weather)
        congestion_input = make_congestion_model_input(target_date, weather)
        prediction = call_ml(visitor_input, congestion_input, predictor)
    except Exception as exc:
        return {
            "date": target_date,
            "status": "model_error",
            "message": str(exc),
            "visitor_model_input": (
                visitor_input.iloc[0].to_dict()
                if "visitor_input" in locals()
                else None
            ),
            "congestion_model_input": (
                congestion_input.iloc[0].to_dict()
                if "congestion_input" in locals()
                else None
            ),
            "data": None,
        }

    visitor_row = visitor_input.iloc[0]
    congestion_row = congestion_input.iloc[0]

    return {
        "date": target_date,
        "status": "success",
        "data": {
            "holiday": bool(visitor_row["holiday"]),
            "weekend": is_weekend(target_date),
            "weekday": {name: bool(visitor_row[name]) for name in WEEKDAYS},
            "temperature": float(visitor_row["temperature"]),
            "rain": float(visitor_row["rain"]),
            "humidity": float(visitor_row["humidity"]),
            "predicted_visitors": prediction["predicted_visitors"],
            "congestion": prediction["congestion"],
            "user_score": prediction["user_score"],
            "weather_score": prediction["weather_score"],
        },
        "meta": {
            "weather_summary": weather.get("weather"),
            "weather_icon": weather.get("icon"),
            "weather_source": weather.get("source", WEATHER_SOURCE),
            "forecast_time": weather.get("forecast_time", "12:00"),
            "congestion_dayoff": bool(congestion_row["dayoff"]),
            "congestion_nextdayoff": bool(congestion_row["nextdayoff"]),
            "congestion_month": target_date[5:7],
        },
    }


def build_ui_payload(
    days: int = 10,
    predictor: Callable[[pd.DataFrame, pd.DataFrame], dict[str, Any]] | None = None,
    weather_provider: Callable[[int], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """UI 팀에 전달할 오늘 포함 최대 10일치 데이터를 만듭니다."""
    provider = weather_provider or get_forecast_weather_range
    weather_days = provider(days)
    results = [make_day_output(item, predictor) for item in weather_days]

    return {
        "generated_at": datetime.now(SEOUL).isoformat(timespec="seconds"),
        "weather_source": WEATHER_SOURCE,
        # 기존 UI 호환을 위해 방문객 모델 Feature 목록은 그대로 둡니다.
        "feature_columns": VISITOR_FEATURES,
        "congestion_feature_columns": CONGESTION_FEATURES,
        "days": results,
    }


def get_selected_day(
    payload: dict[str, Any],
    selected_date: str,
) -> dict[str, Any] | None:
    """UI에서 선택한 날짜의 이미 계산된 결과를 반환합니다."""
    return next(
        (item for item in payload.get("days", []) if item.get("date") == selected_date),
        None,
    )


if __name__ == "__main__":
    import json

    print(f"pipeline version: {PIPELINE_VERSION}")
    print(json.dumps(build_ui_payload(10), ensure_ascii=False, indent=2))
