from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

import holidays
import pandas as pd

from weather import get_forecast_weather_range

PIPELINE_VERSION = "14.0-direct-notice-functions"
SEOUL = ZoneInfo("Asia/Seoul")
WEATHER_SOURCE = "Open-Meteo / 대전 오월드 좌표 / 낮 12시 예보"
NOTICE_SOURCE = (
    "https://www.oworld.kr/newkfsweb/kfi/kfs/linkage/"
    "selectDccoLinkageList.do?mn=KFS_34_01_09_01&bbscd=9&menucd=916"
)
EVENT_BOARD_URL = (
    "https://www.oworld.kr/newkfsweb/kfi/kfs/event/"
    "selectDccoEventList.do?mn=KFS_34_02_03_01"
)

# 학습모델 Feature 17개
MODEL_FEATURES = [
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

# UI 표시용 요일 값 생성
WEEKDAYS = ["mon", "tue", "wed", "thur", "fri", "sat", "sun"]

def _holiday_calendar(*dates: date):
    years = sorted({value.year for value in dates})
    return holidays.KR(years=years)


def _is_public_holiday(target: date) -> bool:
    return target in _holiday_calendar(target)


def _is_dayoff(target: date) -> bool:
    return target.weekday() >= 5 or _is_public_holiday(target)


def make_model_date_features(target_date: str) -> dict[str, int]:
    target = date.fromisoformat(target_date)
    next_date = target + timedelta(days=1)

    result = {
        "dayoff": int(_is_dayoff(target)),
        "nextdayoff": int(_is_dayoff(next_date)),
    }

    for month_number, name in enumerate(MONTHS, start=1):
        result[name] = int(target.month == month_number)

    return result

#UI 표시용 공휴일·주말·요일 One-Hot
def make_ui_date_info(target_date: str) -> dict[str, Any]:
    target = date.fromisoformat(target_date)
    weekday_index = target.weekday()

    return {
        "holiday": _is_public_holiday(target),
        "weekend": weekday_index >= 5,
        "weekday": {
            name: index == weekday_index
            for index, name in enumerate(WEEKDAYS)
        },
    }

#학습모델이 사용하는 날씨값을 검증하고 숫자로 변환
def _validate_weather(weather: dict[str, Any]) -> dict[str, float]:
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
        "temperature": round(temperature, 1),
        "rain": round(rain, 1),
        "humidity": round(humidity, 1),
    }

# 학습모델용 input 생성
def make_model_input(
    target_date: str,
    weather: dict[str, Any],
) -> pd.DataFrame:
    row = make_model_date_features(target_date)
    row.update(_validate_weather(weather))
    return pd.DataFrame([row], columns=MODEL_FEATURES).astype(float)

#학습모델에서 방문객·혼잡도·두 점수를 받아 형식 검증
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

#공지 모듈 에러처리 UI payload 유지
def _notice_error(message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "major_summary": message,
        "notices_list": [],
        "event_board_url": EVENT_BOARD_URL,
    }

# 공지 함수 호출
def call_notice(
    notice_provider: Callable[[], dict[str, Any] | str] | None = None,
) -> dict[str, Any]:
    try:
        if notice_provider is None:
            try:
                from notice import get_oworld_notices, get_urgent_notice_json
            except ImportError as exc:
                return _notice_error(
                    "notice.py의 get_oworld_notices 또는 "
                    f"get_urgent_notice_json을 불러오지 못했습니다: {exc}"
                )

            notices = get_oworld_notices(NOTICE_SOURCE)
            raw_result = get_urgent_notice_json(notices)
        else:
            # 테스트에서는 공지 전체 결과를 반환하는 함수를 주입할 수 있습니다.
            raw_result = notice_provider()

        if isinstance(raw_result, str):
            result = json.loads(raw_result)
        elif isinstance(raw_result, dict):
            result = raw_result
        else:
            raise TypeError("공지 분석 결과는 dict 또는 JSON 문자열이어야 합니다.")

        notices_list = result.get("notices_list")
        if notices_list is None:
            # 이전 형식의 notice.py도 최소한 호환합니다.
            notices_list = result.get("notices", [])

        if not isinstance(notices_list, list):
            raise TypeError("notices_list는 list여야 합니다.")

        major_summary = (
            result.get("major_summary")
            or result.get("summary")
            or "현재 공지사항 요약을 불러오지 못했습니다."
        )

        return {
            "status": str(result.get("status", "error")),
            "major_summary": str(major_summary),
            "notices_list": notices_list[:5],
            "event_board_url": str(
                result.get("event_board_url") or EVENT_BOARD_URL
            ),
        }
    except json.JSONDecodeError as exc:
        return _notice_error(f"공지 AI 결과가 올바른 JSON이 아닙니다: {exc}")
    except Exception as exc:
        return _notice_error(f"공지사항 처리 실패: {exc}")

#날짜 한 개의 UI용 결과
def make_day_output(
    weather: dict[str, Any],
    predictor: Callable[[pd.DataFrame], dict[str, Any]] | None = None,
) -> dict[str, Any]:
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

    model_input: pd.DataFrame | None = None

    try:
        model_input = make_model_input(target_date, weather)
        prediction = call_ml(model_input, predictor)
    except Exception as exc:
        return {
            "date": target_date,
            "status": "model_error",
            "message": str(exc),
            "model_input": (
                model_input.iloc[0].to_dict()
                if model_input is not None
                else None
            ),
            "data": None,
        }

    ui_date = make_ui_date_info(target_date)
    model_row = model_input.iloc[0]

    return {
        "date": target_date,
        "status": "success",
        "data": {
            "holiday": ui_date["holiday"],
            "weekend": ui_date["weekend"],
            "weekday": ui_date["weekday"],
            "temperature": float(model_row["temperature"]),
            "rain": float(model_row["rain"]),
            "humidity": float(model_row["humidity"]),
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
            "model_dayoff": bool(model_row["dayoff"]),
            "model_nextdayoff": bool(model_row["nextdayoff"]),
            "model_month": target_date[5:7],
        },
    }

    """UI 팀에 전달할 최대 10일치 예측 결과와 공지 결과 제작

    notice는 전체 10일 예측에 공통으로 적용되는 정보이므로 날짜별 days가
    아니라 payload 최상위의 notice 항목으로 한 번만 전달
    """
def build_ui_payload(
    days: int = 10,
    predictor: Callable[[pd.DataFrame], dict[str, Any]] | None = None,
    weather_provider: Callable[[int], list[dict[str, Any]]] | None = None,
    notice_provider: Callable[[], dict[str, Any] | str] | None = None,
    include_notice: bool = True,
) -> dict[str, Any]:
    if not 1 <= days <= 10:
        raise ValueError("days는 1~10이어야 합니다.")

    provider = weather_provider or get_forecast_weather_range
    weather_days = provider(days)
    results = [make_day_output(item, predictor) for item in weather_days]

    notice_result = (
        call_notice(notice_provider)
        if include_notice
        else {
            "status": "skipped",
            "major_summary": "",
            "notices_list": [],
            "event_board_url": EVENT_BOARD_URL,
        }
    )

    return {
        "generated_at": datetime.now(SEOUL).isoformat(timespec="seconds"),
        "weather_source": WEATHER_SOURCE,
        "notice_source": NOTICE_SOURCE,
        "feature_columns": MODEL_FEATURES,
        "notice": notice_result,
        "days": results,
    }

#UI에서 선택한 날짜의 이미 계산된 결과를 반환
def get_selected_day(
    payload: dict[str, Any],
    selected_date: str,
) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in payload.get("days", [])
            if item.get("date") == selected_date
        ),
        None,
    )


if __name__ == "__main__":
    print(f"pipeline version: {PIPELINE_VERSION}")
    print(json.dumps(build_ui_payload(10), ensure_ascii=False, indent=2))
