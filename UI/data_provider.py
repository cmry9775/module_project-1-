"""
data_provider.py - 다른 팀에서 넘어오는 데이터의 수신 창구

UI는 이 파일의 fetch_forecast() 하나만 호출한다.
실제 연동 시 USE_MOCK = False 로 바꾸면 되고, app.py 는 손대지 않는다.

────────────────────────────────────────────────────────────
[ API팀 payload ]  build_ui_payload(days=N) 가 주는 원본 형식

  {"days": [
      {
        "date": "2026-08-13",
        "status": "success",
        "data": {
          "holiday": bool, "weekend": bool,
          "weekday": {"mon": bool, ..., "thur": bool, ..., "sun": bool},
          "temperature": float,          # °C
          "rain": float,                 # 강수확률 %
          "humidity": float,             # %
          "predicted_visitors": int,     # ML팀 Regression
          "congestion": int,             # ML팀 Classification 0/1/2
          "user_score": int,             # 0~100, 높을수록 한산
          "weather_score": int           # 0~100, 높을수록 좋은 날씨
        },
        "meta": {
          "weather_summary": str, "weather_icon": str, "weather_source": str,
          "forecast_time": str,
          "congestion_dayoff": bool,     # 그날이 쉬는 날인지
          "congestion_nextdayoff": bool, # 다음 날이 쉬는 날인지
          "congestion_month": str        # "08"
        }
      }, ...
  ]}

[ UI 수신 스키마 ]  fetch_forecast() 는 위 payload 를 아래 dict 리스트로 펴서
                    날짜 오름차순으로 반환한다.

  date            : datetime.date  날짜
  status          : str            수신 상태 ("success" 등)
  ok              : bool           status == "success"
  weekday         : str            요일 ('월'~'일')  ※ 원핫을 표시용 문자열로 변환
  is_weekend      : bool           주말 여부
  is_holiday      : bool           공휴일 여부
  holiday_name    : str            공휴일명 (payload 에 없어 로컬 목록에서 조회)
  temp            : float          평균기온 (°C)
  rain_prob       : float          강수확률 (%)
  humidity        : int            평균습도 (%)
  pred_visitors   : int            예측 이용자 수 (명)
  congestion      : int            혼잡도 원본 정수 (0/1/2, 알 수 없으면 -1)
  crowd_level     : str            혼잡도 라벨 (UI가 계산하지 않고 정수만 번역)
  visitor_score   : int            이용자 Score 0~100 (높을수록 한산)
  weather_score   : int            날씨 Score 0~100 (높을수록 좋은 날씨)
  weather_summary : str            날씨 요약 ("구름 조금")
  weather_icon    : str            날씨 이모지
  weather_source  : str            기상 출처 ("Open-Meteo")
  forecast_time   : str            예보 기준 시각 ("12:00")
  is_dayoff       : bool           그날이 쉬는 날인지
  is_next_dayoff  : bool           다음 날이 쉬는 날인지
  month           : str            예측 기준 월 ("08")
────────────────────────────────────────────────────────────
"""

import datetime as dt

import numpy as np

# 실제 모델/API 연동 시 False 로 변경
USE_MOCK = True

WEEKDAY_KR = ["월", "화", "수", "목", "금", "토", "일"]

# payload 의 요일 원핫 키. 목요일은 thur 철자를 쓴다.
WEEKDAY_KEYS = ["mon", "tue", "wed", "thur", "fri", "sat", "sun"]

# congestion 정수 → 표시 라벨. ML팀 분류 기준은 예측 인원 0~700 / 700~3000 / 3001~ 이다.
# 분류 모델과 회귀 모델이 따로 학습되어 두 값이 어긋나는 날이 있을 수 있는데,
# UI는 받은 값을 고치지 않고 그대로 표시한다.
CONGESTION_LABELS = {0: "여유", 1: "보통", 2: "혼잡"}
CROWD_LEVEL_UNKNOWN = "알 수 없음"

# app.py 의 색상 매핑(theme.LEVEL_COLOR)과 범례가 이 목록을 따른다.
CROWD_LEVELS = list(CONGESTION_LABELS.values())

# TODO(전처리팀): 아래 딕셔너리는 임시본이다.
# payload 에 공휴일 '이름'은 없어서 배지 문구용으로만 쓰고, 공휴일 여부 자체는 payload 값을 따른다.
HOLIDAYS = {
    dt.date(2026, 1, 1): "신정",
    dt.date(2026, 2, 16): "설날 연휴",
    dt.date(2026, 2, 17): "설날",
    dt.date(2026, 2, 18): "설날 연휴",
    dt.date(2026, 3, 1): "삼일절",
    dt.date(2026, 3, 2): "삼일절 대체공휴일",
    dt.date(2026, 5, 5): "어린이날",
    dt.date(2026, 5, 24): "부처님오신날",
    dt.date(2026, 5, 25): "부처님오신날 대체공휴일",
    dt.date(2026, 6, 6): "현충일",
    dt.date(2026, 8, 15): "광복절",
    dt.date(2026, 8, 17): "광복절 대체공휴일",
    dt.date(2026, 9, 24): "추석 연휴",
    dt.date(2026, 9, 25): "추석",
    dt.date(2026, 9, 26): "추석 연휴",
    dt.date(2026, 10, 3): "개천절",
    dt.date(2026, 10, 5): "개천절 대체공휴일",
    dt.date(2026, 10, 9): "한글날",
    dt.date(2026, 12, 25): "성탄절",
}

# 학습에서 이상치로 다루기로 논의된 날 (월, 일) → 안내 문구
OUTLIER_DAYS = {
    (5, 5): "어린이날에는 실제 입장객이 예측값보다 2배 가까이 많을 수 있어요. "
            "예측값은 참고용으로만 봐주세요.",
}


# ---------------------------------------------------------------------------
# 공개 함수
# ---------------------------------------------------------------------------
def fetch_forecast(start_date: dt.date, n_days: int = 10) -> list[dict]:
    """start_date 부터 n_days 일치 예측 데이터를 반환한다.

    실제 API(build_ui_payload)는 시작일 인자를 받지 않고 오늘부터 days 일을 주므로,
    start_date 는 mock 생성과 캐시 키에만 쓰인다.
    """
    payload = (
        _mock_payload(start_date, n_days)
        if USE_MOCK
        else _real_payload(start_date, n_days)
    )
    return normalize_payload(payload)


def _real_payload(start_date: dt.date, n_days: int) -> dict:
    """API팀 파이프라인 호출. 반환 형식은 파일 상단 [ API팀 payload ] 참고."""
    from oworld_openai_pipeline import build_ui_payload

    return build_ui_payload(days=n_days)


def normalize_payload(payload) -> list[dict]:
    """payload(dict 또는 days 리스트)를 UI 수신 스키마로 변환한다."""
    days = payload.get("days", []) if isinstance(payload, dict) else payload
    rows = [row for row in (_to_row(day) for day in days or []) if row is not None]
    rows.sort(key=lambda r: r["date"])
    return rows


def outlier_note(date: dt.date) -> str:
    """학습에서 이상치로 다루는 날의 안내 문구. 해당 없으면 빈 문자열."""
    return OUTLIER_DAYS.get((date.month, date.day), "")


def crowd_label(congestion) -> str:
    """혼잡도 정수를 표시 라벨로 옮긴다. 모르는 값은 UI가 임의 판단하지 않는다."""
    try:
        return CONGESTION_LABELS.get(int(congestion), CROWD_LEVEL_UNKNOWN)
    except (TypeError, ValueError):
        return CROWD_LEVEL_UNKNOWN


# ---------------------------------------------------------------------------
# payload → UI row 변환
# ---------------------------------------------------------------------------
def _parse_date(value) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _weekday_kr(onehot, date: dt.date) -> str:
    """요일 원핫을 표시용 문자열로 바꾼다. 비었거나 날짜와 어긋나면 날짜를 따른다."""
    if isinstance(onehot, dict):
        picked = [i for i, key in enumerate(WEEKDAY_KEYS) if onehot.get(key)]
        if len(picked) == 1 and picked[0] == date.weekday():
            return WEEKDAY_KR[picked[0]]
    return WEEKDAY_KR[date.weekday()]


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _to_row(day) -> dict | None:
    if not isinstance(day, dict):
        return None
    date = _parse_date(day.get("date"))
    if date is None:
        return None

    data = day.get("data") or {}
    meta = day.get("meta") or {}
    status = str(day.get("status", "success"))

    return {
        "date": date,
        "status": status,
        "ok": status.lower() == "success",
        "weekday": _weekday_kr(data.get("weekday"), date),
        "is_weekend": bool(data.get("weekend", date.weekday() >= 5)),
        "is_holiday": bool(data.get("holiday", date in HOLIDAYS)),
        "holiday_name": HOLIDAYS.get(date, ""),
        "temp": round(_as_float(data.get("temperature")), 1),
        "rain_prob": _as_float(data.get("rain")),
        "humidity": _as_int(data.get("humidity")),
        "pred_visitors": _as_int(data.get("predicted_visitors")),
        "congestion": _as_int(data.get("congestion"), -1),
        "crowd_level": crowd_label(data.get("congestion")),
        "visitor_score": _as_int(data.get("user_score")),
        "weather_score": _as_int(data.get("weather_score")),
        "weather_summary": str(meta.get("weather_summary") or ""),
        "weather_icon": str(meta.get("weather_icon") or ""),
        "weather_source": str(meta.get("weather_source") or ""),
        "forecast_time": str(meta.get("forecast_time") or ""),
        "is_dayoff": bool(meta.get("congestion_dayoff", date.weekday() >= 5)),
        "is_next_dayoff": bool(meta.get("congestion_nextdayoff", False)),
        "month": str(meta.get("congestion_month") or f"{date.month:02d}"),
    }


# UI가 기대하는 컬럼 목록. _to_row 에서 직접 뽑아 두 곳이 어긋날 수 없게 한다.
# 예측이 0건이어도 app.py 가 이 컬럼들로 빈 DataFrame 을 만들 수 있다.
ROW_COLUMNS = list(_to_row({"date": "2000-01-01"}))


# ---------------------------------------------------------------------------
# 이하 mock 전용 - 실제 연동 후 통째로 삭제 가능
# API 와 같은 payload 를 만들어 두어야 위 변환 코드가 양쪽에서 똑같이 검증된다.
# ---------------------------------------------------------------------------
# 특정 날짜 강수확률 고정 (월, 일) → %  — 데모용 더미 오버라이드
_RAIN_OVERRIDES = {
    (8, 12): 0,
    (8, 14): 100,
    (8, 17): 0,
}

# 특정 날짜 평균기온 고정 (월, 일) → °C  — 데모용 더미 오버라이드
_TEMP_OVERRIDES = {
    (8, 17): 23.1,
}


def _mock_weather(date: dt.date) -> dict:
    """계절 기반으로 그럴듯한 기상값을 결정적으로 생성 (같은 날짜 = 같은 값)."""
    rng = np.random.default_rng(date.toordinal())
    m = date.month
    season = "봄" if m in (3, 4, 5) else "여름" if m in (6, 7, 8) else "가을" if m in (9, 10, 11) else "겨울"
    base_temp = {"봄": 17, "여름": 28, "가을": 16, "겨울": 2}[season]
    temp = round(base_temp + rng.normal(0, 3), 1)
    if (date.month, date.day) in _TEMP_OVERRIDES:
        temp = float(_TEMP_OVERRIDES[(date.month, date.day)])
    base_rain = {"봄": 25, "여름": 45, "가을": 20, "겨울": 20}[season]
    rain_prob = float(np.clip(base_rain + rng.normal(0, 20), 0, 100))
    rain_prob = round(rain_prob / 10) * 10  # 기상청 표기처럼 10% 단위
    if (date.month, date.day) in _RAIN_OVERRIDES:
        rain_prob = float(_RAIN_OVERRIDES[(date.month, date.day)])
    humidity = int(np.clip(55 + rng.normal(0, 12), 30, 95))
    return {"temp": temp, "rain_prob": rain_prob, "humidity": humidity, "season": season}


def _mock_visitors(date: dt.date, w: dict) -> int:
    """회의에서 확인한 상관관계(주말 > 기온 > 계절 > 강수)를 반영한 임시 규칙."""
    base = 600.0

    if date.weekday() >= 5:
        base *= 2.3
    if date in HOLIDAYS:
        base *= 2.6

    base *= {"봄": 1.25, "여름": 0.85, "가을": 1.35, "겨울": 0.70}[w["season"]]

    t = w["temp"]
    if t < 0:
        base *= 0.50
    elif t < 10:
        base *= 0.75
    elif t <= 25:
        base *= 1.10
    elif t <= 30:
        base *= 0.90
    else:
        base *= 0.65

    p = w["rain_prob"]
    if p >= 70:
        base *= 0.45
    elif p >= 40:
        base *= 0.75
    elif p >= 20:
        base *= 0.92

    rng = np.random.default_rng(date.toordinal() + 7)
    base *= 1 + rng.normal(0, 0.05)
    return int(max(base, 0))


def _mock_congestion(pred: int) -> int:
    """ML팀 Classification 자리를 대신하는 임시 구간 분류 (0~700 / ~3000 / 3001~)."""
    if pred > 3000:
        return 2
    if pred > 700:
        return 1
    return 0


def _mock_visitor_score(pred: int) -> int:
    """이용자 Score: 높을수록 한산. API 표본 3건에 로그로 맞춘 근사식."""
    return int(np.clip(108.8 - 25.2 * np.log10(max(pred, 1)), 0, 100))


def _mock_weather_score(w: dict) -> int:
    """날씨 Score: 높을수록 좋은 날씨."""
    temp_score = max(0, 100 - abs(w["temp"] - 21) * 5)  # 21도 최적
    rain_penalty = w["rain_prob"] * 0.6
    humid_penalty = max(0, w["humidity"] - 70) * 1.2
    return int(np.clip(temp_score - rain_penalty - humid_penalty, 0, 100))


def _mock_sky(w: dict) -> tuple[str, str]:
    """강수확률로 날씨 요약과 아이콘을 고른다."""
    p = w["rain_prob"]
    if p >= 60:
        return "비", "🌧️"
    if p >= 30:
        return "구름 많음", "⛅"
    if p >= 10:
        return "구름 조금", "🌤️"
    return "맑음", "☀️"


def _mock_dayoff(date: dt.date) -> bool:
    return date.weekday() >= 5 or date in HOLIDAYS


def _mock_payload(start_date: dt.date, n_days: int) -> dict:
    days = []
    for i in range(n_days):
        d = start_date + dt.timedelta(days=i)
        w = _mock_weather(d)
        pred = _mock_visitors(d, w)
        summary, icon = _mock_sky(w)
        days.append({
            "date": d.isoformat(),
            "status": "success",
            "data": {
                "holiday": d in HOLIDAYS,
                "weekend": d.weekday() >= 5,
                "weekday": {key: (j == d.weekday()) for j, key in enumerate(WEEKDAY_KEYS)},
                "temperature": w["temp"],
                "rain": w["rain_prob"],
                "humidity": float(w["humidity"]),
                "predicted_visitors": pred,
                "congestion": _mock_congestion(pred),
                "user_score": _mock_visitor_score(pred),
                "weather_score": _mock_weather_score(w),
            },
            "meta": {
                "weather_summary": summary,
                "weather_icon": icon,
                "weather_source": "Mock",
                "forecast_time": "12:00",
                "congestion_dayoff": _mock_dayoff(d),
                "congestion_nextdayoff": _mock_dayoff(d + dt.timedelta(days=1)),
                "congestion_month": f"{d.month:02d}",
            },
        })
    return {"days": days}
