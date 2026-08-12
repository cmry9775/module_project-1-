"""
data_provider.py - 다른 팀에서 넘어오는 데이터의 수신 창구

UI는 이 파일의 fetch_forecast() 하나만 호출한다.
실제 연동 시 USE_MOCK = False 로 바꾸고 _real_forecast() 안만 채우면 되고,
app.py 는 손대지 않는다.

────────────────────────────────────────────────────────────
[ 수신 스키마 ]  fetch_forecast() 는 아래 dict 의 리스트를 날짜 오름차순으로 반환한다.

  date            : datetime.date  날짜
  weekday         : str            요일 ('월'~'일')  ※ 모델 입력용 원핫 7개가 아니라 표시용 문자열
  is_weekend      : bool           주말 여부
  is_holiday      : bool           공휴일 여부
  holiday_name    : str            공휴일명 (없으면 "")
  temp            : float          평균기온 (°C)
  rain_prob       : float          강수확률 (%)          ※ 강수량(mm) 아님
  humidity        : int            평균습도 (%)
  pred_visitors   : int            예측 이용자 수 (명)    ← ML팀 Regression
  crowd_level     : str            예측 혼잡도            ← ML팀 Classification (UI가 계산하지 않음)
  visitor_score   : int            이용자 Score 0~100     ← ML팀 정규화 함수 (높을수록 한산)
  weather_score   : int            날씨 Score 0~100       ← ML팀 정규화 함수 (높을수록 좋은 날씨)
────────────────────────────────────────────────────────────
"""

import datetime as dt

import numpy as np

# 실제 모델/API 연동 시 False 로 변경
USE_MOCK = True

WEEKDAY_KR = ["월", "화", "수", "목", "금", "토", "일"]

# ML팀이 확정한 라벨로 교체할 것. app.py 의 색상 매핑도 이 목록을 따른다.
CROWD_LEVELS = ["여유", "보통", "혼잡"]

# TODO(전처리팀): 아래 딕셔너리는 임시본이다.
# 설날/추석은 음력이라 매년 달라지고 대체공휴일 규칙도 있으므로,
# 전처리 단계에서 이미 만든 공휴일 목록을 그대로 받아 쓰는 편이 안전하다.
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
    (5, 5): "어린이날은 방문객이 평소 예측을 크게 웃도는 날이라, 예측값은 참고용으로만 봐주세요.",
}


# ---------------------------------------------------------------------------
# 공개 함수
# ---------------------------------------------------------------------------
def fetch_forecast(start_date: dt.date, n_days: int = 10) -> list[dict]:
    """start_date 부터 n_days 일치 예측 데이터를 반환한다."""
    if USE_MOCK:
        return _mock_forecast(start_date, n_days)
    return _real_forecast(start_date, n_days)


def _real_forecast(start_date: dt.date, n_days: int) -> list[dict]:
    """
    TODO(연동 담당): OpenAI팀이 넘겨주는 결과를 위 스키마의 dict 리스트로 변환해 반환.

    예)
        raw = openai_client.get_forecast(start_date, n_days)   # 팀 합의된 호출 방식
        return [
            {
                "date": dt.date.fromisoformat(r["date"]),
                "weekday": WEEKDAY_KR[dt.date.fromisoformat(r["date"]).weekday()],
                "is_weekend": r["is_weekend"],
                "is_holiday": r["is_holiday"],
                "holiday_name": r.get("holiday_name", ""),
                "temp": float(r["temp"]),
                "rain_prob": float(r["rain_prob"]),
                "humidity": int(r["humidity"]),
                "pred_visitors": int(r["pred_visitors"]),
                "crowd_level": r["crowd_level"],
                "visitor_score": int(r["visitor_score"]),
                "weather_score": int(r["weather_score"]),
            }
            for r in raw
        ]
    """
    raise NotImplementedError("실제 모델/API 연동이 아직 준비되지 않았습니다.")


# ---------------------------------------------------------------------------
# 이하 mock 전용 - 실제 연동 후 통째로 삭제 가능
# ---------------------------------------------------------------------------
# 이용자 Score 정규화 기준 상한 (mock 전용). 실제로는 ML팀 정규화 함수가 담당.
_VISITOR_CAP = 14000

# 특정 날짜 강수확률 고정 (월, 일) → %  — 데모용 더미 오버라이드
_RAIN_OVERRIDES = {
    (8, 12): 0,
    (8, 14): 100,
}


def _mock_weather(date: dt.date) -> dict:
    """계절 기반으로 그럴듯한 기상값을 결정적으로 생성 (같은 날짜 = 같은 값)."""
    rng = np.random.default_rng(date.toordinal())
    m = date.month
    season = "봄" if m in (3, 4, 5) else "여름" if m in (6, 7, 8) else "가을" if m in (9, 10, 11) else "겨울"
    base_temp = {"봄": 17, "여름": 28, "가을": 16, "겨울": 2}[season]
    temp = round(base_temp + rng.normal(0, 3), 1)
    base_rain = {"봄": 25, "여름": 45, "가을": 20, "겨울": 20}[season]
    rain_prob = float(np.clip(base_rain + rng.normal(0, 20), 0, 100))
    rain_prob = round(rain_prob / 10) * 10  # 기상청 표기처럼 10% 단위
    if (date.month, date.day) in _RAIN_OVERRIDES:
        rain_prob = float(_RAIN_OVERRIDES[(date.month, date.day)])
    humidity = int(np.clip(55 + rng.normal(0, 12), 30, 95))
    return {"temp": temp, "rain_prob": rain_prob, "humidity": humidity, "season": season}


def _mock_visitors(date: dt.date, w: dict) -> int:
    """회의에서 확인한 상관관계(주말 > 기온 > 계절 > 강수)를 반영한 임시 규칙."""
    base = 2200.0

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


def _mock_crowd_level(pred: int) -> str:
    """ML팀 Classification 모델 자리를 대신하는 임시 구간 분류."""
    if pred >= 5000:
        return "혼잡"
    if pred >= 2000:
        return "보통"
    return "여유"


def _mock_visitor_score(pred: int) -> int:
    """이용자 Score: 높을수록 한산."""
    ratio = min(pred, _VISITOR_CAP) / _VISITOR_CAP
    return int(round((1 - ratio) * 100))


def _mock_weather_score(w: dict) -> int:
    """날씨 Score: 높을수록 좋은 날씨."""
    temp_score = max(0, 100 - abs(w["temp"] - 21) * 5)  # 21도 최적
    rain_penalty = w["rain_prob"] * 0.6
    humid_penalty = max(0, w["humidity"] - 70) * 1.2
    return int(np.clip(temp_score - rain_penalty - humid_penalty, 0, 100))


def _mock_forecast(start_date: dt.date, n_days: int) -> list[dict]:
    rows = []
    for i in range(n_days):
        d = start_date + dt.timedelta(days=i)
        w = _mock_weather(d)
        pred = _mock_visitors(d, w)
        rows.append({
            "date": d,
            "weekday": WEEKDAY_KR[d.weekday()],
            "is_weekend": d.weekday() >= 5,
            "is_holiday": d in HOLIDAYS,
            "holiday_name": HOLIDAYS.get(d, ""),
            "temp": w["temp"],
            "rain_prob": w["rain_prob"],
            "humidity": w["humidity"],
            "pred_visitors": pred,
            "crowd_level": _mock_crowd_level(pred),
            "visitor_score": _mock_visitor_score(pred),
            "weather_score": _mock_weather_score(w),
        })
    return rows
