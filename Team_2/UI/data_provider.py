"""
파이프라인 흐름
oworld_openai_pipeline.py(API팀) => [[data_provider.py]](백앤드 최종) => UI(프론트앤드) 출력

데이터 흐름
oworld_openai_pipeline.py → data_provider.py → app.py

data : 날씨 및 모델 예측 결과
meta : 날씨 설명, 휴일, 예측 월 등의 부가정보

fetch_forecast() / fatch_notice 호출 => app.py


파이프라인의 예측 결과를 UI에서 사용할 수 있는 형태로 변환

주요 변수명 변환
temperature → temp
predicted_visitors → pred_visitors
congestion → crowd_level  => 출력값도 변환 (0~2 => 여유~혼잡)
"""

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
import datetime as dt
from oworld_openai_pipeline import build_ui_payload

# UI 호환용 변수 => 더미 데이터가 삭제되어 현재는 실제 데이터만 사용
USE_MOCK = False

# pipeline 통해 받은 요일 변수
WEEKDAY_KEYS = ["mon", "tue", "wed", "thur", "fri", "sat", "sun"]

# UI 출력용 한글 요일
WEEKDAY_KR = ["월", "화", "수", "목", "금", "토", "일"]

# congestion 정수 → 표시 라벨(0~700 / 700~3000 / 3000명 초과 분류 모델) <-> 회귀 모델과 별개
# UI에서는 분류 모델이 예측한 혼잡도를 그대로 표시
CONGESTION_LABELS = {0: "여유", 1: "보통", 2: "혼잡"}
CROWD_LEVEL_UNKNOWN = "알 수 없음"

# 혼잡도 값만 추출 (여유/보통/혼잡)
CROWD_LEVELS = list(CONGESTION_LABELS.values())

# UI 표시용 공휴일 이름
# 공휴일 여부는 파이프라인의 holiday 값을 사용
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

# 공지 상세 URL을 가져오지 못했을 때 사용할 게시판 목록 주소
NOTICE_BOARD_URL = (
    "https://www.oworld.kr/newkfsweb/kfi/kfs/linkage/selectDccoLinkageList.do?mn=KFS_34_01_09_01"
)

# ---------------------------------------------------------------------------
# 공개 / 보조 함수
# ---------------------------------------------------------------------------

# app.py의 요청을 받아 전체 데이터 처리를 실행 (본체)
# 오늘부터 10일 간의 예측 데이터를 반환 / start_date(UI) => 날짜 캐시 갱신
def fetch_forecast(start_date: dt.date, n_days: int = 10) -> list[dict]:
    return normalize_payload(_real_payload(start_date, n_days))

# 부품 1
# oworld_openai_pipeline -> fetch_forecast로 데이터 전달
def _real_payload(start_date: dt.date, n_days: int) -> dict:
    # 공지 호출은 제외하고 날씨·방문객·혼잡도 예측만 요청
    return build_ui_payload(days=n_days, include_notice=False)

# 부품 2
# 파이프라인 days 날짜별 데이터를 추출 -> _to_row()로 변환
def normalize_payload(payload: dict) -> list[dict]:
    days = payload.get("days", [])
    rows = []

    for day in days:
        row = _to_row(day)

        if row is not None:
            rows.append(row)

    rows.sort(key=lambda row: row["date"])
    return rows


# 이상치 안내 문구
def outlier_note(date: dt.date) -> str:
    return OUTLIER_DAYS.get((date.month, date.day), "")


# 혼잡도 분류 모델 정수 라벨 => UI 출력
def crowd_label(congestion) -> str:
    try:
        return CONGESTION_LABELS.get(int(congestion), CROWD_LEVEL_UNKNOWN)
    except (TypeError, ValueError):
        return CROWD_LEVEL_UNKNOWN


# notice.py 데이터 => UI 출력
def fetch_notices(
    _today: dt.date | None = None,
    include_ended: bool = False,
) -> list[dict]:
    """실제 오월드 공지 AI 결과를 UI 형식으로 변환한다."""
    from oworld_openai_pipeline import call_notice

    result = call_notice()
    raw_notices = result.get("notices_list", [])

    rows = []

    for notice in raw_notices:
        status_tag = str(notice.get("status_tag") or "상시")

        if status_tag in ("진행 중", "D-Day"):
            state = "ongoing"
        elif status_tag.startswith("D-"):
            state = "upcoming"
        elif status_tag == "종료":
            continue
        else:
            state = "always"

        rows.append(
            {
                "category": str(notice.get("category") or "안내"),
                "title": str(notice.get("title") or "제목 없음"),
                "url": str(notice.get("url") or NOTICE_BOARD_URL),
                "period_text": str(notice.get("date_string") or "상시"),
                "state": state,
                "state_label": status_tag,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# payload → 날짜/숫자/요일 형식 UI로 변환
# ---------------------------------------------------------------------------

# 날짜 변환 => date(2026, 8, 14)
def _parse_date(value) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


# 날짜 변환 => True -> 요일 출력
def _weekday_kr(onehot, date: dt.date) -> str:
    if isinstance(onehot, dict):
        picked = [i for i, key in enumerate(WEEKDAY_KEYS) if onehot.get(key)]
        if len(picked) == 1 and picked[0] == date.weekday():
            return WEEKDAY_KR[picked[0]]
    return WEEKDAY_KR[date.weekday()]

# 실수 반환 (강수, 기온)
def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

# 정수 반환 (점수, 혼잡도)
def _as_int(value, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


# 위 모든 함수 => _to_row() 통합
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
        "is_dayoff": bool(meta["model_dayoff"]),
        "is_next_dayoff": bool(meta["model_nextdayoff"]),
        "month": str(meta["model_month"]),
    }


# app.py DataFrame의 컬럼 이름과 순서를 고정
# 예측 데이터가 없어도 동일한 컬럼 구조 유지
# ROW_COLUMNS = list(_to_row({"date": "2000-01-01"}))
ROW_COLUMNS = [
    "date",
    "status",
    "ok",
    "weekday",
    "is_weekend",
    "is_holiday",
    "holiday_name",
    "temp",
    "rain_prob",
    "humidity",
    "pred_visitors",
    "congestion",
    "crowd_level",
    "visitor_score",
    "weather_score",
    "weather_summary",
    "weather_icon",
    "weather_source",
    "forecast_time",
    "is_dayoff",
    "is_next_dayoff",
    "month",
]
