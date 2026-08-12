import json

from oworld_openai_pipeline import (
    build_ui_payload,
    search_weather
)


def dummy_predictor(model_input):
    # 입력 Feature에 따라 값이 조금 다르게 나오도록 임시 처리
    row = model_input.iloc[0]

    visitors = 2000

    if row["weekend"] == 1:
        visitors = visitors + 2000

    if row["holiday"] == 1:
        visitors = visitors + 1500

    if row["rain"] >= 50:
        visitors = visitors - 500

    # 임시 혼잡도
    if visitors < 2500:
        congestion = 0
    elif visitors < 5000:
        congestion = 1
    else:
        congestion = 2

    return {
        "predicted_visitors": visitors,
        "congestion": congestion,
        "user_score": 70,
        "weather_score": 80
    }


# 이전 검색 결과 캐시 제거
search_weather.cache_clear()


# 실제 Web Search 실행, ML 부분만 임시 함수 사용
payload = build_ui_payload(
    days=3,
    predictor=dummy_predictor
)


print(
    json.dumps(
        payload,
        ensure_ascii=False,
        indent=2
    )
)