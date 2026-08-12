"""CatBoost 모델과 점수 함수 연결 파일.

프로젝트 파일 구조
- visitor_estimation_model.cbm
- congestion_model.cbm
- weather_score.py
- visitor_score.py

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from dotenv import load_dotenv

from visitor_score import score_single
from weather_score import weather_score

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

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

VISITOR_MODEL_PATH = Path(
    os.getenv(
        "VISITOR_MODEL_PATH",
        str(BASE_DIR / "visitor_estimation_model.cbm"),
    )
)
CONGESTION_MODEL_PATH = Path(
    os.getenv(
        "CONGESTION_MODEL_PATH",
        str(BASE_DIR / "congestion_model.cbm"),
    )
)

_visitor_model: CatBoostRegressor | None = None
_congestion_model: CatBoostClassifier | None = None


def _validate_feature_names(model: Any, model_name: str) -> None:
    names = list(getattr(model, "feature_names_", []) or [])

    # DataFrame으로 학습한 CatBoost 모델은 Feature 이름이 저장됩니다.
    if names and names != MODEL_FEATURES:
        raise ValueError(
            f"{model_name} Feature가 현재 입력과 다릅니다.\n"
            f"모델: {names}\n"
            f"입력: {MODEL_FEATURES}"
        )


def _get_visitor_model() -> CatBoostRegressor:
    global _visitor_model

    if _visitor_model is None:
        if not VISITOR_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"방문객 예측 모델을 찾을 수 없습니다: {VISITOR_MODEL_PATH}"
            )

        model = CatBoostRegressor()
        model.load_model(str(VISITOR_MODEL_PATH), format="cbm")
        _validate_feature_names(model, "방문객 예측 모델")
        _visitor_model = model

    return _visitor_model


def _get_congestion_model() -> CatBoostClassifier:
    global _congestion_model

    if _congestion_model is None:
        if not CONGESTION_MODEL_PATH.exists():
            raise FileNotFoundError(
                "혼잡도 0/1/2를 반환할 분류 모델이 필요합니다: "
                f"{CONGESTION_MODEL_PATH}"
            )

        model = CatBoostClassifier()
        model.load_model(str(CONGESTION_MODEL_PATH), format="cbm")
        _validate_feature_names(model, "혼잡도 모델")
        _congestion_model = model

    return _congestion_model


def _prepare_input(model_input: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(model_input, pd.DataFrame):
        raise TypeError("model_input은 pandas DataFrame이어야 합니다.")
    if len(model_input) != 1:
        raise ValueError("한 번에 날짜 한 개의 1행 데이터만 전달하세요.")

    missing = [name for name in MODEL_FEATURES if name not in model_input.columns]
    extra = [name for name in model_input.columns if name not in MODEL_FEATURES]

    if missing or extra:
        raise ValueError(
            f"입력 Feature가 다릅니다. missing={missing}, extra={extra}"
        )

    return model_input[MODEL_FEATURES].astype(float)

#방문객 수와 업로드된 두 점수 함수를 연결
def predict_visitor_and_scores(model_input: pd.DataFrame) -> dict[str, int]:
    x = _prepare_input(model_input)
    row = x.iloc[0]

    raw_prediction = _get_visitor_model().predict(x)
    predicted_visitors = max(
        0,
        round(float(np.asarray(raw_prediction).reshape(-1)[0])),
    )

    calculated_weather_score = weather_score(
        avg_temp_c=float(row["temperature"]),
        rain_chance_pct=float(row["rain"]),
        humidity_pct=float(row["humidity"]),
    )
    calculated_user_score = score_single(predicted_visitors)

    return {
        "predicted_visitors": int(predicted_visitors),
        "user_score": int(calculated_user_score),
        "weather_score": int(calculated_weather_score),
    }

#파이프라인이 요구하는 방문객·혼잡도·이용자점수·날씨점수를 반환
def predict_all(model_input: pd.DataFrame) -> dict[str, int]:
    x = _prepare_input(model_input)
    result = predict_visitor_and_scores(x)

    raw_congestion = _get_congestion_model().predict(x)
    congestion = int(np.asarray(raw_congestion).reshape(-1)[0])

    if congestion not in (0, 1, 2):
        raise ValueError(
            f"혼잡도 모델 결과는 0, 1, 2 중 하나여야 합니다: {congestion}"
        )

    return {
        **result,
        "congestion": congestion,
    }
