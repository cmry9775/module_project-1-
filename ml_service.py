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

VISITOR_MODEL_PATH = Path(
    os.getenv(
        "VISITOR_MODEL_PATH",
        str(BASE_DIR / "visitor_estimation_model.cbm"),
    )
)

CONGESTION_MODEL_PATH = Path(
    os.getenv(
        "CONGESTION_MODEL_PATH",
        str(BASE_DIR / "catboost_classifier.cbm"),
    )
)

_visitor_model: CatBoostRegressor | None = None
_congestion_model: CatBoostClassifier | None = None


def _ensure_model_file(path: Path, model_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{model_name} 파일을 찾을 수 없습니다: {path}")

    if path.stat().st_size < 100:
        raise ValueError(
            f"{model_name} 파일이 비어 있거나 손상되었습니다: "
            f"{path} ({path.stat().st_size} bytes)"
        )


def _validate_feature_names(model: Any, model_name: str) -> None:
    names = list(getattr(model, "feature_names_", []) or [])

    if names and names != MODEL_FEATURES:
        raise ValueError(
            f"{model_name} Feature가 현재 입력과 다릅니다.\n"
            f"모델: {names}\n"
            f"입력: {MODEL_FEATURES}"
        )


def _get_visitor_model() -> CatBoostRegressor:
    global _visitor_model

    if _visitor_model is None:
        _ensure_model_file(VISITOR_MODEL_PATH, "방문객 예측 모델")

        model = CatBoostRegressor()
        model.load_model(str(VISITOR_MODEL_PATH), format="cbm")
        _validate_feature_names(model, "방문객 예측 모델")
        _visitor_model = model

    return _visitor_model


def _get_congestion_model() -> CatBoostClassifier:
    global _congestion_model

    if _congestion_model is None:
        _ensure_model_file(CONGESTION_MODEL_PATH, "혼잡도 분류 모델")

        model = CatBoostClassifier()
        model.load_model(str(CONGESTION_MODEL_PATH), format="cbm")
        _validate_feature_names(model, "혼잡도 분류 모델")

        classes = [int(value) for value in np.asarray(model.classes_).reshape(-1)]
        if classes != [0, 1, 2]:
            raise ValueError(
                "혼잡도 모델 클래스는 [0, 1, 2]여야 합니다. "
                f"현재 클래스: {classes}"
            )

        _congestion_model = model

    return _congestion_model


def _prepare_input(model_input: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(model_input, pd.DataFrame):
        raise TypeError("model_input은 pandas DataFrame이어야 합니다.")

    if len(model_input) != 1:
        raise ValueError("model_input은 날짜 한 개의 1행 데이터여야 합니다.")

    missing = [
        name for name in MODEL_FEATURES
        if name not in model_input.columns
    ]
    extra = [
        name for name in model_input.columns
        if name not in MODEL_FEATURES
    ]

    if missing or extra:
        raise ValueError(
            "model_input Feature가 다릅니다. "
            f"missing={missing}, extra={extra}"
        )

    return model_input[MODEL_FEATURES].astype(float)

#방문객 수와 이용자·날씨 점수를 반환
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

#공통 입력으로 두 모델을 실행해 최종 네 값을 반환
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

# 모델 파일 상태와 패치 모델의 공통 Feature를 반환
def get_model_status() -> dict[str, Any]:
    return {
        "model_features": MODEL_FEATURES,
        "visitor_model": {
            "path": str(VISITOR_MODEL_PATH),
            "exists": VISITOR_MODEL_PATH.exists(),
            "size": (
                VISITOR_MODEL_PATH.stat().st_size
                if VISITOR_MODEL_PATH.exists()
                else None
            ),
        },
        "congestion_model": {
            "path": str(CONGESTION_MODEL_PATH),
            "exists": CONGESTION_MODEL_PATH.exists(),
            "size": (
                CONGESTION_MODEL_PATH.stat().st_size
                if CONGESTION_MODEL_PATH.exists()
                else None
            ),
        },
    }
