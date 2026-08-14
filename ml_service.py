from __future__ import annotations

import copy
import json
import logging
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

logger = logging.getLogger(__name__)

ML_TOOL_NAME = "predict_visitor_and_classify_congestion"
ML_FUNCTION_TOOLS = [
    {
        "type": "function",
        "name": ML_TOOL_NAME,
        "description": (
            "검증된 오월드 입력으로 CatBoost 방문객 수 회귀 예측과 "
            "혼잡도 0·1·2 분류를 함께 실행합니다."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


class MLToolOrchestrationError(RuntimeError):
    """OpenAI 툴 선택/응답 형식 단계에서 발생한 오류입니다."""


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


def get_ml_function_tools() -> list[dict[str, Any]]:
    """OpenAI 요청에 넣을 ML function tool 정의를 반환합니다."""
    return copy.deepcopy(ML_FUNCTION_TOOLS)


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if arguments in (None, ""):
        return {}

    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise MLToolOrchestrationError(
                f"ML 커스텀 툴 인자가 올바른 JSON이 아닙니다: {exc}"
            ) from exc
    elif isinstance(arguments, dict):
        parsed = arguments
    else:
        raise MLToolOrchestrationError(
            "ML 커스텀 툴 인자는 JSON 객체여야 합니다."
        )

    if not isinstance(parsed, dict):
        raise MLToolOrchestrationError(
            "ML 커스텀 툴 인자는 JSON 객체여야 합니다."
        )

    # 17개 Feature는 파이프라인이 이미 검증한 DataFrame을 그대로 사용
    if parsed:
        raise MLToolOrchestrationError(
            "ML 커스텀 툴은 외부 인자를 받지 않습니다."
        )

    return parsed

def dispatch_ml_tool(
    tool_name: str,
    arguments: Any,
    model_input: pd.DataFrame,
) -> dict[str, int]:
    _parse_tool_arguments(arguments)

    if tool_name != ML_TOOL_NAME:
        raise MLToolOrchestrationError(
            f"등록되지 않은 ML 커스텀 툴입니다: {tool_name}"
        )

    return predict_all(model_input.copy())


def _item_value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def _get_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise MLToolOrchestrationError(
            "openai 패키지를 불러오지 못했습니다."
        ) from exc

    try:
        timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20"))
        max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "1"))
        return OpenAI(timeout=timeout, max_retries=max_retries)
    except Exception as exc:
        raise MLToolOrchestrationError(
            f"OpenAI 클라이언트를 만들지 못했습니다: {exc}"
        ) from exc


def _request_ml_tool_call(
    batch_size: int,
    client: Any = None,
    model: str | None = None,
) -> tuple[Any, Any, Any, str, list[dict[str, str]]]:
    selected_model = model or os.getenv("OPENAI_MODEL", "gpt-5.5")

    if client is None and not os.getenv("OPENAI_API_KEY"):
        raise MLToolOrchestrationError(
            "OPENAI_API_KEY가 없어 ML 커스텀 툴 호출을 시작할 수 없습니다."
        )

    runtime_client = client or _get_openai_client()
    request_input = [
        {
            "role": "user",
            "content": (
                f"애플리케이션에서 검증한 {batch_size}개 날짜의 입력이 "
                "준비되었습니다. 등록된 ML 도구를 실행하세요."
            ),
        }
    ]

    try:
        response = runtime_client.responses.create(
            model=selected_model,
            instructions=(
                "반드시 등록된 오월드 ML 도구를 정확히 한 번 호출하세요. "
                "예측값을 직접 만들지 마세요."
            ),
            input=request_input,
            tools=get_ml_function_tools(),
            tool_choice={"type": "function", "name": ML_TOOL_NAME},
            parallel_tool_calls=False,
        )
    except Exception as exc:
        raise MLToolOrchestrationError(
            f"OpenAI ML 커스텀 툴 요청에 실패했습니다: {exc}"
        ) from exc

    response_output = _item_value(response, "output", []) or []
    tool_calls = [
        item
        for item in response_output
        if _item_value(item, "type") == "function_call"
        and _item_value(item, "name") == ML_TOOL_NAME
    ]

    if len(tool_calls) != 1:
        raise MLToolOrchestrationError(
            "OpenAI가 등록된 ML 커스텀 툴을 정확히 한 번 호출하지 않았습니다."
        )

    tool_call = tool_calls[0]
    _parse_tool_arguments(_item_value(tool_call, "arguments", "{}"))

    if not _item_value(tool_call, "call_id"):
        raise MLToolOrchestrationError(
            "ML 커스텀 툴 응답에 call_id가 없습니다."
        )

    return runtime_client, response, tool_call, selected_model, request_input


def _complete_ml_tool_call(
    client: Any,
    response: Any,
    tool_call: Any,
    result: Any,
    model: str,
    request_input: list[dict[str, str]],
) -> None:
    """툴 결과를 Responses API에 돌려주되 UI 수치는 로컬 결과를 사용합니다."""
    try:
        input_items: list[Any] = list(request_input)
        input_items.extend(list(_item_value(response, "output", []) or []))
        input_items.append(
            {
                "type": "function_call_output",
                "call_id": _item_value(tool_call, "call_id"),
                "output": json.dumps(result, ensure_ascii=False),
            }
        )

        client.responses.create(
            model=model,
            instructions="도구 실행 결과를 확인했다는 짧은 문장만 반환하세요.",
            input=input_items,
            tools=get_ml_function_tools(),
            tool_choice="none",
            parallel_tool_calls=False,
        )
    except Exception as exc:
        logger.warning("ML 커스텀 툴 결과 확인 요청 실패: %s", exc)


def predict_all_with_custom_tool(
    model_input: pd.DataFrame,
    client: Any = None,
    model: str | None = None,
) -> dict[str, int]:
    prepared_input = _prepare_input(model_input)

    if not _env_flag("OWORLD_USE_ML_CUSTOM_TOOL", True):
        return predict_all(prepared_input)

    try:
        runtime_client, response, tool_call, selected_model, request_input = (
            _request_ml_tool_call(
                batch_size=1,
                client=client,
                model=model,
            )
        )
    except MLToolOrchestrationError as exc:
        if not _env_flag("OWORLD_ML_TOOL_FALLBACK", True):
            raise

        logger.warning(
            "OpenAI ML 커스텀 툴을 사용할 수 없어 로컬 모델로 실행합니다: %s",
            exc,
        )
        return predict_all(prepared_input)

    result = dispatch_ml_tool(
        tool_name=_item_value(tool_call, "name"),
        arguments=_item_value(tool_call, "arguments", "{}"),
        model_input=prepared_input,
    )

    _complete_ml_tool_call(
        client=runtime_client,
        response=response,
        tool_call=tool_call,
        result=result,
        model=selected_model,
        request_input=request_input,
    )
    return result

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
