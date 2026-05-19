from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np


DEFAULT_OCR_BASE_URL = "http://localhost:1234/v1"
DEFAULT_OCR_MODEL = "local-vlm"
DEFAULT_OCR_IMAGE_PREPROCESS = "none"
OCR_IMAGE_PREPROCESS_MODES = {"none", "grayscale", "clahe", "vlm-board"}

PRICE_TAG_OCR_FIELDS = (
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
)

PRICE_TAG_OCR_PROMPT = """
Перед тобой фотография магазинного ценника Лента.

Верни только JSON без Markdown. Все значения - строки. Если поле не видно, оставь "".
Не придумывай данные.
Не пытайся декодировать QR-код по изображению: QR разбирает отдельный модуль.

JSON schema:
{
  "product_name": "",
  "price_default": "",
  "price_card": "",
  "price_discount": "",
  "barcode": "",
  "discount_amount": "",
  "id_sku": "",
  "print_datetime": "",
  "code": "",
  "additional_info": "",
  "color": "",
  "special_symbols": "",
  "wholesale_level_1_count": "",
  "wholesale_level_1_price": "",
  "wholesale_level_2_count": "",
  "wholesale_level_2_price": ""
}

Подсказки:
- product_name: полное видимое название товара без цены; если название видно частично, верни читаемую видимую часть, но не придумывай скрытые слова;
- price_default: цена без карты; визуально обычно находится под QR-кодом и над ценой по карте, рядом с подписью "Без карты";
- price_card: цена рядом с подписью "С картой", "По карте" или большая основная цена на красном/оранжевом блоке;
- price_discount: отдельная цена по акции, только если она явно отличается от цены по карте;
- barcode: длинные цифры под линейным штрихкодом, без пробелов; это не QR-код;
- discount_amount: размер скидки в круге/плашке, проценты или сумма в рублях;
- id_sku: маленький артикул/код под процентной плашкой или в нижней части ценника; часто две группы цифр; это не barcode и не price_default;
- print_datetime: полная дата и время печати ценника внизу в формате день.месяц.год часы:минуты;
- code: отдельный код выкладки/зоны, только если он явно виден и это не цена без карты и не id_sku; если не уверен, оставь "";
- additional_info: доп. текст на ценнике, включая номер на весах, вид упаковки, тип напитка или шелфтокер; не включай сюда подписи цен "Без карты", "С картой", "По карте" и символ рубля; если нет - "";
- color: основной цвет ценовой зоны на английском, например red, orange, white, yellow, green;
- special_symbols: маленький символ типа выкладки в круге или рядом с нижней зоной; верни только "Ш", "Л" или "К"; если нет - "";
- wholesale_level_1_count: количество из явной надписи оптового порога "от N шт"; верни только N;
- wholesale_level_1_price: цена для первого оптового порога, если рядом явно указано "от N шт";
- wholesale_level_2_count: второй оптовый порог, если явно виден;
- wholesale_level_2_price: цена для второго оптового порога, если явно видна.

Цены возвращай полностью с копейками: если рубли и копейки напечатаны отдельными группами, объедини их в одну цену.
Если изображение составлено из нескольких частей, используй полный ценник и увеличенные нижние фрагменты как один и тот же ценник.
""".strip()

IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>\s*", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class OcrConfig:
    base_url: str = DEFAULT_OCR_BASE_URL
    api_key: str = "not-needed"
    model: str = DEFAULT_OCR_MODEL
    max_tokens: int = 900
    temperature: float = 0.0
    timeout: float = 180.0
    retries: int = 1
    response_format: bool = False
    disable_reasoning: bool = True
    image_preprocess: str = DEFAULT_OCR_IMAGE_PREPROCESS


@dataclass(frozen=True)
class OcrResult:
    image_name: str
    image_path: str
    raw_response: str
    parsed_response: dict[str, str]
    parse_error: str | None = None


class OcrError(RuntimeError):
    """Base OCR error."""


class OcrDownstreamUnavailableError(OcrError):
    """Raised when the OCR service cannot be reached."""


class OcrDownstreamProtocolError(OcrError):
    """Raised when the OCR service returns an invalid payload."""


class OpenAICompatibleOcrClient:
    def __init__(self, config: OcrConfig) -> None:
        self._config = config

    @property
    def config(self) -> OcrConfig:
        return self._config

    def predict(self, prompt: str, image_data_url: str) -> str:
        messages: list[dict[str, Any]] = []
        if self._config.disable_reasoning:
            messages.append(
                {
                    "role": "system",
                    "content": "/no_think\nТы OCR-экстрактор. Не рассуждай. Верни только финальный JSON.",
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        )

        payload: dict[str, Any] = {
            "model": self._config.model,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "messages": messages,
        }
        if self._config.response_format:
            payload["response_format"] = {"type": "json_object"}
        if self._config.disable_reasoning:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        response_payload = self._request_json(
            "/chat/completions",
            payload=payload,
            timeout=self._config.timeout,
            retries=self._config.retries,
        )
        try:
            return str(response_payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:
            raise OcrDownstreamProtocolError(
                f"downstream returned an invalid completion payload: {error}"
            ) from error

    def check_health(self, *, validate_model: bool = True) -> list[str]:
        response_payload = self._request_json(
            "/models",
            payload=None,
            timeout=min(self._config.timeout, 15.0),
            retries=0,
        )
        models = response_payload.get("data")
        if not isinstance(models, list):
            raise OcrDownstreamProtocolError("downstream returned an invalid models payload")

        model_ids: list[str] = []
        for item in models:
            if isinstance(item, dict) and item.get("id") is not None:
                model_ids.append(str(item["id"]))

        if validate_model and self._config.model and model_ids and self._config.model not in model_ids:
            raise OcrDownstreamUnavailableError(
                f"configured model '{self._config.model}' is not available downstream; "
                f"available models: {', '.join(model_ids)}"
            )
        return model_ids

    def _request_json(
        self,
        path: str,
        *,
        payload: dict[str, Any] | None,
        timeout: float,
        retries: int,
    ) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = Request(
            f"{self._config.base_url.rstrip('/')}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST" if payload is not None else "GET",
        )

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                with urlopen(request, timeout=timeout) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(response_payload, dict):
                    raise OcrDownstreamProtocolError("downstream returned a non-object JSON payload")
                return response_payload
            except OcrDownstreamProtocolError:
                raise
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
                if attempt < retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                break

        raise OcrDownstreamUnavailableError(
            f"downstream request to '{path}' failed after {retries + 1} attempts: {last_error}"
        )


def build_ocr_config_from_env() -> OcrConfig:
    return OcrConfig(
        base_url=_first_env("PRICE_TAG_OCR_BASE_URL", "LLAMA_CPP_BASE_URL", default=DEFAULT_OCR_BASE_URL),
        api_key=_first_env("PRICE_TAG_OCR_API_KEY", "LLAMA_CPP_API_KEY", default="not-needed"),
        model=_first_env("PRICE_TAG_OCR_MODEL", "LLAMA_CPP_MODEL", default=DEFAULT_OCR_MODEL),
        max_tokens=int(_first_env("PRICE_TAG_OCR_MAX_TOKENS", "LLAMA_CPP_MAX_TOKENS", default="900")),
        temperature=float(_first_env("PRICE_TAG_OCR_TEMPERATURE", "LLAMA_CPP_TEMPERATURE", default="0.0")),
        timeout=float(_first_env("PRICE_TAG_OCR_TIMEOUT", "LLAMA_CPP_TIMEOUT", default="180.0")),
        retries=int(_first_env("PRICE_TAG_OCR_RETRIES", "LLAMA_CPP_RETRIES", default="1")),
        response_format=_env_bool("PRICE_TAG_OCR_RESPONSE_FORMAT", _env_bool("LLAMA_CPP_RESPONSE_FORMAT", False)),
        disable_reasoning=_env_bool("PRICE_TAG_OCR_DISABLE_REASONING", True),
        image_preprocess=_first_env("PRICE_TAG_OCR_IMAGE_PREPROCESS", "LLAMA_CPP_IMAGE_PREPROCESS", default=DEFAULT_OCR_IMAGE_PREPROCESS),
    )


def image_bytes_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def image_path_to_ocr_bytes(crop_path: Path, *, mode: str) -> tuple[bytes, str]:
    normalized_mode = mode.strip().casefold()
    if normalized_mode not in OCR_IMAGE_PREPROCESS_MODES:
        raise ValueError(
            f"unknown OCR image preprocess mode '{mode}'; "
            f"expected one of: {', '.join(sorted(OCR_IMAGE_PREPROCESS_MODES))}"
        )
    if normalized_mode == "none":
        return crop_path.read_bytes(), IMAGE_MIME_TYPES.get(crop_path.suffix.casefold(), "application/octet-stream")

    image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return crop_path.read_bytes(), IMAGE_MIME_TYPES.get(crop_path.suffix.casefold(), "application/octet-stream")

    if normalized_mode == "grayscale":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif normalized_mode == "clahe":
        processed = _enhance_text_contrast(image)
    else:
        processed = _build_vlm_board(image)

    ok, encoded = cv2.imencode(".jpg", processed, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return crop_path.read_bytes(), IMAGE_MIME_TYPES.get(crop_path.suffix.casefold(), "application/octet-stream")
    return encoded.tobytes(), "image/jpeg"


def _build_vlm_board(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h < 4 or w < 4:
        return image

    enhanced = _enhance_text_contrast(image)
    target_width = 960
    full = _resize_to_width(image, target_width)
    bottom = _resize_to_width(enhanced[int(h * 0.50) :, :], target_width)
    lower_left = _resize_to_width(enhanced[int(h * 0.55) :, : max(1, int(w * 0.62))], target_width)
    lower_right = _resize_to_width(enhanced[int(h * 0.55) :, max(0, int(w * 0.38)) :], target_width)
    return _stack_vertical([full, bottom, lower_left, lower_right], gap=18)


def _enhance_text_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)


def _resize_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image
    scale = target_width / float(w)
    target_height = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def _stack_vertical(images: list[np.ndarray], *, gap: int) -> np.ndarray:
    non_empty = [image for image in images if image.size > 0]
    if not non_empty:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    width = max(image.shape[1] for image in non_empty)
    height = sum(image.shape[0] for image in non_empty) + gap * (len(non_empty) - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    y_offset = 0
    for image in non_empty:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        x_offset = (width - image.shape[1]) // 2
        canvas[y_offset : y_offset + image.shape[0], x_offset : x_offset + image.shape[1]] = image
        y_offset += image.shape[0] + gap
    return canvas


def parse_model_json(raw_response: str) -> tuple[dict[str, Any], str | None]:
    text = THINK_BLOCK_RE.sub("", raw_response).strip()
    fence_match = FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        first_error: json.JSONDecodeError | None = None
        for start, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError as error:
                if first_error is None:
                    first_error = error
                continue
            if isinstance(parsed, dict):
                return parsed, None
        if "{" not in text:
            return {}, "response does not contain a JSON object"
        if first_error is not None:
            return {}, f"invalid JSON: {first_error}"
        return {}, "response does not contain a JSON object"

    if not isinstance(parsed, dict):
        return {}, "parsed JSON is not an object"
    return parsed, None


def normalize_price_tag_payload(payload: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key in PRICE_TAG_OCR_FIELDS:
        value = payload.get(key, "")
        normalized[key] = "" if value is None else str(value).strip()
    return normalized


def empty_price_tag_payload() -> dict[str, str]:
    return {key: "" for key in PRICE_TAG_OCR_FIELDS}


def extract_price_tag_bytes(
    image_bytes: bytes,
    *,
    client: OpenAICompatibleOcrClient,
    image_name: str = "upload",
    content_type: str = "application/octet-stream",
    image_path_label: str = "",
) -> OcrResult:
    raw_response = ""
    parsed_response: dict[str, Any] = {}
    parse_error: str | None = None
    for attempt in range(client.config.retries + 1):
        raw_response = client.predict(
            PRICE_TAG_OCR_PROMPT,
            image_bytes_to_data_url(image_bytes, content_type),
        )
        parsed_response, parse_error = parse_model_json(raw_response)
        if parse_error is None:
            break
        if attempt < client.config.retries:
            time.sleep(min(2**attempt, 4))
            continue
    if parse_error is not None:
        raise OcrDownstreamProtocolError(f"downstream returned invalid OCR JSON: {parse_error}")

    return OcrResult(
        image_name=image_name or "upload",
        image_path=image_path_label,
        raw_response=raw_response,
        parsed_response=normalize_price_tag_payload(parsed_response),
        parse_error=parse_error,
    )


def extract_price_tag_text(
    crop_path: Path,
    *,
    client: OpenAICompatibleOcrClient,
    image_name: str | None = None,
    image_path_label: str | None = None,
) -> OcrResult:
    image_bytes, mime_type = image_path_to_ocr_bytes(crop_path, mode=client.config.image_preprocess)
    return extract_price_tag_bytes(
        image_bytes,
        client=client,
        image_name=image_name or crop_path.name,
        content_type=mime_type,
        image_path_label=image_path_label or str(crop_path),
    )


def _first_env(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}
