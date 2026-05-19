from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NO_VALUE = "нет"

STRUCTURED_PRICE_TAG_FIELDS = (
    "filename",
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
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_coun",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
)

OPTIONAL_NO_VALUE_FIELDS = {
    "price_discount",
    "additional_info",
    "special_symbols",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_coun",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
}


@dataclass(frozen=True)
class StructuredPriceTagCsvRow:
    values: dict[str, str]

    @classmethod
    def from_payloads(
        cls,
        *,
        video_path: Path,
        track_row: dict[str, Any],
        frame_timestamp: str,
        coordinates: tuple[int | float | None, int | float | None, int | float | None, int | float | None],
        ocr_payload: dict[str, Any],
        qr_payload: dict[str, Any],
    ) -> StructuredPriceTagCsvRow:
        x_min, y_min, x_max, y_max = coordinates
        barcode = ocr_payload.get("barcode", "")
        code_value = ocr_payload.get("code", "")
        price_default = _first_non_empty(
            ocr_payload.get("price_default", ""),
            _price_from_misplaced_code(code_value),
        )
        price_card = ocr_payload.get("price_card", "")
        raw = {key: "" for key in STRUCTURED_PRICE_TAG_FIELDS}
        raw.update(
            {
                "filename": video_path.name,
                "product_name": ocr_payload.get("product_name", ""),
                "price_default": price_default,
                "price_card": price_card,
                "price_discount": ocr_payload.get("price_discount", ""),
                "barcode": barcode,
                "discount_amount": ocr_payload.get("discount_amount", ""),
                "id_sku": ocr_payload.get("id_sku", ""),
                "print_datetime": ocr_payload.get("print_datetime", ""),
                "code": code_value,
                "additional_info": ocr_payload.get("additional_info", ""),
                "color": ocr_payload.get("color", ""),
                "special_symbols": ocr_payload.get("special_symbols", ""),
                "frame_timestamp": frame_timestamp,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "qr_code_barcode": _first_non_empty(
                    qr_payload.get("qr_code_barcode", ""),
                    ocr_payload.get("qr_code_barcode", ""),
                    barcode,
                ),
                "price1_qr": _first_non_empty(
                    qr_payload.get("price1_qr", ""),
                    ocr_payload.get("price1_qr", ""),
                    price_default,
                ),
                "price2_qr": _first_non_empty(
                    qr_payload.get("price2_qr", ""),
                    ocr_payload.get("price2_qr", ""),
                ),
                "price3_qr": _first_non_empty(
                    qr_payload.get("price3_qr", ""),
                    ocr_payload.get("price3_qr", ""),
                ),
                "price4_qr": _first_non_empty(
                    qr_payload.get("price4_qr", ""),
                    ocr_payload.get("price4_qr", ""),
                    price_card,
                ),
                "wholesale_level_1_coun": _first_non_empty(
                    qr_payload.get("wholesale_level_1_coun", ""),
                    qr_payload.get("wholesale_level_1_count", ""),
                    ocr_payload.get("wholesale_level_1_coun", ""),
                    ocr_payload.get("wholesale_level_1_count", ""),
                ),
                "wholesale_level_1_price": _first_non_empty(
                    qr_payload.get("wholesale_level_1_price", ""),
                    ocr_payload.get("wholesale_level_1_price", ""),
                ),
                "wholesale_level_2_count": _first_non_empty(
                    qr_payload.get("wholesale_level_2_count", ""),
                    ocr_payload.get("wholesale_level_2_count", ""),
                ),
                "wholesale_level_2_price": _first_non_empty(
                    qr_payload.get("wholesale_level_2_price", ""),
                    ocr_payload.get("wholesale_level_2_price", ""),
                ),
                "action_price_qr": _first_non_empty(
                    qr_payload.get("action_price_qr", ""),
                    ocr_payload.get("action_price_qr", ""),
                ),
                "action_code_qr": _first_non_empty(
                    qr_payload.get("action_code_qr", ""),
                    ocr_payload.get("action_code_qr", ""),
                ),
            }
        )
        return cls(normalize_structured_row(raw))

    def to_csv_row(self) -> dict[str, str]:
        return {key: self.values.get(key, "") for key in STRUCTURED_PRICE_TAG_FIELDS}


def normalize_structured_row(raw: dict[str, Any]) -> dict[str, str]:
    row = {key: _clean_text(raw.get(key, "")) for key in STRUCTURED_PRICE_TAG_FIELDS}

    for key in ("price_default", "price_card", "price_discount"):
        row[key] = _format_price_comma(row[key], empty_value=NO_VALUE if key == "price_discount" else "")

    for key in ("price1_qr", "price2_qr", "price3_qr", "price4_qr"):
        row[key] = _format_price_dot(row[key], empty_value=NO_VALUE)

    for key in ("wholesale_level_1_price", "wholesale_level_2_price", "action_price_qr"):
        row[key] = _format_price_dot(row[key], empty_value=NO_VALUE)

    row["barcode"] = _digits_only(row["barcode"])
    row["id_sku"] = _format_sku(row["id_sku"])
    row["qr_code_barcode"] = _digits_only(row["qr_code_barcode"])
    row["print_datetime"] = _format_print_datetime(row["print_datetime"])
    row["code"] = _format_zone_code(row["code"])
    row["discount_amount"] = _format_discount_amount(row["discount_amount"])
    row["frame_timestamp"] = _format_int(row["frame_timestamp"])

    for key in ("x_min", "y_min", "x_max", "y_max"):
        row[key] = _format_coordinate(row[key])

    for key in ("wholesale_level_1_coun", "wholesale_level_2_count"):
        row[key] = _format_int_or_no_value(row[key])

    for key in ("color",):
        row[key] = row[key].casefold()

    row["special_symbols"] = _format_special_symbol(row["special_symbols"])

    for key in OPTIONAL_NO_VALUE_FIELDS:
        if not row[key]:
            row[key] = NO_VALUE

    return {key: row.get(key, "") for key in STRUCTURED_PRICE_TAG_FIELDS}


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if not _is_empty_or_no_value(value):
            return value
    return ""


def _price_from_misplaced_code(value: Any) -> str:
    text = _clean_text(value)
    if not _looks_like_price_value(text):
        return ""
    return text


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    text = str(value).strip()
    if text.casefold() in {"none", "null", "nan", "n/a", "na", "-", "—"}:
        return ""
    return re.sub(r"\s+", " ", text)


def _format_price_comma(value: Any, *, empty_value: str) -> str:
    number = _extract_number(value)
    if number is None:
        return empty_value if _is_empty_or_no_value(value) else _clean_text(value)
    return f"{number:.2f}".replace(".", ",")


def _format_price_dot(value: Any, *, empty_value: str) -> str:
    number = _extract_number(value)
    if number is None:
        return empty_value if _is_empty_or_no_value(value) else _clean_text(value)
    return f"{number:.2f}"


def _extract_number(value: Any) -> float | None:
    text = _clean_text(value)
    if not text or text.casefold() == NO_VALUE:
        return None

    explicit_decimal = re.search(r"-?\d[\d\s\u00a0]*(?:[,.]\d{1,3})", text)
    if explicit_decimal is not None:
        number_text = explicit_decimal.group(0).replace("\u00a0", " ").replace(" ", "")
        try:
            return float(number_text.replace(",", "."))
        except ValueError:
            return None

    spaced_cents = re.search(r"-?\d+(?:[\s\u00a0]\d{3})*[\s\u00a0]\d{2}(?!\d)", text)
    if spaced_cents is not None:
        groups = re.findall(r"\d+", spaced_cents.group(0))
        if len(groups) >= 2:
            integer_part = "".join(groups[:-1])
            cents = groups[-1]
            sign = "-" if spaced_cents.group(0).lstrip().startswith("-") else ""
            try:
                return float(f"{sign}{integer_part}.{cents}")
            except ValueError:
                return None

    match = re.search(r"-?\d+", text)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _digits_only(value: Any) -> str:
    return re.sub(r"\D", "", _clean_text(value))


def _format_sku(value: Any) -> str:
    digits = _digits_only(value)
    if len(digits) < 8:
        return ""
    return digits


def _format_print_datetime(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    match = re.search(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\s+(\d{1,2}):(\d{2})\b", text)
    if match is None:
        return ""
    day, month, year, hour, minute = match.groups()
    if len(year) == 2:
        year = f"20{year}"
    return f"{int(day):02d}.{int(month):02d}.{year} {int(hour):02d}:{minute}"


def _format_zone_code(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    if re.search(r"[A-Za-zА-Яа-яЁё]", text):
        return ""
    if _looks_like_price_value(text):
        return ""
    if not re.search(r"\d", text):
        return ""
    if not re.search(r"[_-]", text):
        return ""
    return re.sub(r"\s+", " ", text)


def _looks_like_price_value(value: Any) -> bool:
    text = _clean_text(value)
    if not text:
        return False
    if re.search(r"[A-Za-zА-Яа-яЁё_%]", text):
        return False
    if "_" in text:
        return False
    if re.search(r"\d\s*-\s*\d", text):
        return False

    digits = re.findall(r"\d+", text)
    if not digits:
        return False
    integer_part = "".join(digits[:-1]) if len(digits) > 1 and len(digits[-1]) == 2 else digits[0]
    if len(integer_part) > 5:
        return False

    if re.search(r"\d+[,.]\d{1,2}", text):
        return True
    if len(digits) == 1:
        return 2 <= len(digits[0]) <= 5
    return len(digits[-1]) == 2


def _format_discount_amount(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    if text.casefold() == NO_VALUE:
        return NO_VALUE
    match = re.search(r"-?\d+(?:[,.]\d+)?", text)
    if match is None:
        return text
    number = match.group(0).replace(".", ",")
    if not number.startswith("-"):
        number = f"-{number}"
    folded = text.casefold()
    if "%" in text or "процент" in folded:
        suffix = "%"
    elif "руб" in folded:
        suffix = " руб"
    else:
        suffix = ""
    return f"{number}{suffix}"


def _format_int(value: Any) -> str:
    number = _extract_number(value)
    if number is None:
        return ""
    return str(int(round(number)))


def _format_int_or_no_value(value: Any) -> str:
    if _is_empty_or_no_value(value):
        return NO_VALUE
    formatted = _format_int(value)
    return formatted or _clean_text(value)


def _format_coordinate(value: Any) -> str:
    number = _extract_number(value)
    if number is None:
        return ""
    if abs(number - round(number)) < 1e-6:
        return str(int(round(number)))
    return f"{number:.1f}".replace(".", ",")


def _format_special_symbol(value: Any) -> str:
    text = _clean_text(value)
    if _is_empty_or_no_value(text):
        return ""

    folded = text.casefold()
    mappings = (
        ("Ш", ("ш", "шт", "штуки")),
        ("Л", ("л", "лоток")),
        ("К", ("к", "короб", "коробка")),
    )
    for symbol, aliases in mappings:
        if folded in aliases:
            return symbol

    match = re.search(r"(?:^|[^а-яa-z])([шлк])(?:$|[^а-яa-z])", folded)
    if match is None:
        return text
    return match.group(1).upper()


def _is_empty_or_no_value(value: Any) -> bool:
    text = _clean_text(value)
    return not text or text.casefold() == NO_VALUE
