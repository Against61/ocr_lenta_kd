from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from price_tag_pipeline.codes.reader import (
    CodeDetection,
    detect_codes,
    empty_qr_payload,
    merge_barcode_payload,
    merge_qr_payloads,
    parse_qr_payload,
)


class CodesReaderTests(unittest.TestCase):
    def test_parse_qr_payload_supports_main_aliases(self) -> None:
        payload = parse_qr_payload(
            "https://example.test/?b=4603552017456&p1=415,79&p4=316.99&wL1C=5&wL1P=299.90&aC=A12"
        )

        self.assertEqual(payload["qr_code_barcode"], "4603552017456")
        self.assertEqual(payload["price1_qr"], "415,79")
        self.assertEqual(payload["price4_qr"], "316.99")
        self.assertEqual(payload["wholesale_level_1_count"], "5")
        self.assertEqual(payload["wholesale_level_1_price"], "299.90")
        self.assertEqual(payload["action_code_qr"], "A12")

    def test_parse_qr_payload_supports_output_field_names(self) -> None:
        payload = parse_qr_payload(
            '{"QR_CODE_BARCODE": "2999990013252", "price2_qr": "72.59", '
            '"wholesale_level_1_coun": "3", "action_price_qr": "55.39"}'
        )

        self.assertEqual(payload["qr_code_barcode"], "2999990013252")
        self.assertEqual(payload["price2_qr"], "72.59")
        self.assertEqual(payload["wholesale_level_1_count"], "3")
        self.assertEqual(payload["action_price_qr"], "55.39")

    def test_merge_qr_payloads_keeps_first_non_empty_values(self) -> None:
        detections = [
            CodeDetection(
                source_path=Path("crop.jpg"),
                code_type="QR",
                value="",
                parsed_fields={"price1_qr": "100.00", "price2_qr": ""},
            ),
            CodeDetection(
                source_path=Path("crop.jpg"),
                code_type="QR",
                value="",
                parsed_fields={"price1_qr": "200.00", "price2_qr": "150.00"},
            ),
        ]

        payload = merge_qr_payloads(detections)

        self.assertEqual(payload["price1_qr"], "100.00")
        self.assertEqual(payload["price2_qr"], "150.00")
        self.assertEqual(set(payload), set(empty_qr_payload()))

    def test_detect_codes_returns_empty_for_unreadable_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not_an_image.txt"
            path.write_text("not an image", encoding="utf-8")

            self.assertEqual(detect_codes(path), [])

    def test_merge_barcode_payload_uses_first_non_qr_digits(self) -> None:
        payload = merge_barcode_payload(
            [
                CodeDetection(source_path=Path("crop.jpg"), code_type="QR", value="b=111"),
                CodeDetection(source_path=Path("crop.jpg"), code_type="DATAMATRIX", value="b=222"),
                CodeDetection(source_path=Path("crop.jpg"), code_type="EAN_13", value="4 603552 017456"),
            ]
        )

        self.assertEqual(payload, {"barcode": "4603552017456"})


if __name__ == "__main__":
    unittest.main()
