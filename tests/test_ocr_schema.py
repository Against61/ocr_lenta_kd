from __future__ import annotations

import unittest
from pathlib import Path

from price_tag_pipeline.ocr.schema import STRUCTURED_PRICE_TAG_FIELDS, StructuredPriceTagCsvRow


class OcrSchemaTests(unittest.TestCase):
    def test_normalizes_vlm_values_to_csv_contract(self) -> None:
        row = StructuredPriceTagCsvRow.from_payloads(
            video_path=Path("/data/43_15.mp4"),
            track_row={},
            frame_timestamp="2472.4",
            coordinates=(3194.5, 1722.9, 3298, 1924.0),
            ocr_payload={
                "product_name": "  Мед   тестовый  ",
                "price_default": "415.79 руб",
                "price_card": 316.99,
                "price_discount": "",
                "barcode": "4 603552017456",
                "discount_amount": "23%",
                "id_sku": "370 204 501 518",
                "additional_info": "",
                "color": "Red",
            },
            qr_payload={
                "qr_code_barcode": "4603552017456",
                "price1_qr": "415,79",
                "price2_qr": "394.99",
                "price3_qr": "",
                "price4_qr": "316,99",
                "wholesale_level_1_count": "",
            },
        ).to_csv_row()

        self.assertEqual(tuple(row.keys()), STRUCTURED_PRICE_TAG_FIELDS)
        self.assertEqual(row["filename"], "43_15.mp4")
        self.assertEqual(row["product_name"], "Мед тестовый")
        self.assertEqual(row["price_default"], "415,79")
        self.assertEqual(row["price_card"], "316,99")
        self.assertEqual(row["price_discount"], "нет")
        self.assertEqual(row["barcode"], "4603552017456")
        self.assertEqual(row["discount_amount"], "-23%")
        self.assertEqual(row["id_sku"], "370204501518")
        self.assertEqual(row["additional_info"], "нет")
        self.assertEqual(row["color"], "red")
        self.assertEqual(row["frame_timestamp"], "2472")
        self.assertEqual(row["x_min"], "3194,5")
        self.assertEqual(row["y_min"], "1722,9")
        self.assertEqual(row["x_max"], "3298")
        self.assertEqual(row["y_max"], "1924")
        self.assertEqual(row["price1_qr"], "415.79")
        self.assertEqual(row["price2_qr"], "394.99")
        self.assertEqual(row["price3_qr"], "нет")
        self.assertEqual(row["price4_qr"], "316.99")
        self.assertEqual(row["wholesale_level_1_coun"], "нет")

    def test_uses_validation_csv_wholesale_header_typo(self) -> None:
        self.assertIn("wholesale_level_1_coun", STRUCTURED_PRICE_TAG_FIELDS)
        self.assertNotIn("wholesale_level_1_count", STRUCTURED_PRICE_TAG_FIELDS)

    def test_fills_visual_fallbacks_when_qr_is_missing(self) -> None:
        row = StructuredPriceTagCsvRow.from_payloads(
            video_path=Path("/data/43_15.mp4"),
            track_row={},
            frame_timestamp="0",
            coordinates=(1, 2, 3, 4),
            ocr_payload={
                "product_name": "Сыр тестовый",
                "price_default": "72 59",
                "price_card": "55 39",
                "barcode": "2 999990 013252",
                "discount_amount": "50 руб",
                "special_symbols": "штуки",
                "wholesale_level_1_count": "от 5 шт",
                "wholesale_level_1_price": "168 76",
            },
            qr_payload={},
        ).to_csv_row()

        self.assertEqual(row["price_default"], "72,59")
        self.assertEqual(row["price_card"], "55,39")
        self.assertEqual(row["barcode"], "2999990013252")
        self.assertEqual(row["discount_amount"], "-50 руб")
        self.assertEqual(row["special_symbols"], "Ш")
        self.assertEqual(row["qr_code_barcode"], "2999990013252")
        self.assertEqual(row["price1_qr"], "72.59")
        self.assertEqual(row["price2_qr"], "нет")
        self.assertEqual(row["price3_qr"], "нет")
        self.assertEqual(row["price4_qr"], "55.39")
        self.assertEqual(row["wholesale_level_1_coun"], "5")
        self.assertEqual(row["wholesale_level_1_price"], "168.76")

    def test_drops_unreliable_small_field_hallucinations(self) -> None:
        row = StructuredPriceTagCsvRow.from_payloads(
            video_path=Path("/data/25_12-20.mp4"),
            track_row={},
            frame_timestamp="0",
            coordinates=(1, 2, 3, 4),
            ocr_payload={
                "product_name": "Тест",
                "price_card": "199",
                "id_sku": "315",
                "print_datetime": "дата печати не видна",
                "code": "Зона_123",
            },
            qr_payload={},
        ).to_csv_row()

        self.assertEqual(row["id_sku"], "")
        self.assertEqual(row["print_datetime"], "")
        self.assertEqual(row["code"], "")

    def test_moves_price_default_misread_as_code(self) -> None:
        row = StructuredPriceTagCsvRow.from_payloads(
            video_path=Path("/data/26_12-20.mp4"),
            track_row={},
            frame_timestamp="0",
            coordinates=(1, 2, 3, 4),
            ocr_payload={
                "product_name": "Вино тестовое",
                "price_card": "1899,99",
                "code": "2631 57",
            },
            qr_payload={},
        ).to_csv_row()

        self.assertEqual(row["price_default"], "2631,57")
        self.assertEqual(row["code"], "")
        self.assertEqual(row["price1_qr"], "2631.57")

    def test_keeps_real_zone_code_patterns(self) -> None:
        for code in ("13_043015", "026005 - 026007", "024 017_1_6_2"):
            with self.subTest(code=code):
                row = StructuredPriceTagCsvRow.from_payloads(
                    video_path=Path("/data/26_12-20.mp4"),
                    track_row={},
                    frame_timestamp="0",
                    coordinates=(1, 2, 3, 4),
                    ocr_payload={
                        "product_name": "Вино тестовое",
                        "price_default": "1999,99",
                        "price_card": "1399,99",
                        "code": code,
                    },
                    qr_payload={},
                ).to_csv_row()

                self.assertEqual(row["price_default"], "1999,99")
                self.assertEqual(row["code"], code)


if __name__ == "__main__":
    unittest.main()
