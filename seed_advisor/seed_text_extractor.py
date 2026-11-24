import re

import numpy as np

from .ocr_engine import OcrEngine, OcrSettings
from .packet_regions import PacketField
from .seed_packet_data import FieldRecognition, SeedPacketData


class SeedTextExtractor:
    def __init__(self, ocr_engine: OcrEngine) -> None:
        self.ocr_engine = ocr_engine

        self.settings = {
            PacketField.SEED_COUNT: OcrSettings(
                language="eng",
                psm=7,
                whitelist="0123456789",
            ),
            PacketField.PRICE_CATEGORY: OcrSettings(
                language="eng",
                psm=8,
                whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            ),
            PacketField.NAME: OcrSettings(
                language="eng",
                psm=7,
            ),
            PacketField.VARIETY: OcrSettings(
                language="eng",
                psm=7,
            ),
            PacketField.DESCRIPTION_HU: OcrSettings(
                language="hun",
                psm=6,
            ),
        }

    def extract(
        self,
        crops: dict[PacketField, np.ndarray],
    ) -> SeedPacketData:
        packet = SeedPacketData()

        for field, settings in self.settings.items():
            recognition = self._recognize_field(
                crops.get(field),
                settings,
            )

            packet.recognition[field.value] = recognition

            value = self._parse_field(
                field,
                recognition.raw_text,
                recognition.warnings,
            )

            setattr(packet, field.value, value)

        return packet

    def _recognize_field(
        self,
        image: np.ndarray | None,
        settings: OcrSettings,
    ) -> FieldRecognition:
        if image is None or image.size == 0:
            return FieldRecognition(
                raw_text="",
                mean_confidence=None,
                warnings=["A mező képrészlete hiányzik vagy üres."],
            )

        try:
            result = self.ocr_engine.recognize(image, settings)
        except RuntimeError as exc:
            return FieldRecognition(
                raw_text="",
                mean_confidence=None,
                warnings=[f"OCR-hiba: {exc}"],
            )

        warnings = []

        if not result.text.strip():
            warnings.append("Nem sikerült szöveget felismerni.")
        elif (
            result.mean_confidence is not None
            and result.mean_confidence < 60
        ):
            warnings.append(
                "Alacsony átlagos OCR-konfidencia; ellenőrzés szükséges."
            )

        return FieldRecognition(
            raw_text=result.text,
            mean_confidence=result.mean_confidence,
            warnings=warnings,
        )

    @staticmethod
    def _parse_field(
        field: PacketField,
        text: str,
        warnings: list[str],
    ) -> str | int | None:
        normalized = " ".join(text.split())

        if not normalized:
            return None

        if field is PacketField.SEED_COUNT:
            if re.fullmatch(r"[0-9]+", normalized):
                value = int(normalized)

                if value > 0:
                    return value

            warnings.append(
                "A magszám nem értelmezhető egyértelmű pozitív egész számként."
            )
            return None

        if field is PacketField.PRICE_CATEGORY:
            if re.fullmatch(r"[A-Z]", normalized):
                return normalized

            warnings.append(
                "Az árkategória nem egyetlen nagybetű."
            )
            return None

        return normalized