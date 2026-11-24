from dataclasses import dataclass, field
from pathlib import Path

from enum import Enum
import re
from .depth_extractor import DepthExtractor, DepthResult

import numpy as np

from .ocr_engine import OcrEngine, OcrSettings
from .packet_regions import NormalizedRoi
from .region_extractor import PacketRegionExtractor

class CaptionType(str, Enum):
    MONTHS = "months"
    SPACING = "spacing"

@dataclass(frozen=True)
class MonthParseResult:
    normalized_text: str | None
    months: list[int] | None
    warning: str | None = None


class RomanMonthParser:
    ROMAN_MONTHS = {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "VII": 7,
        "VIII": 8,
        "IX": 9,
        "X": 10,
        "XI": 11,
        "XII": 12,
    }

    def parse(self, text: str) -> MonthParseResult:
        if not text.strip():
            return MonthParseResult(
                normalized_text=None,
                months=None,
                warning="Nem sikerült hónapfeliratot kiolvasni.",
            )

        normalized = text.strip().upper()

        for dash in ("–", "—", "−"):
            normalized = normalized.replace(dash, "-")

        normalized = re.sub(r"\s+", "", normalized)
        normalized = normalized.replace(".", "")
        if not normalized:
            return MonthParseResult(
                normalized_text=None,
                months=None,
                warning=(
                    "A felismert felirat csak elválasztójeleket tartalmaz; "
                    "hónapszámot nem."
                ),
            )

        # Csak egy hónap vagy egy hónaptartomány fogadható el.
        if not re.fullmatch(r"[IVX]+(?:-[IVX]+)?", normalized):
            return MonthParseResult(
                normalized_text=normalized,
                months=None,
                warning=(
                    "A felirat nem értelmezhető egyértelmű "
                    "római számos hónapként."
                ),
            )

        parts = normalized.split("-")

        if any(part not in self.ROMAN_MONTHS for part in parts):
            return MonthParseResult(
                normalized_text=normalized,
                months=None,
                warning="Érvénytelen római hónapszám.",
            )

        start = self.ROMAN_MONTHS[parts[0]]

        if len(parts) == 1:
            return MonthParseResult(
                normalized_text=normalized,
                months=[start],
            )

        end = self.ROMAN_MONTHS[parts[1]]

        if start <= end:
            months = list(range(start, end + 1))
        else:
            # Évfordulón átnyúló tartomány, például XI-II.
            months = (
                list(range(start, 13))
                + list(range(1, end + 1))
            )

        return MonthParseResult(
            normalized_text=normalized,
            months=months,
        )

@dataclass(frozen=True)
class PictogramRegion:
    identifier: str
    bounds: NormalizedRoi
    picture_bounds: NormalizedRoi
    caption_bounds: NormalizedRoi
    caption_type: CaptionType

    depth_bounds: NormalizedRoi | None = None
    depth_light_text: bool = False
    depth_single_digit: bool = False

@dataclass
class PictogramResult:
    identifier: str
    picture_text: str
    caption_text: str
    picture_confidence: float | None
    caption_confidence: float | None

    caption_type: str
    caption_normalized: str | None = None
    months: list[int] | None = None

    depth: DepthResult | None = None

    meaning: str | None = None
    red_x_detected: bool | None = None

    warnings: list[str] = field(default_factory=list)


class PictogramTemplateRepository:
    def get(self, name: str) -> list[PictogramRegion]:
        if name != "moravoseed_limba_right":
            raise ValueError(
                f"Nincs piktogramsablon ehhez: {name}"
            )

        definitions = [
            (
                0.005,
                0.165,
                CaptionType.MONTHS,
                NormalizedRoi(0.16, 0.57, 0.78, 0.72),
                True,
                False,
            ),
            (
                0.210,
                0.370,
                CaptionType.MONTHS,
                NormalizedRoi(0.11, 0.07, 0.66, 0.23),
                False,
                True,
            ),
            (
                0.415,
                0.575,
                CaptionType.MONTHS,
                None,
                False,
                False,
            ),
            (
                0.620,
                0.780,
                CaptionType.SPACING,
                None,
                False,
                False,
            ),
            (
                0.825,
                0.990,
                CaptionType.MONTHS,
                None,
                False,
                False,
            ),
        ]

        return [
            PictogramRegion(
                identifier=f"pictogram_{index:02d}",
                bounds=NormalizedRoi(
                    left, 0.0, right, 1.0
                ),
                picture_bounds=NormalizedRoi(
                    0.0, 0.0, 1.0, 0.77
                ),
                caption_bounds=NormalizedRoi(
                    0.0, 0.77, 1.0, 1.0
                ),
                caption_type=caption_type,
                depth_bounds=depth_bounds,
                depth_light_text=depth_light_text,
                depth_single_digit=depth_single_digit,
            )
            for index, (
                left,
                right,
                caption_type,
                depth_bounds,
                depth_light_text,
                depth_single_digit,
            ) in enumerate(definitions, start=1)
        ]

class PictogramExtractor:
    def __init__(self, ocr_engine: OcrEngine) -> None:
        self.ocr_engine = ocr_engine
        self.month_parser = RomanMonthParser()
        self.depth_extractor = DepthExtractor(ocr_engine)

    def extract(
        self,
        row_image: np.ndarray,
        regions: list[PictogramRegion],
        output_directory: str | Path,
    ) -> list[PictogramResult]:
        if row_image is None or row_image.size == 0:
            raise ValueError("A piktogramsor képe üres.")

        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)

        results: list[PictogramResult] = []

        for region in regions:
            # 1. Képrészletek kivágása
            item = self._crop(row_image, region.bounds)
            picture = self._crop(item, region.picture_bounds)
            caption = self._crop(item, region.caption_bounds)

            # 2. Képrészletek mentése
            PacketRegionExtractor.save_image(
                directory / f"{region.identifier}.png",
                item,
            )

            PacketRegionExtractor.save_image(
                directory / f"{region.identifier}_picture.png",
                picture,
            )

            PacketRegionExtractor.save_image(
                directory / f"{region.identifier}_caption.png",
                caption,
            )

            warnings: list[str] = []

            # 3. Mélységfelirat kivágása és felismerése
            depth_result: DepthResult | None = None

            if region.depth_bounds is not None:
                depth_result: DepthResult | None = None

                if region.depth_bounds is not None:
                    depth_image = self._crop(
                        item,
                        region.depth_bounds,
                    )

                    PacketRegionExtractor.save_image(
                        directory / f"{region.identifier}_depth.png",
                        depth_image,
                    )

                    prepared_depth = self.depth_extractor.prepare_image(
                        depth_image,
                        light_text=region.depth_light_text,
                    )

                    PacketRegionExtractor.save_image(
                        directory / f"{region.identifier}_depth_prepared.png",
                        prepared_depth,
                    )

                    depth_result = self.depth_extractor.extract(
                        depth_image,
                        light_text=region.depth_light_text,
                        single_digit=region.depth_single_digit,
                    )

            # 4. A piktogram képébe írt szöveg felismerése
            picture_text, picture_confidence = self._read(
                picture,
                OcrSettings(
                    language="eng",
                    psm=11,
                    scale=4.0,
                ),
                warnings,
                "Képbe írt szöveg",
            )

            # 5. Az alsó felirat OCR-beállításainak kiválasztása
            if region.caption_type is CaptionType.MONTHS:
                caption_settings = OcrSettings(
                    language="eng",
                    psm=7,
                    scale=5.0,
                    whitelist="IVX.-",
                )
            else:
                caption_settings = OcrSettings(
                    language="eng",
                    psm=7,
                    scale=4.0,
                    whitelist="0123456789xXcmCM.,",
                )

            # 6. Az előkészített feliratkép mentése
            prepared_caption = self.ocr_engine.prepare_image(
                caption,
                scale=caption_settings.scale,
            )

            PacketRegionExtractor.save_image(
                directory / f"{region.identifier}_caption_prepared.png",
                prepared_caption,
            )

            # 7. Az alsó felirat felismerése a nyers kivágásból
            caption_text, caption_confidence = self._read(
                caption,
                caption_settings,
                warnings,
                "Alsó felirat",
            )

            # 8. Hónapfelirat értelmezése
            caption_normalized: str | None = None
            months: list[int] | None = None

            if region.caption_type is CaptionType.MONTHS:
                parsed = self.month_parser.parse(caption_text)

                caption_normalized = parsed.normalized_text
                months = parsed.months

                if parsed.warning:
                    warnings.append(parsed.warning)

            # 9. Konfidencia ellenőrzése
            if (
                caption_confidence is not None
                and caption_confidence < 60
            ):
                warnings.append(
                    "Az alsó felirat OCR-konfidenciája alacsony; "
                    "a feldolgozott értéket is ellenőrizni kell."
                )

            # 10. Eredmény összeállítása
            results.append(
                PictogramResult(
                    identifier=region.identifier,
                    picture_text=picture_text,
                    caption_text=caption_text,
                    picture_confidence=picture_confidence,
                    caption_confidence=caption_confidence,
                    caption_type=region.caption_type.value,
                    caption_normalized=caption_normalized,
                    months=months,
                    depth=depth_result,
                    warnings=warnings,
                )
            )

        return results
    
    def _read(
        self,
        image: np.ndarray,
        settings: OcrSettings,
        warnings: list[str],
        label: str,
    ) -> tuple[str, float | None]:
        try:
            result = self.ocr_engine.recognize(image, settings)
        except RuntimeError as exc:
            warnings.append(f"{label}: OCR-hiba: {exc}")
            return "", None

        if not result.text:
            warnings.append(
                f"{label}: nincs felismert szöveg. "
                "Ez nem bizonyítja, hogy a régióban nincs felirat."
            )

        return result.text, result.mean_confidence

    @staticmethod
    def _crop(
        image: np.ndarray,
        roi: NormalizedRoi,
    ) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = roi.to_pixels(width, height)

        return image[y1:y2, x1:x2].copy()