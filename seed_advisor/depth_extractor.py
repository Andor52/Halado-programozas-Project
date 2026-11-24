from dataclasses import dataclass, field
from email.mime import image
import re

import numpy as np
import cv2

from .ocr_engine import OcrEngine, OcrSettings
from .depth_image_cleaner import DepthImageCleaner
from .single_digit_depth_reader import SingleDigitDepthReader

@dataclass
class DepthResult:
    raw_text: str = ""
    normalized_text: str | None = None

    min_cm: float | None = None
    max_cm: float | None = None

    mean_confidence: float | None = None

    method: str = "full_text"

    number_raw_text: str | None = None
    unit_raw_text: str | None = None

    number_confidence: float | None = None
    unit_confidence: float | None = None

    warnings: list[str] = field(default_factory=list)


class DepthExtractor:
    NUMBER = r"[0-9]+(?:\.[0-9]+)?"

    DEPTH_PATTERN = re.compile(
        rf"(?P<minimum>{NUMBER})"
        rf"(?:-(?P<maximum>{NUMBER}))?"
        r"(?P<unit>cm|mm)",
        re.IGNORECASE,
    )

    def __init__(self, ocr_engine: OcrEngine) -> None:
        self.ocr_engine = ocr_engine
        self.image_cleaner = DepthImageCleaner()
        self.single_digit_reader = SingleDigitDepthReader()

    def extract(
        self,
        image: np.ndarray,
        light_text: bool = False,
        single_digit: bool = False,
    ) -> DepthResult:
        method = (
            "split_single_digit"
            if single_digit
            else "full_text"
        )

        if image is None or image.size == 0:
            return DepthResult(
                method=method,
                warnings=["A mélységfelirat képrészlete üres."],
            )

        prepared = self.prepare_image(
            image,
            light_text=light_text,
        )

        # Külön feldolgozás az engedélyezett egyjegyű mezőkhöz.
        if single_digit:
            reading = self.single_digit_reader.read(prepared)

            if reading.valid:
                combined_text = (
                    f"{reading.number_text} {reading.unit_text}"
                )
                result = self.parse(combined_text)
            else:
                result = DepthResult()

            result.method = "split_single_digit"

            result.number_raw_text = reading.number_text
            result.unit_raw_text = reading.unit_text

            result.number_confidence = reading.number_confidence
            result.unit_confidence = reading.unit_confidence

            result.warnings.extend(reading.warnings)

            return result

        # A korábban működő teljesfelirat-felismerés.
        try:
            ocr_result = self.ocr_engine.recognize(
                prepared,
                OcrSettings(
                    language="eng",
                    psm=7,
                    scale=1.0,
                    whitelist="0123456789.,-cCmM",
                ),
            )
        except RuntimeError as exc:
            return DepthResult(
                method="full_text",
                warnings=[f"Mélység-OCR hiba: {exc}"],
            )

        result = self.parse(ocr_result.text)
        result.method = "full_text"
        result.mean_confidence = ocr_result.mean_confidence

        if (
            result.mean_confidence is not None
            and result.mean_confidence < 60
        ):
            result.warnings.append(
                "Alacsony OCR-konfidencia; "
                "a mélységadatot ellenőrizni kell."
            )

        return result

    def parse(self, text: str) -> DepthResult:
        result = DepthResult(raw_text=text)

        if not text.strip():
            result.warnings.append(
                "Nem sikerült mélységfeliratot kiolvasni."
            )
            return result

        normalized = text.strip().lower()

        for dash in ("–", "—", "−"):
            normalized = normalized.replace(dash, "-")

        normalized = normalized.replace(",", ".")
        normalized = re.sub(r"\s+", "", normalized)

        result.normalized_text = normalized

        match = self.DEPTH_PATTERN.fullmatch(normalized)

        if match is None:
            result.warnings.append(
                "A felirat nem értelmezhető egyértelmű "
                "mélységadatként. Elvárt forma például: "
                "'1 cm' vagy '1-2 cm'."
            )
            return result

        minimum = float(match.group("minimum"))

        maximum_text = match.group("maximum")
        maximum = (
            float(maximum_text)
            if maximum_text is not None
            else minimum
        )

        if minimum > maximum:
            result.warnings.append(
                "A mélységtartomány alsó értéke "
                "nagyobb a felső értéknél."
            )
            return result

        unit = match.group("unit").lower()

        if unit == "mm":
            minimum /= 10
            maximum /= 10

        result.min_cm = minimum
        result.max_cm = maximum

        return result

    def prepare_image(
        self,
        image: np.ndarray,
        light_text: bool = False,
    ) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("A mélységfelirat képrészlete üres.")

        if image.ndim == 2:
            gray = image.copy()
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(
            gray,
            None,
            fx=6.0,
            fy=6.0,
            interpolation=cv2.INTER_CUBIC,
        )

        # Mindkét esetben sötét karaktereket szeretnénk
        # világos háttéren.
        if light_text:
            gray = cv2.bitwise_not(gray)

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            9,
        )
        cv2.imwrite("debug_depth_before_clean.png", binary)
        cleaned = self.image_cleaner.clean(binary)
        cv2.imwrite("debug_depth_after_clean.png", cleaned)

        return cv2.copyMakeBorder(
            cleaned,
            15,
            15,
            15,
            15,
            cv2.BORDER_CONSTANT,
            value=255,
        )