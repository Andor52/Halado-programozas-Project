from dataclasses import dataclass, field
import re
from statistics import mean

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


@dataclass
class SplitDepthReading:
    number_text: str = ""
    unit_text: str = ""

    number_confidence: float | None = None
    unit_confidence: float | None = None

    valid: bool = False
    warnings: list[str] = field(default_factory=list)


class SingleDigitDepthReader:
    """Egy számjegy + cm/mm felirat felismerése.

    Csak olyan sablonmezőnél használjuk, amelyhez
    kifejezetten engedélyeztük az egyjegyű feldolgozást.

    Nem általános olvasó többjegyű számhoz,
    tizedeshez vagy mélységtartományhoz.
    """

    def read(self, prepared: np.ndarray) -> SplitDepthReading:
        result = SplitDepthReading()

        try:
            number_image, unit_image = self._split(prepared)

            number_image = self._prepare_number(number_image)
            unit_image = self._crop_and_pad(unit_image)

            (
                result.number_text,
                result.number_confidence,
            ) = self._read_part(
                number_image,
                whitelist="0123456789",
            )

            (
                result.unit_text,
                result.unit_confidence,
            ) = self._read_part(
                unit_image,
                whitelist="cCmM",
            )

        except (ValueError, RuntimeError) as exc:
            result.warnings.append(
                f"Külön mélységfelismerés sikertelen: {exc}"
            )
            return result

        result.unit_text = result.unit_text.lower()

        if not re.fullmatch(r"[0-9]", result.number_text):
            result.warnings.append(
                "A számrész nem pontosan egy számjegy."
            )

        if result.unit_text not in {"cm", "mm"}:
            result.warnings.append(
                "A mértékegység nem cm vagy mm."
            )

        result.valid = (
            re.fullmatch(r"[0-9]", result.number_text) is not None
            and result.unit_text in {"cm", "mm"}
        )

        for label, confidence in (
            ("Szám", result.number_confidence),
            ("Mértékegység", result.unit_confidence),
        ):
            if confidence is not None and confidence < 60:
                result.warnings.append(
                    f"{label}: alacsony OCR-konfidencia; "
                    "ellenőrzés szükséges."
                )

        return result

    @staticmethod
    def _split(
        gray: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if gray is None or gray.size == 0:
            raise ValueError("Üres mélységkép.")

        if gray.ndim != 2:
            raise ValueError("Egysávos előkészített kép szükséges.")

        foreground = (gray < 128).astype(np.uint8) * 255

        count, _, stats, _ = cv2.connectedComponentsWithStats(
            foreground,
            connectivity=8,
        )

        if count <= 1:
            raise ValueError("Nincs vizsgálható előtér.")

        max_height = int(
            stats[1:, cv2.CC_STAT_HEIGHT].max()
        )

        intervals = []

        for label in range(1, count):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])

            if height >= max_height * 0.5:
                intervals.append((x, x + width))

        intervals.sort()

        merged: list[tuple[int, int]] = []

        for left, right in intervals:
            if merged and left <= merged[-1][1]:
                merged[-1] = (
                    merged[-1][0],
                    max(merged[-1][1], right),
                )
            else:
                merged.append((left, right))

        gaps = [
            (next_left - right, right, next_left)
            for (_, right), (next_left, _) in zip(
                merged,
                merged[1:],
            )
            if next_left > right
        ]

        if not gaps:
            raise ValueError(
                "Nem választható külön a szám és a mértékegység."
            )

        _, gap_left, gap_right = max(
            gaps,
            key=lambda item: item[0],
        )

        split_x = (gap_left + gap_right) // 2

        return (
            gray[:, :split_x].copy(),
            gray[:, split_x:].copy(),
        )

    def _prepare_number(self, gray: np.ndarray) -> np.ndarray:
        foreground = (gray < 128).astype(np.uint8) * 255

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground,
            connectivity=8,
        )

        if count <= 1:
            raise ValueError("Üres számkivágás.")

        largest_label = 1 + int(
            np.argmax(stats[1:, cv2.CC_STAT_AREA])
        )

        largest_area = int(
            stats[largest_label, cv2.CC_STAT_AREA]
        )
        largest_height = int(
            stats[largest_label, cv2.CC_STAT_HEIGHT]
        )

        # Ne töröljünk észrevétlenül egy második,
        # jelentősebb karaktert.
        for label in range(1, count):
            if label == largest_label:
                continue

            area = int(stats[label, cv2.CC_STAT_AREA])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])

            if (
                area >= largest_area * 0.20
                or height >= largest_height * 0.50
            ):
                raise ValueError(
                    "A számrész több jelentős komponenst tartalmaz; "
                    "az egyjegyű feldolgozás nem biztonságos."
                )

        retained = (
            (labels == largest_label).astype(np.uint8)
            * 255
        )

        cleaned = cv2.bitwise_not(retained)

        return self._crop_and_pad(cleaned)

    @staticmethod
    def _crop_and_pad(gray: np.ndarray) -> np.ndarray:
        foreground = (gray < 128).astype(np.uint8) * 255
        points = cv2.findNonZero(foreground)

        if points is None:
            raise ValueError("Üres képrészlet.")

        x, y, width, height = cv2.boundingRect(points)
        cropped = gray[y:y + height, x:x + width]

        return cv2.copyMakeBorder(
            cropped,
            15,
            15,
            15,
            15,
            cv2.BORDER_CONSTANT,
            value=255,
        )

    @staticmethod
    def _read_part(
        image: np.ndarray,
        whitelist: str,
    ) -> tuple[str, float | None]:
        data = pytesseract.image_to_data(
            image,
            lang="eng",
            config=(
                "--oem 3 --psm 7 "
                f"-c tessedit_char_whitelist={whitelist}"
            ),
            output_type=Output.DICT,
            timeout=20,
        )

        words = []
        confidences = []

        for index, raw_text in enumerate(data["text"]):
            text = raw_text.strip()

            if not text:
                continue

            words.append(text)

            confidence = float(data["conf"][index])

            if confidence >= 0:
                confidences.append(confidence)

        return (
            "".join(words),
            round(mean(confidences), 2)
            if confidences
            else None,
        )