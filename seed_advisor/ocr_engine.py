from dataclasses import dataclass
from statistics import mean

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


@dataclass(frozen=True)
class OcrSettings:
    language: str = "eng"
    psm: int = 7
    scale: float = 3.0
    whitelist: str | None = None
    timeout_seconds: float = 20.0


@dataclass(frozen=True)
class OcrResult:
    text: str
    mean_confidence: float | None


class OcrEngine:
    def __init__(self, tesseract_command: str | None = None) -> None:
        if tesseract_command:
            pytesseract.pytesseract.tesseract_cmd = tesseract_command

    def check_environment(self) -> None:
        try:
            pytesseract.get_tesseract_version()
            available = set(pytesseract.get_languages(config=""))
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "A Tesseract nincs telepítve, vagy nem található a PATH-ban."
            ) from exc

        missing = {"hun", "eng"} - available

        if missing:
            raise RuntimeError(
                "Hiányzó Tesseract nyelvi csomagok: "
                + ", ".join(sorted(missing))
            )

    def recognize(
        self,
        image: np.ndarray,
        settings: OcrSettings,
    ) -> OcrResult:
        prepared = self.prepare_image(image, settings.scale)

        config = f"--oem 3 --psm {settings.psm}"

        if settings.whitelist:
            config += (
                " -c tessedit_char_whitelist="
                + settings.whitelist
            )

        data = pytesseract.image_to_data(
            prepared,
            lang=settings.language,
            config=config,
            output_type=Output.DICT,
            timeout=settings.timeout_seconds,
        )

        lines: dict[tuple[int, int, int, int], list[str]] = {}
        confidences: list[float] = []

        for index, value in enumerate(data["text"]):
            word = value.strip()

            if not word:
                continue

            line_key = (
                int(data["page_num"][index]),
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
            )

            lines.setdefault(line_key, []).append(word)

            confidence = float(data["conf"][index])

            if confidence >= 0:
                confidences.append(confidence)

        text = "\n".join(
            " ".join(words)
            for words in lines.values()
        )

        return OcrResult(
            text=text,
            mean_confidence=(
                round(mean(confidences), 2)
                if confidences
                else None
            ),
        )

    def prepare_image(
        self,
        image: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Az OCR bemeneti képe üres.")

        if scale <= 0:
            raise ValueError("A nagyítási tényezőnek pozitívnak kell lennie.")

        if image.ndim == 2:
            gray = image.copy()
        elif image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("Nem támogatott képformátum.")

        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # Egyszerű kezdőheurisztika:
        # sötét háttér esetén fordítsuk meg a képet.
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        return cv2.copyMakeBorder(
            binary,
            20,
            20,
            20,
            20,
            cv2.BORDER_CONSTANT,
            value=255,
        )