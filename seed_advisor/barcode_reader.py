from dataclasses import dataclass, field

import cv2
import numpy as np
import zxingcpp


@dataclass
class BarcodeResult:
    value: str | None = None
    format: str | None = None
    warnings: list[str] = field(default_factory=list)


class BarcodeReader:
    def read(self, image: np.ndarray | None) -> BarcodeResult:
        if image is None or image.size == 0:
            return BarcodeResult(
                warnings=["A vonalkód képrészlete hiányzik vagy üres."]
            )

        # Először az eredeti képpel próbálkozunk.
        results = zxingcpp.read_barcodes(
            np.ascontiguousarray(image)
        )

        # Ha nem sikerült, próbáljuk nagyított szürkeárnyalatos képpel.
        if not results:
            if image.ndim == 2:
                gray = image
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            enlarged = cv2.resize(
                gray,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC,
            )

            results = zxingcpp.read_barcodes(
                np.ascontiguousarray(enlarged)
            )

        # Azonos találatok összevonása.
        unique = {
            (result.text, str(result.format)): result
            for result in results
        }

        if not unique:
            return BarcodeResult(
                warnings=["Nem sikerült vonalkódot dekódolni."]
            )

        if len(unique) > 1:
            return BarcodeResult(
                warnings=[
                    "Több különböző vonalkód található a ROI-ban; "
                    "pontosabb kijelölés szükséges."
                ]
            )

        result = next(iter(unique.values()))

        return BarcodeResult(
            value=result.text,
            format=str(result.format),
        )