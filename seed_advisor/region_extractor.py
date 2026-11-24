from pathlib import Path

import cv2
import numpy as np

from .packet_regions import PacketField, PacketTemplate


class PacketRegionExtractor:
    def extract(
        self,
        image: np.ndarray,
        template: PacketTemplate,
    ) -> dict[PacketField, np.ndarray]:
        self._validate_image(image)

        height, width = image.shape[:2]
        crops: dict[PacketField, np.ndarray] = {}

        for field, roi in template.regions.items():
            x1, y1, x2, y2 = roi.to_pixels(width, height)
            crops[field] = image[y1:y2, x1:x2].copy()

        return crops

    def draw_preview(
        self,
        image: np.ndarray,
        template: PacketTemplate,
    ) -> np.ndarray:
        self._validate_image(image)

        preview = image.copy()
        height, width = image.shape[:2]

        for field, roi in template.regions.items():
            x1, y1, x2, y2 = roi.to_pixels(width, height)

            cv2.rectangle(
                preview,
                (x1, y1),
                (x2 - 1, y2 - 1),
                (0, 255, 255),
                2,
            )

            cv2.putText(
                preview,
                field.value,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return preview

    def save(
        self,
        crops: dict[PacketField, np.ndarray],
        output_directory: str | Path,
    ) -> None:
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)

        for field, crop in crops.items():
            self.save_image(
                directory / f"{field.value}.png",
                crop,
            )

    @staticmethod
    def save_image(path: str | Path, image: np.ndarray) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        success, encoded = cv2.imencode(".png", image)

        if not success:
            raise OSError(f"Nem sikerült a kép kódolása: {path}")

        path.write_bytes(encoded.tobytes())

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image is None or image.size == 0:
            raise ValueError("Üres vagy hiányzó kép.")