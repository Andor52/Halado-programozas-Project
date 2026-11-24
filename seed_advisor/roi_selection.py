from dataclasses import dataclass

import cv2
import numpy as np

from .config import WINDOW_NAME


@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    width: int
    height: int

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0


class RoiSelector:
    """Interaktív ROI-kiválasztás OpenCV ablakban."""

    def select(
        self,
        image: np.ndarray,
        title: str,
        scale: float = 1.0,
    ) -> Roi:
        display_image = self._resize(image, scale)

        selection = cv2.selectROI(
            windowName=f"{WINDOW_NAME} - {title}",
            img=display_image,
            showCrosshair=True,
            fromCenter=False,
        )
        cv2.destroyAllWindows()

        x, y, width, height = map(int, selection)
        roi = Roi(x=x, y=y, width=width, height=height)

        if scale != 1.0:
            roi = Roi(
                x=int(roi.x / scale),
                y=int(roi.y / scale),
                width=int(roi.width / scale),
                height=int(roi.height / scale),
            )

        return roi

    def crop(self, image: np.ndarray, roi: Roi) -> np.ndarray:
        if roi.is_empty:
            raise ValueError("Üres ROI nem vágható ki.")

        return image[
            roi.y:roi.y + roi.height,
            roi.x:roi.x + roi.width,
        ].copy()

    @staticmethod
    def _resize(image: np.ndarray, scale: float) -> np.ndarray:
        if scale == 1.0:
            return image

        return cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )