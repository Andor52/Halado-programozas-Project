import cv2
import numpy as np


class DepthImageCleaner:
    """Vízszintes vonalmaradványok szűrése mélységfeliratból.

    Bemenet:
        fekete karakterek fehér háttéren, bináris képen.

    Ez mélységfeliratokhoz készült heurisztika.
    Hónapfeliratokra ne használjuk, mert azok kötőjelét
    is eltávolíthatná.
    """

    def clean(self, binary: np.ndarray) -> np.ndarray:
        if binary is None or binary.size == 0:
            raise ValueError("Üres kép nem tisztítható.")

        if binary.ndim != 2:
            raise ValueError(
                "A tisztító egysávos bináris képet vár."
            )

        cleaned = binary.copy()

        # A komponenselemzéshez az előtér legyen fehér.
        foreground = cv2.bitwise_not(binary)

        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                foreground,
                connectivity=8,
            )
        )

        image_height, image_width = binary.shape

        for label in range(1, count):
            width = int(
                stats[label, cv2.CC_STAT_WIDTH]
            )
            height = int(
                stats[label, cv2.CC_STAT_HEIGHT]
            )

            aspect_ratio = width / max(height, 1)

            is_horizontal_fragment = (
                aspect_ratio >= 3.0
                and width >= image_width * 0.15
                and height <= image_height * 0.20
            )

            if is_horizontal_fragment:
                cleaned[labels == label] = 255

        return cleaned