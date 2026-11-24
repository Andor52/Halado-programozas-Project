from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


class ImageLoadError(Exception):
    """Hiba történik a magfotó betöltése közben."""


@dataclass(frozen=True)
class LoadedPacketImage:
    image: np.ndarray
    source_path: Path
    width: int
    height: int


class PacketImageLoader:
    """Magcsomag-fotó betöltése és normalizálása."""

    def load(self, image_path: str | Path) -> LoadedPacketImage:
        path = Path(image_path)

        if not path.is_file():
            raise ImageLoadError(f"A megadott fájl nem található: {path}")

        try:
            pil_image = Image.open(path)
            pil_image = ImageOps.exif_transpose(pil_image)
        except Exception as exc:
            raise ImageLoadError(f"Nem sikerült a kép megnyitása: {path}") from exc

        rgb_image = np.array(pil_image.convert("RGB"))
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        height, width = bgr_image.shape[:2]

        return LoadedPacketImage(
            image=bgr_image,
            source_path=path,
            width=width,
            height=height,
        )