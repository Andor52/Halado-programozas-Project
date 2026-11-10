import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output


class DepthSizeTest:
    def __init__(self, tesseract_command: str) -> None:
        pytesseract.pytesseract.tesseract_cmd = (
            tesseract_command
        )

    def run(self, image_path: Path) -> None:
        with Image.open(image_path) as image:
            gray = np.array(image.convert("L"))

        # A bemenet már előkészített kép.
        # Itt csak a fekete előtér befoglaló téglalapját keressük.
        foreground = (gray < 128).astype(np.uint8) * 255
        points = cv2.findNonZero(foreground)

        if points is None:
            raise ValueError("Nincs sötét előtér a képen.")

        x, y, width, height = cv2.boundingRect(points)
        cropped = gray[y:y + height, x:x + width]

        output_directory = Path("output") / "depth_size_test"
        output_directory.mkdir(parents=True, exist_ok=True)

        for target_height in (24, 32, 48, 64):
            scale = target_height / cropped.shape[0]
            target_width = max(
                1,
                round(cropped.shape[1] * scale),
            )

            resized = cv2.resize(
                cropped,
                (target_width, target_height),
                interpolation=(
                    cv2.INTER_AREA
                    if scale < 1
                    else cv2.INTER_CUBIC
                ),
            )

            prepared = cv2.copyMakeBorder(
                resized,
                10,
                10,
                10,
                10,
                cv2.BORDER_CONSTANT,
                value=255,
            )

            output_path = (
                output_directory
                / f"depth_height_{target_height}.png"
            )

            success, encoded = cv2.imencode(".png", prepared)

            if not success:
                raise OSError("Nem sikerült a tesztkép kódolása.")

            output_path.write_bytes(encoded.tobytes())

            for whitelist in (None, "0123456789.,-cCmM"):
                config = "--oem 3 --psm 7"

                if whitelist:
                    config += (
                        " -c tessedit_char_whitelist="
                        + whitelist
                    )

                data = pytesseract.image_to_data(
                    prepared,
                    lang="eng",
                    config=config,
                    output_type=Output.DICT,
                    timeout=20,
                )

                words = [
                    (
                        text.strip(),
                        float(data["conf"][index]),
                    )
                    for index, text in enumerate(data["text"])
                    if text.strip()
                ]

                print(
                    f"Magasság={target_height:2d}, "
                    f"whitelist={whitelist is not None}: "
                    f"{words}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--tesseract",
        default=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    )

    args = parser.parse_args()

    DepthSizeTest(args.tesseract).run(args.image)