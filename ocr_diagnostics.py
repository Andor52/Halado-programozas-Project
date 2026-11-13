import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output


class OcrDiagnostics:
    def __init__(self, tesseract_command: str) -> None:
        pytesseract.pytesseract.tesseract_cmd = (
            tesseract_command
        )

    def inspect(
        self,
        image_path: Path,
        whitelist: str,
    ) -> None:
        with Image.open(image_path) as image:
            pixels = np.array(image.convert("L"))

        print(f"\n{'=' * 60}")
        print(f"Kép: {image_path.name}")
        print(f"Méret: {pixels.shape[1]} × {pixels.shape[0]}")

        for psm in (7, 8, 13):
            for use_whitelist in (False, True):
                config = f"--oem 3 --psm {psm}"

                if use_whitelist:
                    config += (
                        " -c tessedit_char_whitelist="
                        + whitelist
                    )

                data = pytesseract.image_to_data(
                    pixels,
                    lang="eng",
                    config=config,
                    output_type=Output.DICT,
                    timeout=20,
                )

                words = []

                for index, raw_text in enumerate(data["text"]):
                    text = raw_text.strip()

                    if text:
                        words.append(
                            (
                                text,
                                float(data["conf"][index]),
                            )
                        )

                combined = " ".join(
                    text for text, _ in words
                )

                print(
                    f"PSM={psm:2d}, "
                    f"whitelist={str(use_whitelist):5s}: "
                    f"{combined!r}"
                )
                print(f"  Szavak és konfidenciák: {words}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Előkészített OCR-képek összehasonlító tesztje."
    )

    parser.add_argument(
        "directory",
        type=Path,
        help="A piktogramképeket tartalmazó könyvtár",
    )

    parser.add_argument(
        "--tesseract",
        default=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    )

    args = parser.parse_args()

    diagnostics = OcrDiagnostics(args.tesseract)

    tests = [
        (
            "pictogram_01_caption_prepared.png",
            "IVX.-",
        ),
        (
            "pictogram_02_caption_prepared.png",
            "IVX.-",
        ),
        (
            "pictogram_01_depth_prepared.png",
            "0123456789.,-cCmM",
        ),
        (
            "pictogram_02_depth_prepared.png",
            "0123456789.,-cCmM",
        ),
    ]

    for filename, whitelist in tests:
        path = args.directory / filename

        if not path.is_file():
            print(f"Hiányzó fájl: {path}")
            continue

        diagnostics.inspect(path, whitelist)