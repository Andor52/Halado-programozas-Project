import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytesseract


class DepthSplitTest:
    def __init__(self, tesseract_command: str) -> None:
        pytesseract.pytesseract.tesseract_cmd = (
            tesseract_command
        )

    def run(self, image_path: Path) -> None:
        with Image.open(image_path) as image:
            gray = np.array(image.convert("L"))

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

        # A kis pontokat, zajokat nem használjuk a vágási hely
        # meghatározásához, de a képről nem töröljük őket.
        intervals = []

        for label in range(1, count):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])

            if height >= max_height * 0.5:
                intervals.append((x, x + width))

        intervals.sort()

        if len(intervals) < 2:
            raise ValueError(
                "Nem különíthető el két szövegrész."
            )

        # Az egymást átfedő vízszintes intervallumok összevonása.
        merged = []

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
            raise ValueError("Nincs alkalmas elválasztó rés.")

        _, gap_left, gap_right = max(
            gaps,
            key=lambda item: item[0],
        )

        split_x = (gap_left + gap_right) // 2

        parts = {
            "number": gray[:, :split_x],
            "unit": gray[:, split_x:],
        }

        directory = Path("output") / "depth_split_test"
        directory.mkdir(parents=True, exist_ok=True)

        for name, part in parts.items():
            prepared = self._crop_and_pad(
                part,
                keep_largest_component=(name == "number"),
            )

            success, encoded = cv2.imencode(
                ".png",
                prepared,
            )

            if not success:
                raise OSError("Nem sikerült a kép kódolása.")

            (directory / f"{name}.png").write_bytes(
                encoded.tobytes()
            )

            if name == "number":
                whitelist = "0123456789.,-"
                modes = (7, 8, 10, 13)
            else:
                whitelist = "cCmM"
                modes = (7, 8, 13)

            print(f"\n{name}:")

            for psm in modes:
                text = pytesseract.image_to_string(
                    prepared,
                    lang="eng",
                    config=(
                        f"--oem 3 --psm {psm} "
                        f"-c tessedit_char_whitelist={whitelist}"
                    ),
                    timeout=20,
                ).strip()

                print(f"  PSM={psm}: {text!r}")

    @staticmethod
    def _crop_and_pad(
        gray: np.ndarray,
        keep_largest_component: bool = False,
    ) -> np.ndarray:
        foreground = (gray < 128).astype(np.uint8) * 255

        if keep_largest_component:
            count, labels, stats, _ = (
                cv2.connectedComponentsWithStats(
                    foreground,
                    connectivity=8,
                )
            )

            if count <= 1:
                raise ValueError(
                    "Nincs felismerhető előtér a számkivágáson."
                )

            largest_label = 1 + int(
                np.argmax(
                    stats[1:, cv2.CC_STAT_AREA]
                )
            )

            foreground = (
                (labels == largest_label).astype(np.uint8)
                * 255
            )

        points = cv2.findNonZero(foreground)

        if points is None:
            raise ValueError("Az egyik kivágás üres.")

        x, y, width, height = cv2.boundingRect(points)

        # A megtartott előtérből új bináris képet készítünk.
        # Így a kizárt zaj a kivágás belsejében sem marad meg.
        binary = cv2.bitwise_not(foreground)
        cropped = binary[y:y + height, x:x + width]

        return cv2.copyMakeBorder(
            cropped,
            15,
            15,
            15,
            15,
            cv2.BORDER_CONSTANT,
            value=255,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--tesseract",
        default=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    )

    args = parser.parse_args()

    DepthSplitTest(args.tesseract).run(args.image)