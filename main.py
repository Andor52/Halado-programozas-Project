import argparse
from pathlib import Path

import cv2
import numpy as np

from seed_advisor.image_loading import PacketImageLoader
from seed_advisor.ocr_engine import OcrEngine
from seed_advisor.packet_regions import PacketTemplateRepository
from seed_advisor.region_extractor import PacketRegionExtractor
from seed_advisor.seed_packet_data import SeedPacketData
from seed_advisor.seed_text_extractor import SeedTextExtractor
from seed_advisor.barcode_reader import BarcodeReader
from seed_advisor.packet_regions import PacketField
from seed_advisor.pictogram_extractor import (
    PictogramExtractor,
    PictogramTemplateRepository,
)


class SeedPacketApplication:
    def __init__(self) -> None:
        self.loader = PacketImageLoader()
        self.templates = PacketTemplateRepository()
        self.region_extractor = PacketRegionExtractor()

        self.ocr_engine = OcrEngine(
            tesseract_command=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
        self.text_extractor = SeedTextExtractor(self.ocr_engine)
        self.barcode_reader = BarcodeReader()

        self.pictogram_templates = PictogramTemplateRepository()
        self.pictogram_extractor = PictogramExtractor(
            self.ocr_engine
        )

    def run(
        self,
        image_path: str,
        template_name: str,
        show_preview: bool = True,
    ) -> None:
        # Környezet ellenőrzése
        self.ocr_engine.check_environment()

        # Kép és ROI-sablon betöltése
        packet = self.loader.load(image_path)
        template = self.templates.get(template_name)

        output_directory = (
            Path("output")
            / packet.source_path.stem
            / template.name
        )
        output_directory.mkdir(parents=True, exist_ok=True)

        print(f"Kép: {packet.source_path.name}")
        print(f"Méret: {packet.width} × {packet.height} pixel")
        print(f"Sablon: {template.name}")
        print(f"Adatpanel oldala: {template.data_side.value}")

        # Régiók kivágása és mentése
        crops = self.region_extractor.extract(
            packet.image,
            template,
        )
        self.region_extractor.save(
            crops,
            output_directory,
        )

        print(f"Kivágott mezők száma: {len(crops)}")

        # Keretekkel ellátott ellenőrző kép mentése
        preview = self.region_extractor.draw_preview(
            packet.image,
            template,
        )
        self.region_extractor.save_image(
            output_directory / "preview.png",
            preview,
        )

        # Szöveges mezők felismerése
        print("\nOCR-feldolgozás...")

        seed_data = self.text_extractor.extract(crops)

        # Vonalkód dekódolása
        seed_data.barcode = self.barcode_reader.read(
            crops.get(PacketField.BARCODE)
        )

        # Piktogramok kivágása és felirataik OCR-feldolgozása
        pictogram_regions = self.pictogram_templates.get(
            template_name
        )

        seed_data.pictograms = self.pictogram_extractor.extract(
            row_image=crops[PacketField.PICTOGRAM_ROW],
            regions=pictogram_regions,
            output_directory=output_directory / "pictograms",
        )

        # Csak a teljes feldolgozás után mentünk.
        json_path = output_directory / "seed_data.json"
        seed_data.save_json(json_path)

        # Eredmények és diagnosztika
        self._print_results(seed_data)
        self._print_diagnostics(seed_data)

        print(f"\nKimeneti könyvtár: {output_directory.resolve()}")
        print(f"JSON-adatok: {json_path.resolve()}")

        # Opcionális grafikus előnézet
        if show_preview:
            self._show_preview(preview)

    @staticmethod
    def _print_results(seed_data: SeedPacketData) -> None:
        print("\n--- Felismert adatok ---")
        print(f"Magszám: {seed_data.seed_count}")
        print(f"Árkategória: {seed_data.price_category}")
        print(f"Név: {seed_data.name}")
        print(f"Fajta: {seed_data.variety}")
        print(f"Magyar leírás: {seed_data.description_hu}")
        print("\n--- Vonalkód ---")

        if seed_data.barcode is not None:
            print(f"Érték: {seed_data.barcode.value}")
            print(f"Formátum: {seed_data.barcode.format}")

            for warning in seed_data.barcode.warnings:
                print(f"Figyelmeztetés: {warning}")

        print("\n--- Piktogramok ---")

        for pictogram in seed_data.pictograms:
            print(f"\n{pictogram.identifier}")
            print(f"  Képbe írt szöveg: {pictogram.picture_text!r}")
            print(f"  Alsó felirat: {pictogram.caption_text!r}")
            print(f"  Jelentés: {pictogram.meaning}")
            print(f"  Piros X: {pictogram.red_x_detected}")

            for warning in pictogram.warnings:
                print(f"  Figyelmeztetés: {warning}")

        print(f"  Felirattípus: {pictogram.caption_type}")
        print(f"  Normalizált felirat: {pictogram.caption_normalized}")
        print(f"  Hónapok: {pictogram.months}")
        if pictogram.depth is not None:
            depth = pictogram.depth

            print(f"  Mélység módszere: {depth.method}")
            print(f"  Mélység nyers szöveg: {depth.raw_text!r}")
            print(f"  Mélység minimum: {depth.min_cm} cm")
            print(f"  Mélység maximum: {depth.max_cm} cm")

            if depth.method == "split_single_digit":
                print(
                    f"  Szám OCR: {depth.number_raw_text!r}, "
                    f"konfidencia: {depth.number_confidence}"
                )
                print(
                    f"  Mértékegység OCR: {depth.unit_raw_text!r}, "
                    f"konfidencia: {depth.unit_confidence}"
                )
            else:
                print(
                    f"  OCR-konfidencia: {depth.mean_confidence}"
                )

            for warning in depth.warnings:
                print(f"  Mélységfigyelmeztetés: {warning}")

    @staticmethod
    def _print_diagnostics(seed_data: SeedPacketData) -> None:
        print("\n--- OCR-diagnosztika ---")

        for field_name, recognition in seed_data.recognition.items():
            print(f"\n{field_name}")
            print(f"  Nyers szöveg: {recognition.raw_text!r}")
            print(
                "  Átlagos konfidencia: "
                f"{recognition.mean_confidence}"
            )

            for warning in recognition.warnings:
                print(f"  Figyelmeztetés: {warning}")

    @staticmethod
    def _show_preview(image: np.ndarray) -> None:
        height, width = image.shape[:2]

        scale = min(
            1100 / width,
            800 / height,
            1.0,
        )

        display = cv2.resize(
            image,
            (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

        try:
            cv2.imshow(
                "ROI preview - press any key",
                display,
            )
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Magcsomag mezőinek kivágása ROI-sablonnal "
            "és a szöveges adatok felismerése OCR-rel."
        )
    )

    parser.add_argument(
        "image",
        help="A magcsomagfotó elérési útja",
    )

    parser.add_argument(
        "--template",
        default="moravoseed_limba_right",
        help=(
            "A használni kívánt ROI-sablon neve "
            "(alapértelmezett: moravoseed_limba_right)"
        ),
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Ne nyisson meg grafikus előnézeti ablakot",
    )

    args = parser.parse_args()

    application = SeedPacketApplication()
    application.run(
        image_path=args.image,
        template_name=args.template,
        show_preview=not args.no_preview,
    )