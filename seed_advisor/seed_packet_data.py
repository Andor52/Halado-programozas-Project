from dataclasses import asdict, dataclass, field
from .barcode_reader import BarcodeResult
from .pictogram_extractor import PictogramResult
import json
from pathlib import Path


@dataclass
class FieldRecognition:
    raw_text: str
    mean_confidence: float | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class SeedPacketData:
    seed_count: int | None = None
    price_category: str | None = None
    name: str | None = None
    variety: str | None = None
    description_hu: str | None = None

    barcode: BarcodeResult | None = None
    pictograms: list[PictogramResult] = field(default_factory=list)

    recognition: dict[str, FieldRecognition] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
        )

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")