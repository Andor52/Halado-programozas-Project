from dataclasses import dataclass
from enum import Enum


class PacketSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class PacketField(str, Enum):
    SEED_COUNT = "seed_count"
    PRICE_CATEGORY = "price_category"
    NAME = "name"
    VARIETY = "variety"
    DESCRIPTION_HU = "description_hu"
    PICTOGRAM_ROW = "pictogram_row"
    BARCODE = "barcode"


@dataclass(frozen=True)
class NormalizedRoi:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if not 0 <= self.left < self.right <= 1:
            raise ValueError("Érvénytelen vízszintes ROI-koordináták.")

        if not 0 <= self.top < self.bottom <= 1:
            raise ValueError("Érvénytelen függőleges ROI-koordináták.")

    def to_pixels(
        self,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("A képméreteknek pozitívnak kell lenniük.")

        x1 = min(image_width - 1, int(self.left * image_width))
        y1 = min(image_height - 1, int(self.top * image_height))

        x2 = min(
            image_width,
            max(x1 + 1, round(self.right * image_width)),
        )
        y2 = min(
            image_height,
            max(y1 + 1, round(self.bottom * image_height)),
        )

        return x1, y1, x2, y2


@dataclass
class PacketTemplate:
    name: str
    data_side: PacketSide
    regions: dict[PacketField, NormalizedRoi]


class PacketTemplateRepository:
    """A csomagolásokhoz tartozó ROI-sablonok tárolója."""

    def __init__(self) -> None:
        self._templates: dict[str, PacketTemplate] = {
            "moravoseed_limba_right": PacketTemplate(
                name="moravoseed_limba_right",
                data_side=PacketSide.RIGHT,
                regions={
                    PacketField.SEED_COUNT:
                        NormalizedRoi(0.005, 0.010, 0.130, 0.055),

                    PacketField.PRICE_CATEGORY:
                        NormalizedRoi(0.385, 0.005, 0.462, 0.100),

                    PacketField.NAME:
                        NormalizedRoi(0.155, 0.795, 0.315, 0.831),

                    PacketField.VARIETY:
                        NormalizedRoi(0.190, 0.830, 0.285, 0.864),

                    PacketField.DESCRIPTION_HU:
                        NormalizedRoi(0.510, 0.613, 0.990, 0.665),

                    PacketField.PICTOGRAM_ROW:
                        NormalizedRoi(0.509, 0.751, 0.987, 0.870),

                    PacketField.BARCODE:
                        NormalizedRoi(0.772, 0.894, 0.980, 0.988),
                },
            )
        }

    def get(self, name: str) -> PacketTemplate:
        try:
            return self._templates[name]
        except KeyError as exc:
            raise ValueError(f"Ismeretlen csomagolássablon: {name}") from exc

    def register(self, template: PacketTemplate) -> None:
        self._templates[template.name] = template