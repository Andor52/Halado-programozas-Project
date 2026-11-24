from dataclasses import dataclass
from enum import Enum

import numpy as np


class PacketSide(Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class PacketLayout:
    side: PacketSide

    @property
    def data_is_on_left(self) -> bool:
        return self.side is PacketSide.LEFT

    @property
    def data_is_on_right(self) -> bool:
        return self.side is PacketSide.RIGHT


class PacketLayoutDetector:
    """A magcsomag adatpaneljének oldalát kezeli.

    Az MVP-ben a felhasználó adja meg. Később itt lehet
    képalapú automatikus felismerés, például szövegsűrűség,
    vonalkód-detektálás vagy YOLO alapján.
    """

    def detect(self, side: str | PacketSide) -> PacketLayout:
        if isinstance(side, PacketSide):
            return PacketLayout(side=side)

        normalized = side.strip().lower()

        if normalized in {"bal", "left", "l"}:
            return PacketLayout(side=PacketSide.LEFT)

        if normalized in {"jobb", "jobbos", "right", "r"}:
            return PacketLayout(side=PacketSide.RIGHT)

        raise ValueError("Az oldal csak 'bal' vagy 'jobb' lehet.")