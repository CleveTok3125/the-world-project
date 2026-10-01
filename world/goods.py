"""Commodity definitions shared by every village."""

from __future__ import annotations

from enum import Enum


class Goods(Enum):
    """The three commodity families traded in the world."""

    FARM = "farm"
    MINERAL = "mineral"
    CRAFT = "craft"

    @property
    def label(self) -> str:
        """Human readable name of the commodity."""
        return {
            Goods.FARM: "Agriculture",
            Goods.MINERAL: "Minerals",
            Goods.CRAFT: "Handicrafts",
        }[self]

    @property
    def base_value(self) -> float:
        """Reference price of one unit when supply and demand are balanced."""
        return {
            Goods.FARM: 1.0,
            Goods.MINERAL: 1.5,
            Goods.CRAFT: 2.5,
        }[self]

    @property
    def daily_consumption(self) -> float:
        """Units one person uses per day."""
        return 1.0

    @property
    def storage_limit(self) -> float:
        """Maximum units a village can store before spoilage losses start."""
        return 1e9


ALL_GOODS: tuple[Goods, ...] = tuple(Goods)


def zero_stock() -> dict[Goods, float]:
    """A stock dictionary with every commodity set to zero."""
    return {goods: 0.0 for goods in Goods}
