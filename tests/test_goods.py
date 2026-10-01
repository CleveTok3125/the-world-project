import pytest

from world.goods import ALL_GOODS, Goods, zero_stock


def test_three_commodity_families():
    assert {goods for goods in Goods} == {
        Goods.FARM,
        Goods.MINERAL,
        Goods.CRAFT,
    }
    assert len(ALL_GOODS) == 3


def test_every_goods_has_the_expected_label():
    assert {goods: goods.label for goods in Goods} == {
        Goods.FARM: "Agriculture",
        Goods.MINERAL: "Minerals",
        Goods.CRAFT: "Handicrafts",
    }


def test_every_goods_has_distinct_label_and_positive_value():
    labels = [goods.label for goods in Goods]
    values = [goods.base_value for goods in Goods]

    assert len(set(labels)) == len(Goods)
    assert all(label for label in labels)
    assert all(value > 0 for value in values)


def test_daily_consumption_is_positive():
    assert all(goods.daily_consumption > 0 for goods in Goods)


def test_values_order_minerals_above_farm_and_crafts_above_minerals():
    assert Goods.MINERAL.base_value > Goods.FARM.base_value
    assert Goods.CRAFT.base_value > Goods.MINERAL.base_value


def test_zero_stock_covers_every_goods():
    stock = zero_stock()

    assert set(stock) == set(Goods)
    assert all(qty == 0 for qty in stock.values())


def test_zero_stock_returns_independent_dicts():
    first = zero_stock()
    second = zero_stock()
    first[Goods.FARM] = 5.0

    assert second[Goods.FARM] == 0.0


def test_goods_value_is_distinct_per_member():
    with pytest.raises(ValueError):
        Goods("unknown")
