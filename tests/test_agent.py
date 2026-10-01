import numpy as np
import pytest
from sklearn.linear_model import SGDRegressor

from world.agent import NegotiationBrain, Observation
from world.goods import Goods


def make_observation(stock, target=10.0, production=0.0, consumption=0.0):
    return Observation(
        stock=stock, target_stock=target, production=production, consumption=consumption
    )


class FixedModel:
    """Stub regressor returning a constant, to isolate the pricing rule."""

    def __init__(self, value):
        self.value = value
        self.coef_ = np.zeros((1, 4))
        self.fitted = []

    def predict(self, features):
        return np.array([self.value] * len(features))

    def partial_fit(self, features, label):
        self.fitted.append((features, label))
        return self


class UnfittedModel:
    def predict(self, features):
        raise AssertionError("predict must not be called before fitting")

    def partial_fit(self, features, label):
        return self


def test_features_have_bias_stock_surplus_and_flow():
    features = make_observation(stock=15.0, target=10.0, production=3.0, consumption=1.0).features()

    assert features.shape == (1, 4)
    assert features[0][0] == 1.0
    assert features[0][1] == pytest.approx(1.5)
    assert features[0][2] == pytest.approx(0.5)
    assert features[0][3] == pytest.approx(0.2)


def test_scarcity_label_is_positive_when_short():
    assert make_observation(stock=5.0, target=10.0).scarcity_label() == pytest.approx(0.5)


def test_scarcity_label_is_zero_at_target():
    assert make_observation(stock=10.0, target=10.0).scarcity_label() == pytest.approx(0.0)


def test_scarcity_label_clamps_below_minus_one():
    assert make_observation(stock=100.0, target=10.0).scarcity_label() == pytest.approx(-1.0)


def test_zero_target_does_not_divide_by_zero():
    observation = make_observation(stock=5.0, target=0.0)

    assert observation.stock_ratio == 0.0
    assert observation.surplus_ratio == 0.0
    assert observation.net_flow_ratio == 0.0
    assert np.isfinite(observation.features()).all()


def test_predict_falls_back_to_label_before_any_update():
    brain = NegotiationBrain()
    brain.set_model(UnfittedModel())

    assert brain.predict_scarcity(make_observation(stock=5.0, target=10.0)) == pytest.approx(0.5)


def test_observe_sends_feature_and_label_to_the_model():
    brain = NegotiationBrain()
    model = FixedModel(0.0)
    brain.set_model(model)
    observation = make_observation(stock=5.0, target=10.0)

    brain.observe(observation)

    assert len(model.fitted) == 1
    features, label = model.fitted[0]
    assert features[0][1] == pytest.approx(0.5)
    assert label[0] == pytest.approx(0.5)


def test_weights_start_at_zero_and_change_after_observation():
    brain = NegotiationBrain()
    assert np.allclose(brain.weights, 0.0)

    for _ in range(20):
        brain.observe(make_observation(stock=4.0, target=10.0, production=0.0, consumption=2.0))

    assert not np.allclose(brain.weights, 0.0)


def test_price_rises_when_the_model_predicts_scarcity():
    scarce = NegotiationBrain()
    scarce.set_model(FixedModel(1.0))
    abundant = NegotiationBrain()
    abundant.set_model(FixedModel(0.0))
    observation = make_observation(stock=10.0, target=10.0)

    assert scarce.reservation_price(Goods.FARM, observation) > abundant.reservation_price(
        Goods.FARM, observation
    )


def test_price_falls_when_the_village_is_surplus():
    brain = NegotiationBrain()
    brain.set_model(FixedModel(0.0))
    balanced = make_observation(stock=10.0, target=10.0)
    surplus = make_observation(stock=30.0, target=10.0)

    assert brain.reservation_price(Goods.FARM, surplus) < brain.reservation_price(
        Goods.FARM, balanced
    )


def test_price_never_falls_below_the_floor():
    brain = NegotiationBrain()
    brain.set_model(FixedModel(-1.0))
    price = brain.reservation_price(Goods.FARM, make_observation(stock=1000.0, target=10.0))

    assert price == pytest.approx(Goods.FARM.base_value * brain.min_price_ratio)


def test_price_scales_with_base_value():
    brain = NegotiationBrain()
    brain.set_model(FixedModel(0.5))
    observation = make_observation(stock=10.0, target=10.0)

    farm = brain.reservation_price(Goods.FARM, observation)
    craft = brain.reservation_price(Goods.CRAFT, observation)

    assert craft / farm == pytest.approx(Goods.CRAFT.base_value / Goods.FARM.base_value)


def test_fair_rate_is_ratio_of_reservation_prices():
    brain = NegotiationBrain()
    brain.set_model(FixedModel(0.0))
    observation = make_observation(stock=10.0, target=10.0)

    rate = brain.fair_rate(Goods.FARM, Goods.CRAFT, observation, observation)
    expected = brain.reservation_price(Goods.CRAFT, observation) / brain.reservation_price(
        Goods.FARM, observation
    )

    assert rate == pytest.approx(expected)


def test_fair_rate_is_infinite_when_the_offered_goods_are_worthless():
    class WorthlessBrain(NegotiationBrain):
        """Values the offered goods at nothing, which the price floor normally prevents."""

        def reservation_price(self, goods, observation):
            if goods is Goods.FARM:
                return 0.0
            return super().reservation_price(goods, observation)

    brain = WorthlessBrain()
    brain.set_model(FixedModel(0.0))
    observation = make_observation(stock=10.0, target=10.0)

    assert brain.fair_rate(Goods.FARM, Goods.CRAFT, observation, observation) == float("inf")


def test_fair_rate_stays_finite_even_at_the_price_floor():
    brain = NegotiationBrain()
    brain.set_model(FixedModel(-1.0))
    observation = make_observation(stock=1000.0, target=10.0)

    rate = brain.fair_rate(Goods.FARM, Goods.CRAFT, observation, observation)

    assert np.isfinite(rate)


def test_brain_learns_scarcity_direction_over_repeated_scarcity():
    brain = NegotiationBrain(seed=7)
    for _ in range(50):
        brain.observe(make_observation(stock=2.0, target=10.0, production=0.0, consumption=1.0))

    prediction = brain.predict_scarcity(make_observation(stock=2.0, target=10.0))

    assert prediction > 0.5


def test_same_seed_gives_the_same_weights():
    def train(seed):
        brain = NegotiationBrain(seed=seed)
        for day in range(30):
            brain.observe(make_observation(stock=3.0 + day % 4, target=10.0, production=1.0))
        return brain.weights

    assert np.allclose(train(11), train(11))


def test_model_is_a_real_sgd_regressor_by_default():
    brain = NegotiationBrain()
    brain.observe(make_observation(stock=5.0, target=10.0))

    assert isinstance(brain._model, SGDRegressor)
