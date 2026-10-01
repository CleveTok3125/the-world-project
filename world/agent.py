"""Adaptive negotiation model owned by each village.

The brain is a small online linear regressor. Every day it is fed one observation
per commodity and it updates its own weight vector, so the price a village is
willing to pay drifts with the scarcity it actually experiences.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import SGDRegressor

from world.goods import Goods

FEATURE_COUNT = 4
BIAS_INDEX = 0
SCARCITY_INDEX = 1
SUPPLY_INDEX = 2
FLOW_INDEX = 3


@dataclass(frozen=True)
class Observation:
    """Everything a village knows about one commodity on one day."""

    stock: float
    target_stock: float
    production: float
    consumption: float

    @property
    def stock_ratio(self) -> float:
        return self.stock / self.target_stock if self.target_stock else 0.0

    @property
    def surplus_ratio(self) -> float:
        if not self.target_stock:
            return 0.0
        return (self.stock - self.target_stock) / self.target_stock

    @property
    def net_flow_ratio(self) -> float:
        if not self.target_stock:
            return 0.0
        return (self.production - self.consumption) / self.target_stock

    def features(self) -> np.ndarray:
        return np.array(
            [
                [1.0, self.stock_ratio, self.surplus_ratio, self.net_flow_ratio],
            ],
            dtype=float,
        )

    def scarcity_label(self) -> float:
        """Observed shortage, positive when the village is short of the commodity."""
        return max(-1.0, 1.0 - self.stock_ratio)


@dataclass
class NegotiationBrain:
    """Online regressor plus the pricing rule derived from its prediction.

    Attributes:
        seed: Controls the regressor so a run can be reproduced exactly.
        scarcity_gain: How strongly a predicted shortage raises the offered price.
        supply_discount: How strongly a surplus lowers the demanded price.
        min_price_ratio: Floor for a price, expressed as a share of the base value.
        _model: The underlying regressor, replaced wholesale by the tests.
    """

    seed: int = 0
    scarcity_gain: float = 1.2
    supply_discount: float = 0.8
    min_price_ratio: float = 0.25
    _model: SGDRegressor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._model = self._build_model()

    def _build_model(self) -> SGDRegressor:
        return SGDRegressor(
            loss="squared_error",
            learning_rate="adaptive",
            eta0=0.05,
            power_t=0.5,
            random_state=self.seed,
            max_iter=1,
            tol=None,
            shuffle=False,
        )

    @property
    def weights(self) -> np.ndarray:
        if not hasattr(self._model, "coef_"):
            return np.zeros(FEATURE_COUNT, dtype=float)
        return self._model.coef_.ravel()

    def set_model(self, model: SGDRegressor) -> None:
        """Replace the regressor, used by the tests to inject a fixed model."""
        self._model = model

    def predict_scarcity(self, observation: Observation) -> float:
        """Estimated shortage of the observed commodity, clamped to ``[-1, 1]``."""
        if not hasattr(self._model, "coef_"):
            return observation.scarcity_label()
        raw = float(self._model.predict(observation.features())[0])
        return float(np.clip(raw, -1.0, 1.0))

    def observe(self, observation: Observation) -> float:
        """Update the weights from one day of experience, then return the prediction."""
        features = observation.features()
        label = observation.scarcity_label()
        self._model.partial_fit(features, np.array([label], dtype=float))
        return self.predict_scarcity(observation)

    def reservation_price(self, goods: Goods, observation: Observation) -> float:
        """Lowest price this village accepts for one unit of ``goods``."""
        scarcity = self.predict_scarcity(observation)
        pressure = max(0.0, observation.surplus_ratio)
        ratio = 1.0 + self.scarcity_gain * scarcity - self.supply_discount * pressure
        return goods.base_value * max(self.min_price_ratio, ratio)

    def fair_rate(
        self,
        offered: Goods,
        observed: Goods,
        own_offer_state: Observation,
        own_want_state: Observation,
    ) -> float:
        """Units of ``observed`` this village deems fair per unit of ``offered``."""
        own_value = self.reservation_price(offered, own_offer_state)
        wanted_value = self.reservation_price(observed, own_want_state)
        if own_value <= 0:
            return float("inf")
        return max(0.0, wanted_value / own_value)
