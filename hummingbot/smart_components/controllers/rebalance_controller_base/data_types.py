from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Set


class WeightingStrategyType(Enum):
    EQUAL = "EQUAL"
    MARKET_CAP = "MARKET_CAP"
    LIQUIDITY = "LIQUIDITY"


class WeightingStrategy(ABC):
    """
    An abstract base class that defines the interface for asset weighting strategies.
    """

    @abstractmethod
    def calculate_weights(self, assets: Set[str], data: Optional[Any]) -> Dict[str, float]:
        """
        Calculate and return the weights for the given assets based on provided data.

        :param assets: list of asset symbols
        :param data: dictionary with asset symbols as keys and relevant data as values
        :return: dict with asset symbols as keys and calculated weights as values
        """
        raise NotImplementedError


class EqualWeighting(WeightingStrategy):
    def calculate_weights(self, assets: Set[str], data: Optional[Any]) -> Dict[str, float]:
        n = len(assets)
        weight = 1 / n if n > 0 else 0  # Avoid division by zero
        return {asset: weight for asset in assets}


class MarketCapWeighting(WeightingStrategy):
    def calculate_weights(self, assets: Set[str], data: Optional[Any]) -> Dict[str, float]:
        if not data:
            raise ValueError("Market cap data is required for MarketCapWeighting")
        market_caps = data["market_caps"]
        total_market_cap = sum(market_caps.values())
        return {asset: market_caps[asset] / total_market_cap for asset in assets if total_market_cap > 0}


class LiquidityWeighting(WeightingStrategy):
    def calculate_weights(self, assets: Set[str], data: Optional[Any]) -> Dict[str, float]:
        if not data:
            raise ValueError("Trading volume data is required for LiquidityWeighting")
        trading_volumes = data["trading_volumes"]
        total_trading_volume = sum(trading_volumes.values())
        return {asset: trading_volumes[asset] / total_trading_volume for asset in assets if total_trading_volume > 0}


def get_weighting_strategy(strategy_type: WeightingStrategyType):
    if strategy_type == WeightingStrategyType.EQUAL:
        return EqualWeighting()
    elif strategy_type == WeightingStrategyType.MARKET_CAP:
        return MarketCapWeighting()
    elif strategy_type == WeightingStrategyType.LIQUIDITY:
        return LiquidityWeighting()
    else:
        raise ValueError(
            f"Invalid weighting strategy: {strategy_type}. Valid options are: {get_weighting_strategy_members()}"
        )


def get_weighting_strategy_members() -> str:
    return ", ".join(WeightingStrategyType.__members__)
