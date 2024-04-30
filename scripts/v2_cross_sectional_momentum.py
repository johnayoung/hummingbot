import os
from abc import ABC, abstractmethod
from typing import Dict, List

from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig
from hummingbot.smart_components.models.executor_actions import CreateExecutorAction
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase


class WeightingStrategy(ABC):
    """
    An abstract base class that defines the interface for asset weighting strategies.
    """

    @abstractmethod
    def calculate_weights(self, assets, data):
        """
        Calculate and return the weights for the given assets based on provided data.

        :param assets: list of asset symbols
        :param data: dictionary with asset symbols as keys and relevant data as values
        :return: dict with asset symbols as keys and calculated weights as values
        """
        pass


class EqualWeighting(WeightingStrategy):
    def calculate_weights(self, assets, data):
        n = len(assets)
        weight = 1 / n if n > 0 else 0  # Avoid division by zero
        return {asset: weight for asset in assets}


class MarketCapWeighting(WeightingStrategy):
    def calculate_weights(self, assets, market_caps):
        total_market_cap = sum(market_caps.values())
        return {asset: market_caps[asset] / total_market_cap for asset in assets if total_market_cap > 0}


class LiquidityWeighting(WeightingStrategy):
    def calculate_weights(self, assets, trading_volumes):
        total_volume = sum(trading_volumes.values())
        return {asset: trading_volumes[asset] / total_volume for asset in assets if total_volume > 0}


class CrossSectionalMomentumConfig(StrategyV2ConfigBase):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    candles_config: List[CandlesConfig] = []
    controllers_config: List[str] = []

    # Momentum duration calculation
    lookback_period: int = Field(
        default=30,
        gt=0,
        client_data=ClientFieldData(prompt=lambda mi: "Enter the lookback period (e.g. 30): ", prompt_on_new=True),
    )
    # The frequency with which the portfolio is rebalanced to include new top-performing assets
    rebalance_frequency: int = Field(
        default=1,
        gt=0,
        client_data=ClientFieldData(prompt=lambda mi: "Enter the rebalance frequency (e.g. 1): ", prompt_on_new=True),
    )
    # The number of top assets to include in the portfolio + rebalancing
    number_of_top_assets: int = Field(
        default=5,
        gt=0,
        client_data=ClientFieldData(prompt=lambda mi: "Enter the number of top assets (e.g. 5): ", prompt_on_new=True),
    )
    # Minimum average daily trading volume threshold to ensure selected assets have sufficient liquidity
    volume_filter: int = Field(
        default=1000000,
        gt=0,
        client_data=ClientFieldData(prompt=lambda mi: "Enter the volume filter (e.g. 1000000): ", prompt_on_new=True),
    )
    # The method by which assets are weighted within the portfolio.
    weighting_strategy: WeightingStrategy = EqualWeighting()

    max_drawdown: float = Field(
        default=0.1,
        gt=0,
        client_data=ClientFieldData(prompt=lambda mi: "Enter the maximum drawdown (e.g. 0.1): ", prompt_on_new=True),
    )

    # rsi_period: int = Field(
    #     default=14,
    #     gt=0,
    #     client_data=ClientFieldData(prompt=lambda mi: "Enter the RSI period (e.g. 14): ", prompt_on_new=True),
    # )


class CrossSectionalMomentum(StrategyV2Base):
    account_config_set = False

    def __init__(self, connectors: Dict[str, ConnectorBase], config: CrossSectionalMomentumConfig):
        super().__init__(connectors, config)
        self.config = config

    def start(self, clock: Clock, timestamp: float) -> None:
        """
        Start the strategy.
        :param clock: Clock to use.
        :param timestamp: Current time.
        """
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        create_actions = []

        return create_actions

    def apply_initial_setting(self):
        pass
