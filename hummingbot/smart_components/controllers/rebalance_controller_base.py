from abc import ABC, abstractmethod
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, validator

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.client.hummingbot_application import HummingbotApplication
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.smart_components.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.smart_components.executors.position_executor.data_types import TripleBarrierConfig
from hummingbot.smart_components.models.executor_actions import ExecutorAction, StopExecutorAction


class WeightingStrategyType(Enum):
    EQUAL = "EQUAL"
    MARKET_CAP = "MARKET_CAP"
    LIQUIDITY = "LIQUIDITY"


class WeightingStrategy(ABC):
    """
    An abstract base class that defines the interface for asset weighting strategies.
    """

    @staticmethod
    def get_weighting_strategy(strategy_type: WeightingStrategyType):
        return {
            WeightingStrategyType.EQUAL: EqualWeighting(),
            WeightingStrategyType.MARKET_CAP: MarketCapWeighting(),
            WeightingStrategyType.LIQUIDITY: LiquidityWeighting(),
        }[strategy_type]

    @staticmethod
    def get_weighting_strategy_members() -> str:
        return ", ".join(WeightingStrategyType.__members__)

    @abstractmethod
    def calculate_weights(self, assets: List[str], data: Optional[Any]) -> Dict[str, float]:
        """
        Calculate and return the weights for the given assets based on provided data.

        :param assets: list of asset symbols
        :param data: dictionary with asset symbols as keys and relevant data as values
        :return: dict with asset symbols as keys and calculated weights as values
        """
        raise NotImplementedError


class RebalanceControllerConfigBase(ControllerConfigBase):
    controller_type = "rebalance"
    connector_name: str = Field(
        default="kraken",
        client_data=ClientFieldData(
            prompt_on_new=True, prompt=lambda mi: "Enter the name of the exchange to trade on (e.g., kraken):"
        ),
    )
    max_long_positions: int = Field(
        default=5,
        client_data=ClientFieldData(
            prompt_on_new=True, prompt=lambda mi: "Enter the maximum number of long positions to hold:"
        ),
    )
    weighting_strategy: WeightingStrategyType = Field(
        default="EQUAL",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the weighting strategy to use (EQUAL/MARKET_CAP/LIQUIDITY):",
        ),
    )
    target_quote_asset: str = Field(
        default="USD",
        client_data=ClientFieldData(
            prompt_on_new=True, prompt=lambda mi: "Enter the target quote asset for the portfolio:"
        ),
    )
    executor_amount_quote: Decimal = Field(
        default=100.0,
        client_data=ClientFieldData(
            prompt_on_new=True, prompt=lambda mi: "Enter the amount of quote asset to use per executor (e.g., 100):"
        ),
    )

    # Test mode always returns 1 in get_signal, and does not execute orders
    test_mode: bool = Field(
        default=False,
        client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Run in test mode? (Yes/No) "),
    )

    @property
    def triple_barrier_config(self) -> TripleBarrierConfig:
        return TripleBarrierConfig(
            stop_loss=None,
            take_profit=None,
            time_limit=None,
            trailing_stop=None,
            open_order_type=OrderType.MARKET,
            take_profit_order_type=OrderType.MARKET,
            stop_loss_order_type=OrderType.MARKET,  # Defaulting to MARKET as per requirement
            time_limit_order_type=OrderType.MARKET,  # Defaulting to MARKET as per requirement
        )

    @validator("weighting_strategy", pre=True, allow_reuse=True, always=True)
    def validate_weighting_strategy(cls, v) -> WeightingStrategyType:
        if isinstance(v, WeightingStrategyType):
            return v
        elif v is None:
            return WeightingStrategyType.EQUAL
        elif isinstance(v, str):
            try:
                return WeightingStrategyType[v.upper()]
            except KeyError:
                raise ValueError(
                    f"Invalid weighting strategy: {v}. Valid options are: {WeightingStrategy.get_weighting_strategy_members()}"
                )
        raise ValueError(
            f"Invalid weighting strategy: {v}. Valid options are: {WeightingStrategy.get_weighting_strategy_members()}"
        )

    @validator("test_mode", pre=True, always=True)
    def parse_test_mode(cls, v: Any, values: Dict[str, Any]) -> bool:
        return str(v).lower() in {"true", "yes", "y"}


class RebalanceController(ControllerBase):
    def __init__(self, config: RebalanceControllerConfigBase, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config

    def get_weighting_strategy(self) -> WeightingStrategy:
        return WeightingStrategy.get_weighting_strategy(self.config.weighting_strategy)

    def determine_executor_actions(self) -> List[ExecutorAction]:
        """
        Determine actions based on the provided executor handler report.
        """
        actions = []
        actions.extend(self.create_actions_proposal())
        actions.extend(self.stop_actions_proposal())
        return actions

    async def update_processed_data(self):
        """
        Update the processed data based on the current state of the strategy.
        """
        signal = self.get_signal()
        self.processed_data = {"signal": signal}

    def get_signal(self) -> int:
        """
        Get the signal for the strategy. 1 to indicate a rebalance, 0 for no action, -1 to stop the strategy.
        """
        raise NotImplementedError

    def get_portfolio_assets(self) -> Dict[str, Decimal]:
        """
        Get the assets in the portfolio.
        """
        hb = HummingbotApplication.main_application()
        balances = hb.markets[self.config.connector_name].get_all_balances()
        return {asset: Decimal(balance) for asset, balance in balances.items()}

    def get_portfolio_total_value(self) -> Decimal:
        """
        Get the total value of the portfolio.
        """
        raise NotImplementedError

    @abstractmethod
    def get_target_assets(self) -> List[str]:
        """
        Get the rebalance assets for the strategy.
        """
        raise NotImplementedError

    @abstractmethod
    def get_weighting_strategy_data(self) -> Optional[Any]:
        """
        Get additional data needed for the strategy to calculate weightings.
        """
        raise NotImplementedError

    def calculate_weights(self):
        """
        Calculate and return the weights for the given assets based on provided data.

        :param assets: list of asset symbols
        :param data: dictionary with asset symbols as keys and relevant data as values
        :return: dict with asset symbols as keys and calculated weights as values
        """
        target_assets = self.get_target_assets()
        data = self.get_weighting_strategy_data()
        weighting_strategy = self.get_weighting_strategy()
        weights = weighting_strategy.calculate_weights(assets=target_assets, data=data)
        return weights

    def create_actions_proposal(self) -> List[ExecutorAction]:
        """
        Create actions based on signal.
        """
        create_actions = []
        signal = self.processed_data["signal"]
        if signal > 0:
            weights = self.calculate_weights()
            for asset, target_weight in weights.items():
                pair = f"{asset}-{self.config.target_quote_asset}"
                # Get the latest price of the asset
                price = self.market_data_provider.get_price_by_type(
                    self.config.connector_name, pair, PriceType.MidPrice
                )

                # Calculate the current value of the asset
                current_value = self.get_portfolio_assets().get(asset, Decimal(0))

                # Calculate the target value of the asset
                target_value = current_value * Decimal(target_weight)

                # Calculate the amount of the asset to buy
                target_amount = target_value / price
                self.logger().info(
                    f"Creating buy action for {asset} with target amount {target_amount} at price {price}"
                )

                # Create the action

        return create_actions

    def can_create_executor(self, signal: int) -> bool:
        active_long_positions = self.filter_executors(
            executors=self.executors_info, filter_func=lambda x: x.is_active and x.side == TradeType.BUY
        )
        return len(active_long_positions) < self.config.max_long_positions

    def stop_actions_proposal(self) -> List[ExecutorAction]:
        """
        Stop actions based on the provided executor handler report.
        """
        stop_actions = []

        # Get the signal
        signal = self.processed_data["signal"]

        # Get the active long positions
        active_long_positions = self.filter_executors(
            executors=self.executors_info, filter_func=lambda x: x.is_active and x.side == TradeType.BUY
        )

        # Stop the long positions if the signal is negative
        if signal < 0:
            stop_actions.extend([StopExecutorAction(executor_id=e.id) for e in active_long_positions])

        return stop_actions


class EqualWeighting(WeightingStrategy):
    def calculate_weights(self, assets, data):
        n = len(assets)
        weight = 1 / n if n > 0 else 0  # Avoid division by zero
        return {asset: weight for asset in assets}


class MarketCapWeighting(WeightingStrategy):
    def calculate_weights(self, assets, data):
        if not data:
            raise ValueError("Market cap data is required for MarketCapWeighting")
        market_caps = data["market_caps"]
        total_market_cap = sum(market_caps.values())
        return {asset: market_caps[asset] / total_market_cap for asset in assets if total_market_cap > 0}


class LiquidityWeighting(WeightingStrategy):
    def calculate_weights(self, assets, data):
        if not data:
            raise ValueError("Trading volume data is required for LiquidityWeighting")
        trading_volumes = data["trading_volumes"]
        total_trading_volume = sum(trading_volumes.values())
        return {asset: trading_volumes[asset] / total_trading_volume for asset in assets if total_trading_volume > 0}
