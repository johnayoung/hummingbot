import time
from abc import ABC, abstractmethod
from decimal import Decimal
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Set

from pydantic import Field, validator

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.client.hummingbot_application import HummingbotApplication
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.smart_components.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.smart_components.executors.position_executor.data_types import (
    PositionExecutorConfig,
    TripleBarrierConfig,
)
from hummingbot.smart_components.models.executor_actions import CreateExecutorAction, ExecutorAction


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
    target_quote_weight: float = Field(
        default=1.0,
        client_data=ClientFieldData(
            prompt_on_new=True, prompt=lambda mi: "Enter the target weight for the quote asset:"
        ),
    )
    minimum_order_size: Decimal = Field(
        default=Decimal("0.01"),
        client_data=ClientFieldData(
            prompt_on_new=True, prompt=lambda mi: "Enter the minimum order size in quote asset for the exchange:"
        ),
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


@unique
class PortfolioStatus(str, Enum):
    IDLE = "IDLE"
    INITIATE_REBALANCE = "INITIATE_REBALANCE"
    REBALANCE_INITIATE_CLOSE_EXISTING_POSITIONS = "REBALANCE_INITIATE_CLOSE_EXISTING_POSITIONS"
    REBALANCE_CLOSE_EXISTING_POSITIONS_FINALIZED = "REBALANCE_CLOSE_EXISTING_POSITIONS_FINALIZED"
    REBALANCE_INITIATE_OPEN_NEW_POSITIONS = "REBALANCE_INITIATE_OPEN_NEW_POSITIONS"
    REBALANCE_OPEN_NEW_POSITIONS_FINALIZED = "REBALANCE_OPEN_NEW_POSITIONS_FINALIZED"
    INITIATE_CLOSE_TO_QUOTE = "INITIATE_CLOSE_TO_QUOTE"
    CLOSING_POSITIONS = "CLOSING_POSITIONS"


class RebalanceControllerBase(ControllerBase):
    def __init__(self, config: RebalanceControllerConfigBase, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.portfolio_status: PortfolioStatus = PortfolioStatus.IDLE

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

    def get_weighting_strategy(self) -> WeightingStrategy:
        return WeightingStrategy.get_weighting_strategy(self.config.weighting_strategy)

    async def update_processed_data(self):
        """
        Update the processed data for the controller. This includes checking the status of orders
        and deciding if a rebalance should be initiated.
        """
        pass

    def determine_executor_actions(self) -> List[ExecutorAction]:
        actions = []
        actions.extend(self.create_rebalance_proposal())

        # Strip actions of None values
        return [action for action in actions if action is not None]

    def create_rebalance_proposal(self) -> List[ExecutorAction]:
        """
        Rebalance the portfolio.
        """
        signal = self.processed_data.get("signal", 0)
        initiate_rebalance_condition = signal > 0 and self.portfolio_status == PortfolioStatus.IDLE
        initiate_close_to_quote_condition = signal < 0 and self.portfolio_status == PortfolioStatus.IDLE

        # ------------------------------------------------------------
        # Rebalance the portfolio
        # ------------------------------------------------------------

        if initiate_rebalance_condition:
            target_weights = self.calculate_weights()  # Get the new weights for the assets

            # Assets to close
            current_positions = self.get_portfolio_assets()
            target_symbols = list(target_weights.keys())
            symbols_to_close = self.symbols_to_close(current_positions, target_symbols)

            # Close positions
            if len(symbols_to_close) > 0:
                actions = []
                for symbol in symbols_to_close:
                    actions.append(self.rebalance_asset(symbol, 0, self.get_rebalance_equity()))

                self.portfolio_status = PortfolioStatus.REBALANCE_INITIATE_CLOSE_EXISTING_POSITIONS
                return actions

        if self.portfolio_status == PortfolioStatus.REBALANCE_INITIATE_CLOSE_EXISTING_POSITIONS:
            active_close_positions = self.get_active_close_positions()
            if len(active_close_positions) == 0:
                self.portfolio_status = PortfolioStatus.REBALANCE_CLOSE_EXISTING_POSITIONS_FINALIZED
                return []

        if self.portfolio_status == PortfolioStatus.REBALANCE_CLOSE_EXISTING_POSITIONS_FINALIZED:
            target_weights = self.calculate_weights()  # Get the new weights for the assets
            actions = []
            for asset, target_weight in target_weights.items():
                actions.append(self.rebalance_asset(asset, target_weight, self.get_rebalance_equity()))

            self.portfolio_status = PortfolioStatus.REBALANCE_INITIATE_OPEN_NEW_POSITIONS
            return actions

        if self.portfolio_status == PortfolioStatus.REBALANCE_INITIATE_OPEN_NEW_POSITIONS:
            active_open_positions = self.get_active_open_positions()
            if len(active_open_positions) == 0:
                self.portfolio_status = PortfolioStatus.REBALANCE_OPEN_NEW_POSITIONS_FINALIZED
                return []

        if self.portfolio_status == PortfolioStatus.REBALANCE_OPEN_NEW_POSITIONS_FINALIZED:
            self.portfolio_status = PortfolioStatus.IDLE
            return []

        # ------------------------------------------------------------
        # Close all positions to quote asset
        # ------------------------------------------------------------

        if initiate_close_to_quote_condition:
            self.portfolio_status = PortfolioStatus.CLOSING_POSITIONS
            return self.close_all_positions()

        if self.portfolio_status == PortfolioStatus.CLOSING_POSITIONS:
            active_close_positions = self.get_active_close_positions()
            if len(active_close_positions) == 0:
                self.portfolio_status = PortfolioStatus.IDLE
                return []

        return []

    def close_all_positions(self):
        """
        Close all active positions.
        """
        active_positions = self.get_portfolio_assets()
        actions = []

        for asset, _ in active_positions.items():
            if asset == self.config.target_quote_asset:
                continue
            actions.append(self.rebalance_asset(asset, 0, self.get_rebalance_equity()))

        return actions

    def get_rebalance_action_details(self, asset: str, target_weight: float, rebalance_value: Decimal):
        """
        Get the details for rebalancing the asset to the target weight.
        """
        pair = self.generate_trading_pair(asset)
        price = self.market_data_provider.get_price_by_type(self.config.connector_name, pair, PriceType.MidPrice)
        current_value = self.get_portfolio_assets().get(asset, Decimal(0))
        target_value = rebalance_value * Decimal(target_weight)
        delta_value = target_value - current_value
        target_amount = abs(delta_value / price)
        trade_type = TradeType.BUY if delta_value > 0 else TradeType.SELL
        return {
            "pair": pair,
            "price": price,
            "current_value": current_value,
            "target_value": target_value,
            "delta_value": delta_value,
            "target_amount": target_amount,
            "trade_type": trade_type,
            "can_rebalance": target_amount >= self.config.minimum_order_size,
        }

    def rebalance_asset(self, asset: str, target_weight: float, rebalance_value: Decimal):
        """
        Create an action to rebalance the asset to the target weight.
        """
        details = self.get_rebalance_action_details(asset, target_weight, rebalance_value)
        if not details["can_rebalance"]:
            self.logger().info(f"Skipping rebalance for {asset} as target amount is below minimum order size")
            return None

        # Create the action
        action = CreateExecutorAction(
            controller_id=self.config.id,
            executor_config=PositionExecutorConfig(
                type="position",
                timestamp=time.time(),
                connector_name=self.config.connector_name,
                trading_pair=details["pair"],
                side=details["trade_type"],
                entry_price=details["price"],
                amount=details["target_amount"],
                triple_barrier_config=self.config.triple_barrier_config,
                leverage=1,
            ),
        )
        self.logger().info(
            f"Proposing {'buy' if details['trade_type'] == TradeType.BUY else 'sell'} action for {asset} with target amount {details['target_amount']} at price {details['price']}"
        )
        return action

    def get_active_open_positions(self):
        return self.filter_executors(
            executors=self.executors_info, filter_func=lambda x: x.is_active and x.side == TradeType.BUY
        )

    def get_active_close_positions(self):
        return self.filter_executors(
            executors=self.executors_info, filter_func=lambda x: x.is_active and x.side == TradeType.SELL
        )

    def generate_trading_pair(self, asset: str) -> str:
        """
        Get the trading pairs for the strategy.
        """
        return f"{asset}-{self.config.target_quote_asset}"

    def get_asset_price(self, connector_name: str, trading_pair: str) -> Decimal:
        """
        Fetches the asset price, ensuring the trading pair is initialized.
        """
        try:
            price = self.market_data_provider.get_price_by_type(connector_name, trading_pair, PriceType.MidPrice)
            return price
        except Exception as e:
            self.logger().error(f"Failed to fetch price for {trading_pair}: {str(e)}")
            raise

    def get_portfolio_assets(self) -> Dict[str, Decimal]:
        """
        Get the assets in the portfolio.
        """
        hb = HummingbotApplication.main_application()
        return hb.markets[self.config.connector_name].get_all_balances()

    def get_portfolio_total_equity(self) -> Decimal:
        """
        Calculate the total value of the portfolio by summing up the value of each asset. Value is cash + long market value + short market value.
        """
        total_value = Decimal(0)
        portfolio_assets = self.get_portfolio_assets()

        for asset, balance in portfolio_assets.items():
            if asset == self.config.target_quote_asset:
                total_value += balance
            else:
                pair = self.generate_trading_pair(asset)
                try:
                    price = self.get_asset_price(self.config.connector_name, pair)
                    asset_value = balance * price
                    total_value += asset_value
                    self.logger().info(f"Asset: {asset}, Balance: {balance}, Price: {price}, Value: {asset_value}")
                except Exception as e:
                    self.logger().error(f"Error calculating value for {asset}: {e}")

        return total_value

    def get_rebalance_equity(self) -> Decimal:
        """
        Calculate the total value of the portfolio by summing up the value of each asset. Value is cash + long market value + short market value.
        """
        portfolio_equity = self.get_portfolio_total_equity()
        return portfolio_equity * Decimal(self.config.target_quote_weight)

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

    def symbols_to_close(self, current_positions, target_symbols) -> Set[str]:
        """
        Get the symbols that need to be closed.
        """
        symbols_to_close = set(current_positions.keys()) - set(target_symbols)
        return symbols_to_close

    def to_format_status(self) -> List[str]:
        return super().to_format_status() + [f"Portfolio Status: {self.portfolio_status}"]


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
