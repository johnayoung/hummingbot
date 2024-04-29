import os
from decimal import Decimal
from typing import Dict, List, Set

from pydantic import Field

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, PositionMode, PriceType, TradeType
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig
from hummingbot.smart_components.executors.position_executor.data_types import (
    PositionExecutorConfig,
    TripleBarrierConfig,
)
from hummingbot.smart_components.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase


class V2PositionExecutorConfig(StrategyV2ConfigBase):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    candles_config: List[CandlesConfig] = []
    controllers_config: List[str] = []
    markets: Dict[str, Set[str]] = {}
    connector_name: str = "binance_perpetual"
    trading_pair: str = "DOGE-USDT"
    stop_loss: Decimal = Decimal("0.02")
    take_profit: Decimal = Decimal("0.03")
    time_limit: int = 60
    amount_quote: Decimal = Decimal("20")


class V2PositionExecutor(StrategyV2Base):
    def __init__(self, connectors: Dict[str, ConnectorBase], config: V2PositionExecutorConfig):
        super().__init__(connectors, config)
        self.config = config  # Only for type checking

    def start(self, clock: Clock, timestamp: float) -> None:
        """
        Start the strategy.
        :param clock: Clock to use.
        :param timestamp: Current time.
        """
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    @classmethod
    def init_markets(cls, config: V2PositionExecutorConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        actions = []
        active_executors = self.filter_executors(executors=self.get_all_executors(), filter_func=lambda x: x.is_active)
        if len(active_executors) == 0:
            mid_price = self.market_data_provider.get_price_by_type(
                self.config.connector_name, self.config.trading_pair, PriceType.MidPrice
            )
            action = CreateExecutorAction(
                executor_config=PositionExecutorConfig(
                    timestamp=self.current_timestamp,
                    trading_pair=self.config.trading_pair,
                    connector_name=self.config.connector_name,
                    side=TradeType.BUY,
                    entry_price=mid_price,
                    amount=self.config.amount_quote / mid_price,
                    triple_barrier_config=TripleBarrierConfig(
                        stop_loss=self.config.stop_loss,
                        take_profit=self.config.take_profit,
                        time_limit=self.config.time_limit,
                        open_order_type=OrderType.MARKET,
                        take_profit_order_type=OrderType.MARKET,
                    ),
                )
            )
            actions.append(action)
        return actions

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        """
        Create a list of actions to stop the executors based on order refresh and early stop conditions.
        """
        stop_actions = []
        return stop_actions

    def apply_initial_setting(self):
        for connector_name, connector in self.connectors.items():
            if self.is_perpetual(connector_name):
                connector.set_position_mode(PositionMode.HEDGE)
                for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                    connector.set_leverage(trading_pair, 20)
