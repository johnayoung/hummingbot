import asyncio
from decimal import Decimal
from typing import Dict, List

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.logger.logger import HummingbotLogger
from hummingbot.smart_components.executors.executor_base import ExecutorBase
from hummingbot.smart_components.executors.rebalance_executor.data_types import (
    RebalanceExecutorConfig,
    RebalanceExecutorStatus,
)
from hummingbot.smart_components.models.executors import TrackedOrder
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class RebalanceAction:
    asset: str
    amount: Decimal
    side: TradeType

    def __init__(self, asset: str, amount: Decimal, side: TradeType):
        self.asset = asset
        self.amount = amount
        self.side = side


class RebalanceExecutor(ExecutorBase):
    _logger = None

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = HummingbotLogger(__name__)
        return cls._logger

    @property
    def is_closed(self):
        return self.rebalance_status in [RebalanceExecutorStatus.COMPLETED, RebalanceExecutorStatus.FAILED]

    def __init__(self, strategy: ScriptStrategyBase, config: RebalanceExecutorConfig, update_interval: float = 1.0):
        super().__init__(
            strategy=strategy, connectors=[config.connector_name], config=config, update_interval=update_interval
        )
        self.config = config
        self.target_weights = {asset: Decimal(weight) for asset, weight in config.target_weights.items()}
        self.quote_asset = config.quote_asset
        self.quote_weight = config.quote_weight
        self.min_order_amount_to_rebalance_quote = config.min_order_amount_to_rebalance_quote
        self.rebalance_status = RebalanceExecutorStatus.INITIALIZING
        self.tracked_orders: Dict[str, TrackedOrder] = {}

    @property
    def current_balances(self) -> Dict[str, Decimal]:
        cb = self.connectors[self.config.connector_name].get_all_balances()
        return {asset: Decimal(amount) for asset, amount in cb.items()}

    def validate_sufficient_balance(self):
        # TODO
        pass

    def get_net_pnl_quote(self) -> Decimal:
        """
        TODO: Returns the net profit or loss in quote currency.
        """
        return Decimal(0)

    def get_net_pnl_pct(self) -> Decimal:
        """
        TODO: Returns the net profit or loss in percentage.
        """
        return Decimal(0)

    def get_cum_fees_quote(self) -> Decimal:
        """
        Returns the cumulative fees in quote currency.
        """
        return Decimal(0)

    def get_trading_pair(self, asset: str) -> str:
        return f"{asset}-{self.quote_asset}"

    def get_asset_value_in_quote(self, asset: str, amount: Decimal) -> Decimal:
        # Fetches the price of the asset in terms of the quote asset
        price = self.get_price(
            connector_name=self.config.connector_name,
            trading_pair=self.get_trading_pair(asset),
            price_type=PriceType.MidPrice,
        )
        return amount * price

    def calculate_total_portfolio_value(self):
        total_value = Decimal(0)
        for asset, amount in self.current_balances.items():
            if asset in self.target_weights or asset == self.quote_asset:
                if asset != self.quote_asset:
                    total_value += self.get_asset_value_in_quote(asset, amount)
                else:
                    total_value += amount
        return total_value

    def calculate_total_rebalance_value(self):
        total_value = self.calculate_total_portfolio_value()
        rebalance_weight = Decimal((1 - self.quote_weight))
        return total_value * rebalance_weight

    def calculate_rebalance_actions(self) -> List[RebalanceAction]:
        total_rebalance_value = self.calculate_total_rebalance_value()
        target_values = {asset: total_rebalance_value * weight for asset, weight in self.target_weights.items()}

        trade_actions = []
        for asset, target_value in target_values.items():
            if asset == self.quote_asset:
                continue  # Skip the quote asset

            current_value = self.get_asset_value_in_quote(asset, self.current_balances.get(asset, Decimal(0)))
            amount_in_quote = target_value - current_value

            asset_price = self.get_price(
                connector_name=self.config.connector_name,
                trading_pair=self.get_trading_pair(asset),
                price_type=PriceType.MidPrice,
            )
            amount = amount_in_quote / asset_price

            is_non_zero_amount = abs(amount) > 0
            is_above_min_order_amount = abs(amount_in_quote) >= self.min_order_amount_to_rebalance_quote
            if is_non_zero_amount and is_above_min_order_amount:
                side = TradeType.BUY if amount > 0 else TradeType.SELL
                trade_actions.append(RebalanceAction(asset=asset, amount=amount, side=side))

        return trade_actions

    async def control_task(self):
        try:
            if self.is_closed:
                return

            self.rebalance_status = RebalanceExecutorStatus.SELLING
            actions = self.calculate_rebalance_actions()
            sell_actions = [action for action in actions if action.side == TradeType.SELL]
            buy_actions = [action for action in actions if action.side == TradeType.BUY]

            sell_tasks = [self.place_order_and_wait(action) for action in sell_actions]
            await asyncio.gather(*sell_tasks)

            self.rebalance_status = RebalanceExecutorStatus.BUYING
            buy_tasks = [self.place_order_and_wait(action) for action in buy_actions]
            await asyncio.gather(*buy_tasks)

            self.rebalance_status = RebalanceExecutorStatus.COMPLETED
        except Exception as e:
            self.logger().error(f"Error in rebalance executor: {str(e)}")
            self.rebalance_status = RebalanceExecutorStatus.FAILED

    async def place_order_and_wait(self, action: RebalanceAction):
        asset = action.asset
        amount = abs(action.amount)
        side = action.side
        try:
            order_id = self.place_order(
                connector_name=self.config.connector_name,
                trading_pair=self.get_trading_pair(asset),
                order_type=OrderType.MARKET,
                side=side,
                amount=amount,
            )
            if order_id:
                self.update_tracked_order(order_id)
            await self.wait_for_order_completion(order_id)
        except Exception as e:
            self.logger().error(f"Error placing order for {asset}: {str(e)}")
            self.rebalance_status = RebalanceExecutorStatus.FAILED
            raise e

    async def wait_for_order_completion(self, order_id: str):
        while not self.is_order_complete(order_id):
            await asyncio.sleep(1)  # Check order status every second

    def is_order_complete(self, order_id: str) -> bool:
        tracked_order = self.update_tracked_order(order_id)
        if tracked_order and tracked_order.is_done:
            if tracked_order.order and tracked_order.order.is_failure:
                # Raise an exception if the order has failed
                raise Exception(f"Order {order_id} failed: ")
            return True
        return False

    def get_custom_info(self) -> Dict:
        return {
            "rebalance_status": self.rebalance_status,
            "current_balances": self.current_balances,
            "target_weights": self.target_weights,
            "quote_asset": self.quote_asset,
            "quote_weight": self.quote_weight,
            "min_order_amount_to_rebalance_quote": self.min_order_amount_to_rebalance_quote,
        }

    def update_tracked_order(self, order_id: str):
        tracked_order = self.tracked_orders.get(order_id)
        if tracked_order is None:
            tracked_order = TrackedOrder(order_id=order_id)
            self.tracked_orders[order_id] = tracked_order
        if tracked_order.order is None:
            in_flight_order = self.get_in_flight_order(connector_name=self.config.connector_name, order_id=order_id)
            if in_flight_order:
                tracked_order.order = in_flight_order
        return tracked_order
