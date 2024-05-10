import asyncio
from typing import Any, Dict, List

from hummingbot.core.data_type.common import OrderType, PriceType
from hummingbot.logger.logger import HummingbotLogger
from hummingbot.smart_components.executors.executor_base import ExecutorBase
from hummingbot.smart_components.executors.rebalance_executor.data_types import (
    RebalanceExecutorConfig,
    RebalanceExecutorStatus,
)
from hummingbot.smart_components.models.executors import TrackedOrder
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


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

    def __init__(self, strategy: ScriptStrategyBase, config: RebalanceExecutorConfig, **kwargs):
        super().__init__(strategy=strategy, connectors=[config.connector_name], config=config)
        self.config = config
        self.target_weights = config.target_weights
        self.quote_asset = config.quote_asset
        self.rebalance_status = RebalanceExecutorStatus.INITIALIZING
        self.current_balances = self.get_current_balances()
        self.tracked_orders: Dict[str, TrackedOrder] = {}

    def get_trading_pair(self, asset: str) -> str:
        return f"{asset}-{self.quote_asset}"

    def get_current_balances(self) -> Dict[str, float]:
        balances = self._strategy.connectors[self._strategy.connector_name].get_all_balances()
        return {asset: float(balance) for asset, balance in balances.items()}

    def get_asset_value_in_quote(self, asset: str, amount: float) -> float:
        # Fetches the price of the asset in terms of the quote asset
        price = self.get_price(
            connector_name=self.config.connector_name,
            trading_pair=self.get_trading_pair(asset),
            price_type=PriceType.MidPrice,
        )
        return amount * price

    def calculate_total_portfolio_value(self) -> float:
        total_value = 0.0
        for asset, amount in self.current_balances.items():
            if asset != self.quote_asset:
                total_value += self.get_asset_value_in_quote(asset, amount)
            else:
                total_value += amount
        return total_value

    def calculate_rebalance_actions(self) -> List[Dict[str, Any]]:
        total_portfolio_value = self.calculate_total_portfolio_value()
        target_values = {asset: total_portfolio_value * weight for asset, weight in self.target_weights.items()}

        trade_actions = []
        for asset, target_value in target_values.items():
            if asset == self.quote_asset:
                current_value = self.current_balances.get(asset, 0)
            else:
                current_value = self.get_asset_value_in_quote(asset, self.current_balances.get(asset, 0))

            amount_in_quote = target_value - current_value
            if asset != self.quote_asset:
                asset_price = self.get_price(
                    connector_name=self.config.connector_name,
                    trading_pair=self.get_trading_pair(asset),
                    price_type=PriceType.MidPrice,
                )
                amount = amount_in_quote / asset_price
            else:
                amount = amount_in_quote

            if abs(amount) > 0:
                trade_actions.append({"asset": asset, "amount": amount, "side": "buy" if amount > 0 else "sell"})

        return trade_actions

    async def control_task(self):
        try:
            self.rebalance_status = RebalanceExecutorStatus.SELLING
            actions = self.calculate_rebalance_actions()
            sell_actions = [action for action in actions if action["side"] == "sell"]
            buy_actions = [action for action in actions if action["side"] == "buy"]

            sell_tasks = [self.place_order_and_wait(action) for action in sell_actions]
            await asyncio.gather(*sell_tasks)

            self.rebalance_status = RebalanceExecutorStatus.BUYING
            buy_tasks = [self.place_order_and_wait(action) for action in buy_actions]
            await asyncio.gather(*buy_tasks)

            self.rebalance_status = RebalanceExecutorStatus.COMPLETED
        except Exception as e:
            self.logger().error(f"Error in rebalance executor: {str(e)}")
            self.rebalance_status = RebalanceExecutorStatus.FAILED

    async def place_order_and_wait(self, action: Dict[str, Any]):
        asset = action["asset"]
        amount = abs(action["amount"])
        side = action["side"]
        try:
            order_id = self.place_order(
                connector_name=self.config.connector_name,
                trading_pair=self.get_trading_pair(asset),
                order_type=OrderType.MARKET,
                side=side,
                amount=amount,
            )
            in_flight_order = self.get_in_flight_order(connector_name=self.config.connector_name, order_id=order_id)
            tracked_order = TrackedOrder(order_id=order_id)
            tracked_order.order = in_flight_order
            self.tracked_orders[order_id] = tracked_order
            await self.wait_for_order_completion(order_id)
        except Exception as e:
            self.logger().error(f"Error placing order for {asset}: {str(e)}")
            self.rebalance_status = RebalanceExecutorStatus.FAILED

    async def wait_for_order_completion(self, order_id: str):
        while not self.is_order_filled(order_id):
            await asyncio.sleep(1)  # Check order status every second

    def is_order_filled(self, order_id: str) -> bool:
        tracked_order = self.tracked_orders.get(order_id)
        if tracked_order:
            return tracked_order.is_filled
        return False
