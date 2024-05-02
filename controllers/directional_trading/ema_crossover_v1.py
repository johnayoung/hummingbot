from typing import Dict, List, Set

import pandas as pd
from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.smart_components.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.smart_components.models.executor_actions import ExecutorAction


class EMACrossoverControllerConfig(ControllerConfigBase):
    controller_type = "directional_trading"
    controller_name = "ema_crossover_v1"
    connector_name: str = Field(
        default="binance_perpetual",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the name of the exchange to trade on (e.g., binance_perpetual):",
        ),
    )
    trading_pair: str = Field(
        default="WLD-USDT",
        client_data=ClientFieldData(
            prompt_on_new=True, prompt=lambda mi: "Enter the trading pair to trade on (e.g., WLD-USDT):"
        ),
    )
    ema_fast: int = Field(
        default=5,
        gt=0,
        client_data=ClientFieldData(prompt=lambda mi: "Enter the fast EMA period (e.g. 5): ", prompt_on_new=True),
    )
    ema_slow: int = Field(
        default=50,
        gt=0,
        client_data=ClientFieldData(prompt=lambda mi: "Enter the slow EMA period (e.g. 50): ", prompt_on_new=True),
    )

    def update_markets(self, markets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        if self.connector_name not in markets:
            markets[self.connector_name] = set()
        markets[self.connector_name].add(self.trading_pair)
        return markets


class EMACrossoverController(ControllerBase):
    def __init__(self, config: EMACrossoverControllerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config

    async def update_processed_data(self):
        """
        Update the processed data based on the current state of the strategy.
        """
        signal = self.get_signal()
        self.processed_data = {"signal": signal}

    def get_processed_data(self) -> pd.DataFrame:
        df = self.market_data_provider.get_candles_df(self.config.connector_name, self.config.trading_pair, "5m", 500)
        df["fast_ema"] = df["close"].ta.ema(length=self.config.ema_fast)
        df["slow_ema"] = df["close"].ta.ema(length=self.config.ema_slow)
        return df

    def get_signal(self) -> int:
        df = self.get_processed_data()
        last = df.iloc[-1]
        prev = df.iloc[-2]
        if last["fast_ema"] > last["slow_ema"] and prev["fast_ema"] <= prev["slow_ema"]:
            return 1  # Buy signal
        elif last["fast_ema"] < last["slow_ema"] and prev["fast_ema"] >= prev["slow_ema"]:
            return -1  # Sell signal
        return 0  # No action

    def determine_executor_actions(self) -> List[ExecutorAction]:
        """
        Determine actions based on the provided executor handler report.
        """
        actions = []
        actions.extend(self.create_actions_proposal())
        actions.extend(self.stop_actions_proposal())
        return actions

    def create_actions_proposal(self) -> List[ExecutorAction]:
        """
        Create actions based on the provided executor handler report.
        """
        create_actions = []

        return create_actions

    def stop_actions_proposal(self) -> List[ExecutorAction]:
        """
        Stop actions based on the provided executor handler report.
        """
        stop_actions = []
        return stop_actions
