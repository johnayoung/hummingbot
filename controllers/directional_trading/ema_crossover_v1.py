from typing import Any, Dict, List, Optional, Set

import pandas as pd
import pandas_ta as ta  # noqa: F401
from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig
from hummingbot.smart_components.controllers.rebalance_controller_base import (
    RebalanceController,
    RebalanceControllerConfigBase,
)
from hummingbot.smart_components.models.executor_actions import ExecutorAction


class EMACrossoverControllerConfig(RebalanceControllerConfigBase):
    controller_type = "directional_trading"
    controller_name = "ema_crossover_v1"
    connector_name: str = Field(
        default="kraken",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the name of the exchange to trade on (e.g., kraken):",
        ),
    )
    candles_config: List[CandlesConfig] = []
    ema_trading_pair: str = Field(
        default="BTC-USDC",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the trading pair for the candles data: ",
        ),
    )
    ema_candles_interval: str = Field(
        default="1m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ", prompt_on_new=False
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
    lookback_period: int = Field(
        default=30,
        gt=0,
        client_data=ClientFieldData(prompt=lambda mi: "Enter the lookback period: ", prompt_on_new=True),
    )

    @property
    def max_records(self) -> int:
        return self.ema_slow + 30

    def update_markets(self, markets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        if self.connector_name not in markets:
            markets[self.connector_name] = set()
        markets[self.connector_name].add(self.ema_trading_pair)
        return markets


class EMACrossoverController(RebalanceController):
    def __init__(self, config: EMACrossoverControllerConfig, *args, **kwargs):
        self.config = config
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [
                CandlesConfig(
                    connector=config.connector_name,
                    trading_pair=config.ema_trading_pair,
                    interval=config.ema_candles_interval,
                    max_records=config.max_records,
                )
            ]
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        """
        Update the processed data based on the current state of the strategy.
        """
        signal = self.get_signal()
        self.processed_data = {"signal": signal}

    def get_processed_data(self) -> pd.DataFrame:
        df = self.market_data_provider.get_candles_df(
            self.config.connector_name,
            self.config.ema_trading_pair,
            self.config.ema_candles_interval,
            self.config.max_records,
        )

        if df.empty or len(df) < 2:
            self.logger().warning("Empty dataframe received from get_candles_df.")
            return df

        df["fast_ema"] = df.ta.ema(length=self.config.ema_fast)
        df["slow_ema"] = df.ta.ema(length=self.config.ema_slow)
        df["percent_return"] = df.ta.ema(length=self.config.lookback_period, close="close", fillna=True).pct_change()

        return df

    def get_signal(self) -> int:
        if self.config.test_mode:
            self.logger().info("Running in test mode. Skipping signal check.")
            return 1

        df = self.get_processed_data()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        buy_condition = last["fast_ema"] > last["slow_ema"] and prev["fast_ema"] <= prev["slow_ema"]
        sell_condition = last["fast_ema"] < last["slow_ema"] and prev["fast_ema"] >= prev["slow_ema"]

        if buy_condition:
            return 1  # Trigger rebalance
        elif sell_condition:
            return -1  # Trigger position close
        return 0  # No action

    def get_target_assets(self) -> List[str]:
        """
        Get the rebalance assets for the strategy.
        """
        return ["AKT", "CRV", "UNI", "SUSHI", "PAXG", "SNX"]

    def get_weighting_strategy_data(self) -> Optional[Any]:
        """
        Get additional data needed for the strategy to calculate weightings.
        """
        return None

    def determine_executor_actions(self) -> List[ExecutorAction]:
        """
        Determine actions based on the provided executor handler report.
        """
        actions = []
        actions.extend(self.create_actions_proposal())
        actions.extend(self.stop_actions_proposal())
        return actions
