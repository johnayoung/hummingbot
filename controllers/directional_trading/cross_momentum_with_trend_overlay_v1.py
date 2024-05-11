import random
import time
from decimal import Decimal
from typing import Dict, List, Set, Tuple

import numpy as np
from pydantic import Field

from controllers.directional_trading.ema_crossover_v1 import EMACrossoverController, EMACrossoverControllerConfig
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.client.hummingbot_application import HummingbotApplication
from hummingbot.smart_components.controllers.data_types.data_types import EqualWeighting
from hummingbot.smart_components.executors.rebalance_executor.data_types import (
    REBALANCE_EXECUTOR_TYPE,
    RebalanceExecutorConfig,
)
from hummingbot.smart_components.models.executor_actions import CreateExecutorAction, ExecutorAction

DEFAULT_SCREENER_ASSETS = ["USDC", "ETH", "SOL", "DOGE", "NEAR", "RNDR", "ADA", "AVAX", "XRP", "FET", "XMR", "WIF"]


class CrossMomentumWithTrendOverlayControllerConfig(EMACrossoverControllerConfig):
    controller_name = "cross_momentum_with_trend_overlay_v1"
    connector_name: str = Field(
        default="kraken",
        client_data=ClientFieldData(
            prompt_on_new=True, prompt=lambda mi: "Enter the name of the exchange to trade on (e.g., kraken):"
        ),
    )
    quote_asset: str = Field(
        default="USD",
        client_data=ClientFieldData(
            prompt_on_new=True, prompt=lambda mi: "Enter the target quote asset for the portfolio:"
        ),
    )
    quote_weight: float = Field(
        default=0.05,
        client_data=ClientFieldData(
            prompt_on_new=True, prompt=lambda mi: "Enter the target weight of the quote asset in the portfolio:"
        ),
    )
    min_order_amount_to_rebalance_quote: Decimal = Field(
        default=Decimal("0.01"),
        client_data=ClientFieldData(
            prompt_on_new=True, prompt=lambda mi: "Enter the minimum order size in quote asset for the exchange:"
        ),
    )
    screener_assets: str = Field(
        default=",".join(DEFAULT_SCREENER_ASSETS),
        client_data=ClientFieldData(
            prompt_on_new=True, prompt=lambda mi: "Enter the assets to use for the screener universe:"
        ),
    )
    lookback_period: int = Field(
        default=5,
        gt=0,
        client_data=ClientFieldData(prompt=lambda mi: "Enter the lookback period (e.g. 5): ", prompt_on_new=True),
    )

    def update_markets(self, markets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        if self.connector_name not in markets:
            markets[self.connector_name] = set()
        markets[self.connector_name].add(self.ema_trading_pair)

        for asset in self.screener_assets.split(","):
            trading_pair = f"{asset}-{self.quote_asset}"
            markets[self.connector_name].add(trading_pair)

        return markets


class CrossMomentumWithTrendOverlayController(EMACrossoverController):
    def __init__(self, config: CrossMomentumWithTrendOverlayControllerConfig, *args, **kwargs):
        self.config = config
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        await super().update_processed_data()

    def get_current_balances(self) -> Dict[str, float]:
        hb = HummingbotApplication.main_application()
        return hb.markets[self.config.connector_name].get_all_balances()

    def get_current_assets(self) -> Set[str]:
        return set(self.get_current_balances().keys())

    def get_target_assets(self) -> Set[str]:
        # TODO
        return set(random.sample(self.config.screener_assets.split(","), 5))

    def get_assets_to_close(self, current_assets: Set[str], target_assets: Set[str]) -> Set[str]:
        return set(asset for asset in current_assets if asset not in target_assets)

    def calculate_weights_data(self) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
        current_assets = self.get_current_assets()
        target_assets = self.get_target_assets()
        assets_to_close = self.get_assets_to_close(current_assets, target_assets)
        all_assets = current_assets.union(target_assets)
        return current_assets, target_assets, assets_to_close, all_assets

    def calculate_target_weights(self, to_quote: bool = False) -> Dict[str, float]:
        current_assets, target_assets, assets_to_close, all_assets = self.calculate_weights_data()

        if to_quote:
            target_weights = {asset: 0.0 for asset in all_assets}
            target_weights[self.config.quote_asset] = 1.0
            return target_weights

        weighting_strategy = EqualWeighting()
        weights = weighting_strategy.calculate_weights(assets=all_assets, data={"excluded_assets": assets_to_close})

        # Assert that the sum of weights is 1
        assert np.isclose(sum(weights.values()), 1.0), f"Sum of weights is not 1: {sum(weights.values())}"
        return weights

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
        signal = self.processed_data["signal"]
        if signal != 0 and self.can_create_rebalance_executor(signal):
            to_quote_condition = signal < 0
            create_actions.append(
                CreateExecutorAction(
                    controller_id=self.config.id,
                    executor_config=self.get_rebalance_executor_config(
                        target_weights=self.calculate_target_weights(to_quote=to_quote_condition)
                    ),
                )
            )
        return create_actions

    def stop_actions_proposal(self) -> List[ExecutorAction]:
        """
        Stop actions based on the provided executor handler report.
        """
        stop_actions = []
        return stop_actions

    def can_create_rebalance_executor(self, signal: int) -> bool:
        """
        Check if a rebalance executor can be created. Only one rebalance executor is allowed at a time.
        """
        return (
            len(
                self.filter_executors(
                    executors=self.executors_info,
                    filter_func=lambda x: x.is_active and x.type == REBALANCE_EXECUTOR_TYPE,
                )
            )
            == 0
        )

    def get_rebalance_executor_config(self, target_weights: Dict[str, float]) -> RebalanceExecutorConfig:
        """
        Get the rebalance executor config.
        """
        return RebalanceExecutorConfig(
            timestamp=time.time(),
            connector_name=self.config.connector_name,
            current_balances=self.get_current_balances(),
            target_weights=target_weights,
            quote_asset=self.config.quote_asset,
            quote_weight=self.config.quote_weight,
            min_order_amount_to_rebalance_quote=self.config.min_order_amount_to_rebalance_quote,
        )
