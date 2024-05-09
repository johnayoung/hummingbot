import random
from typing import Any, Dict, List, Optional, Set

from pydantic import Field

from controllers.directional_trading.ema_crossover_v1 import EMACrossoverController, EMACrossoverControllerConfig
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.smart_components.controllers.rebalance_controller_base import (
    RebalanceControllerBase,
    RebalanceControllerConfigBase,
)

DEFAULT_SCREENER_ASSETS = ["XBT", "ETH", "SOL", "DOGE", "NEAR", "RNDR", "ADA", "AVAX", "XRP", "FET", "XMR", "WIF"]


class CrossMomentumWithTrendOverlayControllerConfig(EMACrossoverControllerConfig, RebalanceControllerConfigBase):
    controller_name = "cross_momentum_with_trend_overlay_v1"
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
            trading_pair = f"{asset}-{self.target_quote_asset}"
            markets[self.connector_name].add(trading_pair)

        return markets


class CrossMomentumWithTrendOverlayController(EMACrossoverController, RebalanceControllerBase):
    def __init__(self, config: CrossMomentumWithTrendOverlayControllerConfig, *args, **kwargs):
        self.config = config
        super().__init__(config, *args, **kwargs)

    def get_target_assets(self) -> List[str]:
        """
        Get the rebalance assets for the strategy.
        """
        target_assets = random.sample(self.config.screener_assets.split(","), 5)

        return target_assets

    def get_weighting_strategy_data(self) -> Optional[Any]:
        """
        Get additional data needed for the strategy to calculate weightings.
        """
        return None

    async def update_processed_data(self):
        """
        Update the processed data based on the current state of the strategy.
        """
        await EMACrossoverController.update_processed_data(self)  # <-- Fetches the signal
        await RebalanceControllerBase.update_processed_data(self)  # <-- Handles rebalancing
