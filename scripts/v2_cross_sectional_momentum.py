import os
from typing import Dict, List

from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase


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
