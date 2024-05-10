import time
from abc import abstractmethod
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from pydantic import Field, validator

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.client.hummingbot_application import HummingbotApplication
from hummingbot.smart_components.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.smart_components.controllers.rebalance_controller_base.data_types import (
    WeightingStrategyType,
    get_weighting_strategy,
    get_weighting_strategy_members,
)
from hummingbot.smart_components.executors.rebalance_executor.data_types import (
    REBALANCE_EXECUTOR_TYPE,
    RebalanceExecutorConfig,
)
from hummingbot.smart_components.models.executor_actions import CreateExecutorAction, ExecutorAction


class RebalanceControllerConfigBase(ControllerConfigBase):
    controller_type = "rebalance"
    connector_name: str = Field(
        default="kraken",
        client_data=ClientFieldData(
            prompt_on_new=True, prompt=lambda mi: "Enter the name of the exchange to trade on (e.g., kraken):"
        ),
    )
    weighting_strategy: WeightingStrategyType = Field(
        default="EQUAL",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the weighting strategy to use (EQUAL/MARKET_CAP/LIQUIDITY):",
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
                    f"Invalid weighting strategy: {v}. Valid options are: {get_weighting_strategy_members()}"
                )
        raise ValueError(f"Invalid weighting strategy: {v}. Valid options are: {get_weighting_strategy_members()}")


class RebalanceControllerBase(ControllerBase):
    def __init__(self, config: RebalanceControllerConfigBase, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config

    @abstractmethod
    def get_target_assets(self) -> Set[str]:
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

    def get_current_balances(self) -> Dict[str, float]:
        hb = HummingbotApplication.main_application()
        return hb.markets[self.config.connector_name].get_all_balances()

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
        Get the signal for the strategy.
        """
        raise NotImplementedError

    def calculate_target_weights(self, to_quote: bool = False) -> Dict[str, float]:
        target_assets = self.get_target_assets()

        # If to_quote is true, quote_asset gets 1.0 allocation, all else go to 0.0
        if to_quote:
            target_weights = {asset: 0.0 for asset in target_assets}
            target_weights[self.config.quote_asset] = 1.0
            return target_weights

        data = self.get_weighting_strategy_data()
        weighting_strategy = get_weighting_strategy(self.config.weighting_strategy)
        weights = weighting_strategy.calculate_weights(assets=target_assets, data=data)
        return weights

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
