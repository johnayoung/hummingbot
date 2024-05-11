from decimal import Decimal
from enum import Enum
from typing import Dict

from hummingbot.smart_components.executors.data_types import ExecutorConfigBase

REBALANCE_EXECUTOR_TYPE = "rebalance_executor"


class RebalanceExecutorStatus(Enum):
    INITIALIZING = 1
    SELLING = 2
    BUYING = 3
    COMPLETED = 4
    FAILED = 5


class RebalanceExecutorConfig(ExecutorConfigBase):
    type: str = REBALANCE_EXECUTOR_TYPE
    connector_name: str
    target_weights: Dict[str, float]
    quote_asset: str
    quote_weight: float
    min_order_amount_to_rebalance_quote: Decimal
