import asyncio
from decimal import Decimal
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from hummingbot.data_feed.market_data_provider import MarketDataProvider
from hummingbot.smart_components.controllers.rebalance_controller_base import (
    RebalanceControllerBase,
    RebalanceControllerConfigBase,
)


class TestRebalanceControllerBase(IsolatedAsyncioWrapperTestCase):
    def setUp(self):
        # Mocking the DirectionalTradingControllerConfigBase
        self.mock_controller_config = RebalanceControllerConfigBase(
            id="test",
            controller_name="rebalance_test_controller",
            connector_name="kraken",
        )

        # Mocking dependencies
        self.mock_market_data_provider = MagicMock(spec=MarketDataProvider)
        self.mock_actions_queue = AsyncMock(spec=asyncio.Queue)

        # Instantiating the DirectionalTradingControllerBase
        self.controller = RebalanceControllerBase(
            config=self.mock_controller_config,
            market_data_provider=self.mock_market_data_provider,
            actions_queue=self.mock_actions_queue,
        )

    @patch.object(RebalanceControllerBase, "get_signal")
    async def test_update_processed_data(self, get_signal_mock: MagicMock):
        get_signal_mock.return_value = Decimal("1")
        await self.controller.update_processed_data()
        self.assertEqual(self.controller.processed_data["signal"], Decimal("1"))
