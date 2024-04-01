from typing import Dict

import pandas as pd

from datamodel import Order, OrderDepth, TradingState
from trader import Trader

products = ["AMETHYSTS", "STARFRUIT"]


class MarketSimulator:
    df: pd.DataFrame
    trader: Trader
    num_days: int
    player_position: Dict[str, int]

    def __init__(self, day: int) -> None:
        self.df = pd.read_csv(f"day{day}.csv", delimiter=";")
        self.trader = Trader()
        self.num_days = self.df.timestamp.max() // 100
        self.player_position = {product: 0 for product in products}

    def get_order_depth(
        self, df: pd.DataFrame, timestamp: int
    ) -> Dict[str, OrderDepth]:
        """Compute the order depth for each product at a given timestamp
        Extract the order depth from the dataframe for a given timestamp

        Args:
            df (pd.DataFrame): dataframe containing the order depth
            timestamp (int): timestamp to extract the order depth

        Returns:
            Dict[str, OrderDepth]: order depth for each product
        """
        df = df[df.timestamp == timestamp]
        order_depth = {}
        for product in pd["product"].unique():
            product_pd = df[df.product == product]
            assert (
                len(product_pd) == 1
            ), f"len(product_pd) = {len(product_pd)} while it should be one for one day"
            # remove NaN columns
            product_pd.dropna(inplace=True, axis=1)
            buy_orders, sell_orders = {}, {}

            for i in range(1, 4):
                try:
                    buy_orders[product_pd[f"bid_price_{i}"]] = product_pd[
                        f"bid_volume_{i}"
                    ]
                except KeyError:
                    pass
                try:
                    sell_orders[product_pd[f"ask_price_{i}"]] = product_pd[
                        f"ask_volume_{i}"
                    ]
                except KeyError:
                    pass

            order_depth[product] = OrderDepth(
                buy_orders=buy_orders, sell_orders=sell_orders
            )

        return order_depth

    def run(self):
        trader_data = ""
        position = {}
        for i in range(self.num_days):
            row = self.df[self.df.timestamp == i * 100]
            order_depth = self.get_order_depth(self.df, i * 100)

            state = TradingState(
                traderData=trader_data,
                timestamp=i * 100,
                listings={},
                order_depths=order_depth,
                own_trades={},
                market_trades={},
                # TODO: fill in
                position=self.player_position,
                # not supported yet
                observations=None,
            )

            orders, conversions, trader_data = self.loop(row)
            self.compute_trades(orders, state)

    def compute_trades(order, state: TradingState):
        """Computes the trades based on the orders and the state of the market.
        Updates the player's position
        """
        ...
