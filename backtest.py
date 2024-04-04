from typing import Dict

import click
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from datamodel import Order, OrderDepth, TradingState
from trader import Trader

products = ["AMETHYSTS", "STARFRUIT"]
LIMIT_POSITIONS = {
    "AMETHYSTS": 20,
    "STARFRUIT": 20,
}


class MarketSimulator:
    df: pd.DataFrame
    trader: Trader
    num_days: int
    player_position: Dict[str, int]
    player_cash: int = 0
    player_pnl: list[int]
    player_position_history: Dict[str, list[int]]

    def __init__(self, day: int) -> None:
        self.df = pd.read_csv(f"data/day{day}.csv", delimiter=";")
        self._reset()

    def _reset(self):
        self.trader = Trader()
        self.num_days = self.df.timestamp.max() // 100
        self.player_position = {product: 0 for product in products}
        self.player_cash = 0
        self.player_pnl = [0]
        self.player_position_history = {product: [0] for product in products}

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
        for product in df["product"].unique():
            product_pd = df[df["product"] == product]
            assert (
                len(product_pd) == 1
            ), f"len(product_pd) = {len(product_pd)} while it should be one for one day"
            # remove NaN columns
            product_pd = product_pd.dropna(axis=1)
            buy_orders, sell_orders = {}, {}

            for i in range(1, 4):
                try:
                    buy_orders[product_pd[f"bid_price_{i}"].item()] = product_pd[
                        f"bid_volume_{i}"
                    ].item()
                except KeyError:
                    pass
                try:
                    sell_orders[product_pd[f"ask_price_{i}"].item()] = product_pd[
                        f"ask_volume_{i}"
                    ].item()
                except KeyError:
                    pass

            order_depth[product] = OrderDepth(
                buy_orders=buy_orders, sell_orders=sell_orders
            )

        return order_depth

    def run(self):
        trader_data = ""
        for i in tqdm(range(self.num_days // 10)):
            row = self.df[self.df.timestamp == i * 100]
            order_depth = self.get_order_depth(self.df, i * 100)

            state = TradingState(
                traderData=trader_data,
                timestamp=i * 100,
                listings={},
                order_depths=order_depth,
                own_trades={},
                market_trades={},
                position=self.player_position,
                # not supported yet
                observations=None,
            )

            orders, conversions, trader_data = self.trader.run(state)
            self.compute_trades(orders, state)
            self.player_pnl.append(self.compute_pnl(state))

            # add the position to the history
            for product in products:
                self.player_position_history[product].append(
                    self.player_position[product]
                )
        self.plot()

    def compute_pnl(self, state: TradingState):
        """Compute the PnL of the player"""
        pnl = 0
        for product, position in self.player_position.items():
            # TODO: this breaks if there is no buy or sell orders
            mid_price = (
                list(state.order_depths[product].sell_orders.items())[0][0]
                + list(state.order_depths[product].buy_orders.items())[0][0]
            ) / 2
            pnl += mid_price * position
        return self.player_cash + pnl

    def compute_trades(self, order, state: TradingState):
        """Computes the trades based on the orders and the state of the market.
        Updates the player's position
        """
        for product, orders in order.items():
            for order in orders:
                if order.quantity > 0:
                    price, max_amount = list(
                        state.order_depths[product].sell_orders.items()
                    )[0]
                else:
                    price, max_amount = list(
                        state.order_depths[product].buy_orders.items()
                    )[0]

                # assert (
                #     order.price == price
                # ), f"order.price = {order.price} while it should be {price}"
                # # TODO: this works for only one price order
                # assert (
                #     order.quantity <= max_amount
                # ), f"order.quantity = {order.quantity} while it should be <= {max_amount}"

                if order.quantity * self.player_position[product] < 0:
                    max_tradable = max(
                        order.quantity,
                        -(LIMIT_POSITIONS[product] + self.player_position[product]),
                    )
                else:
                    max_tradable = min(
                        order.quantity,
                        LIMIT_POSITIONS[product] - self.player_position[product],
                    )

                self.player_position[product] += max_tradable
                self.player_cash -= order.price * max_tradable

    def plot(self):
        plt.figure()
        plt.plot(self.player_pnl)
        plt.show()


@click.command()
@click.option("--day", default=0, help="Day to backtest")
def main(day: int):
    simulator = MarketSimulator(day)
    simulator.run()


if __name__ == "__main__":
    main()
