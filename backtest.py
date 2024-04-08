import logging
import os
import sys
from pathlib import Path
from typing import Dict

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from datamodel import Order, OrderDepth, TradingState
from trader import Trader, set_debug

logger = logging.getLogger(__name__)
set_debug(True)
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
PRODUCTS = ["AMETHYSTS", "STARFRUIT"]
LIMIT_POSITIONS = {
    "AMETHYSTS": 20,
    "STARFRUIT": 20,
}


class MarketSimulator:
    df: pd.DataFrame
    trader: Trader
    num_days: int
    player_position: Dict[str, int]
    player_cash: Dict[str, list[int]]
    player_pnl: Dict[str, list[int]]
    player_position_history: Dict[str, list[int]]
    markers: Dict[str, list[tuple[int, int]]]
    timestamps: list[int]
    day: int
    round: int

    def __init__(self, round: int, day: int) -> None:
        self.df = pd.read_csv(f"data/round{round}/{day}.csv", delimiter=";")
        self._reset()
        self.round = round
        self.day = day

    def _reset(self):
        self.trader = Trader()
        self.num_days = self.df.timestamp.max() // 100
        self.player_position = {product: 0 for product in PRODUCTS}
        self.player_cash = {product: [0] for product in PRODUCTS}
        self.player_pnl = {product: [0] for product in PRODUCTS}
        self.player_position_history = {product: [0] for product in PRODUCTS}
        self.markers = {product: [] for product in PRODUCTS}
        self.timestamps = []

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
        for product in PRODUCTS:
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
                    sell_orders[product_pd[f"ask_price_{i}"].item()] = -product_pd[
                        f"ask_volume_{i}"
                    ].item()
                except KeyError:
                    pass

            order_depth[product] = OrderDepth(
                buy_orders=buy_orders, sell_orders=sell_orders
            )

        return order_depth

    def get_state(self, i: int, trader_data=None) -> TradingState:
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
        return state

    def iter(self, i: int, trader_data=None):
        state = self.get_state(i, trader_data)
        orders, conversions, trader_data, markers = self.trader.run(state)
        self.compute_trades(orders, state)
        self.store_position(state, i, markers)
        return orders, conversions, trader_data, markers

    def store_position(self, state: TradingState, i: int, markers: Dict[str, tuple]):
        # add the position to the history
        for product in PRODUCTS:
            self.player_position_history[product].append(self.player_position[product])
            self.player_pnl[product].append(self.compute_pnl(state, product))
        for product, marker in markers.items():
            if marker is not None:
                self.markers[product].append((i, *marker))
        self.timestamps.append(i * 100)

    def run(self, max_steps: int):
        trader_data = None
        for i in tqdm(range(min(self.num_days // 1, max_steps))):
            orders, conversions, trader_data, markers = self.iter(i, trader_data)
        self.print()
        self.plot()

    def print(self):
        for product in PRODUCTS:
            print(f"Final PnL {product}: {self.player_pnl[product][-1]}")
        print(
            f"Final PnL: {sum([self.player_pnl[product][-1] for product in PRODUCTS])}"
        )

    def compute_pnl(self, state: TradingState, product: str):
        """Compute the PnL of the player"""
        position = self.player_position[product]
        # TODO: this breaks if there is no buy or sell orders
        mid_price = (
            list(state.order_depths[product].sell_orders.items())[0][0]
            + list(state.order_depths[product].buy_orders.items())[0][0]
        ) / 2
        return self.player_cash[product][-1] + mid_price * position

    def compute_trades(self, order, state: TradingState):
        """Computes the trades based on the orders and the state of the market.
        Updates the player's position
        """
        # TODO: add multi order trade
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
                max_amount = -max_amount
                assert (
                    order.price == price
                ), f"order.price = {order.price} while it should be {price}"
                # TODO: this works for only one price order
                if order.quantity <= max_amount:
                    logger.info(
                        f"order.quantity = {order.quantity} while it should be <= {max_amount}"
                    )

                if self.player_position[product] + order.quantity > 0:
                    max_tradable = min(
                        order.quantity,
                        LIMIT_POSITIONS[product] - self.player_position[product],
                    )
                else:
                    max_tradable = max(
                        order.quantity,
                        -LIMIT_POSITIONS[product] - self.player_position[product],
                    )

                self.player_position[product] += max_tradable
                self.player_cash[product].append(
                    self.player_cash[product][-1] - order.price * max_tradable
                )

    def plot(self):
        base_folder = f"plots/round{self.round}/day{self.day}"
        # make sure path exists
        Path(base_folder).mkdir(parents=True, exist_ok=True)

        # Plot the PnL per product
        for product in PRODUCTS:
            plt.figure(dpi=1200)
            plt.plot(self.player_pnl[product], linewidth=0.5)
            plt.title(f"PnL {product}")
            plt.savefig(f"{base_folder}/pnl_{product}.png")

        # Plot the total
        plt.figure(dpi=1200)
        total_pnl = np.sum(
            [np.array(self.player_pnl[product]) for product in PRODUCTS], axis=0
        )
        plt.plot(total_pnl, linewidth=0.5)
        plt.title("PnL")
        plt.savefig(f"{base_folder}/pnl.png")

        # Plot the position
        plt.figure(dpi=1200)
        for product in PRODUCTS:
            plt.plot(self.player_position_history[product], label=product)
        plt.title("Position")
        plt.legend()
        plt.savefig(f"{base_folder}/position.png")

        # Plot the mid price per product
        for product in PRODUCTS:
            plt.figure(dpi=1200)

            df = self.df[
                (self.df["timestamp"].isin(self.timestamps))
                & (self.df["product"] == product)
            ]
            mid_price = df.mid_price.values
            # plt.plot(
            #     np.array(self.timestamps) / 100, mid_price, label=product, linewidth=0.5
            # )
            plt.plot(
                np.array(self.timestamps) / 100,
                df.bid_price_1,
                label="Bid Price",
                linewidth=0.5,
            )
            plt.plot(
                np.array(self.timestamps) / 100,
                df.ask_price_1,
                label="Ask Price",
                linewidth=0.5,
            )

            x = [xi[0] for xi in self.markers[product]]
            y = [xi[2] for xi in self.markers[product]]
            colors = [COLORS[xi[1]] for xi in self.markers[product]]
            if x:
                plt.scatter(x, y, c=colors)
            plt.legend()
            plt.savefig(f"{base_folder}/{product}.png")


@click.command()
@click.option("--round", default=1, help="Round to backtest")
@click.option("--steps", default=sys.maxsize, help="Number of steps to test against")
def main(round: int, steps: int):
    for file in os.listdir(f"data/round{round}"):
        day = file.split(".")[0]
        simulator = MarketSimulator(round, day)
        simulator.run(steps)


if __name__ == "__main__":
    main()
