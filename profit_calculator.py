from typing import List, Dict
from dataclasses import dataclass, field
from pathlib import Path
from itertools import count
from collections import defaultdict

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from icecream import ic


from visualize import Order as OrderViz, order_visualizer


DEBUG = False


@dataclass
class Profit:
    value: float


## Some Examples
# Order with positive quantity is a buy
#            negative               sell

## Example 1: buy and sell all
# order 10 at 100
# Position = 10@100
# order -10 at 102
# Position = 0@
# Profit = delta_quantity_cover * delta_price = (-10) * (100 - 102) = 20

## Example 2: Buy then sell more
# order 10 at 100
# Position = 10@100
# order -20 at 102
# Position = -10@102
# Profit = delta_quantity_cover * delta_price = (-10) * (100 - 102) = 20
# order 20 at 100
# Position = 10@100
# Profit = delta_quantity_cover * delta_price = 10 * (102 - 100) = 20


@dataclass
class Order:
    price: float
    quantity: float

    def is_valid(self):
        return (not np.isnan(self.price)) and (not np.isnan(self.quantity))

    def area(self):
        return self.price * self.quantity

    def cover(self, other: "Order"):
        if not DEBUG:
            ic = lambda *args, **kwargs: None


        # Special cases: self or other has quantity = 0
        no_pnl = Profit(0)

        if self.quantity == 0:
            return other, no_pnl

        if other.quantity == 0:
            return self, no_pnl

        # Are we covering here or averaging?
        ic(self.quantity * other.quantity)
        if self.quantity * other.quantity > 0:
            # They have the same sign, so we are averaging
            sum_quantities = self.quantity + other.quantity
            price = (self.area() + other.area()) / sum_quantities
            order = Order(price=price, quantity=sum_quantities)
            ic(sum_quantities)
            ic(order)
            return order, no_pnl
        else:
            # The smaller of the two quantities will be covered
            if abs(other.quantity) >= abs(self.quantity):
                cover_quantity = -self.quantity
                new_price = other.price
            else:
                cover_quantity = other.quantity
                new_price = self.price
            ic(cover_quantity)
            ic(new_price)

            remaining_quantity = self.quantity + other.quantity
            ic(remaining_quantity)
            order = Order(price=new_price, quantity=remaining_quantity)
            ic(order)

            price_delta = self.price - other.price

            ic(price_delta)
            profit = Profit(cover_quantity * price_delta)
            ic(profit)

            return order, profit


@dataclass
class Position:
    outstanding: Order = field(default_factory=lambda: Order(0.0, 0.0))
    profits: List = field(default_factory=list)

    def __add__(self, other: Order) -> "Position":
        new_outstanding, profit = self.outstanding.cover(other)
        profits = self.profits + [profit]
        return Position(outstanding=new_outstanding, profits=profits)

    def total_profit(self):
        total_profit = sum([profit.value for profit in self.profits])
        return total_profit


def get_day(file: Path):
    return file.stem.split("_")[4]


@dataclass
class Day:
    day: int
    prices: pd.DataFrame
    trades: pd.DataFrame


def examples():
    pos = Position()

    no = Order(price=0, quantity=0)
    bl = Order(price=100, quantity=10)
    sh = Order(price=150, quantity=-10)

    bh = Order(price=200, quantity=10)
    sl = Order(price=100, quantity=-10)

    ic(pos + bl + bh + sh)

    ic(bl)
    ic(sh)

    ic("[No position]")
    ic(no.cover(bl))

    ic("[buy low, sell high]")
    ic(bl.cover(sh))

    ic("[sell high, buy low]")
    ic(sh.cover(bl))

    ic("[Average up]")
    ic(bl.cover(bh))

    ic("[Average down]")
    ic(bh.cover(bl))

    ic("[buy high, sell low]")
    ic(bh.cover(sl))

    ic("[sell low, buy high]")
    ic(sl.cover(bh))


if __name__ == "__main__":
    # TODO: Change this to the actual product
    PRODUCT = "AMETHYSTS"
    # PRODUCT = "STARFRUIT"

    data_folder = Path("data/round-1-island-data")
    files = sorted(list(data_folder.glob("*.csv")))

    prices_files = [file for file in files if file.stem.startswith("prices")]
    trades_files = [file for file in files if file.stem.startswith("trades")]

    # Parse the csv files into pd.DataFrames
    days = []
    for price_file, trade_file in zip(prices_files, trades_files):
        day = get_day(price_file)
        day_t = get_day(trade_file)
        assert day == day_t
        prices = pd.read_csv(price_file, sep=";")
        trades = pd.read_csv(trade_file, sep=";")
        day = Day(int(day), prices, trades)
        days.append(day)

    days.sort(key=lambda day: day.day)

    # We are only looking at the first day for now
    d = days[0]

    # Parse the DataFrame into Orders
    buy_time2orders: Dict[int, List[Order]] = {}
    sell_time2orders: Dict[int, List[Order]] = {}
    for i, row in d.prices.iterrows():
        if row["product"] != PRODUCT:
            continue
        #
        timestamp = row["timestamp"]
        all_buy_orders = []
        all_sell_orders = []
        for idx in [1, 2, 3]:
            buy_price = row[f"bid_price_{idx}"]
            buy_volume = row[f"bid_volume_{idx}"]
            market_order = Order(price=buy_price, quantity=buy_volume)
            if market_order.is_valid():
                all_buy_orders.append(market_order)

            sell_price = row[f"ask_price_{idx}"]
            sell_volume = row[f"ask_volume_{idx}"]
            market_order = Order(price=sell_price, quantity=-sell_volume)
            if market_order.is_valid():
                all_sell_orders.append(market_order)

        buy_time2orders[timestamp] = all_buy_orders
        sell_time2orders[timestamp] = all_sell_orders

    # Calculate the max theoretical profit (ignoring position limits)
    pos = Position()

    all_buy_orders = [
        order for day_orders in buy_time2orders.values() for order in day_orders
    ]

    all_sell_orders = [
        order for day_orders in sell_time2orders.values() for order in day_orders
    ]

    # We want to "buy low" on the sell orders, so after sorting, the first ones are the best sell orders
    all_sell_orders.sort(key=lambda order: order.price)
    all_buy_orders.sort(key=lambda order: order.price, reverse=True)

    if DEBUG:
        sell_prices = np.array([order.price for order in all_sell_orders])
        buy_prices = np.array([order.price for order in all_buy_orders])
        # Visualize the prices
        plt.plot(sell_prices, label="sell")
        plt.plot(buy_prices, label="buy")
        plt.legend()
        plt.show()

        bids_time2orders = defaultdict(list)
        asks_time2orders = defaultdict(list)
        time2trades = {}
        for i, bid_order in enumerate(all_buy_orders):
            bids_time2orders[i * 100].append(
                OrderViz(price=bid_order.price, volume=bid_order.quantity)
            )

        for i, ask_order in enumerate(all_sell_orders):
            asks_time2orders[i * 100].append(
                OrderViz(price=ask_order.price, volume=ask_order.quantity)
            )

        order_visualizer(bids_time2orders, asks_time2orders, None)

    # Calculate the theoretical max profit of a position

    # We can "buy" on sell orders and "sell" on buy orders
    # The goal is to maximize the profit, so we want to "buy low; sell high"
    # Or "sell high; buy low"
    pos = Position()
    buy_idx_gen = count()
    sell_idx_gen = count()

    while True:
        if pos.outstanding.quantity >= 0:
            # We have a long position, so we want to sell
            # "sell" == act on a buy_order, since our position is positive
            idx: int = next(buy_idx_gen)
            market_order = all_buy_orders[idx]
        else:
            # We want to "buy" == act on a sell_order, since our position is negative
            idx: int = next(sell_idx_gen)
            market_order = all_sell_orders[idx]

        # This is our order, it should be "sell high; buy low", so
        # if it's quantity < 0, the price should be high
        #         quantity > 0,                     low
        our_order = Order(price=market_order.price, quantity=-market_order.quantity)

        next_position = pos + our_order

        # if the profit is still positive at this step, we continue, otherwise stop
        if next_position.profits[-1].value >= 0:
            pos = next_position

        else:
            break

    ic(pos.total_profit())
