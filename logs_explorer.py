from pathlib import Path
from dataclasses import dataclass
import jsonpickle
import json
from io import StringIO

from collections import Counter, defaultdict

from typing import Dict, List
import pandas as pd

from icecream import ic

from visualize import Order, create_order_grid, order_visualizer

import pyvista as pv

import numpy as np


SECTIONS = [
    "Sandbox logs:",
    "Activities log:",
    "Trade History:",
]

SUBMISSIONS = Path("submissions")
BACKTESTS = Path("backtests")


submission = sorted(list(BACKTESTS.iterdir()))[-1]

## this is the result of mine
# submission = SUBMISSIONS / "67e76d9f-f4ce-42dd-a82d-2c47f5c98f62"

## this is the result of Alvise
submission = SUBMISSIONS / "c1ec518f-ca42-4fbf-b607-a848b7aa48ef"


## 

# submission = BACKTESTS / "2024-04-12_03-59-26"

submission = BACKTESTS / "2024-04-12_04-42-46"

## mix submission
submission = SUBMISSIONS / "a36e6a41-6a19-4335-a273-701902790d9f"

# BACKTESTS / 

print(submission)

# PRODUCT = "AMETHYSTS"
PRODUCT = "STARFRUIT"


# csv = submission.with_suffix(".csv")
# orders_df = pd.read_csv(csv, sep=";")

log = submission.with_suffix(".log")


# order_visualizer(bids_time2orders, asks_time2orders)


# ###########################################
# Parse the df into bid and ask orders
# ###########################################
with log.open() as f:
    contents = f.read()

remaining = contents
pre, separator, remaining = remaining.partition("Sandbox logs:")
sandbox_logs, separator, remaining = remaining.partition("Activities log:")
activities_log, separator, remaining = remaining.partition("Trade History:")
trade_history = remaining


orders_df = pd.read_csv(StringIO(activities_log), sep=";")

# ###########################################
# Parse the df into bid and ask orders
# ###########################################
bids_time2orders = {}
asks_time2orders = {}
for i, row in orders_df.iterrows():
    if row["product"] != PRODUCT:
        continue
    #
    timestamp = row["timestamp"]
    bids = []
    asks = []
    #
    for idx in [1, 2, 3]:
        bid_price = row[f"bid_price_{idx}"]
        bid_volume = row[f"bid_volume_{idx}"]
        ask_price = row[f"ask_price_{idx}"]
        ask_volume = row[f"ask_volume_{idx}"]
        # check for nan
        if not np.isnan(bid_price):
            bids.append(Order(price=bid_price, volume=bid_volume))
            asks.append(Order(price=ask_price, volume=ask_volume))
    #
    bids_time2orders[timestamp] = bids
    asks_time2orders[timestamp] = asks


history = eval(trade_history)


@dataclass
class Trade:
    timestamp: int
    buyer: str
    seller: str
    product: str
    price: int
    volume: int


trades = [
    Trade(
        timestamp=h["timestamp"],
        buyer=h["buyer"],
        seller=h["seller"],
        product=h["symbol"],
        price=h["price"],
        volume=h["quantity"],
    )
    for h in history
]

bot_trades = []
own_trades = []

for trade in trades:
    if trade.buyer == "SUBMISSION" or trade.seller == "SUBMISSION":
        own_trades.append(trade)
    else:
        bot_trades.append(trade)

sellers = Counter([t.seller for t in trades])
buyers = Counter([t.buyer for t in trades])

own_trades = [t for t in own_trades if t.product == PRODUCT]
# for trade in own_trades:
#     if trade.buyer == "SUBMISSION":
#         trade.volume = -trade.volume


time2trades = defaultdict(list)

for trade in own_trades:
    time2trades[trade.timestamp].append(Order(price=trade.price, volume=trade.volume))

# # are there 2 trades in any timestamp?
# for timestamp, trades in time2trades.items():
#     if len(trades) > 1:
#         print(timestamp, trades)


order_visualizer(bids_time2orders, asks_time2orders, time2trades)
