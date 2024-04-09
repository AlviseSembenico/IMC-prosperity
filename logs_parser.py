from pathlib import Path
import jsonpickle
import json

from typing import Dict, List

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

submission = SUBMISSIONS / "23566cae-8e27-47fe-86da-42d7df0daf25"


csv = submission.with_suffix(".csv")
log = submission.with_suffix(".log")


if __name__ == "__main__":
    log_file = Path("logs/no_trade_log_dump.log")

    log_file = Path("logs/day1_no_trade_dump.log")
    with log_file.open() as f:
        contents = f.read()

    remaining = contents
    pre, separator, remaining = remaining.partition("Sandbox logs:")
    sandbox_logs, separator, remaining = remaining.partition("Activities log:")
    activities_log, separator, remaining = remaining.partition("Trade History:")
    trade_history = remaining

    def parse_next_sandbox_log(contents: str):
        next, separator, remaining = contents.partition("\n{")
        # wtf = "}"
        return next, remaining

    empty, remaining = parse_next_sandbox_log(sandbox_logs)

    parsed_logs = []
    while remaining:
        next_log, remaining = parse_next_sandbox_log(remaining)
        f = eval(f"{{{next_log}")  # }}
        log_str = f["lambdaLog"]
        try:
            log = eval(log_str)
            parsed_logs.append(log)
        except SyntaxError:
            print(f"Could not parse log: {log_str}")

    log = parsed_logs[0]
    ic(log)

    # %%
    # Visualization starts here

    buy_time2orders = {}
    sell_time2orders = {}
    timestamps = []
    prices = []
    volumes = []
    SAMPLE_N = 1000
    SAMPLE_N = None
    for log in parsed_logs[:SAMPLE_N]:
        # PRODUCT = "AMETHYSTS"
        PRODUCT = "STARFRUIT"
        buy_orders = log["order_depths"][PRODUCT]["buy_orders"]
        sell_orders = log["order_depths"][PRODUCT]["sell_orders"]
        timestamp = log["timestamp"]
        timestamps.append(timestamp)
        buy_order_objects = []
        for price, volume in buy_orders.items():
            buy_order_objects.append(Order(price=price, volume=volume))
            prices.append(int(price))
            volumes.append(volume)
        #
        sell_order_objects = []
        for price, volume in sell_orders.items():
            sell_order_objects.append(Order(price=price, volume=volume))
            prices.append(int(price))
            volumes.append(volume)
        #
        buy_time2orders[timestamp] = buy_order_objects
        sell_time2orders[timestamp] = sell_order_objects

    order_visualizer(buy_time2orders, sell_time2orders)
