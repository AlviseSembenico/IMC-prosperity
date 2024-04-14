import jsonpickle
import numpy as np
from numpy import linalg

from datamodel import Order, OrderDepth, TradingState

DEBUG = False


class LinearRegression:
    def __init__(self, lr: int = 0.01, n_iters: int = 1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        self.weights = linalg.lstsq(X, y, rcond=None)[0]
        return self


def set_debug(debug):
    global DEBUG
    DEBUG = debug


def get_last_price(order_depth: OrderDepth, type, previous: list[int]):
    values = getattr(order_depth, type + "_orders")
    if len(values) == 0:
        return previous[-1]
    return list(values.items())[0][0]


def compute_last_price(order_depth: OrderDepth):
    if len(order_depth.sell_orders) == 0 and len(order_depth.buy_orders) == 0:
        return None
    if len(order_depth.sell_orders) == 0:
        return list(order_depth.buy_orders.keys())[0]
    if len(order_depth.buy_orders) == 0:
        return list(order_depth.sell_orders.keys())[0]
    return (
        list(order_depth.buy_orders.keys())[0] + list(order_depth.sell_orders.keys())[0]
    ) / 2


def amethysts_policy(state: TradingState, order_depth: OrderDepth, previous_info: dict):
    product = "AMETHYSTS"
    orders = []
    acceptable_price = 10000
    info = {}
    marker = None

    current_position = state.position.get(product, 0)

    if len(order_depth.sell_orders) != 0:
        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        if int(best_ask) < acceptable_price:
            orders.append(Order(product, best_ask, -best_ask_amount))

        # # set portfolio to 0 if price is equal to acceptable price
        if int(best_ask) == acceptable_price and current_position < 0:
            # sell all
            orders.append(Order(product, acceptable_price, -current_position))

    if len(order_depth.buy_orders) != 0:
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        if int(best_bid) > acceptable_price:
            orders.append(Order(product, best_bid, -best_bid_amount))
        # set portfolio to 0 if price is equal to acceptable price
        if int(best_bid) == acceptable_price and current_position > 0:
            # discard the short position
            orders.append(Order(product, acceptable_price, -current_position))

    return orders, info, marker


def starfruits_policy(
    state: TradingState, order_depth: OrderDepth, previous_info: dict
):
    product = "STARFRUIT"
    orders = []
    spread_position = 5
    info = {}
    window_size = 10
    marker = None
    threshold = 0.003

    current_price = compute_last_price(order_depth)

    if len(previous_info["last_price"]) >= window_size:
        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

        Y = np.array(previous_info["last_price"][-window_size:]).reshape(-1, 1)
        model = LinearRegression()
        weight = np.ones(window_size)
        model_w = model.fit(
            np.linspace(0, window_size, num=window_size).reshape(-1, 1), Y
        )
        slope = model_w.weights[0]
        last_slopes = np.array(previous_info["generated"].get("last_slopes", []))
        # if 'operation' not in info['generated']
        if abs(slope) < threshold and all(np.abs(last_slopes[-40:]) > threshold):
            if last_slopes[-40:].mean() < -0.03:
                # buy
                # blue color
                orders.append(Order(product, best_ask, -best_ask_amount))
                info["operation"] = best_ask
                marker = (1, best_ask)
            elif last_slopes[-40:].mean() > 0.03:
                # sell
                # red color
                orders.append(Order(product, best_bid, -best_bid_amount))
                info["operation"] = -best_bid
                marker = (0, best_bid)

        slopes = previous_info["generated"].get("last_slopes", [])
        slopes.append(slope)
        previous_info["generated"]["last_slopes"] = slopes
    return orders, info, marker


products_mapping = {"AMETHYSTS": amethysts_policy, "STARFRUIT": starfruits_policy}


def generate_empty_info(products):
    return {
        product: {"last_price": [], "marker": [], "last_ask": [], "last_bid": []}
        for product in products
    }


class Trader:
    def run(self, state: TradingState):
        if not DEBUG:
            # print("State: " + str(state))
            # print("Observations: " + str(state.observations))
            try:
                previous_info = jsonpickle.decode(state.traderData)
            except:
                previous_info = None
        else:
            # We dont encode it during debug
            previous_info = state.traderData
        if previous_info is None:
            previous_info = generate_empty_info(state.order_depths.keys())
        # Orders to be placed on exchange matching engine
        result = {}
        markers = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders = []
            info = {}
            if product not in products_mapping:
                continue
            orders, info, marker = products_mapping[product](
                state, order_depth, previous_info.get(product)
            )
            result[product] = orders
            markers[product] = marker
            previous_info[product]["generated"] = (
                previous_info[product].get("generated", {}) | info
            )
            previous_info[product]["last_price"].append(compute_last_price(order_depth))
            previous_info[product]["last_ask"].append(
                get_last_price(order_depth, "sell", previous_info[product]["last_ask"])
            )
            previous_info[product]["last_bid"].append(
                get_last_price(order_depth, "buy", previous_info[product]["last_bid"])
            )
            previous_info[product]["marker"].append(marker)

        # Sample conversion request. Check more details below.
        conversions = 1
        if not DEBUG:
            previous_info = jsonpickle.encode(previous_info)
            return result, conversions, previous_info
        return result, conversions, previous_info, markers
