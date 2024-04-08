import jsonpickle
import numpy as np

from datamodel import Order, OrderDepth, TradingState

DEBUG = False


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


def forecast_bid(last_bids, n=20):
    x_bid = np.array(
        [
            0.08318381,
            0.08184419,
            0.052289,
            0.06817514,
            0.0789269,
            0.11973523,
            0.12258035,
            0.18677331,
            0.2070171,
        ]
    )
    b_bid = -2.7332801915326854
    x = last_bids[-9:].tolist()
    res = []
    for i in range(n):

        value = np.dot(x_bid, np.array(x) + b_bid)
        res.append(value)
        x.pop(0)
        x.append(int(value))
    return res


def forecast_ask(last_asks, n=20):
    x_ask = np.array(
        [
            0.07261305,
            0.08289478,
            0.04351741,
            0.08481505,
            0.09199201,
            0.10752486,
            0.13582492,
            0.18416474,
            0.1969992,
        ]
    )
    b_ask = -1.8446892638021382
    x = last_asks[-9:].tolist()
    res = []
    for i in range(n):

        value = np.dot(x_ask, np.array(x) + b_ask)
        res.append(value)
        x.pop(0)
        x.append(int(value))
    return res


def starfruits_policy(
    state: TradingState, order_depth: OrderDepth, previous_info: dict
):
    product = "STARFRUIT"
    orders = []
    info = {}
    window_size = 20
    marker = None
    base_position = -15

    # run regression

    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

    forecasted_mid_price = len(previous_info["last_ask"]) * -0.02382699 + 5004.11732383

    if best_ask < forecasted_mid_price:
        orders.append(Order(product, best_ask, -best_ask_amount))
    if best_bid > forecasted_mid_price:
        orders.append(Order(product, best_bid, -best_bid_amount))

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
            print("State: " + str(state))
            print("Observations: " + str(state.observations))
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
