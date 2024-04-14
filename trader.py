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

    return orders, info, marker, 0


def starfruits_policy(
    state: TradingState, order_depth: OrderDepth, previous_info: dict
):
    product = "STARFRUIT"
    orders = []
    spread_position = 5
    info = {}
    window_size = 100
    marker = None
    threshold = 0.003

    # from sklearn.linear_model import LinearRegression

    # current_price = compute_last_price(order_depth)

    # if len(previous_info["last_price"]) >= window_size:
    #     best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
    #     best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

    #     Y = np.array(previous_info["last_price"][-window_size:]).reshape(-1, 1)
    #     model = LinearRegression()
    #     weight = np.ones(window_size)
    #     model_w = model.fit(
    #         np.linspace(0, window_size, num=window_size).reshape(-1, 1), Y, weight
    #     )
    #     slope = model_w.coef_[0]
    #     last_slopes = np.array(previous_info["generated"].get("last_slopes", []))
    #     # if 'operation' not in info['generated']
    #     if abs(slope) < threshold and all(np.abs(last_slopes[-40:]) > threshold):
    #         if last_slopes[-40:].mean() < 0:
    #             # buy
    #             # blue color
    #             orders.append(Order(product, best_ask, -best_ask_amount))
    #             info["operation"] = best_ask
    #             marker = (1, best_ask)
    #         else:
    #             # sell
    #             # red color
    #             orders.append(Order(product, best_bid, -best_bid_amount))
    #             info["operation"] = -best_bid
    #             marker = (0, best_bid)

    #     slopes = previous_info["generated"].get("last_slopes", [])
    #     slopes.append(slope)
    #     previous_info["generated"]["last_slopes"] = slopes
    if len(order_depth.sell_orders) != 0:
        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]

        if len(previous_info["last_ask"]) >= window_size:
            # detect down spikes in ask price
            current_price_change = best_ask - previous_info["last_ask"][-1]
            last_price_changes = np.abs(
                np.diff(previous_info["last_ask"][-window_size:])
            )

            mean_last_bid = np.mean(previous_info["last_bid"][-window_size:])
            mean_last_ask = np.mean(previous_info["last_ask"][-window_size:])
            spread = mean_last_bid - mean_last_ask
            best_ask_n = (mean_last_bid - best_ask) / spread
            if (
                best_ask < previous_info["last_ask"][-1]
                and abs(current_price_change) > (np.mean(last_price_changes) * 2)
                and best_ask_n < 0.3
            ):
                marker = (1, best_ask)
                orders.append(Order(product, best_ask, -best_ask_amount))
                info["last_purchase"] = best_ask

    if len(order_depth.buy_orders) != 0 and marker is None:
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

        if len(previous_info["last_bid"]) >= window_size:
            # detect down spikes in ask price
            current_price_change = best_bid - previous_info["last_bid"][-1]
            last_price_changes = np.abs(
                np.diff(previous_info["last_bid"][-window_size:])
            )

            mean_last_bid = np.mean(previous_info["last_bid"][-window_size:])
            mean_last_ask = np.mean(previous_info["last_ask"][-window_size:])
            spread = mean_last_bid - mean_last_ask
            best_bid_n = (best_bid - mean_last_ask) / spread
            if (
                best_bid > previous_info["last_bid"][-1]
                and abs(current_price_change) > (np.mean(last_price_changes) * 2)
                and best_bid_n < 0.6
            ):
                marker = (2, best_bid)
                orders.append(Order(product, best_bid, -best_bid_amount))
                info["last_sell"] = best_ask

    return orders, info, marker, 0


def orchid_policty(state: TradingState, order_depth: OrderDepth, previous_info: dict):
    orders = []
    info = {}
    marker = None
    conversion = None

    product = "ORCHID"
    # arbitrage
    import_cost = state.observations.conversionObservations[product].importTariff
    export_cost = state.observations.conversionObservations[product].exportTariff
    transport_cost = state.observations.conversionObservations[product].transportFees

    best_external_bid = (
        state.observations.conversionObservations[product].bidPrice
        + import_cost
        + transport_cost
    )
    best_external_ask = (
        state.observations.conversionObservations[product].askPrice
        + export_cost
        + transport_cost
    )

    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]

    if best_ask < best_external_bid:
        orders.append(Order(product, best_ask, -best_ask_amount))
        conversion = best_ask_amount
    if best_bid > best_external_ask:
        orders.append(Order(product, best_bid, -best_bid_amount))
        conversion = -best_bid_amount

    return orders, info, marker, conversion


products_mapping = {
    "AMETHYSTS": amethysts_policy,
    "STARFRUIT": starfruits_policy,
    "ORCHID": orchid_policty,
}


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
        total_conversions = 0
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders = []
            info = {}
            if product not in products_mapping:
                continue
            orders, info, marker, conversion = products_mapping[product](
                state, order_depth, previous_info.get(product)
            )
            total_conversions += conversion
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
        if not DEBUG:
            previous_info = jsonpickle.encode(previous_info)
            return result, total_conversions, previous_info
        return result, total_conversions, previous_info, markers
