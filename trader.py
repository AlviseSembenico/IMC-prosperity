import jsonpickle

from datamodel import Order, OrderDepth, TradingState

TRADER_DATA = "trader_data"


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
    return []
    product = "AMETHYSTS"
    orders = []
    acceptable_price = 10000
    info = {}

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

    return orders, info


def starfruits_policy(
    state: TradingState, order_depth: OrderDepth, previous_info: dict
):
    product = "STARFRUIT"
    orders = []
    acceptable_price = 0
    info = {}

    if len(order_depth.sell_orders) != 0:
        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        if int(best_ask) < acceptable_price:
            orders.append(Order(product, best_ask, -best_ask_amount))

    if len(order_depth.buy_orders) != 0:
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        if int(best_bid) > acceptable_price:
            orders.append(Order(product, best_bid, -best_bid_amount))

    return orders, info


products_mapping = {"AMETHYSTS": amethysts_policy, "STARFRUIT": starfruits_policy}


def generate_empty_info(products):
    return {product: {"last_price": []} for product in products}


class Trader:

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        previous_info = jsonpickle.decode(state.traderData)
        if previous_info is None:
            previous_info = generate_empty_info(state.order_depths.keys())

        # Orders to be placed on exchange matching engine
        result = {}
        product_info = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders = []
            orders, info = products_mapping[product](
                state, order_depth, previous_info.get(product, {})
            )
            result[product] = orders
            product_info[product] = info
            previous_info[product]["last_price"] = compute_last_price(order_depth)

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, jsonpickle.encode(product_info)
