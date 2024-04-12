from typing import List, Any
import json
from dataclasses import dataclass
import jsonpickle
import numpy as np
from statistics import NormalDist

from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)


# fmt: off


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()


# fmt: on


EPS = 1e-4


@dataclass
class Sample:
    value: float
    weight: float

    def __post_init__(self):
        self.value = float(self.value)
        self.weight = float(self.weight)


@dataclass
class NormalDistribution:
    mean: np.floating
    std: np.floating

    @staticmethod
    def from_samples(samples: List[Sample]):
        values = np.array([sample.value for sample in samples])
        weights = np.array([sample.weight for sample in samples])

        mean = np.average(values, weights=weights)
        variance = np.average((values - mean) ** 2, weights=weights)
        std = np.sqrt(variance)

        return NormalDistribution(mean, std)

    def to_normal_dist(self):
        mu = float(self.mean)
        sigma = float(self.std)
        if sigma < EPS:
            sigma = EPS
        return NormalDist(mu=mu, sigma=sigma)


Product = str

HISTORY_LIMIT = 8

EPS = 1e-8


@dataclass
class OrderCandidate:
    price: int
    volume: int
    buy_pdf: float
    sell_pdf: float
    goodness: float = 0.0

    def __post_init__(self):
        # if self.volume > 0:
        #     self.goodness = self.sell_pdf / max(EPS, self.buy_pdf)
        # else:
        #     self.goodness = self.buy_pdf / max(EPS, self.sell_pdf)

        self.goodness = self.sell_pdf - self.buy_pdf
        if self.volume < 0:
            self.goodness = -self.goodness


@dataclass
class HistoricalData:
    timestamp: int
    position: int
    order_depth: OrderDepth
    own_trades: List[Trade]
    market_trades: List[Trade]
    best_order_candidates: List[OrderCandidate]


@dataclass
class History:
    data_points: List[HistoricalData]

    @staticmethod
    def empty():
        return History([])

    def append(self, data_point: HistoricalData):
        self.data_points.append(data_point)

    def prune(self):
        del self.data_points[:-HISTORY_LIMIT]

    def len(self):
        return len(self.data_points)


class Trader:
    def __init__(self):
        self.policies = {
            # "AMETHYSTS": self.policy,
            "AMETHYSTS": self.amethysts_policy,
            "STARFRUIT": self.policy,
        }

        self.limits = 20
        self.warmup = HISTORY_LIMIT
        self.risk = 1.0

    def none_policy(self, history: History, symbol: Product):
        return []

    def amethysts_policy(self, history: History, symbol: Product):
        if history.len() < self.warmup:
            return []

        orders = []

        # Calculate a weighted average of the past prices
        all_samples = []
        for dp in history.data_points:
            for price, volume in dp.order_depth.buy_orders.items():
                all_samples.append(Sample(value=price, weight=volume))

            for price, volume in dp.order_depth.sell_orders.items():
                all_samples.append(Sample(value=price, weight=volume))

        prices = np.array([sample.value for sample in all_samples])
        weights = np.abs(np.array([sample.weight for sample in all_samples]))

        # Calculate the weighted average
        acceptable_price = int(np.round(np.average(prices, weights=weights)))

        current = history.data_points[-1]
        current_position = current.position

        if len(current.order_depth.sell_orders) != 0:
            best_ask, best_ask_amount = list(current.order_depth.sell_orders.items())[0]
            if int(best_ask) < acceptable_price:
                orders.append(Order(symbol, best_ask, -best_ask_amount))

            # # set portfolio to 0 if price is equal to acceptable price
            if int(best_ask) == acceptable_price and current_position < 0:
                # sell all
                orders.append(Order(symbol, acceptable_price, -current_position))

        if len(current.order_depth.buy_orders) != 0:
            best_bid, best_bid_amount = list(current.order_depth.buy_orders.items())[0]
            if int(best_bid) > acceptable_price:
                orders.append(Order(symbol, best_bid, -best_bid_amount))
            # set portfolio to 0 if price is equal to acceptable price
            if int(best_bid) == acceptable_price and current_position > 0:
                # discard the short position
                orders.append(Order(symbol, acceptable_price, -current_position))

        return orders

    def policy(self, history: History, symbol: Product):
        current = history.data_points[-1]
        # print(current.timestamp, symbol)

        if history.len() < self.warmup:
            return []
        # Fit a distribution to the historical orders
        trade_samples = []
        buy_samples = []
        sell_samples = []
        past_goodnesses = []
        for dp in history.data_points[:-1]:
            for ot in dp.own_trades:
                trade_samples.append(Sample(value=ot.price, weight=ot.quantity))

            for mt in dp.market_trades:
                trade_samples.append(Sample(value=mt.price, weight=mt.quantity))

            for price, volume in dp.order_depth.buy_orders.items():
                buy_samples.append(Sample(value=price, weight=volume))

            for price, volume in dp.order_depth.sell_orders.items():
                sell_samples.append(Sample(value=price, weight=volume))

            for order_candidate in dp.best_order_candidates:
                past_goodnesses.append(order_candidate.goodness)

        # trading_distribution = NormalDistribution.from_samples(
        #     trade_samples
        # ).to_normal_dist()
        buy_distribution = NormalDistribution.from_samples(buy_samples).to_normal_dist()
        sell_distribution = NormalDistribution.from_samples(
            sell_samples
        ).to_normal_dist()

        # Calculate the desired orders
        # At a time we should only be buying or selling
        # Check which is better at this time
        # TODO: Also in the next step should we take an opposing action?

        buy_candidates = []
        sell_candidates = []

        for price, volume in current.order_depth.buy_orders.items():
            buy_pdf = buy_distribution.pdf(price)
            sell_pdf = sell_distribution.pdf(price)

            # We can "sell" this order
            buy_candidates.append(OrderCandidate(price, volume, buy_pdf, sell_pdf))

        for price, volume in current.order_depth.sell_orders.items():
            sell_pdf = sell_distribution.pdf(price)
            buy_pdf = buy_distribution.pdf(price)

            # We can "buy" this order
            sell_candidates.append(OrderCandidate(price, volume, buy_pdf, sell_pdf))

        best_buy_candidate = max(buy_candidates, key=lambda x: x.goodness)
        best_sell_candidate = max(sell_candidates, key=lambda x: x.goodness)

        # Store the best candidates
        if best_buy_candidate.goodness > best_sell_candidate.goodness:
            current.best_order_candidates.append(best_buy_candidate)
        else:
            current.best_order_candidates.append(best_sell_candidate)

        # if past_goodnesses:
        #     cutoff = np.mean(past_goodnesses)
        # else:

        cutoff = 0

        # TODO: right now we only act on existing orders
        if (best_buy_candidate.goodness < cutoff) and (
            best_sell_candidate.goodness < cutoff
        ):
            # these are pretty bad, so don't act on this
            ACTION = "NONE"
        elif (best_buy_candidate.goodness > cutoff) and (
            best_sell_candidate.goodness > cutoff
        ):
            # these are both pretty good, so choose one based on our current position
            # Always trying to close our position

            if current.position >= 0:
                ACTION = "SELL"
            else:
                ACTION = "BUY"
        elif best_buy_candidate.goodness > best_sell_candidate.goodness:
            # There's a better bid order than ask, so let's sell on it
            ACTION = "SELL"
        else:
            # There's a better ask order than bid, so let's buy on it
            ACTION = "BUY"

        if ACTION == "NONE":
            orders = []
        elif ACTION == "SELL":
            # Act on the buy order, and let's sell
            max_sell_volume = self.limits + current.position
            # sell_volume = min(max_sell_volume, best_buy_candidate.volume)
            sell_volume = max_sell_volume
            orders = [
                Order(
                    symbol=symbol, price=best_buy_candidate.price, quantity=-sell_volume
                )
            ]
        else:
            # Act on this sell order, and let's buy
            max_buy_volume = self.limits - current.position
            # buy_volume = min(max_buy_volume, -best_sell_candidate.volume)
            buy_volume = max_buy_volume
            orders = [
                Order(
                    symbol=symbol, price=best_sell_candidate.price, quantity=buy_volume
                )
            ]

        # print("Submitting order:", order)
        # __import__("pdb").set_trace()
        return orders

    def run(self, state: TradingState):
        symbols = list(state.listings.keys())
        try:
            previous_info = jsonpickle.decode(state.traderData)
        except:
            previous_info = {symbol: History.empty() for symbol in symbols}

        symbol2orders = {}
        for symbol in symbols:
            history = previous_info[symbol]
            history.append(
                HistoricalData(
                    timestamp=state.timestamp,
                    position=state.position.get(symbol, 0),
                    order_depth=state.order_depths[symbol],
                    own_trades=state.own_trades.get(symbol, []),
                    market_trades=state.market_trades.get(symbol, []),
                    best_order_candidates=[],  # This will be filled in policy
                )
            )

            policy = self.policies[symbol]
            desired_orders = policy(history, symbol)

            symbol2orders[symbol] = desired_orders

            history.prune()

        conversions = 0
        previous_info = jsonpickle.encode(previous_info)

        logger.flush(state, symbol2orders, conversions, "")

        return symbol2orders, conversions, previous_info
