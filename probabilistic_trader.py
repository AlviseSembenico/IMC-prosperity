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

HISTORY_LIMIT = 30


@dataclass
class OrderCandidate:
    price: int
    volume: int
    cdf: float


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
            "AMETHYSTS": self.policy,
            "STARFRUIT": self.policy,
        }

        self.limits = 20
        self.warmup = HISTORY_LIMIT
        self.risk = 1.0

    def none_policy(self, history: History, symbol: Product):
        return []

    def policy(self, history: History, symbol: Product):
        if history.len() < self.warmup:
            return []
        # Fit a distribution to the historical orders
        trade_samples = []
        buy_samples = []
        sell_samples = []
        past_cdfs = []
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
                past_cdfs.append(order_candidate.cdf)

        # trading_distribution = NormalDistribution.from_samples(
        #     trade_samples
        # ).to_normal_dist()
        buy_distribution = NormalDistribution.from_samples(buy_samples).to_normal_dist()
        sell_distribution = NormalDistribution.from_samples(
            sell_samples
        ).to_normal_dist()

        current = history.data_points[-1]

        # Calculate the desired orders
        # At a time we should only be buying or selling
        # Check which is better at this time

        buy_candidates = []
        sell_candidates = []

        # __import__('pdb').set_trace()
        for price, volume in current.order_depth.buy_orders.items():
            # What's the probability of this buy price being
            # high given the sell distribution
            pdf = sell_distribution.pdf(price)
            # cdf = 1 - sell_distribution.cdf(price)
            # We can "sell" this order
            buy_candidates.append(OrderCandidate(price, volume, pdf))

        for price, volume in current.order_depth.sell_orders.items():
            # What's the probability of this sell price being
            # low given the buy distribution
            pdf = buy_distribution.pdf(price)
            # cdf = buy_distribution.cdf(price)
            # We can "buy" this order
            sell_candidates.append(OrderCandidate(price, volume, pdf))

        # Select the best candidate - this will deterimine if we buy or sell
        best_candidate = max(buy_candidates + sell_candidates, key=lambda x: x.cdf)


        current.best_order_candidates.append(best_candidate)

        cdfs = np.array(past_cdfs)
        if len(cdfs) == 0:
            return []

        # Check if we should act on this
        if np.random.uniform(0.0, cdfs.max()) > (best_candidate.cdf * self.risk):
            # print("NoBUY")
            return []

        # TODO: right now we only act on existing orders
        orders = []
        if best_candidate.volume > 0:
            # This is a buy order (volume is positive)
            # So let's sell on this

            max_sell_volume = self.limits + current.position

            sell_volume = min(max_sell_volume, best_candidate.volume)
            orders.append(
                Order(symbol=symbol, price=best_candidate.price, quantity=-sell_volume)
            )
        else:
            # this is a sell order (volume is negative)
            # So let's buy on this
            max_buy_volume = self.limits - current.position
            buy_volume = min(max_buy_volume, -best_candidate.volume)
            orders.append(
                Order(symbol=symbol, price=best_candidate.price, quantity=buy_volume)
            )
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
