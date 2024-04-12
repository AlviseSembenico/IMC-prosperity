import json
from json import JSONEncoder
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import jsonpickle

Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int
Price = int
Volume = int


@dataclass
class Listing:
    symbol: Symbol
    product: Product
    denomination: Product


@dataclass
class ConversionObservation:
    bidPrice: float
    askPrice: float
    transportFees: float
    exportTariff: float
    importTariff: float
    sunlight: float
    humidity: float


@dataclass
class Observation:
    plainValueObservations: Dict[Product, ObservationValue]
    conversionObservations: Dict[Product, ConversionObservation]

    def __str__(self) -> str:
        return f"(plainValueObservations: {jsonpickle.encode(self.plainValueObservations)}, conversionObservations: {jsonpickle.encode(self.conversionObservations)})"


@dataclass
class Order:
    symbol: Symbol
    price: int
    quantity: int

    def __str__(self) -> str:
        return f"({self.symbol}, {self.price}, {self.quantity})"


@dataclass
class OrderDepth:
    buy_orders: Dict[Price, Volume]
    sell_orders: Dict[Price, Volume]

    def __repr__(self) -> str:
        return f"(buy_orders: {jsonpickle.encode(self.buy_orders)}, sell_orders: {jsonpickle.encode(self.sell_orders)})"


@dataclass
class Trade:
    symbol: Symbol
    price: int
    quantity: int
    buyer: Optional[UserId] = None
    seller: Optional[UserId] = None
    timestamp: int = 0

    def __str__(self) -> str:
        return f"({self.symbol}, {self.buyer} << {self.seller}, {self.price}, {self.quantity}, {self.timestamp})"

    def __repr__(self) -> str:
        return f"({self.symbol}, {self.buyer} << {self.seller}, {self.price}, {self.quantity}, {self.timestamp})"


@dataclass
class TradingState(object):
    traderData: str
    timestamp: Time
    listings: Dict[Symbol, Listing]
    order_depths: Dict[Symbol, OrderDepth]
    own_trades: Dict[Symbol, List[Trade]]
    market_trades: Dict[Symbol, List[Trade]]
    position: Dict[Product, Position]
    observations: Observation

    def toJSON(self):
        return json.dumps(self, default=lambda o: asdict(o), sort_keys=True)


class ProsperityEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__
