from dataclasses import dataclass
from typing import Literal

Side = Literal["BID", "ASK"]
OrderType = Literal["LIMIT", "MARKET"]
Liquidity = Literal["MAKER", "TAKER"]

@dataclass
class Order:
    id: int
    side: Side
    px: int #Integer Ticks
    qty: int
    ts: int #Integer Microseconds
    owner: str = "SIM"
    order_type: OrderType = "LIMIT"


@dataclass
class Fill:
    order_id: int
    px: int
    qty: int
    liquidity: Liquidity
    ts: int