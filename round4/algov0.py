from datamodel import OrderDepth, UserId, TradingState, Order, Symbol, Listing, Trade, Observation, ProsperityEncoder
from typing import List, Dict, Any
import string
import json
import jsonpickle
import math
import numpy as np

"""
Update: Code works well for Basket 1 just needs optimising
Now: since Basket 2 doesn't trade well on a synthetic orderbook we need to trade it against
     the other basket since there is correlation
"""

CUR_DAY = 4
class MarketData:
    end_pos: Dict[str, int] = {}
    buy_sum: Dict[str, int] = {}
    sell_sum: Dict[str, int] = {}
    bid_prices: Dict[str, List[float]] = {}
    bid_volumes: Dict[str, List[int]] = {}
    ask_prices: Dict[str, List[float]] = {}
    ask_volumes: Dict[str, List[int]] = {}
    fair: Dict[str, float] = {}


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list:
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

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    DJEMBES = "DJEMBES"
    CROISSANT = "CROISSANTS"
    JAMS = "JAMS"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    ARTIFICAL1 = "ARTIFICAL1"
    ARTIFICAL2 = "ARTIFICAL2"
    SPREAD1 = "SPREAD1"
    SPREAD2 = "SPREAD2"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"


PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 1,
        "soft_position_limit": 50,  # 30
    },
    Product.KELP: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": False,
        "adverse_volume": 15,  # 20 - doesn't work as great
        "reversion_beta": -0.18,  # -0.2184
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 1,
        "ink_adjustment_factor": 0.05,
    },
    Product.SQUID_INK: {
        "take_width": 2,
        "clear_width": 1,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "reversion_beta": -0.228,
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 1,
        "spike_lb": 3,
        "spike_ub": 5.6,
        "offset": 2,
        "reversion_window": 55,  # altered
        "reversion_weight": 0.12,
    },
    Product.SPREAD1: {
        "default_spread_mean": 48.777856,
        "default_spread_std": 85.119723,
        "spread_window": 55,
        "zscore_threshold": 4,
        "target_position": 100,
    },
    Product.SPREAD2: {
        "default_spread_mean": 30.2336,
        "default_spread_std": 59.8536,
        "spread_window": 59,
        "zscore_threshold": 6,
        "target_position": 100,
    },
    Product.PICNIC_BASKET1: {
        "b2_adjustment_factor": 0.05
    },
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "mean_volatility": 0.119077,
        # "threshold": 0.1,
        "strike": 10000,  # this might be silly but profits are much higher with this strike
        # I do not know why that is the case for all of them
        # You can try if you want the actual strike but it makes a loss
        # to note I've uploaded this exactly after I have managed to get the code to run
        # I will look into this now along with mean-reverting deltas for selected vouchers
        "starting_time_to_expiry": 7 / 365,
        "std_window": 6,
        "z_score_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.147417,
        # "threshold": 0,
        "strike": 10000,
        "starting_time_to_expiry": 7 / 365,
        "std_window": 6,
        "z_score_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 0.140554,
        # "threshold": 0.01,
        "strike": 10000,
        "starting_time_to_expiry": 7 / 365,
        "std_window": 6,
        "z_score_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 0.128666,
        # "threshold": 0.728011,
        "strike": 10000,
        "starting_time_to_expiry": 7 / 365,
        "std_window": 6,
        "z_score_threshold": 25,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 0.127146,
        # "threshold": 0.0552,
        "strike": 10000,
        "starting_time_to_expiry": 7 / 365,
        "std_window": 6,
        "z_score_threshold": 25,
    },

}

PICNIC1_WEIGHTS = {
    Product.DJEMBES: 1,
    Product.CROISSANT: 6,
    Product.JAMS: 3,
}
PICNIC2_WEIGHTS = {
    Product.CROISSANT: 4,
    Product.JAMS: 2}

from math import log, sqrt, exp, erf, pi


# from statistics import NormalDist


class BlackScholes:
    @staticmethod
    def NormalDistcdf(x: float) -> float:
        return 0.5 * (1 + erf(x / math.sqrt(2)))

    @staticmethod
    def NormalDistpdf(x: float) -> float:
        return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
                     log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
             ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = (spot * BlackScholes.NormalDistcdf(d1) -
                      strike * BlackScholes.NormalDistcdf(d2))
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
                volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * BlackScholes.NormalDistcdf(-d2) - spot * BlackScholes.NormalDistcdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
                     log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
             ) / (volatility * sqrt(time_to_expiry))
        return BlackScholes.NormalDistcdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (
                     log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
             ) / (volatility * sqrt(time_to_expiry))
        return BlackScholes.NormalDistpdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (
                     log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
             ) / (volatility * sqrt(time_to_expiry))
        # print(f"d1: {d1}")
        # print(f"vol: {volatility}")
        # print(f"spot: {spot}")
        # print(f"strike: {strike}")
        # print(f"time: {time_to_expiry}")
        return BlackScholes.NormalDistpdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(
            call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS

        self.params = params
        self.PRODUCT_LIMIT = {Product.RAINFOREST_RESIN: 50,
                              Product.KELP: 50,
                              Product.SQUID_INK: 50,
                              Product.CROISSANT: 250,
                              Product.JAMS: 350,
                              Product.DJEMBES: 60,
                              Product.PICNIC_BASKET1: 60,
                              Product.PICNIC_BASKET2: 100,
                              Product.VOLCANIC_ROCK: 400,
                              Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
                              Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
                              Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
                              Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
                              Product.VOLCANIC_ROCK_VOUCHER_10500: 200}

    def take_best_orders(self, product: str,
                         fair_value: str, take_width: float,
                         orders: List[Order], order_depth: OrderDepth,
                         position: int, buy_order_volume: int,
                         sell_order_volume: int,
                         prevent_adverse: bool = False, adverse_volume: int = 0, traderObject: dict = None
                         ):
        # Check if squid ink, if it is then check for jumps and bet to go away from jump. (Look for big diff)

        position_limit = self.PRODUCT_LIMIT[product]

        if product == "SQUID_INK":
            if "currentSpike" not in traderObject:
                traderObject["currentSpike"] = False
            prev_price = traderObject["ink_last_price"]
            if traderObject["currentSpike"]:
                if abs(fair_value - prev_price) < self.params[Product.SQUID_INK]["spike_lb"]:
                    traderObject["currentSpike"] = False
                else:
                    if fair_value < traderObject["recoveryValue"]:
                        best_ask = min(order_depth.sell_orders.keys())
                        best_ask_amount = order_depth.sell_orders[best_ask]
                        quantity = max(best_ask_amount, position_limit + position)
                        if quantity > 0:
                            orders.append(Order(product, best_ask, quantity))
                            buy_order_volume += quantity
                            order_depth.sell_orders[best_ask] += quantity
                            if order_depth.sell_orders[best_ask] == 0:
                                del order_depth.buy_orders[best_ask]
                        if best_ask_amount > position + position_limit:
                            # Try second-best bid if leftover space
                            best_ask = max(list(filter(lambda x: x != best_ask, order_depth.sell_orders.keys())))
                            best_ask_amount = order_depth.buy_orders[best_ask]
                            quantity = max(best_ask_amount, position_limit + position)
                            if quantity > 0:
                                orders.append(Order(product, best_ask, quantity))
                                buy_order_volume += quantity
                                order_depth.sell_orders[best_ask] += quantity
                                if order_depth.sell_orders[best_ask] == 0:
                                    del order_depth.buy_orders[best_ask]
                        return buy_order_volume, 0
                    else:
                        best_bid = max(order_depth.buy_orders.keys())
                        best_bid_amount = order_depth.buy_orders[best_bid]
                        quantity = max(best_bid_amount, position_limit + position)
                        if quantity > 0:
                            orders.append(Order(product, best_bid, -1 * quantity))
                            sell_order_volume += quantity
                            order_depth.buy_orders[best_bid] -= quantity
                            if order_depth.buy_orders[best_bid] == 0:
                                del order_depth.buy_orders[best_bid]
                        if best_bid_amount > position + position_limit:
                            # Try second-best bid if leftover space
                            best_bid = max(list(filter(lambda x: x != best_bid, order_depth.buy_orders.keys())))
                            best_bid_amount = order_depth.buy_orders[best_bid]
                            quantity = max(best_bid_amount, position_limit + position)
                            if quantity > 0:
                                orders.append(Order(product, best_bid, -1 * quantity))
                                sell_order_volume += quantity
                                order_depth.buy_orders[best_bid] -= quantity
                                if order_depth.buy_orders[best_bid] == 0:
                                    del order_depth.buy_orders[best_bid]
                        return 0, sell_order_volume
            if abs(fair_value - prev_price) > self.params[Product.SQUID_INK]["spike_ub"]:
                traderObject["currentSpike"] = True
                traderObject["recoveryValue"] = prev_price + self.params[Product.SQUID_INK][
                    "offset"] if fair_value > prev_price else prev_price - self.params[Product.SQUID_INK]["offset"]
                # Main spike
                if fair_value > prev_price:
                    # Spike up, so sell bids until capacity reached
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_amount = order_depth.buy_orders[best_bid]
                    quantity = max(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
                    if best_bid_amount > position + position_limit:
                        # Try second-best bid if leftover space
                        best_bid = max(list(filter(lambda x: x != best_bid, order_depth.buy_orders.keys())))
                        best_bid_amount = order_depth.buy_orders[best_bid]
                        quantity = max(best_bid_amount, position_limit + position)
                        if quantity > 0:
                            orders.append(Order(product, best_bid, -1 * quantity))
                            sell_order_volume += quantity
                            order_depth.buy_orders[best_bid] -= quantity
                            if order_depth.buy_orders[best_bid] == 0:
                                del order_depth.buy_orders[best_bid]
                    return 0, sell_order_volume
                else:
                    # Spike down, so buy asks until capacity reached
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_amount = order_depth.sell_orders[best_ask]
                    quantity = max(best_ask_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.buy_orders[best_ask]
                    if best_ask_amount > position + position_limit:
                        # Try second-best bid if leftover space
                        best_ask = max(list(filter(lambda x: x != best_ask, order_depth.sell_orders.keys())))
                        best_ask_amount = order_depth.buy_orders[best_ask]
                        quantity = max(best_ask_amount, position_limit + position)
                        if quantity > 0:
                            orders.append(Order(product, best_ask, quantity))
                            buy_order_volume += quantity
                            order_depth.sell_orders[best_ask] += quantity
                            if order_depth.sell_orders[best_ask] == 0:
                                del order_depth.buy_orders[best_ask]
                    return buy_order_volume, 0

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(self, product: str,
                    orders: List[Order],
                    bid: int, ask: int, position: int,
                    buy_order_volume: int, sell_order_volume: int,
                    ):
        buy_quantity = self.PRODUCT_LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, math.floor(bid), buy_quantity))  # Buy order

        sell_quantity = self.PRODUCT_LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, math.ceil(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(self, product: str,
                             fair_value: float,
                             width: int, orders: List[Order],
                             order_depth: OrderDepth,
                             position: int, buy_order_volume: int,
                             sell_order_volume: int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.PRODUCT_LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.PRODUCT_LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject,
                        ink_order_depth: OrderDepth):
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            valid_ask = [price for price in order_depth.sell_orders.keys()
                         if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
            valid_buy = [price for price in order_depth.buy_orders.keys()
                         if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]

            mm_ask = min(valid_ask) if len(valid_ask) > 0 else None
            mm_bid = max(valid_buy) if len(valid_buy) > 0 else None
            if valid_ask and valid_buy:
                mmmid_price = (mm_ask + mm_bid) / 2

            else:
                if traderObject.get('kelp_last_price', None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject['kelp_last_price']

            if traderObject.get('kelp_last_price', None) is None:
                fair = mmmid_price
            else:
                ### Alpha-ish - LR forecast
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (last_returns * self.params[Product.KELP]["reversion_beta"])
                fair = mmmid_price + (mmmid_price * pred_returns)

            if traderObject.get("ink_last_price", None) is not None:
                ### Alpha - Neg Corr Ink
                old_ink_price = traderObject["ink_last_price"]
                valid_ask_ink = [price for price in ink_order_depth.sell_orders.keys()
                                 if abs(ink_order_depth.sell_orders[price]) >= self.params[Product.SQUID_INK][
                                     "adverse_volume"]]
                valid_buy_ink = [price for price in ink_order_depth.buy_orders.keys()
                                 if abs(ink_order_depth.buy_orders[price]) >= self.params[Product.SQUID_INK][
                                     "adverse_volume"]]
                if valid_ask_ink and valid_buy_ink:
                    new_ink_mid = (min(valid_ask_ink) + max(valid_buy_ink)) / 2
                else:
                    new_ink_mid = (min(ink_order_depth.sell_orders.keys()) +
                                   max(ink_order_depth.buy_orders.keys())) / 2

                ink_return = (new_ink_mid - old_ink_price) / old_ink_price
                fair = fair - (self.params[Product.KELP].get("ink_adjustment_factor", 0.5) * ink_return * mmmid_price)
                # ink_return = (traderObject["ink_last_price"] - traderObject["prev_ink_price"]) / traderObject["prev_ink_price"]
                # adj = self.params[Product.KELP].get("ink_adjustment_factor", 0.5) * ink_return * mmmid_price
                # fair -= adj

            # traderObject["prev_ink_price"] = traderObject.get("ink_last_price", mmmid_price)
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def ink_fair_value(self, order_depth: OrderDepth, traderObject):
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            valid_ask = [price for price in order_depth.sell_orders.keys()
                         if abs(order_depth.sell_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]
            valid_buy = [price for price in order_depth.buy_orders.keys()
                         if abs(order_depth.buy_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]

            mm_ask = min(valid_ask) if len(valid_ask) > 0 else None
            mm_bid = max(valid_buy) if len(valid_buy) > 0 else None
            if valid_ask and valid_buy:
                mmmid_price = (mm_ask + mm_bid) / 2

            else:
                if traderObject.get('ink_last_price', None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject['ink_last_price']

            if traderObject.get('ink_price_history', None) is None:
                traderObject['ink_price_history'] = []

            traderObject['ink_price_history'].append(mmmid_price)
            if len(traderObject['ink_price_history']) > self.params[Product.SQUID_INK]["reversion_window"]:
                traderObject['ink_price_history'] = traderObject['ink_price_history'][
                                                    -self.params[Product.SQUID_INK]["reversion_window"]:]

            # New Alpha attempt: adaptive mean reversion
            if len(traderObject['ink_price_history']) >= self.params[Product.SQUID_INK]["reversion_window"]:
                prices = np.array(traderObject['ink_price_history'])

                returns = (prices[1:] - prices[:-1]) / prices[:-1]
                X = returns[:-1]
                Y = returns[1:]
                if np.dot(X, X) != 0:
                    estimated_beta = - np.dot(X, Y) / np.dot(X, X)
                else:
                    estimated_beta = self.params[Product.SQUID_INK]["reversion_beta"]

                adaptive_beta = (self.params[Product.SQUID_INK]['reversion_weight'] * estimated_beta
                                 + (1 - self.params[Product.SQUID_INK]['reversion_weight']) *
                                 self.params[Product.SQUID_INK]["reversion_beta"])
            else:
                adaptive_beta = self.params[Product.SQUID_INK]["reversion_beta"]

            if traderObject.get('ink_last_price', None) is None:
                fair = mmmid_price
            else:
                last_price = traderObject["ink_last_price"]
                last_return = (mmmid_price - last_price) / last_price
                pred_return = last_return * adaptive_beta
                fair = mmmid_price + (mmmid_price * pred_return)
            traderObject["ink_last_price"] = mmmid_price
            return fair
        return None

    def take_orders(self, product: str, order_depth: OrderDepth,
                    fair_value: float, take_width: float,
                    position: int, prevent_adverse: bool = False,
                    adverse_volume: int = 0, traderObject: dict = None):
        orders: List[Order] = []

        buy_order_volume, sell_order_volume = 0, 0
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product, fair_value, take_width, orders, order_depth,
            position, buy_order_volume, sell_order_volume, prevent_adverse,
            adverse_volume, traderObject
        )

        return orders, buy_order_volume, sell_order_volume

    def clear_orders(self, product: str, order_depth: OrderDepth,
                     fair_value: float, clear_width: int,
                     position: int, buy_order_volume: int,
                     sell_order_volume: int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product, fair_value, clear_width, orders, order_depth,
            position, buy_order_volume, sell_order_volume
        )

        return orders, buy_order_volume, sell_order_volume

    def make_orders(self, product, order_depth: OrderDepth, fair_value: float,
                    position: int, buy_order_volume: int, sell_order_volume: int,
                    disregard_edge: float, join_edge: float, default_edge: float,
                    manage_position: bool = False, soft_position_limit: int = 0,
                    cur_resin_price: float = None):
        adjustment = 0
        if product == Product.RAINFOREST_RESIN:
            total_buy_volume = sum(order_depth.buy_orders.values()) if order_depth.buy_orders else 0

            total_sell_volume = sum(abs(v) for v in order_depth.sell_orders.values()) if order_depth.sell_orders else 0
            total_volume = total_buy_volume + total_sell_volume if (total_buy_volume + total_sell_volume) > 0 else 1

            imbalance_ratio = (total_buy_volume - total_sell_volume) / total_volume
            scaling_factor = 4.0  # You can tune this: higher means more aggressive adjustment.
            adjustment = round(scaling_factor * imbalance_ratio)

        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair + 1  # join
            else:
                ask = best_ask_above_fair  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def get_microprice(self, order_depth):
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol +
                                                                      best_ask_vol)

    def artifical_order_depth(self, order_depths: Dict[str, OrderDepth],
                              picnic1: bool = True):  # maybe change pinic1 to whether they like Marvin Gaye's music
        if picnic1:
            DJEMBES_PER_PICNIC = PICNIC1_WEIGHTS[Product.DJEMBES]
            CROISSANT_PER_PICNIC = PICNIC1_WEIGHTS[Product.CROISSANT]
            JAM_PER_PICNIC = PICNIC1_WEIGHTS[Product.JAMS]

        else:
            CROISSANT_PER_PICNIC = PICNIC2_WEIGHTS[Product.CROISSANT]
            JAM_PER_PICNIC = PICNIC2_WEIGHTS[Product.JAMS]

        artifical_order_price = OrderDepth()

        croissant_best_bid = (max(order_depths[Product.CROISSANT].buy_orders.keys())
                              if order_depths[Product.CROISSANT].buy_orders
                              else 0)

        croissant_best_ask = (min(order_depths[Product.CROISSANT].sell_orders.keys())
                              if order_depths[Product.CROISSANT].sell_orders
                              else float("inf"))

        jams_best_bid = (max(order_depths[Product.JAMS].buy_orders.keys())
                         if order_depths[Product.JAMS].buy_orders
                         else 0)

        jams_best_ask = (min(order_depths[Product.JAMS].sell_orders.keys())
                         if order_depths[Product.JAMS].sell_orders
                         else float("inf"))

        if picnic1:
            djembes_best_bid = (max(order_depths[Product.DJEMBES].buy_orders.keys())
                                if order_depths[Product.DJEMBES].buy_orders
                                else 0)

            djembes_best_ask = (min(order_depths[Product.DJEMBES].sell_orders.keys())
                                if order_depths[Product.DJEMBES].sell_orders
                                else float("inf"))

            art_bid = (djembes_best_bid * DJEMBES_PER_PICNIC +
                       croissant_best_bid * CROISSANT_PER_PICNIC +
                       jams_best_bid * JAM_PER_PICNIC)
            art_ask = (djembes_best_ask * DJEMBES_PER_PICNIC +
                       croissant_best_ask * CROISSANT_PER_PICNIC +
                       jams_best_ask * JAM_PER_PICNIC)
        else:
            art_bid = (croissant_best_bid * CROISSANT_PER_PICNIC +
                       jams_best_bid * JAM_PER_PICNIC)
            art_ask = (croissant_best_ask * CROISSANT_PER_PICNIC +
                       jams_best_ask * JAM_PER_PICNIC)

        if art_bid > 0:
            croissant_bid_volume = (order_depths[Product.CROISSANT].buy_orders[croissant_best_bid]
                                    // CROISSANT_PER_PICNIC)
            jams_bid_volume = (order_depths[Product.JAMS].buy_orders[jams_best_bid]
                               // JAM_PER_PICNIC)

            if picnic1:
                djembes_bid_volume = (order_depths[Product.DJEMBES].buy_orders[djembes_best_bid]
                                      // DJEMBES_PER_PICNIC)

                artifical_bid_volume = min(djembes_bid_volume, croissant_bid_volume,
                                           jams_bid_volume)
            else:
                artifical_bid_volume = min(croissant_bid_volume, jams_bid_volume)
            artifical_order_price.buy_orders[art_bid] = artifical_bid_volume

        if art_ask < float("inf"):
            croissant_ask_volume = (-order_depths[Product.CROISSANT].sell_orders[croissant_best_ask]
                                    // CROISSANT_PER_PICNIC)
            jams_ask_volume = (-order_depths[Product.JAMS].sell_orders[jams_best_ask]
                               // JAM_PER_PICNIC)

            if picnic1:
                djembes_ask_volume = (-order_depths[Product.DJEMBES].sell_orders[djembes_best_ask]
                                      // DJEMBES_PER_PICNIC)

                artifical_ask_volume = min(
                    djembes_ask_volume, croissant_ask_volume, jams_ask_volume
                )
            else:
                artifical_ask_volume = min(croissant_ask_volume, jams_ask_volume)
            artifical_order_price.sell_orders[art_ask] = -artifical_ask_volume

        return artifical_order_price

    def convert_orders(self, artifical_orders: List[Order],
                       order_depths: Dict[str, OrderDepth],
                       picnic1: bool = True):
        if picnic1:
            component_orders = {
                Product.DJEMBES: [],
                Product.CROISSANT: [],
                Product.JAMS: [],
            }
        else:
            component_orders = {
                Product.CROISSANT: [],
                Product.JAMS: [],
            }

        artfical_order_depth = self.artifical_order_depth(order_depths, picnic1)
        best_bid = (max(artfical_order_depth.buy_orders.keys())
                    if artfical_order_depth.buy_orders else 0)
        best_ask = (min(artfical_order_depth.sell_orders.keys())
                    if artfical_order_depth.sell_orders else float("inf"))

        for order in artifical_orders:
            price = order.price
            quantity = order.quantity

            if quantity > 0 and price >= best_ask:
                croissant_price = min(order_depths[Product.CROISSANT].sell_orders.keys())
                jams_price = min(order_depths[Product.JAMS].sell_orders.keys())
                if picnic1:
                    djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                croissant_price = max(order_depths[Product.CROISSANT].buy_orders.keys())
                jams_price = max(order_depths[Product.JAMS].buy_orders.keys())
                if picnic1:
                    djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                continue

            croissaint_order = Order(
                Product.CROISSANT,
                croissant_price,
                (quantity * (PICNIC1_WEIGHTS[Product.CROISSANT])
                 if picnic1 else quantity * (PICNIC2_WEIGHTS[Product.CROISSANT])
                 ),
            )
            jams_order = Order(
                Product.JAMS,
                jams_price,
                (quantity * (PICNIC1_WEIGHTS[Product.JAMS])
                 if picnic1 else quantity * (PICNIC2_WEIGHTS[Product.JAMS])
                 ),
            )
            if picnic1:
                djembes_order = Order(
                    Product.DJEMBES,
                    djembes_price,
                    quantity * (PICNIC1_WEIGHTS[Product.DJEMBES]),
                )
                component_orders[Product.DJEMBES].append(djembes_order)

            component_orders[Product.CROISSANT].append(croissaint_order)
            component_orders[Product.JAMS].append(jams_order)

        return component_orders

    def execute_spreads(self, target_position: int,
                        picnic_position: int,
                        order_depths: Dict[str, OrderDepth],
                        picnic1: bool = True):
        if target_position == picnic_position:
            return None

        target_quantity = abs(target_position - picnic_position)
        picnic_order_depth = (order_depths[Product.PICNIC_BASKET1] if picnic1
                              else order_depths[Product.PICNIC_BASKET2])
        artifical_order_depth = self.artifical_order_depth(order_depths, picnic1)

        if target_position > picnic_position:
            picnic_ask_price = min(picnic_order_depth.sell_orders.keys())
            picnic_ask_vol = abs(picnic_order_depth.sell_orders[picnic_ask_price])
            artifical_bid_price = min(artifical_order_depth.buy_orders.keys())
            artifical_bid_vol = abs(artifical_order_depth.buy_orders[artifical_bid_price])

            orderbook_volume = min(picnic_ask_vol, artifical_bid_vol)
            execute_volume = min(orderbook_volume, target_quantity)

            picnic_orders = [
                (Order(Product.PICNIC_BASKET1, picnic_ask_price, execute_volume)
                 if picnic1
                 else Order(Product.PICNIC_BASKET2, picnic_ask_price, execute_volume))
            ]
            artifical_orders = [
                (Order(Product.ARTIFICAL1, artifical_bid_price, -execute_volume)
                 # tbh does it matter if we used two artifical names
                 )
            ]

            aggregate_orders = self.convert_orders(
                artifical_orders, order_depths, picnic1
            )
            if picnic1:
                aggregate_orders[Product.PICNIC_BASKET1] = picnic_orders
            else:
                aggregate_orders[Product.PICNIC_BASKET2] = picnic_orders
            return aggregate_orders
        else:
            picnic_bid_price = min(picnic_order_depth.buy_orders.keys())
            picnic_bid_vol = abs(picnic_order_depth.buy_orders[picnic_bid_price])
            artifical_ask_price = min(artifical_order_depth.sell_orders.keys())
            artifical_ask_vol = abs(artifical_order_depth.sell_orders[artifical_ask_price])

            orderbook_volume = min(picnic_bid_vol, artifical_ask_vol)
            execute_volume = min(orderbook_volume, target_quantity)

            picnic_orders = [
                (Order(Product.PICNIC_BASKET1, picnic_bid_price, -execute_volume)
                 if picnic1
                 else Order(Product.PICNIC_BASKET2, picnic_bid_price, -execute_volume))
            ]
            artifical_orders = [
                (Order(Product.ARTIFICAL1, artifical_ask_price, -execute_volume)
                 # tbh does it matter if we used two artifical names
                 )
            ]

            aggregate_orders = self.convert_orders(
                artifical_orders, order_depths, picnic1
            )
            if picnic1:
                aggregate_orders[Product.PICNIC_BASKET1] = picnic_orders
            else:
                aggregate_orders[Product.PICNIC_BASKET2] = picnic_orders
            return aggregate_orders

    def spread_orders(self, order_depths: Dict[str, OrderDepth],
                      product: Product, picnic_position: int,
                      spread_data: Dict[str, Any],
                      SPREAD,
                      picnic1: bool = True,
                      ):
        if (Product.PICNIC_BASKET1 not in order_depths.keys() or
                Product.PICNIC_BASKET2 not in order_depths.keys()):
            return None

        picnic_order_depth = (order_depths[Product.PICNIC_BASKET1] if picnic1
                              else order_depths[Product.PICNIC_BASKET2])
        artifical_order_depth = self.artifical_order_depth(order_depths, picnic1)
        picnic_mprice = self.get_microprice(picnic_order_depth)
        artifical_mprice = self.get_microprice(artifical_order_depth)
        spread = picnic_mprice - artifical_mprice
        spread_data["spread_history"].append(spread)

        if (len(spread_data["spread_history"])
                < self.params[SPREAD]["spread_window"]):
            return None
        elif len(spread_data["spread_history"]) > self.params[SPREAD]["spread_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])

        zscore = (spread - self.params[SPREAD]["default_spread_mean"]) / spread_std

        if zscore >= self.params[SPREAD]["zscore_threshold"]:
            if picnic_position != -self.params[SPREAD]["target_position"]:
                return self.execute_spreads(
                    -self.params[SPREAD]["target_position"],
                    picnic_position,
                    order_depths,
                    picnic1
                )

        if zscore <= -self.params[SPREAD]["zscore_threshold"]:
            if picnic_position != self.params[SPREAD]["target_position"]:
                return self.execute_spreads(
                    self.params[SPREAD]["target_position"],
                    picnic_position,
                    order_depths,
                    picnic1
                )

        spread_data["prev_zscore"] = zscore
        return None

    def trade_resin(self, state, market_data):
        product = "RAINFOREST_RESIN"
        end_pos = state.position.get(product, 0)
        buy_sum = 50 - end_pos
        sell_sum = 50 + end_pos
        orders = []
        order_depth: OrderDepth = state.order_depths[product]
        bids = order_depth.buy_orders
        asks = order_depth.sell_orders
        bid_prices = list(bids.keys())
        bid_volumes = list(bids.values())
        ask_prices = list(asks.keys())
        ask_volumes = list(asks.values())

        # for each buy order level, if > fair, fill completely SELLING
        if sell_sum > 0:
            for i in range(0, len(bid_prices)):
                if bid_prices[i] > 10000:
                    fill = min(bid_volumes[i], sell_sum)
                    orders.append(Order(product, bid_prices[i], -fill))
                    sell_sum -= fill
                    end_pos -= fill
                    bid_volumes[i] -= fill

        # remove prices that were matched against
        bid_prices, bid_volumes = zip(*[(ai, bi) for ai, bi in zip(bid_prices, bid_volumes) if bi != 0])
        bid_prices = list(bid_prices)
        bid_volumes = list(bid_volumes)

        # for each sell order level, if < fair, fill completely BUYING
        if buy_sum > 0:
            for i in range(0, len(ask_prices)):
                if ask_prices[i] < 10000:
                    fill = min(-ask_volumes[i], buy_sum)
                    orders.append(Order(product, ask_prices[i], fill))
                    buy_sum -= fill
                    end_pos += fill
                    ask_volumes[i] += fill

        # remove prices that were matched against
        ask_prices, ask_volumes = zip(*[(ai, bi) for ai, bi in zip(ask_prices, ask_volumes) if bi != 0])
        ask_prices = list(ask_prices)
        ask_volumes = list(ask_volumes)

        # # Fair = 10000, MM around
        # if abs(ask_volumes[0]) > 1 and ask_prices[0] == 10002:
        #     orders.append(Order(product, 10000+1, -min(14, sell_sum))) # ask
        # else:
        #     orders.append(Order(product, max(10000+3, ask_prices[0]-1), -min(14, sell_sum))) # ask
        # sell_sum -= min(14, sell_sum)

        # if bid_volumes[0] > 1 and bid_prices[0] == 9998:
        #     orders.append(Order(product, 10000-1, min(14, buy_sum))) # bid
        # else:
        #     orders.append(Order(product, min(10000-3, bid_prices[0]+1), min(14, buy_sum))) # bid
        # buy_sum -= min(14, buy_sum)

        # Fair = 10000, MM around
        if abs(ask_volumes[0]) > 1:
            orders.append(Order(product, max(ask_prices[0] - 1, 10000 + 1), -min(14, sell_sum)))  # ask
        else:
            orders.append(Order(product, max(10000 + 1, ask_prices[0]), -min(14, sell_sum)))  # ask
        sell_sum -= min(14, sell_sum)

        if bid_volumes[0] > 1:
            orders.append(Order(product, min(bid_prices[0] + 1, 10000 - 1), min(14, buy_sum)))  # bid
        else:
            orders.append(Order(product, min(10000 - 1, bid_prices[0]), min(14, buy_sum)))  # bid
        buy_sum -= min(14, buy_sum)

        # orders.append(Order(product, 10000-2, min(14, buy_sum))) # bid
        # orders.append(Order(product, 10000+2, -min(14, sell_sum))) # ask

        if end_pos > 0:  # sell to bring pos closer to 0
            for i in range(0, len(bid_prices)):
                if bid_prices[i] == 10000:
                    fill = min(bid_volumes[i], sell_sum)
                    orders.append(Order(product, bid_prices[i], -fill))
                    sell_sum -= fill
                    end_pos -= fill

        if end_pos < 0:  # buy to bring pos closer to 0
            for i in range(0, len(ask_prices)):
                if ask_prices[i] == 10000:
                    fill = min(-ask_volumes[i], buy_sum)
                    orders.append(Order(product, ask_prices[i], fill))
                    buy_sum -= fill
                    end_pos += fill

        return orders


    def norm_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def black_scholes_call(self, S: float, K: float, T_days: float, r: float, sigma: float) -> float:
        """Black-Scholes price of a European call option."""
        T = T_days / 365.0
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)

    def implied_vol_call(self, market_price, S, K, T_days, r, tol=0.00000000000001, max_iter=250):
        """
        Calculate implied volatility from market call option price using bisection.

        Parameters:
        - market_price: observed market price of the option
        - S: spot price
        - K: strike price
        - T_days: time to maturity in days
        - r: risk-free interest rate
        - tol: convergence tolerance
        - max_iter: maximum number of iterations

        Returns:
        - Implied volatility (sigma)
        """
        # Set reasonable initial bounds
        sigma_low = 0.01
        sigma_high = 0.35

        for _ in range(max_iter):
            sigma_mid = (sigma_low + sigma_high) / 2
            price = self.black_scholes_call(S, K, T_days, r, sigma_mid)

            if abs(price - market_price) < tol:
                return sigma_mid

            if price > market_price:
                sigma_high = sigma_mid
            else:
                sigma_low = sigma_mid

        return (sigma_low + sigma_high) / 2  # Final estimate

    def call_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate the Black-Scholes delta of a European call option.

        Parameters:
        - S: Current stock price
        - K: Strike price
        - T: Time to maturity (in days)
        - r: Risk-free interest rate
        - sigma: Volatility (annual)

        Returns:
        - delta: Call option delta
        """
        r = 0
        T = T / 365
        if T == 0 or sigma == 0:
            return 1.0 if S > K else 0.0

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return 0.5 * (1 + math.erf(d1 / math.sqrt(2)))

    def trade_10000(self, state, market_data, traderObject):
        product = "VOLCANIC_ROCK_VOUCHER_10000"
        delta_sum = 0
        orders = {}
        for p in ["VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_10000"]:
            orders[p] = []
        fair = market_data.fair[product]
        underlying_fair = market_data.fair["VOLCANIC_ROCK"]
        dte = CUR_DAY - state.timestamp / 1_000_000
        v_t = self.implied_vol_call(fair, underlying_fair, 10000, dte, 0)
        delta = self.call_delta(fair, underlying_fair, dte, v_t)
        m_t = np.log(10000 / underlying_fair) / np.sqrt(dte / 365)
        # print(f"m_t = {m_t}")

        # , ,
        base_coef = 0.14786181  # 0.13571776890273662 + 0.00229274*dte
        linear_coef = 0.00099561  # -0.03685812200491957 + 0.0072571*dte
        squared_coef = 0.23544086  # 0.16277746617792221 + 0.01096456*dte
        fair_iv = base_coef + linear_coef * m_t + squared_coef * (m_t ** 2)
        # print(fair_iv)

        diff = v_t - fair_iv
        if "prices_10000" not in traderObject:
            traderObject["prices_10000"] = [diff]
        else:
            traderObject["prices_10000"].append(diff)
        threshold = 0.0035
        # print(diff)
        if len(traderObject["prices_10000"]) > 20:
            diff -= np.mean(traderObject["prices_10000"])
            traderObject["prices_10000"].pop(0)
            if diff > threshold:  # short vol so sell option, buy und
                amount = market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_10000"]
                amount = min(amount, sum(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_10000"]))
                option_amount = amount
                rock_amount = amount

                # print(rock_amount)
                '''for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK"])):
                    fill = min(-market_data.ask_volumes["VOLCANIC_ROCK"][i], rock_amount)
                    #print(fill)
                    if fill != 0:
                        orders["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", market_data.ask_prices["VOLCANIC_ROCK"][i], fill))
                        market_data.buy_sum["VOLCANIC_ROCK"] -= fill
                        market_data.end_pos["VOLCANIC_ROCK"] += fill
                        rock_amount -= fill
                        #print(fill)'''

                for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK_VOUCHER_10000"])):
                    fill = min(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_10000"][i], option_amount)
                    delta_sum -= delta * fill
                    if fill != 0:
                        orders["VOLCANIC_ROCK_VOUCHER_10000"].append(Order("VOLCANIC_ROCK_VOUCHER_10000",
                                                                           market_data.bid_prices[
                                                                               "VOLCANIC_ROCK_VOUCHER_10000"][i],
                                                                           -fill))
                        market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_10000"] -= fill
                        market_data.end_pos["VOLCANIC_ROCK_VOUCHER_10000"] -= fill
                        option_amount -= fill

            elif diff < -threshold:  # long vol
                # print("LONG")
                # print("----")
                amount = market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_10000"]
                # print(amount)
                amount = min(amount, -sum(market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_10000"]))
                # print(amount)
                option_amount = amount
                rock_amount = amount
                # print(f"{rock_amount} rocks")
                for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK_VOUCHER_10000"])):
                    fill = min(-market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_10000"][i], option_amount)
                    delta_sum += delta * fill
                    if fill != 0:
                        orders["VOLCANIC_ROCK_VOUCHER_10000"].append(Order("VOLCANIC_ROCK_VOUCHER_10000",
                                                                           market_data.ask_prices[
                                                                               "VOLCANIC_ROCK_VOUCHER_10000"][i], fill))
                        market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_10000"] -= fill
                        market_data.end_pos["VOLCANIC_ROCK_VOUCHER_10000"] += fill
                        option_amount -= fill

                '''for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK"])):
                        fill = min(market_data.bid_volumes["VOLCANIC_ROCK"][i], rock_amount)
                        #print(fill)
                        if fill != 0:
                            orders["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", market_data.bid_prices["VOLCANIC_ROCK"][i], -fill))
                            market_data.sell_sum["VOLCANIC_ROCK"] -= fill
                            market_data.end_pos["VOLCANIC_ROCK"] -= fill
                            rock_amount -= fill'''

        return orders["VOLCANIC_ROCK_VOUCHER_10000"]

    def trade_10500(self, state, market_data, traderObject):
        product = "VOLCANIC_ROCK_VOUCHER_10500"
        delta_sum = 0
        orders = {}
        for p in ["VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_10500"]:
            orders[p] = []
        fair = market_data.fair[product]
        underlying_fair = market_data.fair["VOLCANIC_ROCK"]
        dte = CUR_DAY - state.timestamp / 1_000_000
        v_t = self.implied_vol_call(fair, underlying_fair, 10500, dte, 0)
        try:
            delta = self.call_delta(fair, underlying_fair, dte, v_t)
        except:
            return [], []
        m_t = np.log(10500 / underlying_fair) / np.sqrt(dte / 365)
        # print(f"m_t = {m_t}")

        # , ,
        base_coef = 0.264416  # 0.13571776890273662 + 0.00229274*dte
        linear_coef = 0.010031  # -0.03685812200491957 + 0.0072571*dte
        squared_coef = 0.147604  # 0.16277746617792221 + 0.01096456*dte
        fair_iv = base_coef + linear_coef * m_t + squared_coef * (m_t ** 2)
        # print(fair_iv)

        diff = v_t - fair_iv
        if "prices_10500" not in traderObject:
            traderObject["prices_10500"] = [diff]
        else:
            traderObject["prices_10500"].append(diff)
        # print(diff)
        if len(traderObject["prices_10500"]) > 13:
            diff -= np.mean(traderObject["prices_10500"])
            traderObject["prices_10500"].pop(0)
        threshold = 0.001
        if diff > threshold:  # short vol so sell option, buy und
            amount = min(market_data.buy_sum["VOLCANIC_ROCK"], market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_10500"])
            amount = min(amount, -sum(market_data.ask_volumes["VOLCANIC_ROCK"]),
                         sum(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_10500"]))
            option_amount = amount
            if np.mean(traderObject["prices_10500"]) > 0:
                rock_amount = amount
            else:
                rock_amount = amount // 2

            # print(rock_amount)
            for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK"])):
                fill = min(-market_data.ask_volumes["VOLCANIC_ROCK"][i], rock_amount)
                # print(fill)
                if fill != 0:
                    orders["VOLCANIC_ROCK"].append(
                        Order("VOLCANIC_ROCK", market_data.ask_prices["VOLCANIC_ROCK"][i], fill))
                    market_data.buy_sum["VOLCANIC_ROCK"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK"] += fill
                    rock_amount -= fill
                    # print(fill)

            for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK_VOUCHER_10500"])):
                fill = min(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_10500"][i], option_amount)
                delta_sum -= delta * fill
                if fill != 0:
                    orders["VOLCANIC_ROCK_VOUCHER_10500"].append(Order("VOLCANIC_ROCK_VOUCHER_10500",
                                                                       market_data.bid_prices[
                                                                           "VOLCANIC_ROCK_VOUCHER_10500"][i],
                                                                       -fill))
                    market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_10500"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK_VOUCHER_10500"] -= fill
                    option_amount -= fill

        elif diff < -threshold:  # long vol
            # print("LONG")
            # print("----")
            amount = min(market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_10500"], market_data.sell_sum["VOLCANIC_ROCK"])
            # print(amount)
            amount = min(amount, -sum(market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_10500"]),
                         sum(market_data.bid_volumes["VOLCANIC_ROCK"]))
            # print(amount)
            option_amount = amount
            if np.mean(traderObject["prices_10500"]) < 0:
                rock_amount = amount
            else:
                rock_amount = amount // 2
                raise Exception(state.timestamp, option_amount)
            # print(f"{rock_amount} rocks")
            for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK_VOUCHER_10500"])):
                fill = min(-market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_10500"][i], option_amount)
                delta_sum += delta * fill
                if fill != 0:
                    orders["VOLCANIC_ROCK_VOUCHER_10500"].append(Order("VOLCANIC_ROCK_VOUCHER_10500",
                                                                       market_data.ask_prices[
                                                                           "VOLCANIC_ROCK_VOUCHER_10500"][i], fill))
                    market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_10500"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK_VOUCHER_10500"] += fill
                    option_amount -= fill

            for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK"])):
                fill = min(market_data.bid_volumes["VOLCANIC_ROCK"][i], rock_amount)
                # print(fill)
                if fill != 0:
                    orders["VOLCANIC_ROCK"].append(
                        Order("VOLCANIC_ROCK", market_data.bid_prices["VOLCANIC_ROCK"][i], -fill))
                    market_data.sell_sum["VOLCANIC_ROCK"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK"] -= fill
                    rock_amount -= fill

        return orders["VOLCANIC_ROCK"], orders["VOLCANIC_ROCK_VOUCHER_10500"]

    def trade_9500(self, state, market_data, traderObject):
        product = "VOLCANIC_ROCK_VOUCHER_9500"
        delta_sum = 0
        orders = {}
        for p in ["VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9500"]:
            orders[p] = []
        fair = market_data.fair[product]
        underlying_fair = market_data.fair["VOLCANIC_ROCK"]
        dte = CUR_DAY - state.timestamp / 1_000_000
        v_t = self.implied_vol_call(fair, underlying_fair, 9500, dte, 0)
        try:
            delta = self.call_delta(fair, underlying_fair, dte, v_t)
        except:
            return [], []
        m_t = np.log(9500 / underlying_fair) / np.sqrt(dte / 365)
        # print(f"m_t = {m_t}")

        # , ,
        base_coef = 0.264416  # 0.13571776890273662 + 0.00229274*dte
        linear_coef = 0.010031  # -0.03685812200491957 + 0.0072571*dte
        squared_coef = 0.147604  # 0.16277746617792221 + 0.01096456*dte
        fair_iv = base_coef + linear_coef * m_t + squared_coef * (m_t ** 2)
        # print(fair_iv)

        diff = v_t - fair_iv
        if "prices_9500" not in traderObject:
            traderObject["prices_9500"] = [diff]
        else:
            traderObject["prices_9500"].append(diff)
        # print(diff)
        if len(traderObject["prices_9500"]) > 13:
            diff -= np.mean(traderObject["prices_9500"])
            traderObject["prices_9500"].pop(0)
        threshold = 0.0005
        if diff > threshold:  # short vol so sell option, buy und
            amount = min(market_data.buy_sum["VOLCANIC_ROCK"], market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_9500"])
            amount = min(amount, -sum(market_data.ask_volumes["VOLCANIC_ROCK"]),
                         sum(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_9500"]))
            option_amount = amount
            rock_amount = amount
            # print(rock_amount)
            for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK"])):
                fill = min(-market_data.ask_volumes["VOLCANIC_ROCK"][i], rock_amount)
                # print(fill)
                if fill != 0:
                    orders["VOLCANIC_ROCK"].append(
                        Order("VOLCANIC_ROCK", market_data.ask_prices["VOLCANIC_ROCK"][i], fill))
                    market_data.buy_sum["VOLCANIC_ROCK"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK"] += fill
                    rock_amount -= fill
                    # print(fill)

            for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK_VOUCHER_9500"])):
                fill = min(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_9500"][i], option_amount)
                delta_sum -= delta * fill
                if fill != 0:
                    orders["VOLCANIC_ROCK_VOUCHER_9500"].append(Order("VOLCANIC_ROCK_VOUCHER_9500",
                                                                      market_data.bid_prices[
                                                                          "VOLCANIC_ROCK_VOUCHER_9500"][i],
                                                                      -fill))
                    market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_9500"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK_VOUCHER_9500"] -= fill
                    option_amount -= fill

        elif diff < -threshold:  # long vol
            # print("LONG")
            # print("----")
            amount = min(market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_9500"], market_data.sell_sum["VOLCANIC_ROCK"])
            # print(amount)
            amount = min(amount, -sum(market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_9500"]),
                         sum(market_data.bid_volumes["VOLCANIC_ROCK"]))
            # print(amount)
            option_amount = amount
            rock_amount = amount

            # print(f"{rock_amount} rocks")
            for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK_VOUCHER_9500"])):
                fill = min(-market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_9500"][i], option_amount)
                delta_sum += delta * fill
                if fill != 0:
                    orders["VOLCANIC_ROCK_VOUCHER_9500"].append(Order("VOLCANIC_ROCK_VOUCHER_9500",
                                                                      market_data.ask_prices[
                                                                          "VOLCANIC_ROCK_VOUCHER_9500"][i], fill))
                    market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_9500"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK_VOUCHER_9500"] += fill
                    option_amount -= fill

            for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK"])):
                fill = min(market_data.bid_volumes["VOLCANIC_ROCK"][i], rock_amount)
                # print(fill)
                if fill != 0:
                    orders["VOLCANIC_ROCK"].append(
                        Order("VOLCANIC_ROCK", market_data.bid_prices["VOLCANIC_ROCK"][i], -fill))
                    market_data.sell_sum["VOLCANIC_ROCK"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK"] -= fill
                    rock_amount -= fill

        return orders["VOLCANIC_ROCK"], orders["VOLCANIC_ROCK_VOUCHER_9500"]

    def trade_9750(self, state, market_data, traderObject):
        product = "VOLCANIC_ROCK_VOUCHER_9750"
        orders = {}
        for p in ["VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9750"]:
            orders[p] = []
        fair = market_data.fair[product]
        underlying_fair = market_data.fair["VOLCANIC_ROCK"]
        dte = CUR_DAY - state.timestamp / 1_000_000

        v_t = self.implied_vol_call(fair, underlying_fair, 9750, dte, 0)
        m_t = np.log(9750 / underlying_fair) / np.sqrt(dte / 365)
        # print(f"m_t = {m_t}")

        # , ,
        base_coef = 0.264416  # 0.13571776890273662 + 0.00229274*dte
        linear_coef = 0.010031  # -0.03685812200491957 + 0.0072571*dte
        squared_coef = 0.147604  # 0.16277746617792221 + 0.01096456*dte
        fair_iv = base_coef + linear_coef * m_t + squared_coef * (m_t ** 2)
        # print(fair_iv)
        diff = v_t - fair_iv
        if "prices_9750" not in traderObject:
            traderObject["prices_9750"] = [diff]
        else:

            traderObject["prices_9750"].append(diff)
        threshold = 0.0055
        # print(diff)
        if len(traderObject["prices_9750"]) > 13:
            diff -= np.mean(traderObject["prices_9750"])
            traderObject["prices_9750"].pop(0)
        if diff > threshold:  # short vol so sell option, buy und
            amount = market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_9750"]
            amount = min(amount, sum(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_9750"]))
            option_amount = amount

            rock_amount = amount

            for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK_VOUCHER_9750"])):
                fill = min(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_9750"][i], option_amount)
                if fill != 0:
                    orders["VOLCANIC_ROCK_VOUCHER_9750"].append(Order("VOLCANIC_ROCK_VOUCHER_9750",
                                                                      market_data.bid_prices[
                                                                          "VOLCANIC_ROCK_VOUCHER_9750"][i],
                                                                      -fill))
                    market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_9750"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK_VOUCHER_9750"] -= fill
                    option_amount -= fill

        elif diff < -threshold:  # long vol
            # print("LONG")
            # print("----")
            amount = market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_9750"]
            # print(amount)
            amount = min(amount, -sum(market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_9750"]))
            # print(amount)
            option_amount = amount
            rock_amount = amount
            # print(f"{rock_amount} rocks")
            for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK_VOUCHER_9750"])):
                fill = min(-market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_9750"][i], option_amount)
                if fill != 0:
                    orders["VOLCANIC_ROCK_VOUCHER_9750"].append(Order("VOLCANIC_ROCK_VOUCHER_9750",
                                                                      market_data.ask_prices[
                                                                          "VOLCANIC_ROCK_VOUCHER_9750"][i], fill))
                    market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_9750"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK_VOUCHER_9750"] += fill
                    option_amount -= fill

            """for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK"])):
                    fill = min(market_data.bid_volumes["VOLCANIC_ROCK"][i], rock_amount)
                    #print(fill)
                    if fill != 0:
                        orders["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", market_data.bid_prices["VOLCANIC_ROCK"][i], -fill))
                        market_data.sell_sum["VOLCANIC_ROCK"] -= fill
                        market_data.end_pos["VOLCANIC_ROCK"] -= fill
                        rock_amount -= fill"""
        return orders["VOLCANIC_ROCK_VOUCHER_9750"]


    def trade_10250(self, state, market_data, traderObject):
        product = "VOLCANIC_ROCK_VOUCHER_10250"
        delta_sum = 0
        orders = {}
        for p in ["VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_10250"]:
            orders[p] = []
        fair = market_data.fair[product]
        underlying_fair = market_data.fair["VOLCANIC_ROCK"]
        dte = CUR_DAY - state.timestamp / 1_000_000
        v_t = self.implied_vol_call(fair, underlying_fair, 10250, dte, 0)
        try:
            delta = self.call_delta(fair, underlying_fair, dte, v_t)
        except:
            return [], []
        m_t = np.log(10250 / underlying_fair) / np.sqrt(dte / 365)
        # print(f"m_t = {m_t}")

        # , ,
        base_coef = 0.264416  # 0.13571776890273662 + 0.00229274*dte
        linear_coef = 0.010031  # -0.03685812200491957 + 0.0072571*dte
        squared_coef = 0.147604  # 0.16277746617792221 + 0.01096456*dte
        fair_iv = base_coef + linear_coef * m_t + squared_coef * (m_t ** 2)
        # print(fair_iv)

        diff = v_t - fair_iv
        if "prices_10250" not in traderObject:
            traderObject["prices_10250"] = [diff]
        else:
            traderObject["prices_10250"].append(diff)
        # print(diff)
        if len(traderObject["prices_10250"]) > 13:
            diff -= np.mean(traderObject["prices_10250"])
            traderObject["prices_10250"].pop(0)
        threshold = 0.00055
        if diff > threshold:  # short vol so sell option, buy und
            amount = min(market_data.buy_sum["VOLCANIC_ROCK"], market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_10250"])
            amount = min(amount, -sum(market_data.ask_volumes["VOLCANIC_ROCK"]),
                         sum(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_10250"]))
            option_amount = amount
            if np.mean(traderObject["prices_10250"]) > 0:
                rock_amount = amount
            else:
                rock_amount = amount // 2

            # print(rock_amount)
            for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK"])):
                fill = min(-market_data.ask_volumes["VOLCANIC_ROCK"][i], rock_amount)
                # print(fill)
                if fill != 0:
                    orders["VOLCANIC_ROCK"].append(
                        Order("VOLCANIC_ROCK", market_data.ask_prices["VOLCANIC_ROCK"][i], fill))
                    market_data.buy_sum["VOLCANIC_ROCK"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK"] += fill
                    rock_amount -= fill
                    # print(fill)

            for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK_VOUCHER_10250"])):
                fill = min(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_10250"][i], option_amount)
                delta_sum -= delta * fill
                if fill != 0:
                    orders["VOLCANIC_ROCK_VOUCHER_10250"].append(Order("VOLCANIC_ROCK_VOUCHER_10250",
                                                                       market_data.bid_prices[
                                                                           "VOLCANIC_ROCK_VOUCHER_10250"][i],
                                                                       -fill))
                    market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_10250"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK_VOUCHER_10250"] -= fill
                    option_amount -= fill

        elif diff < -threshold:  # long vol
            # print("LONG")
            # print("----")
            amount = min(market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_10250"], market_data.sell_sum["VOLCANIC_ROCK"])
            # print(amount)
            amount = min(amount, -sum(market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_10250"]),
                         sum(market_data.bid_volumes["VOLCANIC_ROCK"]))
            # print(amount)
            option_amount = amount
            if np.mean(traderObject["prices_10250"]) < 0:
                rock_amount = amount
            else:
                rock_amount = amount // 2
                raise Exception(state.timestamp, option_amount)
            # print(f"{rock_amount} rocks")
            for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK_VOUCHER_10250"])):
                fill = min(-market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_10250"][i], option_amount)
                delta_sum += delta * fill
                if fill != 0:
                    orders["VOLCANIC_ROCK_VOUCHER_10250"].append(Order("VOLCANIC_ROCK_VOUCHER_10250",
                                                                       market_data.ask_prices[
                                                                           "VOLCANIC_ROCK_VOUCHER_10250"][i], fill))
                    market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_10250"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK_VOUCHER_10250"] += fill
                    option_amount -= fill

            for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK"])):
                fill = min(market_data.bid_volumes["VOLCANIC_ROCK"][i], rock_amount)
                # print(fill)
                if fill != 0:
                    orders["VOLCANIC_ROCK"].append(
                        Order("VOLCANIC_ROCK", market_data.bid_prices["VOLCANIC_ROCK"][i], -fill))
                    market_data.sell_sum["VOLCANIC_ROCK"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK"] -= fill
                    rock_amount -= fill

        return orders["VOLCANIC_ROCK"], orders["VOLCANIC_ROCK_VOUCHER_10250"]

    def run(self, state: TradingState):
        traderObject = {}
        result = {}
        market_data = MarketData()
        products = ["RAINFOREST_RESIN", "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
                    "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250",
                    "VOLCANIC_ROCK_VOUCHER_10500", "VOLCANIC_ROCK"]
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        for product in products:
            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]
            bids, asks = order_depth.buy_orders, order_depth.sell_orders
            if order_depth.buy_orders:
                mm_bid = max(bids.items(), key=lambda tup: tup[1])[0]
            if order_depth.sell_orders:
                mm_ask = min(asks.items(), key=lambda tup: tup[1])[0]
            if order_depth.sell_orders and order_depth.buy_orders:
                fair_price = (mm_ask + mm_bid) / 2
            elif order_depth.sell_orders:
                fair_price = mm_ask
            elif order_depth.buy_orders:
                fair_price = mm_bid
            else:
                fair_price = traderObject[f"prev_fair_{product}"]
            traderObject[f"prev_fair_{product}"] = fair_price

            market_data.end_pos[product] = position
            market_data.buy_sum[product] = self.PRODUCT_LIMIT[product] - position
            market_data.sell_sum[product] = self.PRODUCT_LIMIT[product] + position
            market_data.bid_prices[product] = list(bids.keys())
            market_data.bid_volumes[product] = list(bids.values())
            market_data.ask_prices[product] = list(asks.keys())
            market_data.ask_volumes[product] = list(asks.values())
            market_data.fair[product] = fair_price

        result = {}
        result[Product.VOLCANIC_ROCK_VOUCHER_9750] = self.trade_9750(state, market_data, traderObject)
        result["RAINFOREST_RESIN"] = self.trade_resin(state, market_data)

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = (state.position[Product.KELP]
                             if Product.KELP in state.position
                             else 0)
            kelp_fair_value = self.kelp_fair_value(state.order_depths[Product.KELP], traderObject,
                                                   state.order_depths[Product.SQUID_INK])
            #      kelp_position = state.position.get(Product.KELP, 0)
            #      kelp_fair_value = self.kelp_fair_value(state.order_depths[Product.KELP], traderObject)
            kelp_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(Product.KELP,
                                 state.order_depths[Product.KELP],
                                 kelp_fair_value,
                                 self.params[Product.KELP]['take_width'],
                                 kelp_position,
                                 self.params[Product.KELP]['prevent_adverse'],
                                 self.params[Product.KELP]['adverse_volume'],
                                 traderObject)
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(Product.KELP,
                                  state.order_depths[Product.KELP],
                                  kelp_fair_value,
                                  self.params[Product.KELP]['clear_width'],
                                  kelp_position,
                                  buy_order_volume,
                                  sell_order_volume, )
            )
            kelp_make_orders, _, _ = self.make_orders(Product.KELP,
                                                      state.order_depths[Product.KELP],
                                                      kelp_fair_value,
                                                      kelp_position,
                                                      buy_order_volume,
                                                      sell_order_volume,
                                                      self.params[Product.KELP]['disregard_edge'],
                                                      self.params[Product.KELP]['join_edge'],
                                                      self.params[Product.KELP]['default_edge'],
                                                      )

            result[Product.KELP] = (
                    kelp_take_orders + kelp_clear_orders + kelp_make_orders
            )

        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            ink_position = (state.position[Product.SQUID_INK]
                            if Product.SQUID_INK in state.position
                            else 0)
            ink_fair_value = self.ink_fair_value(state.order_depths[Product.SQUID_INK], traderObject)
            ink_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(Product.SQUID_INK,
                                 state.order_depths[Product.SQUID_INK],
                                 ink_fair_value,
                                 self.params[Product.SQUID_INK]['take_width'],
                                 ink_position,
                                 self.params[Product.SQUID_INK]['prevent_adverse'],
                                 self.params[Product.SQUID_INK]['adverse_volume'],
                                 traderObject)
            )
            ink_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(Product.SQUID_INK,
                                  state.order_depths[Product.SQUID_INK],
                                  ink_fair_value,
                                  self.params[Product.SQUID_INK]['clear_width'],
                                  ink_position,
                                  buy_order_volume,
                                  sell_order_volume, )
            )
            ink_make_orders, _, _ = self.make_orders(Product.SQUID_INK,
                                                     state.order_depths[Product.SQUID_INK],
                                                     ink_fair_value,
                                                     ink_position,
                                                     buy_order_volume,
                                                     sell_order_volume,
                                                     self.params[Product.SQUID_INK]['disregard_edge'],
                                                     self.params[Product.SQUID_INK]['join_edge'],
                                                     self.params[Product.SQUID_INK]['default_edge'],
                                                     )

            result[Product.SQUID_INK] = (
                    ink_take_orders + ink_clear_orders + ink_make_orders
            )

        if Product.SPREAD1 not in traderObject:
            traderObject[Product.SPREAD1] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }
        picnic1_position = (
            state.position[Product.PICNIC_BASKET1]
            if Product.PICNIC_BASKET1 in state.position
            else 0
        )
        spread1_orders = self.spread_orders(
            state.order_depths,
            Product.PICNIC_BASKET1,
            picnic1_position,
            traderObject[Product.SPREAD1],
            SPREAD=Product.SPREAD1,
            picnic1=True
        )
        if spread1_orders:
            result[Product.DJEMBES] = spread1_orders[Product.DJEMBES]
            result[Product.CROISSANT] = spread1_orders[Product.CROISSANT]
            result[Product.JAMS] = spread1_orders[Product.JAMS]
            result[Product.PICNIC_BASKET1] = spread1_orders[Product.PICNIC_BASKET1]

        if Product.SPREAD2 not in traderObject:
            traderObject[Product.SPREAD2] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        picnic2_position = (state.position[Product.PICNIC_BASKET2]
                            if Product.PICNIC_BASKET2 in state.position else 0)
        spread2_orders = self.spread_orders(
            state.order_depths,
            Product.PICNIC_BASKET2,
            picnic2_position,
            traderObject[Product.SPREAD2],
            SPREAD=Product.SPREAD2,
            picnic1=False
        )
        if spread2_orders:
            result[Product.CROISSANT] = spread2_orders[Product.CROISSANT]
            result[Product.JAMS] = spread2_orders[Product.JAMS]
            result[Product.PICNIC_BASKET2] = spread2_orders[Product.PICNIC_BASKET2]

        # Round 3
        result[Product.VOLCANIC_ROCK_VOUCHER_10000] = self.trade_10000(state, market_data, traderObject)

        rock1, result[Product.VOLCANIC_ROCK_VOUCHER_10500] = self.trade_10500(state, market_data, traderObject)
        rock2, result[Product.VOLCANIC_ROCK_VOUCHER_9500] = self.trade_9500(state, market_data, traderObject)
        rock3, result[Product.VOLCANIC_ROCK_VOUCHER_10250] = self.trade_10250(state, market_data, traderObject)
        result[Product.VOLCANIC_ROCK] = rock1 + rock2 + rock3

        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
