import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from dataclasses import dataclass, field
from typing import List, Any
import string
import numpy as np
import math
from typing import Dict, List

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
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
            compressed.append([listing.symbol, listing.product, listing.denomination])

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

    def compress_observations(self, observations: Observation) -> list[Any]:
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

        return value[: max_length - 3] + "..."

logger = Logger()

class MarketData:
    end_pos: Dict[str, int] = {}
    buy_sum: Dict[str, int] = {}
    sell_sum: Dict[str, int] = {}
    bid_prices: Dict[str, List[float]] = {}
    bid_volumes: Dict[str, List[int]] = {}
    ask_prices: Dict[str, List[float]] = {}
    ask_volumes: Dict[str, List[int]] = {}
    fair: Dict[str, float] = {}

class Trader:
    prices = []
    prices2 = []
    prices3 = []
    limits = {
        "KELP": 50,
        "RAINFOREST_RESIN": 50,
        "CROISSANTS": 250,
        "JAMS": 350,
        "DJEMBES": 60,
        "PICNIC_BASKET1": 60,
        "PICNIC_BASKET2": 100,
        "VOLCANIC_ROCK_VOUCHER_9750": 200,
        "VOLCANIC_ROCK_VOUCHER_9500": 200,
        "VOLCANIC_ROCK_VOUCHER_10000": 200,
        "VOLCANIC_ROCK_VOUCHER_10250": 200,
        "VOLCANIC_ROCK_VOUCHER_10500": 200,
        "VOLCANIC_ROCK": 400,
        "MAGNIFICENT_MACARONS": 75
    }
    last_sunlight = 0
    sunlight_history = []  # Track last 5 sunlight values

    def calculate_sunlight_rate_of_change(self):
        """Calculate the average rate of change of sunlight over the last 5 ticks"""
        if len(self.sunlight_history) < 5:
            return 0
        changes = []
        for i in range(1, len(self.sunlight_history)):
            changes.append(self.sunlight_history[i] - self.sunlight_history[i-1])
        return sum(changes) / len(changes)

    def trade_macaron(self, state, market_data):
        product = "MAGNIFICENT_MACARONS"
        orders ={}
        for p in ["MAGNIFICENT_MACARONS"]:
            orders[p] = []
        fair = market_data.fair[product]
        conversions = 0
        #print(state.observations.conversionObservations[product])

        overseas_ask = state.observations.conversionObservations[product].askPrice + state.observations.conversionObservations[product].transportFees + state.observations.conversionObservations[product].importTariff
        overseas_bid = state.observations.conversionObservations[product].bidPrice - state.observations.conversionObservations[product].transportFees - state.observations.conversionObservations[product].exportTariff

        if state.observations.conversionObservations[product].sunlightIndex < self.last_sunlight:
            direction = -1
        elif state.observations.conversionObservations[product].sunlightIndex == self.last_sunlight:
            direction = 0
        else:
            direction = 1
        
        # Update sunlight history
        self.sunlight_history.append(state.observations.conversionObservations[product].sunlightIndex)
        if len(self.sunlight_history) > 5:
            self.sunlight_history.pop(0)
        
        self.last_sunlight = state.observations.conversionObservations[product].sunlightIndex

        # New trading strategy based on bid/ask volumes and sunlight
        total_bids = sum(market_data.bid_volumes[product])
        total_asks = -sum(market_data.ask_volumes[product])
        
        current_sunlight = state.observations.conversionObservations[product].sunlightIndex

        # Calculate z-score for position management
        mean_price = 640
        std_dev = 55  # Based on range 550-750
        current_price = fair  # Using the fair price as current price
        z_score = (current_price - mean_price) / std_dev

        # Strategy for sunlight below 50
        if current_sunlight < 50:
            # Buy if sunlight dropped below 50 and is less than previous day
            if direction == -1 and market_data.buy_sum[product] > 0:
                amount = min(market_data.buy_sum[product], -sum(market_data.ask_volumes[product]))
                for i in range(0, len(market_data.ask_prices[product])):
                    fill = min(-market_data.ask_volumes[product][i], amount)
                    if fill != 0:
                        orders[product].append(Order(product, market_data.ask_prices[product][i], fill))
                        market_data.buy_sum[product] -= fill
                        market_data.end_pos[product] += fill
                        amount -= fill
            # Go short if sunlight is increasing rapidly from below 50
            elif direction == 1 and market_data.sell_sum[product] > 0 and self.calculate_sunlight_rate_of_change() > 0.008:
                amount = min(market_data.sell_sum[product], sum(market_data.bid_volumes[product]))
                for i in range(0, len(market_data.bid_prices[product])):
                    fill = min(market_data.bid_volumes[product][i], amount)
                    if fill != 0:
                        orders[product].append(Order(product, market_data.bid_prices[product][i], -fill))
                        market_data.sell_sum[product] -= fill
                        market_data.end_pos[product] -= fill
                        amount -= fill
            # Close short position if sunlight reaches 49
            elif abs(current_sunlight -49) < 1 and market_data.end_pos[product] < 0:
                amount = min(market_data.buy_sum[product], -market_data.end_pos[product])
                for i in range(0, len(market_data.ask_prices[product])):
                    fill = min(-market_data.ask_volumes[product][i], amount)
                    if fill != 0:
                        orders[product].append(Order(product, market_data.ask_prices[product][i], fill))
                        market_data.buy_sum[product] -= fill
                        market_data.end_pos[product] += fill
                        amount -= fill

        elif current_sunlight > 50:
            # Mean reversion strategy with z-score
            if z_score < -1.2 and market_data.buy_sum[product] > 0:  # Price is significantly below mean
                # Buy when price is significantly below mean
                amount = min(market_data.buy_sum[product], -sum(market_data.ask_volumes[product]))
                for i in range(0, len(market_data.ask_prices[product])):
                    fill = min(-market_data.ask_volumes[product][i], amount)
                    if fill != 0:
                        orders[product].append(Order(product, market_data.ask_prices[product][i], fill))
                        market_data.buy_sum[product] -= fill
                        market_data.end_pos[product] += fill
                        amount -= fill
            elif z_score > 1.2 and market_data.sell_sum[product] > 0:  # Price is significantly above mean
                # Sell when price is significantly above mean
                amount = min(market_data.sell_sum[product], sum(market_data.bid_volumes[product]))
                for i in range(0, len(market_data.bid_prices[product])):
                    fill = min(market_data.bid_volumes[product][i], amount)
                    if fill != 0:
                        orders[product].append(Order(product, market_data.bid_prices[product][i], -fill))
                        market_data.sell_sum[product] -= fill
                        market_data.end_pos[product] -= fill
                        amount -= fill

        return orders["MAGNIFICENT_MACARONS"], conversions

    def run(self, state: TradingState):
        result = {}
        market_data = MarketData()

        for product in list(self.limits.keys()):
            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]
            bids, asks = order_depth.buy_orders, order_depth.sell_orders
            if bids != {}:
                mm_bid = max(bids.items(), key=lambda tup: tup[1])[0]
                mm_ask = min(asks.items(), key=lambda tup: tup[1])[0]
                fair_price = (mm_ask + mm_bid) / 2
            else:
                fair_price = list(asks.keys())[0]

            market_data.end_pos[product] = position
            market_data.buy_sum[product] = self.limits[product] - position
            market_data.sell_sum[product] = self.limits[product] + position
            market_data.bid_prices[product] = list(bids.keys())
            market_data.bid_volumes[product] = list(bids.values())
            market_data.ask_prices[product] = list(asks.keys())
            market_data.ask_volumes[product] = list(asks.values())
            market_data.fair[product] = fair_price

        #result["KELP"] = self.trade_kelp(state, market_data)
        #result["RAINFOREST_RESIN"] = self.trade_resin(state, market_data)
        #result["PICNIC_BASKET1"], result["CROISSANTS"], result["JAMS"], result["DJEMBES"] = self.trade_basket_1(state, market_data)
        #result["VOLCANIC_ROCK_VOUCHER_10250"], result4 = self.trade_option(state, market_data, 10250, 0.001)
        # logger.print(state.observations.conversionObservations)
        # logger.print(state.observations)
        result["MAGNIFICENT_MACARONS"], conversions = self.trade_macaron(state, market_data)
        traderData = "" #json.dumps(prices)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

#python -m prosperity3bt dynamicbook.py 2
#python E:/imc-prosperity-3-backtester-master/prosperity3bt/__main__.py arb.py 4 --print
