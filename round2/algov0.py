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

    limits = {
        "KELP": 50,
        "RAINFOREST_RESIN": 50,
        "CROISSANTS": 250,
        "JAMS": 350,
        "DJEMBES": 60,
        "PICNIC_BASKET1": 60,
        "PICNIC_BASKET2": 100
    }

    # need to optimise, if weve cleared the best layer, send another order in with new best layer
    def trade_kelp(self, state, market_data):
        product = "KELP"
        orders = []
        fair = market_data.fair[product]
        if fair//1 == 0: # fair is a whole number so use it
            edge = 1
        else:
            edge = 0.5

        # for each buy order level, if > fair, fill completely
        if market_data.sell_sum[product] > 0:
            for i in range(0, len(market_data.bid_prices[product])):
                if market_data.bid_prices[product][i] > fair:
                    fill = min(market_data.bid_volumes[product][i], market_data.sell_sum[product])
                    if fill != 0:
                        orders.append(Order(product, market_data.bid_prices[product][i], -fill))
                        market_data.sell_sum[product] -= fill
                        market_data.end_pos[product] -= fill

        # for each sell order level, if < fair, fill completely
        if market_data.buy_sum[product] > 0:
            for i in range(0, len(market_data.ask_prices[product])):
                if market_data.ask_prices[product][i] < fair:
                    fill = min(-market_data.ask_volumes[product][i], market_data.buy_sum[product])
                    if fill != 0:
                        orders.append(Order(product, market_data.ask_prices[product][i], fill))
                        market_data.buy_sum[product] -= fill
                        market_data.end_pos[product] += fill

        for i in range(0, len(market_data.bid_prices[product])):
                if market_data.bid_prices[product][i] == fair and market_data.end_pos[product] > 0:
                    fill = min(market_data.bid_volumes[product][i], market_data.sell_sum[product])
                    if fill != 0:
                        orders.append(Order(product, market_data.bid_prices[product][i], -fill))
                        market_data.sell_sum[product] -= fill
                        market_data.end_pos[product] -= fill


        # If Q @ +-2 ==1: still place at +-2 else dip in
        if market_data.buy_sum[product] !=0:
            orders.append(Order("KELP", min(int(fair-edge), market_data.bid_prices[product][0]+1), min(15, market_data.buy_sum[product]))) # bid
        if market_data.sell_sum[product] != 0:
            orders.append(Order("KELP", max(int(fair+edge), market_data.ask_prices[product][0]-1), -min(15, market_data.sell_sum[product]))) # ask

        market_data.buy_sum[product] -= min(15, market_data.buy_sum[product])
        market_data.sell_sum[product] -= min(15, market_data.sell_sum[product])

        return orders

    def trade_resin(self, state, market_data):
        product = "RAINFOREST_RESIN"
        end_pos = state.position.get(product, 0)
        buy_sum = self.limits[product] - end_pos
        sell_sum = self.limits[product] + end_pos
        orders = []
        order_depth: OrderDepth = state.order_depths[product]
        bids = order_depth.buy_orders
        asks = order_depth.sell_orders
        bid_prices = list(bids.keys())
        bid_volumes = list(bids.values())
        ask_prices = list(asks.keys())
        ask_volumes = list(asks.values())



        #for each buy order level, if > fair, fill completely SELLING
        if sell_sum > 0:
            for i in range(0, len(bid_prices)):
                if bid_prices[i] > 10000:
                    fill = min(bid_volumes[i], sell_sum)
                    orders.append(Order(product, bid_prices[i], -fill))
                    sell_sum -= fill
                    end_pos -= fill
                    bid_volumes[i] -= fill

        #remove prices that were matched against
        bid_prices, bid_volumes = zip(*[(ai, bi) for ai, bi in zip(bid_prices, bid_volumes) if bi != 0])
        bid_prices = list(bid_prices)
        bid_volumes = list(bid_volumes)


        #for each sell order level, if < fair, fill completely BUYING
        if buy_sum > 0:
            for i in range(0, len(ask_prices)):
                if ask_prices[i] < 10000:
                    fill = min(-ask_volumes[i], buy_sum)
                    orders.append(Order(product, ask_prices[i], fill))
                    buy_sum -= fill
                    end_pos += fill
                    ask_volumes[i] +=fill

        #remove prices that were matched against
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
        if abs(ask_volumes[0]) > 1 :
            orders.append(Order(product, max(ask_prices[0]-1, 10000+1), -min(14, sell_sum))) # ask
        else:
            orders.append(Order(product, max(10000+1, ask_prices[0]), -min(14, sell_sum))) # ask
        sell_sum -= min(14, sell_sum)

        if bid_volumes[0] > 1:
            orders.append(Order(product, min(bid_prices[0]+1, 10000-1), min(14, buy_sum))) # bid
        else:
            orders.append(Order(product, min(10000-1, bid_prices[0]), min(14, buy_sum))) # bid
        buy_sum -= min(14, buy_sum)


        # orders.append(Order(product, 10000-2, min(14, buy_sum))) # bid
        # orders.append(Order(product, 10000+2, -min(14, sell_sum))) # ask


        if end_pos > 0: # sell to bring pos closer to 0
            for i in range(0, len(bid_prices)):
                if bid_prices[i] == 10000:
                    fill = min(bid_volumes[i], sell_sum)
                    orders.append(Order(product, bid_prices[i], -fill))
                    sell_sum -= fill
                    end_pos -= fill

        if end_pos < 0: # buy to bring pos closer to 0
            for i in range(0, len(ask_prices)):
                if ask_prices[i] == 10000:
                    fill = min(-ask_volumes[i], buy_sum)
                    orders.append(Order(product, ask_prices[i], fill))
                    buy_sum -= fill
                    end_pos += fill

        return orders

    def trade_ink(self, state, market_data):
        # if significant price drop, buy in
        pass


    def trade_basket_1(self, state, market_data):
        product = "PICNIC_BASKET1"
        prices_history = json.loads(state.traderData) if state.traderData else {}
        orders ={}
        for p in ["JAMS", "CROISSANTS", "PICNIC_BASKET1", "DJEMBES"]:
            orders[p] = []

        fair = market_data.fair[product]

        synthetic = 3*market_data.fair["JAMS"] + 6*market_data.fair["CROISSANTS"] + market_data.fair["DJEMBES"]
        diff = fair - synthetic
        prices = prices_history.get("PICNIC_BASKET1", [])
        prices.append(synthetic)

        def buy_basket_1(units):
            for product in ["PICNIC_BASKET1"]:
                fill = units
                for i in range(0, len(market_data.ask_prices[product])):
                    level_fill = min(-market_data.ask_volumes[product][i], market_data.buy_sum[product], fill)
                    if level_fill != 0:
                        orders[product].append(Order(product, market_data.ask_prices[product][i], level_fill))
                        fill -= level_fill
                        market_data.buy_sum[product] -= level_fill
                        market_data.end_pos[product] += level_fill

            for product in ["JAMS", "CROISSANTS", "DJEMBES"]:
                multiple = 6 if product == "CROISSANTS" else 3 if product == "JAMS" else 1
                fill = units*multiple
                if market_data.sell_sum[product] > 0:
                    for i in range(0, len(market_data.bid_prices[product])):
                        level_fill = min(market_data.bid_volumes[product][i], fill)
                        if level_fill != 0:
                            orders[product].append(Order(product, market_data.bid_prices[product][i], -level_fill))
                            fill -= level_fill
                            market_data.end_pos[product] -= level_fill
                            market_data.sell_sum[product] -= level_fill

        def sell_basket_1(units):
            fill = units
            for product in ["PICNIC_BASKET1"]:
                for i in range(0, len(market_data.bid_prices[product])):
                    level_fill = min(market_data.bid_volumes[product][i], fill)
                    if level_fill != 0:
                        orders[product].append(Order(product, market_data.bid_prices[product][i], -level_fill))
                        fill -= level_fill
                        market_data.end_pos[product] -= level_fill
                        market_data.sell_sum[product] -= level_fill

            for product in ["JAMS", "CROISSANTS", "DJEMBES"]:
                multiple = 6 if product == "CROISSANTS" else 3 if product == "JAMS" else 1
                fill = units*multiple
                if market_data.sell_sum[product] > 0:
                    for i in range(0, len(market_data.ask_prices[product])):
                        level_fill = min(-market_data.ask_volumes[product][i], market_data.buy_sum[product], fill)
                        if level_fill != 0:
                            orders[product].append(Order(product, market_data.ask_prices[product][i], level_fill))
                            fill -= level_fill
                            market_data.buy_sum[product] -= level_fill
                            market_data.end_pos[product] += level_fill


        if diff > 100: # short the gap
            amount = min(market_data.buy_sum["JAMS"]/3, market_data.buy_sum["CROISSANTS"]/6, market_data.buy_sum["DJEMBES"])
            amount = min(amount, -sum(market_data.ask_volumes["JAMS"])/3, -sum(market_data.ask_volumes["CROISSANTS"])/6, -sum(market_data.ask_volumes["DJEMBES"]))
            amount = min(amount, market_data.sell_sum["PICNIC_BASKET1"])
            amount = int(min(amount, sum(market_data.bid_volumes["PICNIC_BASKET1"])))
            if amount != 0:
                sell_basket_1(amount)

        elif -20 <= diff <= 20: # liquidate
            units = market_data.end_pos["PICNIC_BASKET1"]

            if units > 0: # we are long etfs, so sell n etfs and buy 4n croissants and 2n jams
                amount = min(units, sum(market_data.bid_volumes["PICNIC_BASKET1"]), -sum(market_data.ask_volumes["JAMS"])/3, -sum(market_data.ask_volumes["CROISSANTS"])/6, -sum(market_data.ask_volumes["DJEMBES"]))
                amount = min(amount, market_data.sell_sum["PICNIC_BASKET1"], market_data.buy_sum["JAMS"]/3, market_data.buy_sum["CROISSANTS"]/6, market_data.buy_sum["DJEMBES"])
                if amount != 0:
                    sell_basket_1(amount)

            elif units < 0: # we are short etfs, so buy n etfs and sell 4n croissants and 2n jams
                amount = min(-units, -sum(market_data.ask_volumes["PICNIC_BASKET1"]), sum(market_data.bid_volumes["JAMS"])/3, sum(market_data.bid_volumes["CROISSANTS"])/6, sum(market_data.bid_volumes["DJEMBES"]))
                amount = min(amount, market_data.buy_sum["PICNIC_BASKET1"], market_data.sell_sum["JAMS"]/3, market_data.sell_sum["CROISSANTS"]/6, market_data.sell_sum["DJEMBES"])
                if amount != 0:
                    buy_basket_1(amount)

        elif diff < -69: # long the gap or still need
            amount = min(market_data.sell_sum["JAMS"]/3, market_data.sell_sum["CROISSANTS"]/6, market_data.sell_sum["DJEMBES"])
            amount = min(amount, sum(market_data.bid_volumes["JAMS"])/3, sum(market_data.bid_volumes["CROISSANTS"])/6, sum(market_data.bid_volumes["DJEMBES"]))
            amount = min(amount, market_data.buy_sum["PICNIC_BASKET1"])
            amount = int(min(amount, -sum(market_data.ask_volumes["PICNIC_BASKET1"])))
            if amount != 0:
                buy_basket_1(amount)

        if len(prices) > 50:
            mean = np.mean(np.array(prices))
            diff_mean = synthetic-mean
            amount = 1
            if diff_mean > 13:
                # amount = min(market_data.buy_sum["JAMS"]/3, market_data.buy_sum["CROISSANTS"]/6, market_data.buy_sum["DJEMBES"])
                # amount = min(amount, -sum(market_data.ask_volumes["JAMS"])/3, -sum(market_data.ask_volumes["CROISSANTS"])/6, -sum(market_data.ask_volumes["DJEMBES"]))
                # amount = min(amount, market_data.sell_sum["PICNIC_BASKET1"])
                # amount = int(min(amount, sum(market_data.bid_volumes["PICNIC_BASKET1"])))
                sell_basket_1(1)
            elif  diff_mean < -13:
                # amount = min(market_data.sell_sum["JAMS"]/3, market_data.sell_sum["CROISSANTS"]/6, market_data.sell_sum["DJEMBES"])
                # amount = min(amount, sum(market_data.bid_volumes["JAMS"])/3, sum(market_data.bid_volumes["CROISSANTS"])/6, sum(market_data.bid_volumes["DJEMBES"]))
                # amount = min(amount, market_data.buy_sum["PICNIC_BASKET1"])
                # amount = int(min(amount, -sum(market_data.ask_volumes["PICNIC_BASKET1"])))
                buy_basket_1(1)

        prices = prices[-60:]
        prices_history["PICNIC_BASKET1"] = prices

        return orders["PICNIC_BASKET1"], orders["CROISSANTS"], orders["JAMS"], orders["DJEMBES"], prices_history["PICNIC_BASKET1"]

    def trade_basket_2(self, state, market_data):
        product = "PICNIC_BASKET2"
        prices_history = json.loads(state.traderData) if state.traderData else {}
        orders ={}
        for p in ["JAMS", "CROISSANTS", "PICNIC_BASKET2", "PICNIC_BASKET1", "DJEMBES"]:
            # CHECK IF we are at target for some but not others
            orders[p] = []

        fair = market_data.fair[product]

        synthetic = 2*market_data.fair["JAMS"] + 4*market_data.fair["CROISSANTS"]
        diff = fair - synthetic
        prices = prices_history.get("PICNIC_BASKET2", [])
        prices.append(synthetic)

        def buy_basket_2(units):
            for product in ["PICNIC_BASKET2"]:
                fill = units
                print(f"total buy for {product} = {fill}")
                for i in range(0, len(market_data.ask_prices[product])):
                    level_fill = min(-market_data.ask_volumes[product][i], market_data.buy_sum[product], fill)
                    if level_fill != 0:
                        orders[product].append(Order(product, market_data.ask_prices[product][i], level_fill))
                        fill -= level_fill
                        print(level_fill)
                        market_data.buy_sum[product] -= level_fill
                        market_data.end_pos[product] += level_fill

            for product in ["JAMS", "CROISSANTS"]:
                multiple = 4 if product == "CROISSANTS" else 2
                fill = units*multiple
                print(f"total sell for {product} = {fill}")
                if market_data.sell_sum[product] > 0:
                    for i in range(0, len(market_data.bid_prices[product])):
                        level_fill = min(market_data.bid_volumes[product][i], fill)
                        if level_fill != 0:
                            orders[product].append(Order(product, market_data.bid_prices[product][i], -level_fill))
                            fill -= level_fill
                            print(level_fill)
                            market_data.end_pos[product] -= level_fill
                            market_data.sell_sum[product] -= level_fill

        def sell_basket_2(units):
            fill = units
            for product in ["PICNIC_BASKET2"]:
                for i in range(0, len(market_data.bid_prices[product])):
                    level_fill = min(market_data.bid_volumes[product][i], fill)
                    if level_fill != 0:
                        orders[product].append(Order(product, market_data.bid_prices[product][i], -level_fill))
                        fill -= level_fill
                        market_data.end_pos[product] -= level_fill
                        market_data.sell_sum[product] -= level_fill

            for product in ["JAMS", "CROISSANTS"]:
                multiple = 4 if product == "CROISSANTS" else 2
                fill = units*multiple
                if market_data.sell_sum[product] > 0:
                    for i in range(0, len(market_data.ask_prices[product])):
                        level_fill = min(-market_data.ask_volumes[product][i], market_data.buy_sum[product], fill)
                        if level_fill != 0:
                            orders[product].append(Order(product, market_data.ask_prices[product][i], level_fill))
                            fill -= level_fill
                            market_data.buy_sum[product] -= level_fill
                            market_data.end_pos[product] += level_fill


        if diff > 100: # short the gap
            amount = min(market_data.buy_sum["JAMS"]/2, market_data.buy_sum["CROISSANTS"]/4)
            amount = min(amount, -sum(market_data.ask_volumes["JAMS"])/2, -sum(market_data.ask_volumes["CROISSANTS"])/4)
            amount = min(amount, market_data.sell_sum["PICNIC_BASKET2"])
            amount = int(min(amount, sum(market_data.bid_volumes["PICNIC_BASKET2"])))
            if amount != 0:
                sell_basket_2(amount)

        elif -20 <= diff <= 20: # liquidate
            units = market_data.end_pos["PICNIC_BASKET2"]

            if units > 0: # we are long etfs, so sell n etfs and buy 4n croissants and 2n jams
                amount = min(units, sum(market_data.bid_volumes["PICNIC_BASKET2"]), -sum(market_data.ask_volumes["JAMS"])/2, -sum(market_data.ask_volumes["CROISSANTS"])/4)
                amount = min(amount, market_data.sell_sum["PICNIC_BASKET2"], market_data.buy_sum["JAMS"]/2, market_data.buy_sum["CROISSANTS"]/4)
                if amount != 0:
                    sell_basket_2(amount)

            elif units < 0: # we are short etfs, so buy n etfs and sell 4n croissants and 2n jams
                amount = min(-units, -sum(market_data.ask_volumes["PICNIC_BASKET2"]), sum(market_data.bid_volumes["JAMS"])/2, sum(market_data.bid_volumes["CROISSANTS"])/4)
                amount = min(amount, market_data.buy_sum["PICNIC_BASKET2"], market_data.sell_sum["JAMS"]/2, market_data.sell_sum["CROISSANTS"]/4)
                if amount != 0:
                    buy_basket_2(amount)

        elif diff < -69: # long the gap or still need
            amount = min(market_data.sell_sum["JAMS"]/2, market_data.sell_sum["CROISSANTS"]/4)
            amount = min(amount, sum(market_data.bid_volumes["JAMS"])/2, sum(market_data.bid_volumes["CROISSANTS"])/4)
            amount = min(amount, market_data.buy_sum["PICNIC_BASKET2"])
            amount = int(min(amount, -sum(market_data.ask_volumes["PICNIC_BASKET2"])))
            if amount != 0:
                buy_basket_2(amount)

        if len(prices) > 50:
            mean = np.mean(np.array(prices))
            diff_mean = synthetic-mean
            amount = 1
            if diff_mean > 8:
                # amount = min(market_data.buy_sum["JAMS"]/2, market_data.buy_sum["CROISSANTS"]/4)
                # amount = min(amount, -sum(market_data.ask_volumes["JAMS"])/2, -sum(market_data.ask_volumes["CROISSANTS"])/4)
                # amount = min(amount, market_data.sell_sum["PICNIC_BASKET2"])
                # amount = int(min(amount, sum(market_data.bid_volumes["PICNIC_BASKET2"])))
                sell_basket_2(1)
            elif  diff_mean < -8:
                # amount = min(market_data.sell_sum["JAMS"]/2, market_data.sell_sum["CROISSANTS"]/4)
                # amount = min(amount, sum(market_data.bid_volumes["JAMS"])/2, sum(market_data.bid_volumes["CROISSANTS"])/4)
                # amount = min(amount, market_data.buy_sum["PICNIC_BASKET2"])
                # amount = int(min(amount, -sum(market_data.ask_volumes["PICNIC_BASKET2"])))
                buy_basket_2(1)



        prices = prices[-60:]
        prices_history["PICNIC_BASKET2"] = prices
        return orders["PICNIC_BASKET2"], orders["CROISSANTS"], orders["JAMS"], prices_history

    def run(self, state: TradingState):
        result = {}
        market_data = MarketData()

        for product in list(self.limits.keys()):
            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]
            bids, asks = order_depth.buy_orders, order_depth.sell_orders
            mm_bid = max(bids.items(), key=lambda tup: tup[1])[0]
            mm_ask = min(asks.items(), key=lambda tup: tup[1])[0]
            fair_price = (mm_ask + mm_bid) / 2

            market_data.end_pos[product] = position
            market_data.buy_sum[product] = self.limits[product] - position
            market_data.sell_sum[product] = self.limits[product] + position
            market_data.bid_prices[product] = list(bids.keys())
            market_data.bid_volumes[product] = list(bids.values())
            market_data.ask_prices[product] = list(asks.keys())
            market_data.ask_volumes[product] = list(asks.values())
            market_data.fair[product] = fair_price

        result["KELP"] = self.trade_kelp(state, market_data)
        result["RAINFOREST_RESIN"] = self.trade_resin(state, market_data)
        result["PICNIC_BASKET1"], result["CROISSANTS"], result["JAMS"], result["DJEMBES"], prices_history = self.trade_basket_1(state, market_data)
        result["PICNIC_BASKET2"], result["CROISSANTS"], result["JAMS"], prices_history = self.trade_basket_2(state, market_data)


        conversions = 0
        traderData = json.dumps(prices_history)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

#python -m prosperity3bt Tutorial.py 0
#5507 online
