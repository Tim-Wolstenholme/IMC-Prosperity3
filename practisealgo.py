from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
from math import log,sqrt


class Trader:

    def __init__(self):
        # Initialise some stuff, like amount of each product currently kept and ema prices etc.
        self.product_limits = {"RAINFOREST_RESIN": 50,
                               "KELP": 50}
        self.emas = {"RAINFOREST_RESIN": None,
                     "KELP": None}
        self.ema_alpha = 0.5  # Hyperparam
        self.round = 0
        self.PRODUCTS = ["RAINFOREST_RESIN", "KELP"]
        self.defaults = {"RAINFOREST_RESIN": None,
                          "KELP": None}
        self.volatilities = {"RAINFOREST_RESIN": 0.1,
                             "KELP": 0.1} #Should use backtesting data to make these more accurate later on

    def get_defaults(self,state):
        for product in self.PRODUCTS:
            market_bids = state.order_depths[product].buy_orders
            market_asks = state.order_depths[product].sell_orders
            if market_bids:
                best_bid = max(market_bids)
                self.defaults[product] = best_bid
            if market_asks:
                best_ask = min(market_asks)
                self.defaults[product] = best_ask
            if market_bids and market_asks:
                self.defaults[products] = (best_bid + best_ask)*0.5

    def get_mid_prices(self,product,state):
        if product not in state.order_depths:
            return self.emas[product] if self.emas[product] else self.defaults[product]
        market_bids = state.order_depths[product].buy_orders
        market_asks = state.order_depths[product].sell_orders
        #If no bids or asks, return previous price
        if any(len(x) == 0 for x in (market_asks,market_bids)):
            return self.emas[product] if self.emas[product] else self.defaults[product]
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return 0.5*(best_bid+best_ask)

    def update_ema(self,state):
        for product in self.PRODUCTS:
            mid_price = self.get_mid_price(product,state)
            if self.emas[product] is None:
                self.emas[product] = mid_price
            self.emas[product] = self.ema_alpha*mid_price + (1-self.ema_alpha)*self.emas[product]

    def update_volatilites(self,state):
        #Using realised volatility formula
       if self.round == 1:
           return
       for product in self.PRODUCTS:
           prev_price = self.emas[product]
           log_returns = log(self.get_mid_prices(product,state)-prev_price)
           self.volatilities[product] = sqrt((self.round-1/self.round)*(self.volatilies[product]**2+(T/self.round-1)*log_returns**2))

    def kelp_strat(self,state):
        position = state.position["KELP"]
        bid_volume = self.product_limits["KELP"] - position
        ask_volume = -self.product_limits["KELP"] - position
        orders = []
        if self.position == 0:
            #Neither long nor short, so symm
            orders.append(Order("KELP",math.floor(self.emas["KELP"]-1,),bid_volume))
            orders.append(Order("KELP", math.floor(self.emas["KELP"] + 1,), ask_volume))

        if self.position < 0:
            #Prioritise long
            orders.append(Order("KELP", math.floor(self.emas["KELP"] - 2), bid_volume))
            orders.append(Order("KELP", math.floor(self.emas["KELP"]), ask_volume))

        if self.position > 0:
            #prioritise short
            orders.append(Order("KELP", math.floor(self.emas["KELP"]), bid_volume))
            orders.append(Order("KELP", math.floor(self.emas["KELP"] + 2), ask_volume))
        return orders

    def run(self, state: TradingState):
        if self.round == 0:
            self.get_defaults()
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = 10  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(
                len(order_depth.sell_orders)))

            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))

            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))

            result[product] = orders

        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        return result, conversions, traderData
