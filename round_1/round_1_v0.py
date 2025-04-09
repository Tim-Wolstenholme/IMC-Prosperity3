from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math

"""
Market Make around 10_000 for RESIN,
Linear Regression Model on KELP and INK
"""

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 40, # 30
    },
    Product.KELP: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15, # 20 - doesn't work as great
        "reversion_beta": 0.0, # -0.2184
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 1,
        "ink_adjustment_factor": 0.05,
    },
    Product.SQUID_INK: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.2264,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    }

}

class Trader:
  def __init__(self, params=None):
    if params is None:
      params = PARAMS

    self.params = params
    self.PRODUCT_LIMIT = {Product.RAINFOREST_RESIN: 50,
                           Product.KELP: 50,
                           Product.SQUID_INK: 50}

  def take_best_orders(self, product: str,
                       fair_value: str, take_width: float,
                       orders: List[Order], order_depth: OrderDepth,
                       position: int, buy_order_volume: int,
                       sell_order_volume: int,
                       prevent_adverse: bool = False, adverse_volume: int = 0
                       ):

    position_limit = self.PRODUCT_LIMIT[product]

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
        orders.append(Order(product, round(bid), buy_quantity))  # Buy order

    sell_quantity = self.PRODUCT_LIMIT[product] + (position - sell_order_volume)
    if sell_quantity > 0:
        orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
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
      if abs (order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
      valid_buy = [price for price in order_depth.buy_orders.keys()
      if abs (order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]

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
                           if abs(ink_order_depth.sell_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]
          valid_buy_ink = [price for price in ink_order_depth.buy_orders.keys()
                           if abs(ink_order_depth.buy_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]
          if valid_ask_ink and valid_buy_ink:
              new_ink_mid = (min(valid_ask_ink) + max(valid_buy_ink)) / 2
          else:
              new_ink_mid = (min(ink_order_depth.sell_orders.keys()) +
                             max(ink_order_depth.buy_orders.keys())) / 2

          ink_return = (new_ink_mid - old_ink_price) / old_ink_price
          fair = fair - (self.params[Product.KELP].get("ink_adjustment_factor", 0.5) * ink_return * mmmid_price)
          #ink_return = (traderObject["ink_last_price"] - traderObject["prev_ink_price"]) / traderObject["prev_ink_price"]
          #adj = self.params[Product.KELP].get("ink_adjustment_factor", 0.5) * ink_return * mmmid_price
          #fair = fair - adj
    
      #traderObject["prev_ink_price"] = traderObject.get("ink_last_price", mmmid_price)
      traderObject["kelp_last_price"] = mmmid_price
      return fair
    return None

  def ink_fair_value(self, order_depth: OrderDepth, traderObject):
     if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
    
        valid_ask = [price for price in order_depth.sell_orders.keys()
                     if abs (order_depth.sell_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]
        valid_buy = [price for price in order_depth.buy_orders.keys()
                     if abs (order_depth.buy_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]
    
        mm_ask = min(valid_ask) if len(valid_ask) > 0 else None
        mm_bid = max(valid_buy) if len(valid_buy) > 0 else None
        if valid_ask and valid_buy:
            mmmid_price = (mm_ask + mm_bid) / 2
          
        else:
            if traderObject.get('ink_last_price', None) == None:
              mmmid_price = (best_ask + best_bid) / 2
            else:
              mmmid_price = traderObject['ink_last_price']
    
        if traderObject.get('ink_last_price', None) is None:
            fair = mmmid_price
        else:
            ### Alpha
            last_price = traderObject["ink_last_price"]
            last_returns = (mmmid_price - last_price) / last_price
            pred_returns = (last_returns * self.params[Product.SQUID_INK]["reversion_beta"])        
            fair = mmmid_price + (mmmid_price * pred_returns)
        traderObject["ink_last_price"] = mmmid_price
        return fair
     return None

  def take_orders(self, product:str, order_depth: OrderDepth,
                  fair_value: float, take_width: float,
                  position: int, prevent_adverse: bool = False,
                  adverse_volume: int = 0):
    orders: List[Order] = []
    buy_order_volume, sell_order_volume = 0, 0
    buy_order_volume, sell_order_volume = self.take_best_orders(
        product, fair_value, take_width, orders, order_depth,
        position, buy_order_volume, sell_order_volume, prevent_adverse,
        adverse_volume
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
                  ):

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
    if best_ask_above_fair != None:
        if abs(best_ask_above_fair - fair_value) <= join_edge:
            ask = best_ask_above_fair  # join
        else:
            ask = best_ask_above_fair - 1  # penny

    bid = round(fair_value - default_edge)
    if best_bid_below_fair != None:
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


  def run(self, state: TradingState):
    traderObject = {}
    if state.traderData != None and state.traderData != "":
      traderObject = jsonpickle.decode(state.traderData)

    result = {}

    if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
      resin_position = (state.position[Product.RAINFOREST_RESIN]
                        if Product.RAINFOREST_RESIN in state.position
                        else 0)
      resin_take_orders, buy_order_volume, sell_order_volume = (
          self.take_orders(Product.RAINFOREST_RESIN,
                           state.order_depths[Product.RAINFOREST_RESIN],
                           self.params[Product.RAINFOREST_RESIN]['fair_value'],
                           self.params[Product.RAINFOREST_RESIN]['take_width'],
                           resin_position,)
      )
      resin_clear_orders, buy_order_volume, sell_order_volume = (
          self.clear_orders(Product.RAINFOREST_RESIN,
                            state.order_depths[Product.RAINFOREST_RESIN],
                            self.params[Product.RAINFOREST_RESIN]['fair_value'],
                            self.params[Product.RAINFOREST_RESIN]['clear_width'],
                            resin_position,
                            buy_order_volume,
                            sell_order_volume,
                            )
      )
      resin_make_orders, _, _ = self.make_orders(Product.RAINFOREST_RESIN,
                                                  state.order_depths[Product.RAINFOREST_RESIN],
                                                  self.params[Product.RAINFOREST_RESIN]['fair_value'],
                                                  resin_position,
                                                  buy_order_volume,
                                                  sell_order_volume,
                                                  self.params[Product.RAINFOREST_RESIN]['disregard_edge'],
                                                  self.params[Product.RAINFOREST_RESIN]['join_edge'],
                                                  self.params[Product.RAINFOREST_RESIN]['default_edge'],
                                                  True,
                                                  self.params[Product.RAINFOREST_RESIN]['soft_position_limit'],
                                                  )

      result[Product.RAINFOREST_RESIN] = (
          resin_take_orders + resin_clear_orders + resin_make_orders
      )

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
                           self.params[Product.KELP]['adverse_volume'],)
      )
      kelp_clear_orders, buy_order_volume, sell_order_volume = (
          self.clear_orders(Product.KELP,
                            state.order_depths[Product.KELP],
                            kelp_fair_value,
                            self.params[Product.KELP]['clear_width'],
                            kelp_position,
                            buy_order_volume,
                            sell_order_volume,)
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
                           )
      )
      ink_clear_orders, buy_order_volume, sell_order_volume = (
          self.clear_orders(Product.SQUID_INK,
                            state.order_depths[Product.SQUID_INK],
                            ink_fair_value,
                            self.params[Product.SQUID_INK]['clear_width'],
                            ink_position,
                            buy_order_volume,
                            sell_order_volume,)
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

    conversions = 1
    traderData = jsonpickle.encode(traderObject)

    return result, conversions, traderData