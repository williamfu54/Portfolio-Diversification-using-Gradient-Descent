"""
This is a template bot for the CAPM Task.
"""

import copy
from typing import List
from fmclient import Agent, Session, Order, OrderSide, OrderType, Market

# Submission details
SUBMISSION = {"number": "998092", "name": "Boxiang Fu"}


class CAPMBot(Agent):

    def __init__(self, account, email, password, marketplace_id, risk_penalty=0.001, session_time=20):
        """
        Constructor for the Bot
        :param account: Account name
        :param email: Email id
        :param password: password
        :param marketplace_id: id of the marketplace
        :param risk_penalty: Penalty for risk
        :param session_time: Total trading time for one session
        """

        super().__init__(account, email, password, marketplace_id, name="CAPM Bot")

        # Information on the session and risk preferences of the bot
        self._payoffs = {}
        self._risk_penalty = risk_penalty
        self._session_time = session_time
        self._market_ids = {}
        self._price_tick = {}

        # Information on bot performance
        self._current_performance = None

        # Information on cash (in cents), asset holdings, and asset holdings including amount that can be shorted
        self._cash_holdings = None
        self._asset_holdings = {}
        self._short_asset_holdings = {}

        # Information on individual asset expected returns, variance, and covariance
        self._asset_expected_returns = None
        self._asset_variance_covariance_matrix = None

        # Information on current most performance enhancing valid market trade
        self._most_enhancing_trade = None

        # Information on maximum wait time before taking market making action or canceling orders
        self._max_wait_time = None
        self._current_wait_time = 0

        # Information on if a order sent by the bot currently exists on the order book
        self._order_on_market = False

    def initialised(self):

        # Extract payoff distribution, market id, and price tick for each security
        for market_id, market_info in self.markets.items():
            security = market_info.item
            description = market_info.description
            self._payoffs[security] = [int(a) for a in description.split(",")]
            self._market_ids[security] = market_id
            self._price_tick[security] = market_info.price_tick

    def get_potential_performance(self, orders):
        """
        Returns the portfolio performance if the given list of orders is executed.
        The performance as per the following formula:
        Performance = ExpectedPayoff - b * PayoffVariance, where b is the penalty for risk.
        :param orders: List of orders that is of class Order
        :return: Portfolio performance given orders are executed
        """

        # The expected payoff and payoff variance (rescaled to dollar terms)
        curr_asset_holdings = self._asset_holdings.copy()
        expected_payoff = self._cash_holdings / 100
        payoff_variance = 0

        # Updating hypothetical asset holdings and purchase/sell price for given list of orders
        for order in orders:

            if order.order_side == OrderSide.SELL:
                curr_asset_holdings[order.ref] -= 1
                expected_payoff += order.price / 100

            if order.order_side == OrderSide.BUY:
                curr_asset_holdings[order.ref] += 1
                expected_payoff -= order.price / 100

        # Portfolio expected return
        for security, holdings in curr_asset_holdings.items():
            expected_payoff += self._asset_expected_returns[security] * holdings

        # Portfolio payoff variance (payoff variance weights is on number of securities held, not percentage of total
        # portfolio)
        for security_1, holdings_1 in curr_asset_holdings.items():
            for security_2, holdings_2 in curr_asset_holdings.items():
                curr_matrix_key = (security_1, security_2)
                payoff_variance += holdings_1 * holdings_2 * self._asset_variance_covariance_matrix[curr_matrix_key]

        return expected_payoff - (self._risk_penalty * payoff_variance)

    def is_portfolio_optimal(self):
        """
        Returns true if the current holdings are optimal with respect to current best bid/ask (as per the performance
        formula), false otherwise. Mutates the variable self._most_enhancing_trade with the most performance enhancing
        valid trade if holdings are not optimal
        :return: Return True if current holdings are optimal, False otherwise
        """

        # Find best bid/ask price for each market
        best_bid_dict = {}
        best_ask_dict = {}

        for order_id, order in Order.current().items():
            security = order.market.item
            price = order.price

            if order.order_side == OrderSide.SELL:
                if best_ask_dict.get(security) is None:
                    best_ask_dict[security] = price
                else:
                    if price < best_ask_dict[security]:
                        best_ask_dict[security] = price

            if order.order_side == OrderSide.BUY:
                if best_bid_dict.get(security) is None:
                    best_bid_dict[security] = price
                else:
                    if price > best_bid_dict[security]:
                        best_bid_dict[security] = price

        # Finds all performance increasing market price trades (with performance as keys and order as values)
        optimal_trades = {}

        # Determines if buying 1 more security at the market price increases performance
        for security, price in best_ask_dict.items():

            buy_order = Order.create_new()
            buy_order.price = price
            buy_order.order_side = OrderSide.BUY
            buy_order.ref = security

            performance = self.get_potential_performance([buy_order])
            if performance > self._current_performance:
                optimal_trades[performance] = buy_order

        # Determines if selling 1 more security at the market price increases performance
        for security, price in best_bid_dict.items():

            sell_order = Order.create_new()
            sell_order.price = price
            sell_order.order_side = OrderSide.SELL
            sell_order.ref = security

            performance = self.get_potential_performance([sell_order])
            if performance > self._current_performance:
                optimal_trades[performance] = sell_order

        # Finds the best performance enhancing trade that satisfies order validity if there are any
        if len(optimal_trades) != 0:

            best_market_performance = 0
            best_market_trade = None

            for performance, trade in optimal_trades.items():
                if performance > best_market_performance:

                    # Populates the rest of the parameters for the order
                    trade.market = Market(self._market_ids[trade.ref])
                    trade.order_type = OrderType.LIMIT
                    trade.units = 1

                    # Checking for order validity
                    valid_order = self._check_order_validity(trade)

                    if valid_order:

                        best_market_performance = performance
                        best_market_trade = trade

            if best_market_trade is not None:
                self._most_enhancing_trade = best_market_trade
                return False

        self._most_enhancing_trade = None
        return True

    def order_accepted(self, order):

        # Find the order type of the order (i.e. BUY, SELL, or CANCEL order)
        if order.order_side == OrderSide.BUY:
            order_type = "BUY"
        else:
            order_type = "SELL"

        if order.order_type == OrderType.CANCEL:
            order_type = "CANCEL"

        self.inform(f"A {order_type} order is ACCEPTED for security {order.market.item} for 1 unit @ {order.price}")

    def order_rejected(self, info, order):

        # Find the order type of the order (i.e. BUY, SELL, or CANCEL order)
        if order.order_side == OrderSide.BUY:
            order_type = "BUY"
        else:
            order_type = "SELL"

        if order.order_type == OrderType.CANCEL:
            order_type = "CANCEL"

        self.inform(f"A {order_type} order is REJECTED for security {order.market.item} for 1 unit @ {order.price}")

    def received_orders(self, orders: List[Order]):

        # Check if orders sent by the bot is traded. Initiates new trades only when there are no orders sent by the bot
        # on the market
        if self._order_on_market is True:

            for curr_order in orders:

                # When order is immediately executed on the market
                if curr_order.mine:
                    if curr_order.traded_order:
                        self._order_on_market = False
                        self._current_wait_time = 0

                        # Informs the user of a traded order
                        consumed_order = curr_order
                        traded_price = consumed_order.price

                        if consumed_order.order_side == OrderSide.BUY:
                            traded_side = "BUY"
                        else:
                            traded_side = "SELL"

                        self.inform(f"A {traded_side} order is TRADED for the security {consumed_order.market.item}"
                                    f" for 1 unit @ {traded_price}")

                        return None

                # When order is not immediately executed on the market or if the first order received by the client
                # for an immediately executed order is the offsetting order
                if curr_order.traded_order:
                    if curr_order.traded_order.mine:
                        self._order_on_market = False
                        self._current_wait_time = 0

                        # Informs the user of a traded order
                        consumed_order = curr_order.traded_order
                        traded_price = consumed_order.price

                        if consumed_order.order_side == OrderSide.BUY:
                            traded_side = "BUY"
                        else:
                            traded_side = "SELL"

                        self.inform(f"A {traded_side} order is TRADED for the security {consumed_order.market.item}"
                                    f" for 1 unit @ {traded_price}")

                        return None

            # Cancel order if wait time for order exceeds the maximum wait time
            if self._current_wait_time > self._max_wait_time:
                for order_id, order in Order.current().items():
                    if order.mine:
                        self._cancel_order(order)

                self._order_on_market = False
                self._current_wait_time = 0

            return None

        # Trades performance maximizing valid market order if portfolio is not optimal
        optimal_achieved = self.is_portfolio_optimal()

        if not optimal_achieved:
            optimal_trade = self._most_enhancing_trade
            self.send_order(optimal_trade)
            self._order_on_market = True
            self._current_wait_time = 0

        # Trades performance maximizing market making order if portfolio is not optimal and current wait time has
        # exceeded max wait time
        else:
            if self._current_wait_time > self._max_wait_time:

                market_making_optimal = self._market_making_portfolio_optimal()

                if not market_making_optimal:
                    optimal_trade = self._most_enhancing_trade
                    self.send_order(optimal_trade)
                    self._order_on_market = True
                    self._current_wait_time = 0

    def received_session_info(self, session: Session):

        session_id = session.fm_id
        if session.is_open:
            self.inform(f"Marketplace is now open. The session's id is {session_id}")
        else:
            self.inform("Marketplace is now closed")

    def pre_start_tasks(self):

        # Initializes the maximum and tick times for wait times (in seconds). Max wait time is 0.5% of session time
        self._max_wait_time = (self._session_time * 60) // 200
        self.execute_periodically(self._increment_current_wait_time, 1)

        # And his name is ... JOHN CENA!!!
        self.inform("John Cena Bot is ONLINE! (╯°□°)╯")

        # Tries to send market making orders when calls to received_orders is infrequent (i.e. market is illiquid)
        self.execute_periodically(self._illiquid_market_market_making, 5)

    def received_holdings(self, holdings):

        # Find the cash and asset holdings of the bot
        self._cash_holdings = holdings.cash_available
        for market_info, asset_info in holdings.assets.items():
            self._asset_holdings[market_info.item] = asset_info.units
            self._short_asset_holdings[market_info.item] = asset_info.units_available

        # Initialize the expected return, variance, and covariance of securities
        if self._current_performance is None:
            self._initialize_asset_properties()

        # Update current portfolio performance
        self._current_performance = self.get_potential_performance(list())

        if not self._order_on_market:
            self.inform(f"Portfolio performance is now {self._current_performance:.3f}")

    def _illiquid_market_market_making(self):
        """
        Tries to initiate market making orders for an illiquid market
        :return: None. Sends a market making order request to the market
        """

        if self._current_wait_time > self._max_wait_time:
            if not self._order_on_market:

                market_making_optimal = self._market_making_portfolio_optimal()

                if not market_making_optimal:
                    optimal_trade = self._most_enhancing_trade
                    self.send_order(optimal_trade)
                    self._order_on_market = True
                    self._current_wait_time = 0

    def _initialize_asset_properties(self):
        """
        Calculates the expected return and the variance/covariance matrix of securities (rescaled to dollar terms)
        :return: None. Mutates asset expected return and the variance/covariance matrix variables.
        """

        expected_return = {}
        variance_covariance_matrix = {}

        # Expected return
        for security, payoff_dist in self._payoffs.items():
            curr_return = 0

            for state in payoff_dist:
                curr_return += state / 100
            expected_return[security] = (curr_return / len(payoff_dist))

        # Variance/Covariance Matrix
        for first_security in self._payoffs.keys():
            for second_security in self._payoffs.keys():

                curr_matrix = 0

                first_security_payoff = self._payoffs[first_security]
                second_security_payoff = self._payoffs[second_security]
                state_num = len(first_security_payoff)
                state_probability = 1 / state_num

                for state in range(state_num):
                    first_security_diff = (first_security_payoff[state] / 100) - expected_return[first_security]
                    second_security_diff = (second_security_payoff[state] / 100) - expected_return[second_security]

                    curr_matrix += state_probability * (first_security_diff * second_security_diff)
                variance_covariance_matrix[(first_security, second_security)] = curr_matrix

        # Updating the security attributes
        self._asset_expected_returns = expected_return
        self._asset_variance_covariance_matrix = variance_covariance_matrix

    def _check_order_validity(self, order):
        """
        Checks if the order is a valid order. Returns True if order is valid, False otherwise.
        :param order: Object of class Order. Contains all relevant information for sending an order to the server
        :return: True if order is valid, False otherwise
        """

        # Check if price exceeds price ceiling
        if order.price > Market(self._market_ids[order.ref]).max_price:
            return False

        # Check if price exceeds price floor
        if order.price < Market(self._market_ids[order.ref]).min_price:
            return False

        if order.order_side == OrderSide.BUY:

            # Check if there is enough cash
            if order.price > self._cash_holdings:
                return False

        if order.order_side == OrderSide.SELL:

            # Check if there is enough securities
            security = order.ref
            if order.units > self._short_asset_holdings[security]:
                return False

        return True

    def _increment_current_wait_time(self):
        """
        Increments the variable self._current_wait_time by 1 every second
        :return: None. Mutates self._current_wait_time instead
        """

        self._current_wait_time += 1

    def _cancel_order(self, order):
        """
        Cancels existing orders on the market that is sent by the bot
        :param order: Object of class Order that is to be canceled
        :return: None. Sends a request to the market to cancel an order
        """

        cancel_order = copy.copy(order)
        cancel_order.order_type = OrderType.CANCEL
        self.send_order(cancel_order)

    def _market_making_portfolio_optimal(self):
        """
        A market maker type bot that is called if there is insufficient liquidity in the order book. It calculates
        portfolio performance if orders are sent that is 1 price tick better than current market prices
        :return: Return True if portfolio is optimal even when the market making orders are sent, False otherwise
        """

        # Find best bid/ask price for each market
        best_bid_dict = {}
        best_ask_dict = {}

        for order_id, order in Order.current().items():
            security = order.market.item
            price = order.price

            if order.order_side == OrderSide.SELL:
                if best_ask_dict.get(security) is None:
                    best_ask_dict[security] = price
                else:
                    if price < best_ask_dict[security]:
                        best_ask_dict[security] = price

            if order.order_side == OrderSide.BUY:
                if best_bid_dict.get(security) is None:
                    best_bid_dict[security] = price
                else:
                    if price > best_bid_dict[security]:
                        best_bid_dict[security] = price

        # Populates the best bid/ask dictionary with min/max price for securities with no prices
        for market_id, market_info in self.markets.items():
            security = market_info.item
            max_price = market_info.max_price
            min_price = market_info.min_price

            if best_bid_dict.get(security) is None:
                best_bid_dict[security] = min_price

            if best_ask_dict.get(security) is None:
                best_ask_dict[security] = max_price

        # Finds all performance increasing trades that is 1 price tick better than current market prices
        optimal_trades = {}

        # Determines if buying 1 more security at 1 price tick above market bid increases performance
        for security, price in best_bid_dict.items():

            buy_order = Order.create_new()
            buy_order.price = price + self._price_tick[security]
            buy_order.order_side = OrderSide.BUY
            buy_order.ref = security

            performance = self.get_potential_performance([buy_order])
            if performance > self._current_performance:
                optimal_trades[performance] = buy_order

        # Determines if selling 1 more security at 1 price tick below market ask increases performance
        for security, price in best_ask_dict.items():

            sell_order = Order.create_new()
            sell_order.price = price - self._price_tick[security]
            sell_order.order_side = OrderSide.SELL
            sell_order.ref = security

            performance = self.get_potential_performance([sell_order])
            if performance > self._current_performance:
                optimal_trades[performance] = sell_order

        # Finds the best performance enhancing market maker trade that satisfies order validity if there are any
        if len(optimal_trades) != 0:

            best_market_maker_performance = 0
            best_market_maker_trade = None

            for performance, trade in optimal_trades.items():
                if performance > best_market_maker_performance:

                    # Populates the rest of the parameters for the order
                    trade.market = Market(self._market_ids[trade.ref])
                    trade.order_type = OrderType.LIMIT
                    trade.units = 1

                    # Checking for order validity
                    valid_order = self._check_order_validity(trade)

                    if valid_order:

                        best_market_maker_performance = performance
                        best_market_maker_trade = trade

            if best_market_maker_trade is not None:
                self._most_enhancing_trade = best_market_maker_trade
                return False

        self._most_enhancing_trade = None
        return True


if __name__ == "__main__":
    FM_ACCOUNT = 
    FM_EMAIL = 
    FM_PASSWORD = 
    MARKETPLACE_ID = 

    bot = CAPMBot(FM_ACCOUNT, FM_EMAIL, FM_PASSWORD, MARKETPLACE_ID)
    bot.run()
