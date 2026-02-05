import ccxt
import logging

class OrderManager:
    def __init__(self, exchange_id, api_key, secret, testnet=True):
        self.logger = logging.getLogger(__name__)
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}  # Enforce Spot Only
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)
        
        # Double check it is strictly spot
        if self.exchange.options.get('defaultType') != 'spot':
            raise ValueError("CRITICAL: Exchange must be configured for SPOT only.")

    def fetch_ticker(self, symbol):
        return self.exchange.fetch_ticker(symbol)

    def fetch_balance(self):
        return self.exchange.fetch_balance()

    def create_limit_buy(self, symbol, amount, price):
        # Additional safety check for leverage could go here
        return self.exchange.create_order(symbol, 'limit', 'buy', amount, price)

    def create_limit_sell(self, symbol, amount, price):
        # Check inventory?
        return self.exchange.create_order(symbol, 'limit', 'sell', amount, price)

    def cancel_order(self, order_id, symbol):
        return self.exchange.cancel_order(order_id, symbol)

    def fetch_open_orders(self, symbol):
        return self.exchange.fetch_open_orders(symbol)

    def fetch_order(self, order_id, symbol):
        return self.exchange.fetch_order(order_id, symbol)
