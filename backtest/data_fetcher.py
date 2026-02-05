"""
Fetch and cache historical OHLCV data via CCXT.
"""
import os
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache')


def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)


def _cache_path(symbol: str, timeframe: str) -> str:
    """Generate cache file path for a symbol/timeframe combo."""
    safe_symbol = symbol.replace('/', '_')
    return os.path.join(CACHE_DIR, f"{safe_symbol}_{timeframe}.csv")


def load_cached(symbol: str, timeframe: str, max_age_hours: int = 24) -> pd.DataFrame | None:
    """
    Load cached OHLCV data if it exists and is fresh enough.
    
    Args:
        symbol: Trading pair (e.g., 'SOL/USDT')
        timeframe: Candle timeframe (e.g., '1h', '15m')
        max_age_hours: Maximum age of cache before considered stale
        
    Returns:
        DataFrame with OHLCV data or None if cache miss/stale
    """
    path = _cache_path(symbol, timeframe)
    if not os.path.exists(path):
        return None
    
    # Check age
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    if datetime.now() - mtime > timedelta(hours=max_age_hours):
        logger.info(f"Cache stale for {symbol} {timeframe}")
        return None
    
    try:
        df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
        logger.info(f"Loaded {len(df)} candles from cache for {symbol}")
        return df
    except Exception as e:
        logger.error(f"Failed to load cache: {e}")
        return None


def fetch_ohlcv(
    symbol: str,
    timeframe: str = '1h',
    days: int = 30,
    exchange_id: str = 'binance',
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from exchange.
    
    Args:
        symbol: Trading pair (e.g., 'SOL/USDT')
        timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
        days: Number of days of history to fetch
        exchange_id: CCXT exchange ID
        use_cache: Whether to use cached data if available
        
    Returns:
        DataFrame with columns: open, high, low, close, volume
        Index: timestamp (datetime)
    """
    # Try cache first
    if use_cache:
        cached = load_cached(symbol, timeframe)
        if cached is not None and not cached.empty:
            # Check if cache covers the requested duration
            duration = cached.index[-1] - cached.index[0]
            if duration >= timedelta(days=days - 0.1): # Allow small margin
                return cached
    
    logger.info(f"Fetching {days} days of {timeframe} data for {symbol}...")
    
    # Initialize exchange
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True})
    
    # Calculate since timestamp
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=days)).isoformat())
    
    # Fetch in batches (most exchanges limit to 1000 candles per request)
    all_candles = []
    limit = 1000
    
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not candles:
            break
        
        all_candles.extend(candles)
        
        # Move since to after last candle
        since = candles[-1][0] + 1
        
        # Check if we've fetched enough
        if len(candles) < limit:
            break
        
        logger.debug(f"Fetched {len(all_candles)} candles so far...")
    
    if not all_candles:
        raise ValueError(f"No data returned for {symbol}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Remove duplicates (can happen at batch boundaries)
    df = df[~df.index.duplicated(keep='first')]
    
    logger.info(f"Fetched {len(df)} candles for {symbol}")
    
    # Cache the data
    if use_cache:
        _ensure_cache_dir()
        df.to_csv(_cache_path(symbol, timeframe))
        logger.info(f"Cached data to {_cache_path(symbol, timeframe)}")
    
    return df
