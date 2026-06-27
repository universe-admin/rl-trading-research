"""Data acquisition + feature engineering for NSE DRL trading.

Robust to Yahoo 429s: caches each ticker to CSV on first success and
reuses the cache afterwards, with exponential backoff on download.
"""
import os
import time
import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(CACHE_DIR, exist_ok=True)


def fetch(ticker: str, period: str = "10y", interval: str = "1d",
          max_retries: int = 6) -> pd.DataFrame:
    """Download OHLCV with caching + backoff. Returns DataFrame indexed by date."""
    cache = os.path.join(CACHE_DIR, f"{ticker}_{interval}.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if len(df) > 200:
            return df

    import yfinance as yf
    last_err = None
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False, threads=False)
            if df is not None and len(df) > 200:
                # flatten possible MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                df.to_csv(cache)
                return df
            last_err = f"empty/short frame (len={0 if df is None else len(df)})"
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)
        sleep = min(60, 5 * (2 ** attempt))
        print(f"  [{ticker}] attempt {attempt+1} failed ({last_err}); sleeping {sleep}s")
        time.sleep(sleep)
    raise RuntimeError(f"Failed to fetch {ticker}: {last_err}")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append technical-indicator feature columns used as the RL observation."""
    out = df.copy()
    close = out["Close"]

    out["ret1"] = close.pct_change()
    out["ret5"] = close.pct_change(5)

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi14"] = (100 - 100 / (1 + rs)) / 100.0  # scaled 0-1

    # MACD histogram (normalized by price)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = (macd - signal) / close

    # Bollinger %b (20, 2)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    out["bb_pctb"] = (close - lower) / (upper - lower).replace(0, np.nan)

    # volatility + trend ratios
    out["vol20"] = out["ret1"].rolling(20).std()
    out["px_sma20"] = close / sma20 - 1
    out["px_sma50"] = close / close.rolling(50).mean() - 1

    out = out.dropna().reset_index()
    out = out.rename(columns={out.columns[0]: "Date"})  # daily->Date, intraday->Datetime
    return out


FEATURE_COLS = ["ret1", "ret5", "rsi14", "macd_hist", "bb_pctb",
                "vol20", "px_sma20", "px_sma50"]


def chrono_split(df: pd.DataFrame, train=0.70, val=0.15):
    """Chronological train/val/test split (no shuffling -> no lookahead)."""
    n = len(df)
    i_tr = int(n * train)
    i_va = int(n * (train + val))
    return df.iloc[:i_tr], df.iloc[i_tr:i_va], df.iloc[i_va:]


def normalizer(train_df: pd.DataFrame):
    """Fit z-score stats on TRAIN ONLY; return (mean, std) for FEATURE_COLS."""
    mu = train_df[FEATURE_COLS].mean().values
    sd = train_df[FEATURE_COLS].std().replace(0, 1).values
    return mu.astype(np.float32), sd.astype(np.float32)
