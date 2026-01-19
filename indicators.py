import pandas as pd
import numpy as np


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Wilder smoothing (EMA-like)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_indicators(df: pd.DataFrame, cfg=None):
    df = df.copy()
    df = _flatten_columns(df)

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    std20 = df["Close"].rolling(20).std()
    df["BB_UP"] = df["MA20"] + 2 * std20
    df["BB_LOW"] = df["MA20"] - 2 * std20

    df["RSI14"] = _rsi(df["Close"], 14)

    return df
