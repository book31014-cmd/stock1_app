# indicators.py
# =========================================================
# Indicator layer - robust version
# - Flattens MultiIndex columns (yfinance)
# - Ensures a usable "Close" column (fallback to "Adj Close" / "close")
# - Adds MA20_prev / MA50_prev for slope filters
# =========================================================

from __future__ import annotations

import pandas as pd
import numpy as np


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    return df


def _ensure_close(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a 'Close' column usable for indicators.
    Fallback order:
      1) Close
      2) Adj Close
      3) close (lowercase)
    """
    if "Close" in df.columns:
        return df

    if "Adj Close" in df.columns:
        df = df.copy()
        df["Close"] = df["Adj Close"]
        return df

    if "close" in df.columns:
        df = df.copy()
        df["Close"] = df["close"]
        return df

    raise KeyError(f"indicators.add_indicators: 找不到 Close 欄位。現有欄位：{list(df.columns)}")


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_indicators(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        df = df.to_frame()

    df = df.copy()
    df = _flatten_columns(df)

    df = _ensure_close(df)
    close = pd.to_numeric(df["Close"], errors="coerce")

    df["MA20"] = close.rolling(20, min_periods=20).mean()
    df["MA50"] = close.rolling(50, min_periods=50).mean()

    # ✅ for slope filters
    df["MA20_prev"] = df["MA20"].shift(1)
    df["MA50_prev"] = df["MA50"].shift(1)

    std20 = close.rolling(20, min_periods=20).std()
    df["BB_UP"] = df["MA20"] + 2 * std20
    df["BB_LOW"] = df["MA20"] - 2 * std20

    df["RSI14"] = _rsi(close, 14)

    return df
