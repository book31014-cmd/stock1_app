# data.py
# =========================================================
# Data layer (yfinance + optional FinMind fallback)
# - No Streamlit dependency
# - No circular import
# - Robust columns (always returns standard OHLCV columns)
# - Handles yfinance rate limit / empty downloads gracefully
#
# Optional FinMind fallback:
#   - set environment variable FINMIND_TOKEN (recommended)
#   - or set finmind_token parameter in fetch_latest_price(...)
#
# Exposed APIs (used by app.py):
#   - fetch_daily
#   - fetch_recent
#   - fetch_latest_price
#   - fetch_batch_recent
#   - debug_check_has_today_bar
# =========================================================

from __future__ import annotations

import os
import time
import datetime as dt
from typing import Dict, Tuple, Optional, Iterable

import pandas as pd
import yfinance as yf

# Optional fallback
import requests

FINMIND_ENDPOINT = "https://api.finmindtrade.com/api/v4/data"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    return df


def _normalize_market(market: str) -> str:
    if not market:
        return market
    m = str(market)
    if "上市" in m:
        return "上市"
    if "上櫃" in m or "上柜" in m:
        return "上櫃"
    return m


def _ticker_with_suffix(code: str, market: str) -> str:
    code = str(code).strip()
    if code.endswith(".TW") or code.endswith(".TWO"):
        return code

    m = _normalize_market(market)
    suffix = ".TW" if ("上市" in m or ".TW" in m) else ".TWO"
    return f"{code}{suffix}"


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Always return a DataFrame with standard OHLCV columns (even if empty)."""
    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(columns=cols)

    df = _flatten_columns(df).copy()

    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    df = df.sort_index()
    df = df.dropna(how="all")
    return df


def _safe_yf_download(ticker_or_tickers, *, period: str, interval: str = "1d", group_by: str = "column") -> pd.DataFrame:
    """
    yfinance can raise YFRateLimitError or return empty/None.
    We'll retry a couple times with small backoff.
    """
    last_err = None
    for attempt in range(3):
        try:
            df = yf.download(
                tickers=ticker_or_tickers,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                group_by=group_by,
                threads=False,  # reduce burst
            )
            if df is None:
                raise RuntimeError("yfinance returned None")
            return df
        except Exception as e:
            last_err = e
            # backoff: 0.8s, 1.6s, 3.2s
            time.sleep(0.8 * (2 ** attempt))
    raise RuntimeError(f"yfinance download failed: {last_err}")


def _finmind_latest_close(code: str, token: str) -> Optional[Tuple[str, float]]:
    """
    FinMind fallback: dataset TaiwanStockPrice
    Returns (date_str, close) or None.
    """
    if not token:
        return None

    end = dt.date.today()
    start = end - dt.timedelta(days=20)

    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": str(code).strip(),  # 2330 (no .TW)
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "token": token,
    }

    try:
        resp = requests.get(FINMIND_ENDPOINT, params=params, timeout=15)
        resp.raise_for_status()
        j = resp.json()
        if not j.get("status"):
            return None
        data = j.get("data", []) or []
        if not data:
            return None
        df = pd.DataFrame(data)
        if df.empty or "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        last = df.iloc[-1]
        close = last.get("close", None)
        if close is None:
            return None
        return last["date"].date().isoformat(), float(close)
    except Exception:
        return None


# ---------------------------------------------------------
# Public APIs
# ---------------------------------------------------------
def fetch_daily(code: str, period_years: int = 5, market: str = "上市(.TW)") -> Tuple[pd.DataFrame, str]:
    ticker = _ticker_with_suffix(code, market)
    years = max(1, int(period_years))
    try:
        raw = _safe_yf_download(ticker, period=f"{years}y", interval="1d", group_by="column")
        df = _ensure_ohlcv(raw)
    except Exception:
        df = _ensure_ohlcv(pd.DataFrame())
    return df, ticker


def fetch_recent(code: str, days: int = 260, market: str = "上市(.TW)") -> Tuple[pd.DataFrame, str]:
    ticker = _ticker_with_suffix(code, market)
    d = max(5, int(days))
    try:
        raw = _safe_yf_download(ticker, period=f"{d}d", interval="1d", group_by="column")
        df = _ensure_ohlcv(raw)
    except Exception:
        df = _ensure_ohlcv(pd.DataFrame())
    return df, ticker


def fetch_latest_price(
    code: str,
    market: str = "上市(.TW)",
    finmind_token: Optional[str] = None,
) -> Tuple[Dict[str, object], str]:
    """
    Latest close (prefer yfinance; fallback to FinMind if provided).
    finmind_token priority:
      1) finmind_token argument
      2) environment variable FINMIND_TOKEN
    """
    df, ticker = fetch_recent(code, days=30, market=market)

    # If yfinance rate-limited, df may be empty -> try FinMind
    token = (finmind_token or os.environ.get("FINMIND_TOKEN", "")).strip()

    if df is None or df.empty:
        fm = _finmind_latest_close(str(code).strip(), token)
        if fm is not None:
            d, close = fm
            return {"date": d, "close": close, "source": "FinMind"}, ticker
        raise ValueError(f"抓不到最新價格：{ticker}")

    last = df.iloc[-1]

    close = None
    if "Close" in df.columns and pd.notna(last.get("Close")):
        close = float(last["Close"])
    elif "Adj Close" in df.columns and pd.notna(last.get("Adj Close")):
        close = float(last["Adj Close"])

    if close is None:
        # fallback FinMind if available
        fm = _finmind_latest_close(str(code).strip(), token)
        if fm is not None:
            d, close2 = fm
            return {"date": d, "close": close2, "source": "FinMind"}, ticker
        raise ValueError(f"抓不到最新價格（Close/Adj Close 都是 NaN）：{ticker}")

    d = last.name
    if isinstance(d, pd.Timestamp):
        d = d.date().isoformat()
    elif isinstance(d, dt.datetime):
        d = d.date().isoformat()
    elif isinstance(d, dt.date):
        d = d.isoformat()
    else:
        d = str(d)

    return {"date": d, "close": close, "source": "yfinance"}, ticker


def debug_check_has_today_bar(df_recent: pd.DataFrame) -> Tuple[Optional[dt.date], bool]:
    if df_recent is None or df_recent.empty:
        return None, False

    last = df_recent.index[-1]
    if isinstance(last, pd.Timestamp):
        last_date = last.date()
    elif isinstance(last, dt.datetime):
        last_date = last.date()
    elif isinstance(last, dt.date):
        last_date = last
    else:
        try:
            last_date = pd.to_datetime(last).date()
        except Exception:
            return None, False

    today = dt.date.today()
    return last_date, (last_date == today)


def fetch_batch_recent(
    codes: Iterable[str],
    days: int = 260,
    market: str = "上市(.TW)",
) -> Tuple[Dict[str, pd.DataFrame], str]:
    codes = list(codes)
    if not codes:
        return {}, ""

    m = _normalize_market(market)
    resolved_suffix = ".TW" if ("上市" in m or ".TW" in m) else ".TWO"
    tickers = [_ticker_with_suffix(c, market) for c in codes]
    d = max(5, int(days))

    try:
        raw = _safe_yf_download(" ".join(tickers), period=f"{d}d", interval="1d", group_by="ticker")
    except Exception:
        raw = None

    data_map: Dict[str, pd.DataFrame] = {}

    if raw is None or getattr(raw, "empty", True):
        # fallback loop (still may be rate-limited, but we try)
        for c, t in zip(codes, tickers):
            df, _ = fetch_recent(c, days=d, market=market)
            if df is not None and not df.empty:
                data_map[str(c)] = df
        return data_map, resolved_suffix

    if isinstance(raw.columns, pd.MultiIndex):
        for c, t in zip(codes, tickers):
            if t in raw.columns.get_level_values(0):
                one = _ensure_ohlcv(raw[t].copy())
                if not one.empty:
                    data_map[str(c)] = one
        return data_map, resolved_suffix

    # single ticker format
    if len(codes) == 1:
        one = _ensure_ohlcv(raw)
        if not one.empty:
            data_map[str(codes[0])] = one
        return data_map, resolved_suffix

    # unexpected: fallback loop
    for c, t in zip(codes, tickers):
        df, _ = fetch_recent(c, days=d, market=market)
        if df is not None and not df.empty:
            data_map[str(c)] = df

    return data_map, resolved_suffix
