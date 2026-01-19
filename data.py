import yfinance as yf
import pandas as pd
import datetime as dt


def _resolve_ticker(code: str, market: str):
    suffix = ".TW" if market.startswith("上市") else ".TWO"
    return code if "." in code else f"{code}{suffix}"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 單檔有時會是 MultiIndex：('Close','') -> 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    return df


def fetch_daily(code, period_years=5, market="上市(.TW)"):
    ticker = _resolve_ticker(code, market)
    df = yf.download(
        ticker, period=f"{period_years}y", interval="1d",
        auto_adjust=False, progress=False
    )
    if df is None or df.empty:
        raise ValueError("fetch_daily 抓不到資料")
    df = df.dropna().copy()
    df = _flatten_columns(df)
    df.index = pd.to_datetime(df.index)
    return df, ticker


def fetch_recent(code, days=260, market="上市(.TW)"):
    ticker = _resolve_ticker(code, market)
    df = yf.download(
        ticker, period=f"{days}d", interval="1d",
        auto_adjust=False, progress=False
    )
    if df is None or df.empty:
        raise ValueError("fetch_recent 抓不到資料")
    df = df.dropna().copy()
    df = _flatten_columns(df)
    df.index = pd.to_datetime(df.index)
    return df, ticker


def fetch_latest_price(code, market="上市(.TW)"):
    ticker = _resolve_ticker(code, market)
    df = yf.download(
        ticker, period="7d", interval="1d",
        auto_adjust=False, progress=False
    )
    if df is None or df.empty:
        raise ValueError("fetch_latest_price 抓不到資料")
    df = df.dropna().copy()
    df = _flatten_columns(df)

    last = df.iloc[-1]
    return {"close": float(last["Close"]), "date": last.name.date()}, ticker


def fetch_batch_recent(codes, days=260, market="上市(.TW)"):
    """
    一次抓多檔，回傳 dict: code -> df(OHLCV)
    """
    suffix = ".TW" if market.startswith("上市") else ".TWO"
    tickers = [f"{c}{suffix}" for c in codes]

    big = yf.download(
        tickers,
        period=f"{days}d",
        interval="1d",
        group_by="column",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    if big is None or big.empty:
        raise ValueError("fetch_batch_recent 抓不到資料（Yahoo 可能暫時不穩）")

    # big 典型結構：columns MultiIndex (PriceField, Ticker)
    out = {}
    if isinstance(big.columns, pd.MultiIndex):
        # level 1 通常是 ticker
        for code, ticker in zip(codes, tickers):
            try:
                df_one = big.xs(ticker, axis=1, level=1).dropna().copy()
                df_one.index = pd.to_datetime(df_one.index)
                out[code] = df_one
            except Exception:
                continue
    else:
        # 只有一檔時可能會變單層（保險）
        df_one = big.dropna().copy()
        df_one.index = pd.to_datetime(df_one.index)
        out[codes[0]] = df_one

    return out, suffix


def debug_check_has_today_bar(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    last_trade_date = df.index[-1].date()
    today = dt.date.today()
    return last_trade_date, last_trade_date == today
