"""
backtest_v3_2022_2025.py
=========================================================
✅ 目的：不用 Streamlit UI，直接跑 signals_v3.py 在 2022-01-01 ~ 2025-12-31 的回測統計
- 支援：單檔 / 多檔 / 不給參數就跑 app.py 的 BUILTIN_CODES_200
- 輸出：每檔 summary + 全部 trades.csv
- 可調參數：preset（STRICT/BALANCED/ACTIVE/ENV/TUNED/WINRATE）

用法：
  # 單檔
  python backtest_v3_2022_2025.py --code 2330 --market 上市 --preset TUNED

  # 多檔（逗號分隔）
  python backtest_v3_2022_2025.py --codes 2330,2317,2303 --market 上市 --preset BALANCED

  # ✅ 不給 --code/--codes，直接吃 app.py 內 BUILTIN_CODES_200（200檔）
  python backtest_v3_2022_2025.py --market 上市 --preset TUNED

輸出檔案：
  - outputs/summary.csv
  - outputs/trades.csv
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd

from data import fetch_daily
from indicators import add_indicators
from signals_v3 import (
    SignalConfig,
    suggest_price_levels,
    score_row,
    classify_action,
    staged_take_profit_levels,
)

# -------------------------
# Backtest window
# -------------------------
START = dt.date(2022, 1, 1)
END = dt.date(2025, 12, 31)

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume", "MA20", "MA50", "RSI14", "BB_UP", "BB_LOW"]


def _to_date(x) -> dt.date:
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    return pd.to_datetime(x).date()


def _slice_window(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    d0 = pd.Timestamp(START)
    d1 = pd.Timestamp(END) + pd.Timedelta(days=1)
    return df[(df.index >= d0) & (df.index < d1)].copy()


def _extract_list_literal_after(text: str, anchor: str) -> str:
    """
    從整份文字中找到 anchor（例如 BUILTIN_CODES_200），
    然後抓出它後面第一個 list literal 的完整 "[...]"（用括號計數方式找到配對 ']'）
    """
    idx = text.find(anchor)
    if idx < 0:
        raise ValueError(f"找不到 {anchor}")

    lb = text.find("[", idx)
    if lb < 0:
        raise ValueError(f"{anchor} 後面找不到 '['")

    depth = 0
    for i in range(lb, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[lb : i + 1]

    raise ValueError(f"{anchor} 的 list '[]' 括號不完整（找不到配對的 ']'）")


def load_codes_from_app_py(app_path: str = "app.py") -> List[str]:
    """
    ✅ 不 import app.py（避免 Streamlit 亂跑）
    直接讀 app.py 文字，抓第一個 BUILTIN_CODES_200 = [ ... ] 的 list literal 做 literal_eval。
    這樣可以避開 app.py 後面如果有：
      BUILTIN_CODES_200 = BUILTIN_CODES_200[:200]
    這種不能 literal_eval 的語句。
    """
    p = Path(app_path)
    if not p.exists():
        raise ValueError(f"找不到 {app_path}（請確認 backtest_v3_2022_2025.py 與 app.py 在同一資料夾）")

    txt = p.read_text(encoding="utf-8", errors="ignore")
    if "BUILTIN_CODES_200" not in txt:
        raise ValueError(f"{app_path} 內沒有 BUILTIN_CODES_200")

    list_src = _extract_list_literal_after(txt, "BUILTIN_CODES_200")

    try:
        arr = ast.literal_eval(list_src)
    except Exception as e:
        raise ValueError(f"BUILTIN_CODES_200 的清單解析失敗：{e}")

    if not isinstance(arr, list) or not arr:
        raise ValueError(f"{app_path} 的 BUILTIN_CODES_200 不是有效的 list 或是空的")

    out: List[str] = []
    for x in arr:
        if x is None:
            continue
        s = str(x).strip()
        if s:
            out.append(s)

    if not out:
        raise ValueError(f"{app_path} 的 BUILTIN_CODES_200 解析後是空的")

    return out[:200]


def make_cfg(preset: str) -> SignalConfig:
    """
    preset:
      - STRICT / BALANCED / ACTIVE / WINRATE -> signals_v3 的模式（mode=...）
      - ENV -> 交給 signals_v3 的 SIGNAL_MODE / 檔案內 MODE
      - TUNED -> 勝率優先 + 稍微增加出手機會 的微調版（回測檔內覆寫參數）
    """
    p = (preset or "").strip().upper()

    if p == "ENV":
        return SignalConfig()  # mode 由 signals_v3 決定（ENV > 檔案 MODE）

    if p in ("STRICT", "BALANCED", "ACTIVE", "WINRATE"):
        return SignalConfig(mode=p)

    if p == "TUNED":
        cfg = SignalConfig(mode="BALANCED")

        # ✅ 微調：交易多一點、但不要變爛
        cfg.buy_zone_pos_max = 0.70       # BALANCED 0.60
        cfg.ma20_freefall_min = 0.975     # BALANCED 0.98
        cfg.require_rsi_turn = True       # 勝率核心保留
        cfg.rsi_overheat_max = 75.0       # BALANCED 72

        cfg.rr_buy = 1.20                 # BALANCED 1.30
        cfg.rr_strong_buy = 1.85          # BALANCED 2.00

        cfg.score_buy = 63                # BALANCED 65
        cfg.score_strong_buy = 78         # BALANCED 80
        return cfg

    raise ValueError(f"Unknown preset: {preset}")


def backtest_one(
    code: str,
    market: str,
    cfg: SignalConfig,
    max_hold_days: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    回測規則：
    - 訊號日 i 產生 action（BUY/STRONG_BUY）
    - 下一交易日開盤價進場（entry = Open[i+1]）
    - 之後最多持有 max_hold_days 天
    - 出場優先順序（每日檢查）：
        1) stop_loss（用當日 Low）
        2) STRONG_BUY：分段停利（TP1/TP2）
           BUY：到 sell_ref（BB_UP）就出
        3) 到 max_hold_days 仍未出 -> 用當日收盤出
    - 一次只持一筆（不重疊持倉）
    """
    df, resolved = fetch_daily(code, period_years=7, market=market)

    if df is None or df.empty:
        return pd.DataFrame(), {"code": code, "ticker": resolved, "error": "empty_data"}

    df = add_indicators(df, cfg).dropna().copy()
    if df.empty:
        return pd.DataFrame(), {"code": code, "ticker": resolved, "error": "no_indicators"}

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return pd.DataFrame(), {"code": code, "ticker": resolved, "error": f"missing_cols:{missing}"}

    bt_df = _slice_window(df)
    if bt_df is None or bt_df.empty or len(bt_df) < 120:
        return pd.DataFrame(), {"code": code, "ticker": resolved, "error": "too_few_rows_in_window"}

    trades: List[Dict[str, Any]] = []
    i = 0

    while i < len(bt_df) - 2:
        today = bt_df.iloc[i]
        close_i = float(today["Close"])

        rsi_prev = float(bt_df.iloc[i - 1]["RSI14"]) if i - 1 >= 0 else None
        rsi_prev2 = float(bt_df.iloc[i - 2]["RSI14"]) if i - 2 >= 0 else None

        price = suggest_price_levels(today)
        score, _detail = score_row(today, close_i, cfg, rsi_prev=rsi_prev, rsi_prev2=rsi_prev2)
        action, dbg = classify_action(
            score, price, close_i, cfg, today_row=today, rsi_prev=rsi_prev, rsi_prev2=rsi_prev2
        )

        if action not in ("BUY", "STRONG_BUY"):
            i += 1
            continue

        entry_idx = i + 1
        entry_row = bt_df.iloc[entry_idx]
        entry = float(entry_row["Open"])
        entry_date = _to_date(bt_df.index[entry_idx])

        stop = float(price["stop_loss"])
        sell_ref = float(price["sell_ref"])

        staged = (action == "STRONG_BUY")
        tp1 = tp2 = None
        got_tp1 = False

        if staged:
            tps = staged_take_profit_levels(price, close=close_i)
            if tps:
                tp1 = float(tps.get("tp1")) if tps.get("tp1") is not None else None
                tp2 = float(tps.get("tp2")) if tps.get("tp2") is not None else None

        exit_price = None
        exit_date = None
        outcome = "TIME"

        last_j = min(len(bt_df) - 1, entry_idx + max_hold_days)
        exit_j = entry_idx

        for j in range(entry_idx, last_j + 1):
            row = bt_df.iloc[j]
            high = float(row["High"])
            low = float(row["Low"])
            close_j = float(row["Close"])

            # 1) stop
            if low <= stop:
                exit_price = stop
                exit_date = _to_date(bt_df.index[j])
                outcome = "STOP"
                exit_j = j
                break

            # 2) take profit
            if staged:
                if (not got_tp1) and (tp1 is not None) and high >= tp1:
                    got_tp1 = True
                    outcome = "TP1_HIT"
                if (tp2 is not None) and high >= tp2:
                    exit_price = tp2
                    exit_date = _to_date(bt_df.index[j])
                    outcome = "TP2"
                    exit_j = j
                    break
            else:
                if high >= sell_ref:
                    exit_price = sell_ref
                    exit_date = _to_date(bt_df.index[j])
                    outcome = "TP"
                    exit_j = j
                    break

            # time exit candidate
            exit_price = close_j
            exit_date = _to_date(bt_df.index[j])
            exit_j = j

        if exit_price is None or exit_date is None:
            i += 1
            continue

        # returns
        if staged and got_tp1 and (tp1 is not None):
            ret = ((tp1 - entry) / entry) * 0.5 + ((exit_price - entry) / entry) * 0.5
        else:
            ret = (exit_price - entry) / entry

        trades.append({
            "code": str(code),
            "ticker": resolved,
            "mode": cfg.mode,
            "signal_date": _to_date(bt_df.index[i]),
            "entry_date": entry_date,
            "exit_date": exit_date,
            "action": action,
            "entry": round(entry, 4),
            "exit": round(float(exit_price), 4),
            "ret_pct": round(ret * 100, 2),
            "outcome": outcome,
            "win": 1 if ret > 0 else 0,
            "hold_days": int((pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days),
            # debug bits
            "score": int(score),
            "rr": round(float(dbg.get("rr", 0.0)), 2) if isinstance(dbg, dict) else None,
            "pos": dbg.get("pos") if isinstance(dbg, dict) else None,
            "ok_pos": dbg.get("ok_pos<=buy_zone_pos_max") if isinstance(dbg, dict) else None,
            "ok_freefall": dbg.get("ok_close>=MA20*freefall_min") if isinstance(dbg, dict) else None,
            "ok_rsi_turn": dbg.get("ok_rsi_turn") if isinstance(dbg, dict) else None,
            "ok_rsi_cap": dbg.get("ok_rsi<cap") if isinstance(dbg, dict) else None,
            "ok_ma20_slope_up": dbg.get("ok_ma20_slope_up") if isinstance(dbg, dict) else None,
        })

        # jump after exit to avoid overlapping positions
        i = exit_j + 1

    out = pd.DataFrame(trades)
    summary: Dict[str, Any] = {
        "code": str(code),
        "ticker": resolved,
        "mode": cfg.mode,
        "trades": int(len(out)),
    }

    if out.empty:
        summary.update({
            "winrate_pct": None,
            "avg_ret_pct": None,
            "median_ret_pct": None,
            "tp_rate_pct": None,
            "stop_rate_pct": None,
            "max_dd_pct": None,
            "expectancy_pct": None,
        })
        return out, summary

    winrate = float(out["win"].mean()) * 100.0
    avg_ret = float(out["ret_pct"].mean())
    med_ret = float(out["ret_pct"].median())
    tp_rate = float(out["outcome"].isin(["TP", "TP2"]).mean()) * 100.0
    stop_rate = float((out["outcome"] == "STOP").mean()) * 100.0

    rets = (out["ret_pct"].astype(float) / 100.0).fillna(0.0)
    equity = (1.0 + rets).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min()) * 100.0

    win_rate = float(out["win"].mean())
    avg_win = float(out.loc[out["ret_pct"] > 0, "ret_pct"].mean()) if (out["ret_pct"] > 0).any() else 0.0
    avg_loss = float(out.loc[out["ret_pct"] <= 0, "ret_pct"].mean()) if (out["ret_pct"] <= 0).any() else 0.0
    expectancy = avg_win * win_rate + avg_loss * (1.0 - win_rate)

    summary.update({
        "winrate_pct": round(winrate, 2),
        "avg_ret_pct": round(avg_ret, 2),
        "median_ret_pct": round(med_ret, 2),
        "tp_rate_pct": round(tp_rate, 2),
        "stop_rate_pct": round(stop_rate, 2),
        "max_dd_pct": round(max_dd, 2),
        "expectancy_pct": round(float(expectancy), 2),
    })
    return out, summary


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", type=str, default="", help="single code e.g. 2330")
    ap.add_argument("--codes", type=str, default="", help="multi codes comma-separated e.g. 2330,2317")
    ap.add_argument("--market", type=str, default="上市", choices=["上市", "上櫃"], help="market")
    ap.add_argument("--preset", type=str, default="TUNED", help="STRICT/BALANCED/ACTIVE/ENV/TUNED/WINRATE")
    ap.add_argument("--max_hold_days", type=int, default=20, help="max holding days")
    ap.add_argument("--outdir", type=str, default="outputs", help="output directory")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # resolve codes
    if args.code.strip():
        codes = [args.code.strip()]
    else:
        codes = [c.strip() for c in (args.codes or "").split(",") if c.strip()]

    if not codes:
        try:
            codes = load_codes_from_app_py("app.py")
            print(f"ℹ️ 未提供 --code/--codes，改用 app.py 的 BUILTIN_CODES_200（共 {len(codes)} 檔）")
        except Exception as e:
            raise SystemExit("請給 --code 或 --codes，或確認同資料夾有 app.py 且內含 BUILTIN_CODES_200") from e

    market_arg = "上市(.TW)" if args.market == "上市" else "上櫃(.TWO)"
    cfg = make_cfg(args.preset)

    all_trades = []
    summaries = []

    for idx, c in enumerate(codes, 1):
        trades, summary = backtest_one(c, market_arg, cfg, max_hold_days=int(args.max_hold_days))
        summaries.append(summary)

        if trades is not None and not trades.empty:
            all_trades.append(trades)

        if idx % 20 == 0 or idx == len(codes):
            print(f"Progress: {idx}/{len(codes)}")

    summary_df = pd.DataFrame(summaries)
    trades_df = pd.concat(all_trades, axis=0, ignore_index=True) if all_trades else pd.DataFrame()

    summary_path = outdir / "summary.csv"
    trades_path = outdir / "trades.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")

    print("✅ done")
    print(f"- {summary_path}")
    print(f"- {trades_path}")
    print(summary_df.head(10))


if __name__ == "__main__":
    main()
