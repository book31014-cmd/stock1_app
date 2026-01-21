# signals_v3.py
# =========================================================
# Strategy signals with MODE switch:
#   MODE = "STRICT" | "BALANCED" | "ACTIVE" | "WINRATE"
#
# How to switch:
#   1) Edit MODE below
#   2) Or set env var: SIGNAL_MODE=STRICT/BALANCED/ACTIVE/WINRATE
# =========================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import os


# ---------------------------------------------------------
# ✅ One-line switch (file-level default)
# ---------------------------------------------------------
MODE = "BALANCED"  # "STRICT" | "BALANCED" | "ACTIVE" | "WINRATE"


def _get_mode() -> str:
    """Mode priority: ENV > file MODE."""
    m = (os.environ.get("SIGNAL_MODE", "") or "").strip().upper()
    if m in ("STRICT", "BALANCED", "ACTIVE", "WINRATE"):
        return m
    m2 = (MODE or "").strip().upper()
    if m2 in ("STRICT", "BALANCED", "ACTIVE", "WINRATE"):
        return m2
    return "BALANCED"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =========================================================
# Config
# =========================================================
@dataclass
class SignalConfig:
    # --- default values (will be overwritten by __post_init__ based on mode) ---
    mode: str = "BALANCED"

    # thresholds
    score_strong_buy: int = 80
    score_buy: int = 65
    score_sell: int = 40

    rr_buy: float = 1.3
    rr_strong_buy: float = 2.0

    # win-rate filters
    buy_zone_pos_max: float = 0.6
    ma20_freefall_min: float = 0.98
    require_rsi_turn: bool = True

    require_trend_ma50: bool = False
    require_ma20_slope_up: bool = False  # ✅ NEW: MA20 must be rising vs yesterday
    rsi_overheat_max: float = 72.0

    # optional volume confirmation (needs Volume_MA20 from your indicator layer)
    require_vol_confirm: bool = False
    vol_ma20_ratio_min: float = 0.9

    # optional early cut (needs entry_price / holding_days)
    early_cut_days: int = 5
    early_cut_pct: float = 0.03

    def __post_init__(self):
        m = (self.mode or "").strip().upper()
        if m not in ("STRICT", "BALANCED", "ACTIVE", "WINRATE"):
            m = _get_mode()
        self.mode = m

        # -------------------------------
        # STRICT: fewer trades, higher quality
        # -------------------------------
        if m == "STRICT":
            self.score_buy = 68
            self.score_strong_buy = 83
            self.rr_buy = 1.4
            self.rr_strong_buy = 2.2

            self.buy_zone_pos_max = 0.4
            self.ma20_freefall_min = 0.985
            self.require_rsi_turn = True

            self.require_trend_ma50 = True
            self.require_ma20_slope_up = True
            self.rsi_overheat_max = 68.0

        # -------------------------------
        # WINRATE: ✅ higher win rate preset (trades less than BALANCED, but not dead)
        # -------------------------------
        elif m == "WINRATE":
            # 勝率優先：盤整假反彈擋掉、只吃順風盤
            self.score_buy = 66
            self.score_strong_buy = 82
            self.rr_buy = 1.25
            self.rr_strong_buy = 1.95

            self.buy_zone_pos_max = 0.55
            self.ma20_freefall_min = 0.985
            self.require_rsi_turn = True

            self.require_trend_ma50 = True
            self.require_ma20_slope_up = True
            self.rsi_overheat_max = 70.0

        # -------------------------------
        # BALANCED: default (good trade-off)
        # -------------------------------
        elif m == "BALANCED":
            self.score_buy = 65
            self.score_strong_buy = 80
            self.rr_buy = 1.3
            self.rr_strong_buy = 2.0

            self.buy_zone_pos_max = 0.6
            self.ma20_freefall_min = 0.98
            self.require_rsi_turn = True

            self.require_trend_ma50 = False
            self.require_ma20_slope_up = False
            self.rsi_overheat_max = 72.0

        # -------------------------------
        # ACTIVE: more trades (may lower win-rate)
        # -------------------------------
        else:  # ACTIVE
            self.score_buy = 60
            self.score_strong_buy = 75
            self.rr_buy = 1.15
            self.rr_strong_buy = 1.7

            self.buy_zone_pos_max = 0.8
            self.ma20_freefall_min = 0.965
            self.require_rsi_turn = False

            self.require_trend_ma50 = False
            self.require_ma20_slope_up = False
            self.rsi_overheat_max = 78.0


# =========================================================
# Price levels
# =========================================================
def suggest_price_levels(today_row) -> Dict[str, float]:
    """
    buy_low: BB_LOW
    buy_high: min(MA20, BB_LOW*1.06)
    sell_ref: BB_UP

    ✅ stop_loss 微調（勝率友善，少被洗掉一點）：
    - 原本：min(BB_LOW, MA50*0.97)
    - 現在：min(BB_LOW*0.995, MA50*0.975)
    """
    ma20 = float(today_row.get("MA20"))
    ma50 = float(today_row.get("MA50"))
    bb_up = float(today_row.get("BB_UP"))
    bb_low = float(today_row.get("BB_LOW"))

    buy_low = bb_low
    buy_high = min(ma20, bb_low * 1.06)
    sell_ref = bb_up

    stop_loss = min(bb_low * 0.995, ma50 * 0.975)

    return {
        "buy_low": float(buy_low),
        "buy_high": float(buy_high),
        "sell_ref": float(sell_ref),
        "stop_loss": float(stop_loss),
    }


def risk_reward(price: Dict[str, Any], close: float) -> float | None:
    """(sell_ref - close) / (close - stop_loss)"""
    try:
        sell_ref = float(price["sell_ref"])
        stop_loss = float(price["stop_loss"])
        denom = close - stop_loss
        if denom <= 0:
            return None
        rr = (sell_ref - close) / denom
        if rr <= 0:
            return None
        return rr
    except Exception:
        return None


def buy_zone_position(close: float, buy_low: float, buy_high: float) -> float | None:
    """0=near buy_low, 1=near buy_high"""
    w = buy_high - buy_low
    if w <= 0:
        return None
    return (close - buy_low) / w


def rsi_turning_up(rsi_today: float, rsi_prev: float | None, rsi_prev2: float | None = None) -> bool:
    """Default: today > prev (simple)."""
    if rsi_prev is None:
        return False
    return rsi_today > rsi_prev


def pass_winrate_filters(
    today_row,
    close: float,
    price: Dict[str, Any],
    cfg: SignalConfig,
    rsi_prev: float | None = None,
    rsi_prev2: float | None = None,
) -> Tuple[bool, Dict[str, Any]]:
    ma20 = float(today_row.get("MA20"))
    ma50 = float(today_row.get("MA50"))
    rsi_today = float(today_row.get("RSI14"))

    buy_low = float(price["buy_low"])
    buy_high = float(price["buy_high"])

    pos = buy_zone_position(close, buy_low, buy_high)
    ok_pos = (pos is not None) and (pos <= cfg.buy_zone_pos_max)
    ok_freefall = close >= ma20 * cfg.ma20_freefall_min

    ok_trend = True
    if cfg.require_trend_ma50:
        ok_trend = close > ma50

    ok_rsi = True
    if cfg.require_rsi_turn:
        ok_rsi = rsi_turning_up(rsi_today, rsi_prev, rsi_prev2)

    ok_rsi_cap = rsi_today < float(cfg.rsi_overheat_max)

    # ✅ NEW: MA20 slope filter (avoid sideways fake bounces)
    ok_ma20_slope = True
    if cfg.require_ma20_slope_up:
        ma20_prev = today_row.get("MA20_prev", None)
        if ma20_prev is None:
            ok_ma20_slope = False
        else:
            try:
                ok_ma20_slope = float(ma20) > float(ma20_prev)
            except Exception:
                ok_ma20_slope = False

    ok_vol = True
    if cfg.require_vol_confirm:
        vol = today_row.get("Volume", None)
        vol_ma20 = today_row.get("Volume_MA20", None)
        if vol is None or vol_ma20 is None:
            ok_vol = False
        else:
            ok_vol = float(vol) >= float(vol_ma20) * float(cfg.vol_ma20_ratio_min)

    ok = ok_pos and ok_freefall and ok_trend and ok_rsi and ok_rsi_cap and ok_ma20_slope and ok_vol

    dbg = {
        "mode": cfg.mode,
        "pos": None if pos is None else round(float(pos), 3),
        "ok_pos<=buy_zone_pos_max": bool(ok_pos),
        "ok_close>=MA20*freefall_min": bool(ok_freefall),
        "ok_trend_close>MA50": bool(ok_trend),
        "ok_rsi_turn": bool(ok_rsi),
        "ok_rsi<cap": bool(ok_rsi_cap),
        "ok_ma20_slope_up": bool(ok_ma20_slope),
        "ok_vol": bool(ok_vol),
    }
    return ok, dbg


# =========================================================
# Scoring
# =========================================================
def score_row(
    today_row,
    close: float,
    cfg: SignalConfig | None = None,
    rsi_prev: float | None = None,
    rsi_prev2: float | None = None,
) -> Tuple[int, Dict[str, int]]:
    """Score 0~100 (explainable breakdown)."""
    cfg = cfg or SignalConfig()

    ma20 = float(today_row.get("MA20"))
    ma50 = float(today_row.get("MA50"))
    rsi = float(today_row.get("RSI14"))
    bb_up = float(today_row.get("BB_UP"))
    bb_low = float(today_row.get("BB_LOW"))

    dist20 = (ma20 - close) / ma20
    dist50 = (ma50 - close) / ma50
    trend_raw = 0.6 * dist20 + 0.4 * dist50
    trend = int(round(_clamp((trend_raw + 0.02) / 0.12, 0, 1) * 35))

    momentum = int(round(_clamp((70 - rsi) / 45, 0, 1) * 30))

    width = max(1e-9, bb_up - bb_low)
    pos_bb = (close - bb_low) / width
    volatility = int(round(_clamp((1 - pos_bb), 0, 1) * 25))

    price = suggest_price_levels(today_row)
    rr = risk_reward(price, close)
    risk = int(round(_clamp((rr - 1.0) / 2.0, 0, 1) * 10)) if rr is not None else 0

    total = int(_clamp(trend + momentum + volatility + risk, 0, 100))
    detail = {"trend": trend, "momentum": momentum, "volatility": volatility, "risk": risk}
    return total, detail


# =========================================================
# Action classify
# =========================================================
def classify_action(
    score: int,
    price: Dict[str, Any],
    close: float,
    cfg: SignalConfig | None = None,
    today_row=None,
    rsi_prev: float | None = None,
    rsi_prev2: float | None = None,
    entry_price: float | None = None,
    holding_days: int | None = None,
) -> Tuple[str, Dict[str, Any]]:
    cfg = cfg or SignalConfig()
    rr = risk_reward(price, close) or 0.0

    stop_loss = float(price["stop_loss"])
    buy_high = float(price["buy_high"])

    # hard risk cut
    if close <= stop_loss:
        action = "STRONG_SELL" if score <= (cfg.score_sell - 10) else "SELL"
        return action, {"risk_cut": True, "rr": round(float(rr), 2), "score": int(score), "mode": cfg.mode}

    # optional early cut
    if entry_price is not None and holding_days is not None:
        if holding_days <= int(cfg.early_cut_days) and close < float(entry_price) * (1 - float(cfg.early_cut_pct)):
            return "SELL", {"early_cut": True, "holding_days": int(holding_days), "entry": float(entry_price), "mode": cfg.mode}

    # compatibility mode (no filters)
    if today_row is None:
        if score >= cfg.score_strong_buy and rr >= cfg.rr_strong_buy and close <= buy_high * 1.01:
            return "STRONG_BUY", {"compat_mode": True, "rr": round(float(rr), 2), "score": int(score), "mode": cfg.mode}
        if score >= cfg.score_buy and rr >= cfg.rr_buy and close <= buy_high * 1.01:
            return "BUY", {"compat_mode": True, "rr": round(float(rr), 2), "score": int(score), "mode": cfg.mode}
        if score < cfg.score_sell:
            return "SELL", {"compat_mode": True, "rr": round(float(rr), 2), "score": int(score), "mode": cfg.mode}
        return "HOLD", {"compat_mode": True, "rr": round(float(rr), 2), "score": int(score), "mode": cfg.mode}

    ok_filters, dbg = pass_winrate_filters(today_row, close, price, cfg, rsi_prev=rsi_prev, rsi_prev2=rsi_prev2)
    dbg["rr"] = round(float(rr), 2)
    dbg["score"] = int(score)

    if not ok_filters:
        if score < cfg.score_sell:
            return "SELL", dbg
        return "HOLD", dbg

    if score >= cfg.score_strong_buy and rr >= cfg.rr_strong_buy and close <= buy_high * 1.01:
        return "STRONG_BUY", dbg
    if score >= cfg.score_buy and rr >= cfg.rr_buy and close <= buy_high * 1.01:
        return "BUY", dbg
    if score < cfg.score_sell:
        return "SELL", dbg
    return "HOLD", dbg


# =========================================================
# Staged take profit (STRONG_BUY)
# =========================================================
def staged_take_profit_levels(
    price: Dict[str, Any],
    close: float,
    rr1: float = 1.5,
    rr2: float = 2.5,
) -> Dict[str, float] | None:
    try:
        stop_loss = float(price["stop_loss"])
        sell_ref = float(price["sell_ref"])
        risk = close - stop_loss
        if risk <= 0:
            return None

        tp1 = close + rr1 * risk
        tp2 = close + rr2 * risk
        tp1 = min(tp1, sell_ref)
        tp2 = min(tp2, sell_ref)

        return {"tp1": float(tp1), "tp2": float(tp2), "rr1": float(rr1), "rr2": float(rr2)}
    except Exception:
        return None
