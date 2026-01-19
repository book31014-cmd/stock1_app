# signals_v3_winrate.py  (勝率提升版 + 可解釋分數)
# ✅ ① 只買「買入區下半部」：position <= 0.4
# ✅ ② RSI 必須「轉強」：RSI 今日 > 昨日（可選：連2日上升）
# ✅ ③ 價格不能「自由落體」：close >= MA20 * 0.98
# ✅ ④ STRONG_BUY / BUY 分得更嚴（條件全過才給）
# - 分數可解釋（score + breakdown）
# - RR 風報比納入決策
# - 停損：結構型（min(BB_LOW, MA50*0.97)）
#
# ⚠️ 依賴 indicators.add_indicators 產生欄位：MA20, MA50, RSI14, BB_UP, BB_LOW
# （可選）若要啟用量能確認：請另外產生 Volume_MA20

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class SignalConfig:
    # 門檻
    score_strong_buy: int = 80
    score_buy: int = 65
    score_sell: int = 40

    rr_buy: float = 1.3
    rr_strong_buy: float = 2.0

    # 勝率提升濾網
    buy_zone_pos_max: float = 0.4       # ① 買入區下半部
    ma20_freefall_min: float = 0.98     # ③ close >= MA20*0.98
    require_rsi_turn: bool = True       # ② RSI 轉強

    # 勝率再提升（更挑環境、更少追高）
    require_trend_ma50: bool = True     # 只在 close > MA50 的環境考慮買
    rsi_overheat_max: float = 68.0      # RSI 過熱上限（避免追最後一棒）
    require_vol_confirm: bool = False   # 是否啟用量能確認（需有 Volume_MA20）
    vol_ma20_ratio_min: float = 0.9     # Volume >= Volume_MA20 * ratio

    # （可選）早期認錯：需要回測/APP 傳入 entry_price/holding_days 才會生效
    early_cut_days: int = 5
    early_cut_pct: float = 0.03         # 3% 早停損（勝率優先）


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def suggest_price_levels(today_row) -> Dict[str, float]:
    """
    buy_low: BB_LOW
    buy_high: min(MA20, BB_LOW*1.06)
    sell_ref: BB_UP
    stop_loss: min(BB_LOW, MA50*0.97)（結構型）
    """
    ma20 = float(today_row.get("MA20"))
    ma50 = float(today_row.get("MA50"))
    bb_up = float(today_row.get("BB_UP"))
    bb_low = float(today_row.get("BB_LOW"))

    buy_low = bb_low
    buy_high = min(ma20, bb_low * 1.06)
    sell_ref = bb_up

    ma50_sl = ma50 * 0.97
    stop_loss = min(bb_low, ma50_sl) if bb_low > 0 else ma50_sl

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
    """買入區位置：0=貼 buy_low，1=貼 buy_high"""
    width = buy_high - buy_low
    if width <= 0:
        return None
    return (close - buy_low) / width


def rsi_turning_up(rsi_today: float, rsi_prev: float | None, rsi_prev2: float | None) -> bool:
    """② RSI 轉強：今日>昨日；或更嚴：連2日上升（今日>昨日 且 昨日>前日）"""
    if rsi_prev is None:
        return False
    if rsi_today > rsi_prev:
        return True
    # 如果你想更嚴格，可以改成只回傳下面這條（目前保留不啟用）
    # return (rsi_prev2 is not None) and (rsi_today > rsi_prev) and (rsi_prev > rsi_prev2)
    return False


def pass_winrate_filters(
    today_row,
    close: float,
    price: Dict[str, Any],
    cfg: SignalConfig,
    rsi_prev: float | None = None,
    rsi_prev2: float | None = None,
) -> Tuple[bool, Dict[str, Any]]:
    """回傳 (是否通過, debug dict)"""
    ma20 = float(today_row.get("MA20"))
    rsi_today = float(today_row.get("RSI14"))
    buy_low = float(price["buy_low"])
    buy_high = float(price["buy_high"])

    pos = buy_zone_position(close, buy_low, buy_high)
    ok_pos = (pos is not None) and (pos <= cfg.buy_zone_pos_max)
    ok_freefall = close >= ma20 * cfg.ma20_freefall_min

    # 趨勢環境（勝率用）：只在 close > MA50 的環境考慮買
    ok_trend = True
    if cfg.require_trend_ma50:
        ma50 = float(today_row.get("MA50"))
        ok_trend = close > ma50

    # RSI 轉強 + 不過熱（避免追最後一棒）
    ok_rsi = True
    if cfg.require_rsi_turn:
        ok_rsi = rsi_turning_up(rsi_today, rsi_prev, rsi_prev2)
    ok_rsi_cap = rsi_today < float(cfg.rsi_overheat_max)

    # 量能確認（可選）：Volume >= Volume_MA20 * ratio
    ok_vol = True
    if cfg.require_vol_confirm:
        vol = today_row.get("Volume", None)
        vol_ma20 = today_row.get("Volume_MA20", None)
        if vol is None or vol_ma20 is None:
            ok_vol = False
        else:
            ok_vol = float(vol) >= float(vol_ma20) * float(cfg.vol_ma20_ratio_min)

    ok = ok_pos and ok_freefall and ok_trend and ok_rsi and ok_rsi_cap and ok_vol
    dbg = {
        "pos": None if pos is None else round(float(pos), 3),
        "ok_pos<=cfg.buy_zone_pos_max": bool(ok_pos),
        "ok_close>=MA20*cfg.ma20_freefall_min": bool(ok_freefall),
        "ok_trend_close>MA50": bool(ok_trend),
        "ok_rsi_turn": bool(ok_rsi),
        "ok_rsi<cap": bool(ok_rsi_cap),
        "ok_vol": bool(ok_vol),
    }
    return ok, dbg


def score_row(
    today_row,
    close: float,
    cfg: SignalConfig | None = None,
    rsi_prev: float | None = None,
    rsi_prev2: float | None = None,
) -> Tuple[int, Dict[str, int]]:
    """分數 + breakdown（0~100）"""
    cfg = cfg or SignalConfig()

    ma20 = float(today_row.get("MA20"))
    ma50 = float(today_row.get("MA50"))
    rsi = float(today_row.get("RSI14"))
    bb_up = float(today_row.get("BB_UP"))
    bb_low = float(today_row.get("BB_LOW"))

    # 越靠近/低於 MA20/MA50 越便宜（分數高）；但注意：勝率濾網會擋掉自由落體
    dist20 = (ma20 - close) / ma20
    dist50 = (ma50 - close) / ma50
    trend_raw = 0.6 * dist20 + 0.4 * dist50
    trend = int(round(_clamp((trend_raw + 0.02) / 0.12, 0, 1) * 35))

    # RSI 越低分數越高（但勝率濾網會要求轉強 + 不過熱）
    momentum = int(round(_clamp((70 - rsi) / 45, 0, 1) * 30))

    # 越靠近 BB_LOW 分數越高
    width = max(1e-9, bb_up - bb_low)
    pos_bb = (close - bb_low) / width
    volatility = int(round(_clamp((1 - pos_bb), 0, 1) * 25))

    price = suggest_price_levels(today_row)
    rr = risk_reward(price, close)
    if rr is None:
        risk = 0
    else:
        risk = int(round(_clamp((rr - 1.0) / 2.0, 0, 1) * 10))

    total = int(_clamp(trend + momentum + volatility + risk, 0, 100))
    detail = {"trend": trend, "momentum": momentum, "volatility": volatility, "risk": risk}
    return total, detail


def classify_action(
    score: int,
    price: Dict[str, Any],
    close: float,
    cfg: SignalConfig | None = None,
    today_row=None,
    rsi_prev: float | None = None,
    rsi_prev2: float | None = None,
    # 可選：勝率用狀態參數（不傳也能跑）
    entry_price: float | None = None,
    holding_days: int | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """回傳 (action, debug)"""
    cfg = cfg or SignalConfig()

    stop_loss = float(price["stop_loss"])
    buy_high = float(price["buy_high"])
    rr = risk_reward(price, close) or 0.0

    # 風控優先：跌破結構停損直接走
    if close <= stop_loss:
        action = "STRONG_SELL" if score <= (cfg.score_sell - 10) else "SELL"
        return action, {"risk_cut": True, "rr": round(float(rr), 2), "score": int(score)}

    # （可選）早期認錯：需要 backtest/app 傳入 entry_price/holding_days
    if entry_price is not None and holding_days is not None:
        if holding_days <= int(cfg.early_cut_days) and close < float(entry_price) * (1 - float(cfg.early_cut_pct)):
            return "SELL", {"early_cut": True, "holding_days": int(holding_days), "entry": float(entry_price)}

    # 相容模式：如果沒帶 today_row，就退回舊邏輯（不跑勝率濾網）
    if today_row is None:
        if score >= cfg.score_strong_buy and rr >= cfg.rr_strong_buy and close <= buy_high * 1.01:
            return "STRONG_BUY", {"compat_mode": True, "rr": round(float(rr), 2), "score": int(score)}
        if score >= cfg.score_buy and rr >= cfg.rr_buy and close <= buy_high * 1.01:
            return "BUY", {"compat_mode": True, "rr": round(float(rr), 2), "score": int(score)}
        if score < cfg.score_sell:
            return "SELL", {"compat_mode": True, "rr": round(float(rr), 2), "score": int(score)}
        return "HOLD", {"compat_mode": True, "rr": round(float(rr), 2), "score": int(score)}

    ok_filters, dbg = pass_winrate_filters(
        today_row, close, price, cfg, rsi_prev=rsi_prev, rsi_prev2=rsi_prev2
    )
    dbg["rr"] = round(float(rr), 2)
    dbg["score"] = int(score)

    # 不通過勝率濾網：偏 HOLD（少做爛交易 = 勝率↑）
    if not ok_filters:
        if score < cfg.score_sell:
            return "SELL", dbg
        return "HOLD", dbg

    # 通過濾網才給 BUY/STRONG_BUY
    if score >= cfg.score_strong_buy and rr >= cfg.rr_strong_buy and close <= buy_high * 1.01:
        return "STRONG_BUY", dbg
    if score >= cfg.score_buy and rr >= cfg.rr_buy and close <= buy_high * 1.01:
        return "BUY", dbg
    if score < cfg.score_sell:
        return "SELL", dbg
    return "HOLD", dbg


# ========= 分段停利（只建議 STRONG_BUY 用） =========
def staged_take_profit_levels(
    price: Dict[str, Any],
    close: float,
    rr1: float = 1.5,
    rr2: float = 2.5,
) -> Dict[str, float] | None:
    """
    以 stop_loss 當風險距離，做「RR 分段停利」價格：
      TP1 = close + rr1 * (close - stop_loss)
      TP2 = close + rr2 * (close - stop_loss)

    並且不超過 sell_ref（布林上軌）避免給出不合理過高目標。
    """
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

        return {
            "tp1": float(tp1),
            "tp2": float(tp2),
            "rr1": float(rr1),
            "rr2": float(rr2),
        }
    except Exception:
        return None
