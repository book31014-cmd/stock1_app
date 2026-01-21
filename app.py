import datetime as dt
import re
from typing import Optional, Tuple, Dict

import pandas as pd
import streamlit as st
import os


def get_finmind_token(ui_value: str = "") -> str:
    """
    Token 讀取優先順序：
    1) st.secrets["FINMIND_TOKEN"]
    2) UI 手動輸入（備用）
    3) 空字串（fail-open）
    """
    try:
        secret_token = st.secrets.get("FINMIND_TOKEN", "")
        if secret_token:
            return secret_token
    except Exception:
        pass
    return (ui_value or "").strip()

import yfinance as yf
import requests

from data import (
    fetch_daily,
    fetch_recent,
    fetch_latest_price,
    fetch_batch_recent,
    debug_check_has_today_bar,
)
from indicators import add_indicators
from signals_v3 import (
    SignalConfig,
    suggest_price_levels,
    score_row,
    classify_action,
    staged_take_profit_levels,
)

# =========================
# Streamlit 設定
# =========================
st.set_page_config(page_title="股票策略分析", layout="centered", initial_sidebar_state="collapsed")
st.title("📈 股票策略分析工具（單檔 + 200檔掃描）")

# =========================
# 小工具：欄位安全、快取、風險報酬比
# =========================
REQUIRED_INDICATOR_COLS = ["MA20", "MA50", "RSI14", "BB_UP", "BB_LOW"]
CLOSE_COL = "Close"  # 統一使用 Close（yfinance 慣例）

# =========================
# FinMind 即時法人（取代 Yahoo）
# =========================
FINMIND_ENDPOINT = "https://api.finmindtrade.com/api/v4/data"


def _ensure_required_cols(df: pd.DataFrame) -> None:
    """缺欄位就直接擋掉，避免 KeyError（例如 BB_UP 缺失）。"""
    missing = [c for c in REQUIRED_INDICATOR_COLS if c not in df.columns]
    if missing:
        st.warning(f"缺少指標欄位：{missing}（資料不足或指標計算失敗）")
        st.stop()


def _risk_reward(price: dict, close: float) -> float | None:
    """簡易風報比： (賣出目標 - 現價) / (現價 - 停損)。<=0 或除以0 會回傳 None。"""
    try:
        sell_ref = float(price["sell_ref"])
        stop_loss = float(price["stop_loss"])
        denom = (close - stop_loss)
        if denom <= 0:
            return None
        rr = (sell_ref - close) / denom
        if rr <= 0:
            return None
        return rr
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_batch_recent(tickers: list[str], days: int, market: str):
    return fetch_batch_recent(tickers, days=days, market=market)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_add_indicators(df: pd.DataFrame, _cfg: SignalConfig) -> pd.DataFrame:
    return add_indicators(df, _cfg)


def _get_close_from_today_row(today_row: pd.Series, fallback_close: float | None = None) -> float:
    """統一取 Close：若今天列沒有 Close 就用 fallback_close（例如 latest_price）。"""
    if CLOSE_COL in today_row.index and pd.notna(today_row[CLOSE_COL]):
        return float(today_row[CLOSE_COL])
    if fallback_close is not None:
        return float(fallback_close)
    for k in ("close", "Adj Close", "Adj_Close"):
        if k in today_row.index and pd.notna(today_row[k]):
            return float(today_row[k])
    raise KeyError(f"找不到收盤欄位：'{CLOSE_COL}'（也沒有 fallback_close）")


def _ticker_with_suffix(code: str, market: str) -> str:
    """
    將 2330 + 市場轉成 Yahoo / yfinance 常用的 ticker：
      - 上市：2330.TW
      - 上櫃：2330.TWO
    """
    code = code.strip()
    if code.endswith(".TW") or code.endswith(".TWO"):
        return code
    suffix = ".TW" if "上市" in market else ".TWO"
    return f"{code}{suffix}"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_finmind_institutional_latest(code: str, token: str) -> Dict:
    """
    抓 FinMind 最新一個交易日的三大法人買賣超（張）。
    回傳格式：
      {"date": "YYYY-MM-DD", "net": {"foreign": int, "it": int, "dealer": int, "total": int}}
    失敗時：{"error": "..."}
    """
    if not token or not str(token).strip():
        return {"error": "FinMind Token 未提供"}

    code = str(code).strip()
    if not code:
        return {"error": "股票代號為空"}

    # 抓最近 14 天，避免遇到假日抓不到最新交易日
    end = dt.date.today()
    start = end - dt.timedelta(days=14)

    params = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": code,  # 台股用純代號（2330、2357），不要加 .TW
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "token": str(token).strip(),
    }

    try:
        resp = requests.get(FINMIND_ENDPOINT, params=params, timeout=15)
        resp.raise_for_status()
        j = resp.json()
        if not j.get("status"):
            return {"error": j.get("msg", "FinMind 回傳失敗")}

        data = j.get("data", []) or []
        if not data:
            return {"error": "FinMind 沒有回傳資料（可能代號/日期範圍/額度限制）"}

        df = pd.DataFrame(data)
        if df.empty or ("date" not in df.columns) or ("name" not in df.columns):
            return {"error": "FinMind 資料格式不完整"}

        # buy/sell -> net
        for c in ("buy", "sell"):
            if c not in df.columns:
                df[c] = 0
        df["net"] = (
            pd.to_numeric(df["buy"], errors="coerce").fillna(0)
            - pd.to_numeric(df["sell"], errors="coerce").fillna(0)
        )

        # 只取最新交易日
        df["date"] = pd.to_datetime(df["date"]).dt.date
        latest_date = df["date"].max()
        d0 = df[df["date"] == latest_date].copy()

        def _map_name(n: str):
            n = str(n)
            if ("Foreign" in n) or ("外資" in n):
                return "foreign"
            if ("Investment_Trust" in n) or ("投信" in n):
                return "it"
            if ("Dealer" in n) or ("自營商" in n):
                return "dealer"
            return None

        d0["who"] = d0["name"].map(_map_name)
        d0 = d0[d0["who"].notna()].copy()
        if d0.empty:
            return {"error": "FinMind 找不到外資/投信/自營商資料"}

        piv = d0.pivot_table(index="date", columns="who", values="net", aggfunc="sum")
        foreign = int(piv["foreign"].iloc[0]) if "foreign" in piv.columns else 0
        it = int(piv["it"].iloc[0]) if "it" in piv.columns else 0
        dealer = int(piv["dealer"].iloc[0]) if "dealer" in piv.columns else 0
        total = foreign + it + dealer

        return {
            "date": str(latest_date),
            "net": {"foreign": foreign, "it": it, "dealer": dealer, "total": total},
        }
    except Exception as e:
        return {"error": f"FinMind 抓取失敗：{e}"}


def institutional_filter_pass(insti: Dict, th_foreign: int, th_it: int, th_total: int) -> Tuple[bool, Dict]:
    """
    法人濾網：
      - 外資買賣超 <= th_foreign  -> 擋
      - 投信買賣超 <= th_it       -> 擋
      - 三大合計 <= th_total      -> 擋
    th_* 通常是負數（例如 -500、-300、-800）
    """
    dbg = {"ok": True, "reason": None}
    if insti is None or insti.get("error"):
        # fail-open：抓不到就不擋（避免因網站變動整個不能用）
        dbg["ok"] = True
        dbg["reason"] = insti.get("error") if isinstance(insti, dict) else "insti=None"
        return True, dbg

    net = insti.get("net", {}) or {}
    f = net.get("foreign")
    it = net.get("it")
    tot = net.get("total")

    # 有些列抓不到就跳過那條規則
    if f is not None and f <= th_foreign:
        return False, {"ok": False, "reason": f"外資買賣超 {f} ≤ {th_foreign}"}
    if it is not None and it <= th_it:
        return False, {"ok": False, "reason": f"投信買賣超 {it} ≤ {th_it}"}
    if tot is not None and tot <= th_total:
        return False, {"ok": False, "reason": f"三大法人合計 {tot} ≤ {th_total}"}

    return True, {"ok": True, "reason": "法人濾網通過"}


# =========================
# 大盤濾網：加權指數 ^TWII Close >= MA20（修正版：避免 Series truth value）
# =========================
MARKET_INDEX_TICKER = "^TWII"


@st.cache_data(ttl=1800, show_spinner=False)
def cached_fetch_market_index(days: int = 120) -> pd.DataFrame:
    df = yf.download(
        MARKET_INDEX_TICKER,
        period=f"{days}d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    df = df.dropna().copy()
    if df.empty:
        return df

    # ✅ 防 yfinance 回 MultiIndex 欄位（版本差異）
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if "Close" not in df.columns:
        return pd.DataFrame()

    df["MA20"] = df["Close"].rolling(20, min_periods=20).mean()
    return df


def market_filter_pass() -> tuple[bool, dict]:
    dfm = cached_fetch_market_index(120)
    if dfm is None or dfm.empty or ("MA20" not in dfm.columns):
        return True, {"ok": True, "note": "大盤資料抓取失敗，先不擋單（fail-open）"}

    last = dfm.iloc[-1]

    def _to_float(x):
        if isinstance(x, (pd.Series, pd.DataFrame)):
            x = x.squeeze()
        if hasattr(x, "item"):
            try:
                x = x.item()
            except Exception:
                pass
        return float(x)

    try:
        close = _to_float(last["Close"])
        ma20_raw = last["MA20"]

        if isinstance(ma20_raw, (pd.Series, pd.DataFrame)):
            ma20_raw = ma20_raw.squeeze()

        # ✅ 不用 if pd.notna(series) 這種會爆的寫法
        if pd.isna(ma20_raw).any() if isinstance(ma20_raw, pd.Series) else pd.isna(ma20_raw):
            return True, {"ok": True, "note": "MA20 不足，先不擋單（fail-open）"}

        ma20 = _to_float(ma20_raw)
        ok = close >= ma20
        return ok, {
            "ok": ok,
            "index_close": round(close, 2),
            "index_ma20": round(ma20, 2),
            "ticker": MARKET_INDEX_TICKER,
        }
    except Exception as e:
        return True, {"ok": True, "note": f"大盤濾網計算失敗，先不擋單（fail-open）: {e}"}


# =========================
# 200 檔候選池
# =========================
BUILTIN_CODES_200 = [
    "4533","1216","1301","1303","1326","1402","1476","1504","1513",
    "1605","1707","1717","1722","1802","2002","2015","2027","2049","2105",
    "2201","2207","2301","2303","2308","2317","2324","2327","2330","2345",
    "2352","2353","2354","2357","2360","2376","2377","2382","2395","2408",
    "2412","2454","2474","2498","2603","2609","2610","2615","2801","2809",
    "2812","2823","2834","2845","2855","2867","2880","2881","2882","2883",
    "2884","2885","2886","2887","2888","2889","2890","2891","2892","3008",
    "3017","3034","3037","3045","3059","3081","3189","3231","3264","3293",
    "3406","3443","3450","3481","3532","3533","3653","3661","3711","3714",
    "4904","4915","4938","4958","4960","4961","4977","5269","5347","5388",
    "5471","5871","5880","6005","6015","6116","6176","6213","6239","6269",
    "6285","6409","6415","6446","6505","6515","6526","6531","6533","6558",
    "6592","6669","6691","6706","6719","6770","6781","6789","6805","8016",
    "8028","8046","8054","8069","8070","8081","8104","8150","8210","8249",
    "8299","8341","8454","8906","9910","9921","9933","9945","9958","1308",
    "1434","1440","1455","1477","1503","1506","1522","1536","1560","1582",
    "1590","1608","1616","1907","2204","2227","2231","2305","2313","2323",
    "2347","2356","2362","2368","2371","2379","2383","2385","2392","2409",
    "2413","2421","2449","2464","2478","2481","2488","2492","2495","2542",
    "2618","2707","2722","2884","2912","3042","3090","3211","3702","4763",
    "4906","5522","5608","5876","5904","6112","6196","6202","6230","6266",
    "6278","6282","6412","6456","6488","6510","6666","6670","8044","9914"
]
BUILTIN_CODES_200 = BUILTIN_CODES_200[:200]

# =========================
# 手機優先設定（主畫面也能調，不怕 sidebar 在手機被收起）
# =========================
with st.expander("⚙️ 設定（手機建議在這裡調）", expanded=True):
    market = st.selectbox("市場", ["上市", "上櫃"], index=0, key="market")
    years = st.slider("單檔回看年數（只影響統計/指標穩定）", 1, 10, 5, key="years")
    days_scan = st.slider("掃描回看天數（越大越慢）", 120, 400, 260, step=20, key="days_scan")
    top_n = st.slider("掃描顯示前 N 名", 10, 200, 50, step=10, key="top_n")

    use_market_filter = st.checkbox("啟用大盤濾網（^TWII Close ≥ MA20）", value=True, key="use_market_filter")

    st.markdown("### 🧾 法人濾網（FinMind｜即時 + 回測一致）")
    use_insti_filter = st.checkbox("啟用法人濾網（外資/投信/三大合計買賣超）", value=True, key="use_insti_filter")
    finmind_token_live = st.text_input("FinMind Token（即時/掃描用）", value="", type="password")


    c1, c2, c3 = st.columns(3)
    with c1:
        th_foreign = st.number_input("外資門檻（<= 就擋）", value=-500, step=50)
    with c2:
        th_it = st.number_input("投信門檻（<= 就擋）", value=-300, step=50)
    with c3:
        th_total = st.number_input("三大合計門檻（<= 就擋）", value=-800, step=50)


# ✅ finmind_token 優先讀取 st.secrets['FINMIND_TOKEN']，UI 輸入僅作備用（未提供也不擋單 / fail-open）
finmind_token = get_finmind_token(finmind_token_live)

mf_ok, mf_info = market_filter_pass() if use_market_filter else (True, {"ok": True, "note": "未啟用"})

# Sidebar 只放摘要（避免手機看不到設定）
with st.sidebar:
    st.header("⚙️ 設定摘要")
    st.write(f"市場：**{market}**")
    st.write(f"回看年數：**{years}**")
    st.write(f"掃描回看天數：**{days_scan}**")
    st.write(f"顯示前 N 名：**{top_n}**")
    st.caption(f"📊 大盤濾網：{'✅ 通過' if mf_ok else '❌ 不通過'} | {mf_info}")
    st.caption(f"🧾 法人濾網：{'✅ 啟用' if use_insti_filter else '❌ 未啟用'} | 門檻 外資{th_foreign} / 投信{th_it} / 合計{th_total}")

cfg = SignalConfig()

tab1, tab2 = st.tabs(["🔍 單檔分析", "🧲 多檔自動掃描（200檔）"])

# =========================
# 單檔分析
# =========================
with tab1:
    code = st.text_input("股票代號（例如 2330）", value="2330").strip()
    if not code:
        st.stop()

    try:
        df_long, resolved_long = fetch_daily(code, period_years=years, market=market)
        df_recent, resolved_recent = fetch_recent(code, days=days_scan, market=market)
        latest_info, resolved_latest = fetch_latest_price(code, market=market)
    except Exception as e:
        st.error(str(e))
        st.stop()

    if not (resolved_long == resolved_recent == resolved_latest):
        st.error(f"資料來源不一致：long={resolved_long}｜recent={resolved_recent}｜latest={resolved_latest}")
        st.stop()

    latest_close = float(latest_info["close"])
    latest_date = latest_info["date"]

    df = pd.concat([df_long, df_recent], axis=0)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    st.caption(f"📌 ticker：{resolved_latest} ｜ 最新交易日：{latest_date} ｜ 最新收盤：{latest_close:.2f}")

    last_trade_date, is_today_bar = debug_check_has_today_bar(df_recent)
    st.caption(
        f"🧪 本機日期：{dt.date.today()} ｜ 最新交易日：{last_trade_date} ｜ 是否為今日日K：{'✅ 是' if is_today_bar else '❌ 否'}"
    )

    df2 = cached_add_indicators(df, cfg).dropna().copy()
    if len(df2) < 80:
        st.warning("資料太少，指標不穩，請拉長回看年數/天數。")
        st.stop()

    _ensure_required_cols(df2)
    today = df2.iloc[-1]
    rsi_prev = float(df2.iloc[-2]["RSI14"]) if len(df2) >= 2 else None
    rsi_prev2 = float(df2.iloc[-3]["RSI14"]) if len(df2) >= 3 else None

    # ✅ 勝率優先濾網：MA20 斜率（避免橫盤假強）
    ma20_slope = (
        float(today["MA20"]) - float(df2.iloc[-5]["MA20"]) if len(df2) >= 5 else 0.0
    )


    price = suggest_price_levels(today)
    price["close"] = latest_close

    score, score_detail = score_row(today, latest_close, cfg, rsi_prev=rsi_prev, rsi_prev2=rsi_prev2)
    action, filter_dbg = classify_action(
        score, price, latest_close, cfg, today_row=today, rsi_prev=rsi_prev, rsi_prev2=rsi_prev2
    )

    # 大盤濾網：不通過就禁止 BUY/STRONG_BUY（避免系統性風險）
    if use_market_filter and (not mf_ok) and action in ("BUY", "STRONG_BUY"):
        filter_dbg["market_filter_block"] = True
        action = "HOLD"

    # ✅ 勝率優先：MA20 斜率必須 > 0、RSI 必須 ≥ 50（否則不給 BUY）
    if action in ("BUY", "STRONG_BUY"):
        if ma20_slope <= 0:
            filter_dbg["ma20_slope_block"] = True
            filter_dbg["ma20_slope"] = round(float(ma20_slope), 4)
            action = "HOLD"
        if float(today["RSI14"]) < 50:
            filter_dbg["rsi50_block"] = True
            filter_dbg["rsi"] = round(float(today["RSI14"]), 2)
            action = "HOLD"


    # 法人濾網（Yahoo）：外資/投信/合計賣超過大 -> 擋 BUY
    insti = None
    insti_ok = True
    insti_dbg = {"ok": True, "reason": "未啟用"}
    if use_insti_filter:
        insti = fetch_finmind_institutional_latest(code, finmind_token)
        insti_ok, insti_dbg = institutional_filter_pass(insti, int(th_foreign), int(th_it), int(th_total))
        filter_dbg["institutional"] = insti_dbg
        if (not insti_ok) and action in ("BUY", "STRONG_BUY"):
            filter_dbg["institutional_block"] = True
            action = "HOLD"

    rr = _risk_reward(price, latest_close)

    st.divider()

    # =========================
    # 手機優先：改成直向堆疊（避免 3 欄在手機太擠）
    # =========================
    with st.container():
        st.subheader("🎯 分數")
        st.metric("Score (0-100)", int(score))
        st.write(f"建議動作：**{action}**")
        with st.expander("📊 分數拆解（可解釋）"):
            st.json(score_detail)
        with st.expander("🧪 勝率濾網檢查（通過才給 BUY）"):
            st.json(filter_dbg)

        if use_insti_filter:
            with st.expander("🧾 法人買賣超（FinMind）", expanded=False):
                if insti is None:
                    st.caption("法人資料：N/A")
                elif insti.get("error"):
                    st.warning(insti["error"])
                    st.caption("提示：請確認 FinMind Token / API 額度 / 股票代號")
                else:
                    net = insti.get("net", {})
                    cA, cB, cC, cD = st.columns(4)
                    cA.metric("外資買賣超(張)", net.get("foreign"))
                    cB.metric("投信買賣超(張)", net.get("it"))
                    cC.metric("自營商買賣超(張)", net.get("dealer"))
                    cD.metric("三大合計(張)", net.get("total"))
                    st.caption(f"法人濾網：{'✅ 通過' if insti_ok else '❌ 擋單'}｜{insti_dbg.get('reason')}")

        if rr is not None:
            st.metric("風險報酬比 (R/R)", f"{rr:.2f}")
        else:
            st.caption("風險報酬比：N/A（可能賣出目標<=現價或停損>=現價）")

    st.divider()

    with st.container():
        st.subheader("💰 建議價位")
        st.write(f"📌 現價：**{latest_close:.2f}**")
        st.write(f"🟢 買入區間：**{float(price['buy_low']):.2f} ～ {float(price['buy_high']):.2f}**")
        st.write(f"🔴 賣出參考：**{float(price['sell_ref']):.2f}**（布林上軌）")
        st.write(f"⛔ 停損參考：**{float(price['stop_loss']):.2f}**（MA50*0.97）")

        rr1_live = 1.2 if action == "BUY" else 1.5
        tp = staged_take_profit_levels(price, latest_close, rr1=rr1_live, rr2=2.5)
        if tp is not None:
            st.write(f"🟡 TP1（先出50%）：**{tp['tp1']:.2f}**（約 RR≥{tp['rr1']}）")
            st.write(f"🟠 TP2（出剩下）：**{tp['tp2']:.2f}**（約 RR≥{tp['rr2']} 或到 BB_UP）")

    st.divider()

    with st.container():
        st.subheader("📌 指標摘要")
        st.write(f"MA20：**{float(today['MA20']):.2f}**")
        st.write(f"MA50：**{float(today['MA50']):.2f}**")
        st.write(f"RSI14：**{float(today['RSI14']):.1f}**")
        st.write(f"BB_UP：**{float(today['BB_UP']):.2f}** ｜ BB_LOW：**{float(today['BB_LOW']):.2f}**")

    with st.expander("查看最後 10 筆資料"):
        st.dataframe(df2.tail(10), use_container_width=True, height=320)

# =========================
# 回測（3年）
# =========================
with st.expander("🧪 回測（過去 3 年勝率）", expanded=False):
    st.caption("規則：符合 BUY/STRONG_BUY（含勝率濾網）時進場；以『先碰到停損/目標』決定輸贏。")
    st.caption("⚠️ 提醒：法人濾網是『當下』FinMind 即時資料，回測時不使用（避免未來資料偏誤）。")
    hold_days = st.slider("最多持有天數（避免一直拖）", 5, 60, 20, 5)
    tp_buffer = st.slider("目標價觸發比例（例如 0.95=提早到 95% 就算到目標）", 0.80, 1.00, 0.95, 0.01)
    run_bt = st.button("▶️ 開始回測（3 年）")

    if run_bt:
        bt_df = df2.copy()
        if len(bt_df) > 800:
            bt_df = bt_df.iloc[-800:].copy()

        # =========================
        # ✅ 回測用：大盤濾網要用「每一天」的 ^TWII Close >= MA20
        # （避免用「今天的 mf_ok」去擋掉整個 3 年回測，造成失真）
        # =========================
        market_ok_map = None
        if use_market_filter:
            try:
                start_date = bt_df.index.min().date()
                end_date = bt_df.index.max().date()

                dfm = yf.download(
                    MARKET_INDEX_TICKER,
                    start=(start_date - dt.timedelta(days=60)).isoformat(),
                    end=(end_date + dt.timedelta(days=2)).isoformat(),
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    group_by="column",
                ).dropna().copy()

                # 防 yfinance 回 MultiIndex
                if isinstance(dfm.columns, pd.MultiIndex):
                    dfm.columns = [c[0] for c in dfm.columns]

                if "Close" in dfm.columns:
                    dfm["MA20"] = dfm["Close"].rolling(20, min_periods=20).mean()
                    dfm["d"] = dfm.index.date
                    market_ok_map = {}
                    for _, r in dfm.iterrows():
                        d = r.get("d", None)
                        ma20 = r.get("MA20", None)
                        close_m = r.get("Close", None)
                        if d is None:
                            continue
                        # MA20 不足 -> fail-open
                        if pd.isna(ma20):
                            market_ok_map[d] = True
                        else:
                            market_ok_map[d] = float(close_m) >= float(ma20)
            except Exception:
                # fail-open：抓不到大盤就不擋
                market_ok_map = None


        trades = []
        i = 3
        while i < len(bt_df) - 2:
            row = bt_df.iloc[i]
            close_i = float(row.get("Close"))
            rsi_prev = float(bt_df.iloc[i - 1]["RSI14"]) if i - 1 >= 0 else None
            rsi_prev2 = float(bt_df.iloc[i - 2]["RSI14"]) if i - 2 >= 0 else None

            # ✅ 勝率優先濾網（回測）：MA20 斜率 > 0、RSI ≥ 50
            ma20_slope_i = (
                float(row.get("MA20")) - float(bt_df.iloc[i - 5]["MA20"]) if i - 5 >= 0 else 0.0
            )


            price_i = suggest_price_levels(row)
            score_i, _detail = score_row(row, close_i, cfg, rsi_prev=rsi_prev, rsi_prev2=rsi_prev2)
            action_i, _fdbg = classify_action(score_i, price_i, close_i, cfg, today_row=row, rsi_prev=rsi_prev, rsi_prev2=rsi_prev2)

            # ✅ 勝率優先：不符合就直接跳過（不進場）
            if action_i in ("BUY", "STRONG_BUY"):
                if ma20_slope_i <= 0:
                    i += 1
                    continue
                if float(row.get("RSI14", 0)) < 50:
                    i += 1
                    continue

            # ✅ 大盤濾網（回測版）：用「訊號當天」的大盤狀態判斷
            if use_market_filter and action_i in ("BUY", "STRONG_BUY"):
                sig_date = bt_df.index[i]
                sig_d = sig_date.date() if hasattr(sig_date, "date") else None
                if market_ok_map is not None and sig_d is not None:
                    ok_mkt = bool(market_ok_map.get(sig_d, True))  # 找不到日期就 fail-open
                    if not ok_mkt:
                        i += 1
                        continue

            if action_i in ("BUY", "STRONG_BUY"):
                next_row = bt_df.iloc[i + 1]
                buy_low = float(price_i["buy_low"])
                buy_high = float(price_i["buy_high"])
                next_low = float(next_row["Low"]) if "Low" in bt_df.columns else float(next_row.get("Close"))
                # 若隔天最低價都高於買入區上緣 → 視為未成交，跳過（避免回測灌水）
                if next_low > buy_high:
                    i += 1
                    continue
                # 以買入區內「可成交的最差價」作為進場價（保守）
                entry = min(max(buy_low, next_low), buy_high)
                stop = float(price_i["stop_loss"])

                # 分段停利：STRONG_BUY 才用；BUY 照舊用 sell_ref
                staged = True  # ✅ 勝率優先：BUY 也採用 TP1（先出一半），避免贏單回吐
                rr1_bt = 1.5 if action_i == "STRONG_BUY" else 1.2
                tp_levels = staged_take_profit_levels(price_i, entry, rr1=rr1_bt, rr2=2.5) if staged else None

                tp1 = None
                tp2 = None
                if tp_levels is not None:
                    tp1 = float(tp_levels["tp1"]) * float(tp_buffer)
                    tp2 = float(tp_levels["tp2"]) * float(tp_buffer)
                else:
                    tp2 = float(price_i["sell_ref"]) * float(tp_buffer)

                got_tp1 = False
                exit_price = float(next_row.get("Close"))
                exit_idx = i + 1
                outcome = "TIMEOUT"

                end_j = min(len(bt_df) - 1, i + int(hold_days))
                for j in range(i + 1, end_j + 1):
                    rj = bt_df.iloc[j]
                    low = float(rj["Low"]) if "Low" in bt_df.columns else float(rj.get("Close"))
                    high = float(rj["High"]) if "High" in bt_df.columns else float(rj.get("Close"))
                    close_j = float(rj.get("Close"))

                    # 先停損
                    if low <= stop:
                        exit_price = stop
                        exit_idx = j
                        outcome = "STOP"
                        break

                    # TP1（STRONG_BUY 先出一半）
                    if staged and (not got_tp1) and (tp1 is not None) and high >= tp1:
                        got_tp1 = True
                        outcome = "TP1_HIT"

                    # TP2（全出）
                    if (tp2 is not None) and high >= tp2:
                        exit_price = tp2
                        exit_idx = j
                        outcome = "TP2" if staged else "TP"
                        break

                    exit_price = close_j
                    exit_idx = j

                # 報酬：若 hit TP1，假設 50% 在 tp1 出、50% 在 exit_price 出
                if staged and got_tp1 and (tp1 is not None):
                    ret = ((tp1 - entry) / entry) * 0.5 + ((exit_price - entry) / entry) * 0.5
                else:
                    ret = (exit_price - entry) / entry

                win = 1 if ret > 0 else 0
                trades.append({
                    "entry_date": bt_df.index[i + 1],
                    "exit_date": bt_df.index[exit_idx],
                    "action": action_i,
                    "entry": round(entry, 4),
                    "exit": round(exit_price, 4),
                    "ret_pct": round(ret * 100, 2),
                    "outcome": outcome,
                    "win": win,
                })

                i = exit_idx + 1
                continue

            i += 1

        if not trades:
            st.warning("回測期間沒有觸發任何 BUY/STRONG_BUY（可能條件太嚴或資料不足）")
        else:
            out_bt = pd.DataFrame(trades)
            winrate = out_bt["win"].mean() * 100
            avg_ret = out_bt["ret_pct"].mean()
            med_ret = out_bt["ret_pct"].median()
            tp_rate = (out_bt["outcome"].isin(["TP", "TP2"])).mean() * 100
            stop_rate = (out_bt["outcome"] == "STOP").mean() * 100

            # 進一步統計：最大回撤 & 期望值（Expectancy）
            # equity curve：假設每筆用同等資金、按報酬複利串接（用於估計 MDD）
            rets = (out_bt["ret_pct"].astype(float) / 100.0).fillna(0.0)
            equity = (1.0 + rets).cumprod()
            peak = equity.cummax()
            dd = equity / peak - 1.0
            max_dd = float(dd.min()) * 100.0  # 會是負值（例如 -12.3）

            win_rate = float(out_bt["win"].mean())
            avg_win = float(out_bt.loc[out_bt["ret_pct"] > 0, "ret_pct"].mean()) if (out_bt["ret_pct"] > 0).any() else 0.0
            avg_loss = float(out_bt.loc[out_bt["ret_pct"] <= 0, "ret_pct"].mean()) if (out_bt["ret_pct"] <= 0).any() else 0.0
            expectancy = avg_win * win_rate + avg_loss * (1.0 - win_rate)

            cA, cB, cC, cD = st.columns(4)
            cA.metric("交易筆數", int(len(out_bt)))
            cB.metric("勝率", f"{winrate:.1f}%")
            cC.metric("平均報酬", f"{avg_ret:.2f}%")
            cD.metric("中位數報酬", f"{med_ret:.2f}%")

            cE, cF, cG, cH = st.columns(4)
            cE.metric("最大回撤 (Max DD)", f"{max_dd:.2f}%")
            cF.metric("期望值 (Expectancy)", f"{expectancy:.2f}%")
            cG.metric("平均單筆獲利", f"{avg_win:.2f}%")
            cH.metric("平均單筆虧損", f"{avg_loss:.2f}%")


            st.caption(f"TP(含TP2) 比例：{tp_rate:.1f}% ｜ STOP 比例：{stop_rate:.1f}%（其餘為持有到期）")
            st.dataframe(out_bt.tail(30), use_container_width=True, height=480)

# =========================
# 多檔掃描（200檔）
# =========================
with tab2:
    st.write("✅ 會自動掃描內建 200 檔，計算分數 + 建議價位，並排序輸出。")
    st.caption("提示：第一次掃描會比較慢；之後可縮短「掃描回看天數」加快。")
    st.caption("⚠️ 如果你同時啟用『法人濾網』，掃描會更慢（只在 BUY/STRONG_BUY 時才抓 FinMind，已節流），但勝率通常更好。")

    cA, cB = st.columns([1, 1])
    with cA:
        min_rr = st.slider("最低風險報酬比（RR）", 0.5, 3.0, 1.5, 0.1)
    with cB:
        only_buy = st.checkbox("只看可買（BUY / STRONG_BUY）", value=True)

    if st.button("🚀 開始掃描 200 檔"):
        # 大盤濾網：不通過就停止（建議不交易）
        if use_market_filter and (not mf_ok):
            st.warning("❌ 大盤濾網未通過（^TWII Close < MA20）→ 今天建議不交易，停止掃描。")
            st.stop()

        tickers = BUILTIN_CODES_200
        prog = st.progress(0)
        status = st.empty()

        try:
            data_map, _resolved_suffix = cached_fetch_batch_recent(tickers, days=days_scan, market=market)
        except Exception as e:
            st.error(f"批次抓取失敗：{e}")
            st.stop()

        rows = []
        total = len(tickers)

        action_rank = {"STRONG_BUY": 0, "BUY": 1, "HOLD": 2, "SELL": 3, "STRONG_SELL": 4}

        for i, code in enumerate(tickers, start=1):
            status.write(f"掃描中：{code} ({i}/{total})")
            prog.progress(int(i / total * 100))

            df_one = data_map.get(code)
            if df_one is None or df_one.empty:
                continue

            df2 = cached_add_indicators(df_one, cfg).dropna()
            if len(df2) < 80:
                continue

            if any(c not in df2.columns for c in REQUIRED_INDICATOR_COLS) or (CLOSE_COL not in df2.columns):
                continue

            today = df2.iloc[-1]
            rsi_prev = float(df2.iloc[-2]["RSI14"]) if len(df2) >= 2 else None
            rsi_prev2 = float(df2.iloc[-3]["RSI14"]) if len(df2) >= 3 else None

            ma20_slope = (
                float(today["MA20"]) - float(df2.iloc[-5]["MA20"]) if len(df2) >= 5 else 0.0
            )


            last_close = _get_close_from_today_row(today)
            price = suggest_price_levels(today)
            price["close"] = last_close

            score, score_detail = score_row(today, last_close, cfg, rsi_prev=rsi_prev, rsi_prev2=rsi_prev2)
            action, filter_dbg = classify_action(
                score, price, last_close, cfg, today_row=today, rsi_prev=rsi_prev, rsi_prev2=rsi_prev2
            )

            # ✅ 勝率優先：MA20 斜率必須 > 0、RSI 必須 ≥ 50（否則不給 BUY）
            if action in ("BUY", "STRONG_BUY"):
                if ma20_slope <= 0 or float(today.get("RSI14", 0)) < 50:
                    action = "HOLD"
                    filter_dbg["winrate_block"] = True
                    filter_dbg["ma20_slope"] = round(float(ma20_slope), 4)
                    filter_dbg["rsi"] = round(float(today.get("RSI14", 0)), 2)

            # 法人濾網：只要抓得到而且不通過，就把 BUY/STRONG_BUY 改 HOLD
            insti_ok = True
            insti_reason = None
            if use_insti_filter and action in ("BUY", "STRONG_BUY"):
                insti = fetch_finmind_institutional_latest(code, finmind_token)
                insti_ok, insti_dbg = institutional_filter_pass(insti, int(th_foreign), int(th_it), int(th_total))
                insti_reason = insti_dbg.get("reason")
                if not insti_ok:
                    action = "HOLD"
                    filter_dbg["institutional_block"] = True
                    filter_dbg["institutional_reason"] = insti_reason

            rr = _risk_reward(price, last_close)

            rows.append({
                "code": code,
                "close": round(last_close, 2),
                "score": int(score),
                "action": action,
                "action_rank": action_rank.get(action, 99),
                "buy_low": round(float(price["buy_low"]), 2),
                "buy_high": round(float(price["buy_high"]), 2),
                "sell_ref": round(float(price["sell_ref"]), 2),
                "stop_loss": round(float(price["stop_loss"]), 2),
                "RR": round(float(rr), 2) if rr is not None else None,
                "why": (
                    f"T{score_detail.get('trend',0)}/M{score_detail.get('momentum',0)}/V{score_detail.get('volatility',0)}/R{score_detail.get('risk',0)}"
                    f" | pos={filter_dbg.get('pos',None)}"
                    f" | ok={filter_dbg.get('ok_pos<=0.4',False)}&{filter_dbg.get('ok_close>=MA20*0.98',False)}&{filter_dbg.get('ok_rsi_turn',False)}"
                    + (f" | insti={'OK' if insti_ok else 'BLOCK'}" if use_insti_filter else "")
                ),
                "MA20": round(float(today["MA20"]), 2),
                "MA50": round(float(today["MA50"]), 2),
                "RSI14": round(float(today["RSI14"]), 1),
                "last_trade_date": df2.index[-1].date() if hasattr(df2.index[-1], "date") else df2.index[-1],
            })

        prog.progress(100)
        status.write("✅ 掃描完成")

        if not rows:
            st.warning("沒有掃到可用資料（可能 Yahoo 當下不穩或回看天數太短）")
            st.stop()

        out = pd.DataFrame(rows)

        # 過濾：RR 門檻 + 只看可買
        out = out[out["RR"].fillna(0) >= float(min_rr)].copy()
        if only_buy:
            out = out[out["action"].isin(["BUY", "STRONG_BUY"])].copy()

        if out.empty:
            st.warning("目前條件下沒有符合的標的（可降低 RR 門檻或取消只看可買）")
            st.stop()

        # ✅ <3 檔提示：今天高勝率局太少 → 建議不交易
        if len(out) < 3:
            st.info("⚠️ 今天高勝率標的少於 3 檔：建議不交易/少交易，寧可等待高勝率的局。")

        out = out.sort_values(["action_rank", "RR", "score"], ascending=[True, False, False]).reset_index(drop=True)

        st.subheader("🏁 掃描結果")
        st.dataframe(out.head(top_n), use_container_width=True, height=620)

        st.download_button(
            "⬇️ 下載結果 CSV",
            data=out.to_csv(index=False).encode("utf-8-sig"),
            file_name="scan_results.csv",
            mime="text/csv",
        )

# --- patched finmind token reader ---

def get_finmind_token(ui_value: str = "") -> str:
    """
    Token priority:
    1) st.secrets["FINMIND_TOKEN"]
    2) Environment variable FINMIND_TOKEN
    3) UI input
    All whitespace/newlines will be stripped.
    """
    import re
    token = ""

    # 1) secrets
    try:
        token = (st.secrets.get("FINMIND_TOKEN", "") or "").strip()
        if token:
            return re.sub(r"\\s+", "", token)
    except Exception:
        pass

    # 2) env var
    token = (os.environ.get("FINMIND_TOKEN", "") or "").strip()
    if token:
        return re.sub(r"\\s+", "", token)

    # 3) UI
    token = (ui_value or "").strip()
    return re.sub(r"\\s+", "", token)
