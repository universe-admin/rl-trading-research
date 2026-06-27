"""Do company fundamentals affect NSE stock-price outcomes?

Two analyses:

1) Earnings-surprise event study (the most direct fundamental -> price link):
   for each quarterly earnings date, measure the price reaction (return from the
   close before the announcement to the close on/after it) and correlate it with
   the EPS surprise %. Pooled across all 5 stocks (~40 events).

2) Cross-sectional snapshot: current valuation / quality / growth metrics vs the
   stock's realized 2y return (n=5 -> illustrative, not statistically powered).

Honest limits: free fundamentals give only ~8 quarters/stock and 5 names, so
this is a small-sample study. Announcement timing (pre/post market) makes the
1-day reaction approximate.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
import yfinance as yf
import data as D

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

TICKERS = ["RELIANCE.NS", "ICICIBANK.NS", "LT.NS", "TCS.NS", "INFY.NS"]
SNAP_KEYS = ["trailingPE", "priceToBook", "returnOnEquity", "debtToEquity",
             "profitMargins", "revenueGrowth", "earningsGrowth", "dividendYield"]


def snapshot(ticker):
    info = yf.Ticker(ticker).info
    return {k: info.get(k) for k in SNAP_KEYS}


def reaction_events(ticker):
    """Return list of (surprise_pct, react_1d, react_3d) per earnings date."""
    prices = D.fetch(ticker)["Close"]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=16)
    except Exception:  # noqa: BLE001
        return []
    ed = ed.dropna(subset=["Surprise(%)"])
    events = []
    for ts, row in ed.iterrows():
        d = pd.Timestamp(ts).tz_localize(None).normalize()
        pos = prices.index.searchsorted(d)
        if pos < 1 or pos >= len(prices):
            continue
        pre = prices.iloc[pos - 1]
        post1 = prices.iloc[pos]
        r1 = post1 / pre - 1
        r3 = (prices.iloc[min(pos + 2, len(prices) - 1)] / pre - 1)
        events.append((float(row["Surprise(%)"]), float(r1), float(r3)))
    return events


def _fmt(v):
    if v is None:
        return "n/a"
    return str(round(v, 3)) if isinstance(v, (int, float)) else str(v)


def rp(a, b):
    if len(a) < 5:
        return (np.nan, np.nan, len(a))
    r_, p_ = pearsonr(a, b)
    return (round(float(r_), 3), round(float(p_), 4), len(a))


def main():
    out = {"snapshot": {}, "earnings_event_study": {}, "cross_section": {}}

    print("=== FUNDAMENTAL SNAPSHOT ===", flush=True)
    hdr = f"{'Ticker':<14}" + "".join(f"{k[:10]:<12}" for k in SNAP_KEYS)
    print(hdr, flush=True)
    for t in TICKERS:
        s = snapshot(t)
        out["snapshot"][t] = s
        line = f"{t:<14}" + "".join(f"{_fmt(s[k]):<12}" for k in SNAP_KEYS)
        print(line, flush=True)

    print("\n=== EARNINGS-SURPRISE EVENT STUDY ===", flush=True)
    pooled_surp, pooled_r1, pooled_r3 = [], [], []
    for t in TICKERS:
        ev = reaction_events(t)
        if not ev:
            print(f"  {t}: no events", flush=True)
            continue
        s, r1, r3 = zip(*ev)
        pooled_surp += list(s); pooled_r1 += list(r1); pooled_r3 += list(r3)
        pr = rp(list(s), list(r1))
        out["earnings_event_study"][t] = {"n": len(ev), "corr_surprise_vs_1d_react": pr[:2]}
        print(f"  {t}: n={len(ev):<2} corr(surprise, 1d reaction) r={pr[0]} p={pr[1]}", flush=True)

    pr1 = rp(pooled_surp, pooled_r1)
    pr3 = rp(pooled_surp, pooled_r3)
    out["earnings_event_study"]["POOLED"] = {
        "n": pr1[2],
        "corr_surprise_vs_1d_react": pr1[:2],
        "corr_surprise_vs_3d_react": pr3[:2],
    }
    print(f"\n  POOLED (n={pr1[2]}): surprise vs 1d reaction  r={pr1[0]} p={pr1[1]}", flush=True)
    print(f"  POOLED (n={pr3[2]}): surprise vs 3d reaction  r={pr3[0]} p={pr3[1]}", flush=True)

    print("\n=== CROSS-SECTION: valuation/growth vs 2y return ===", flush=True)
    rows = []
    for t in TICKERS:
        px = D.fetch(t)["Close"]
        ret2y = float(px.iloc[-1] / px.iloc[-min(len(px), 498)] - 1)
        s = out["snapshot"][t]
        rows.append((t, s.get("trailingPE"), s.get("earningsGrowth"),
                     s.get("returnOnEquity"), ret2y))
        out["cross_section"][t] = {"trailingPE": s.get("trailingPE"),
                                   "earningsGrowth": s.get("earningsGrowth"),
                                   "returnOnEquity": s.get("returnOnEquity"),
                                   "ret_2y": round(ret2y, 3)}
        print(f"  {t:<14} PE={s.get('trailingPE')}  earnGrow={s.get('earningsGrowth')}  "
              f"ROE={s.get('returnOnEquity')}  ret2y={ret2y:+.1%}", flush=True)
    df = pd.DataFrame(rows, columns=["t", "pe", "eg", "roe", "ret"]).dropna()
    for col, lbl in [("pe", "PE"), ("eg", "earningsGrowth"), ("roe", "ROE")]:
        pr = rp(df[col].tolist(), df["ret"].tolist())
        out["cross_section"][f"corr_{lbl}_vs_ret2y"] = pr[:2]
        print(f"  corr({lbl}, 2y return) r={pr[0]} p={pr[1]} (n={pr[2]})", flush=True)

    with open(os.path.join(RESULTS, "fundamentals.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nDONE -> results/fundamentals.json", flush=True)


if __name__ == "__main__":
    main()
