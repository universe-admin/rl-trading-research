"""Correlate public attention (Google Trends search interest) with NSE returns.

True historical news-sentiment time series for NSE names are not freely
available, so we use Google Trends search interest as a proxy for the
"news / attention trend" around each company. We then test:

  * contemporaneous correlation: does attention move WITH returns / volatility?
  * predictive (lead):  does a change in attention this week predict NEXT week's return?
  * reverse (lag):      does this week's return drive NEXT week's attention?

The lead/lag split matters: if attention only *follows* price (reverse), it has
no forecasting value. That is the usual, honest finding.
"""
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
import data as D

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

KEYWORDS = {
    "RELIANCE.NS": "Reliance Industries",
    "ICICIBANK.NS": "ICICI Bank",
    "LT.NS": "Larsen Toubro",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
}


def get_trend(keyword, retries=4):
    from pytrends.request import TrendReq
    for i in range(retries):
        try:
            p = TrendReq(hl="en-US", tz=330)
            p.build_payload([keyword], timeframe="today 5-y", geo="IN")
            df = p.interest_over_time()
            if df is not None and len(df) > 20:
                s = df[keyword]
                s = s[~df.get("isPartial", pd.Series(False, index=df.index)).astype(bool)]
                return s
        except Exception as e:  # noqa: BLE001
            print(f"    trend retry {i+1} ({keyword}): {repr(e)[:80]}")
        time.sleep(8 * (i + 1))
    raise RuntimeError(f"Google Trends failed for {keyword}")


def weekly_returns(ticker):
    df = D.fetch(ticker)  # cached daily OHLCV
    close = df["Close"]
    wk = close.resample("W").last()
    return wk.pct_change()


def analyze(ticker, keyword):
    trend = get_trend(keyword)
    ret = weekly_returns(ticker)

    t = trend.copy(); t.index = t.index.to_period("W")
    r = ret.copy(); r.index = r.index.to_period("W")
    m = pd.DataFrame({"trend": t, "ret": r}).dropna()
    m["dtrend"] = m["trend"].diff()
    m["absret"] = m["ret"].abs()
    m = m.dropna()

    def rp(a, b):
        if len(a) < 10:
            return (np.nan, np.nan)
        r_, p_ = pearsonr(a, b)
        return (round(float(r_), 3), round(float(p_), 4))

    # contemporaneous
    same_ret = rp(m["dtrend"], m["ret"])
    same_vol = rp(m["dtrend"], m["absret"])
    # predictive: dtrend[t] vs ret[t+1]
    lead = rp(m["dtrend"][:-1], m["ret"].shift(-1).dropna()[: len(m) - 1])
    # reverse: ret[t] vs dtrend[t+1]
    rev = rp(m["ret"][:-1], m["dtrend"].shift(-1).dropna()[: len(m) - 1])

    return {
        "n_weeks": int(len(m)),
        "corr_dtrend_vs_return_same_week": same_ret,
        "corr_dtrend_vs_abs_return_same_week": same_vol,
        "corr_dtrend_predicts_next_return (lead)": lead,
        "corr_return_predicts_next_dtrend (reverse)": rev,
    }


def main():
    out = {}
    for ticker, kw in KEYWORDS.items():
        print(f"=== {ticker}  ('{kw}') ===", flush=True)
        try:
            res = analyze(ticker, kw)
            out[ticker] = res
            for k, v in res.items():
                print(f"   {k}: {v}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"   FAIL: {e}", flush=True)
        time.sleep(5)

    with open(os.path.join(RESULTS, "news_trend.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n==== ATTENTION vs RETURN SUMMARY (Pearson r, p-value) ====", flush=True)
    print(f"{'Ticker':<14}{'same-wk ret':<16}{'same-wk vol':<16}"
          f"{'lead(predict)':<16}{'reverse(follow)':<16}", flush=True)
    print("-" * 78, flush=True)
    for ticker, r in out.items():
        print(f"{ticker:<14}"
              f"{str(r['corr_dtrend_vs_return_same_week']):<16}"
              f"{str(r['corr_dtrend_vs_abs_return_same_week']):<16}"
              f"{str(r['corr_dtrend_predicts_next_return (lead)']):<16}"
              f"{str(r['corr_return_predicts_next_dtrend (reverse)']):<16}", flush=True)
    print("\nDONE -> results/news_trend.json", flush=True)


if __name__ == "__main__":
    main()
