# rl-trading-research

Rigorous backtesting of trading strategies on **NSE (India)** stocks — from
classic technical-analysis rules to **Deep Reinforcement Learning** (PPO / A2C)
— with an emphasis on **honest out-of-sample evaluation** rather than
cherry-picked backtests.

## TL;DR — the honest finding

> Across 6 rule-based strategies and 2 deep-RL algorithms, tested out-of-sample
> on 5 large-cap NSE stocks (daily **and** 1-hour data), **no method reliably
> beats buy-and-hold.** Apparent in-sample winners collapse out-of-sample, and
> validation performance does not predict test performance — the classic
> signature of overfitting to noise.

This repo is deliberately built to *catch* that, because catching it is the
point. It is a methodology you can trust, not a "money machine."

## Why it's rigorous

- **Chronological train/val/test split** — no shuffling, so no look-ahead.
- **Normalization fit on the training set only** — test statistics never leak.
- **Metrics reported on the held-out test set only** (data the model never saw).
- **Multiple random seeds** — to separate skill from luck (high seed variance = luck).
- **Walk-forward validation** for the rule-based strategies (detects overfitting).
- **Validation-based model selection** for the RL agents (select on val, report on test).
- **Transaction costs** applied on every position change.
- **Differential Sharpe Ratio** reward (Moody & Saffell, 1998) — risk-adjusted.
- Agents can go **long, flat, or short** — downside is tradeable.

## What's tested

- **Rule-based:** RSI, Bollinger mean-reversion, MACD, EMA cross, Supertrend, Donchian.
- **Deep RL:** PPO and A2C (Stable-Baselines3), MLP policy, DSR reward.
- **Universe:** RELIANCE, ICICIBANK, LT, TCS, INFY (`.NS`).

## Headline results (out-of-sample)

### Deep RL on daily data (mean of 3 seeds)

| Stock | PPO | A2C | Buy & Hold | Best vs B&H |
|---|---|---|---|---|
| RELIANCE | -12.7% | +3.3% | +5.1% | -1.8% |
| ICICIBANK | +2.7% | -11.6% | +10.6% | -8.0% |
| LT | +7.5% | +15.9% | +19.5% | -3.6% |
| TCS | -41.0% | -42.2% | -45.5% | +4.5% (still a loss) |
| INFY | -16.9% | -43.4% | -43.4% | +26.4% (still a loss) |

On every *rising* stock, buy-and-hold won. The only "outperformance" was on
*falling* stocks — by losing slightly less, which is not earning.

### Deep RL on 1-hour data (validation-selected model)

Finer resolution + proper model selection did **not** break the wall:
validation Sharpe failed to predict test returns (e.g. the best-validation
models on RELIANCE-A2C and TCS-PPO both *lost* money out-of-sample). See
`results/` and the run logs for full numbers.

### News / attention-trend correlation

True historical news-sentiment series for NSE names are not freely available,
so `news_trend.py` uses **Google Trends search interest** (geo: India) as a
proxy for the "news / attention trend" and correlates weekly changes in
attention against weekly returns — 259 weeks (~5 years) per stock.

Pearson r (change in attention vs return); **bold = statistically significant (p < 0.05)**:

| Stock | Same week | Predicts next week (lead) | Follows price (reverse) |
|---|---|---|---|
| RELIANCE | +0.05 | +0.11 | +0.01 |
| ICICIBANK | −0.05 | +0.09 | −0.03 |
| LT | −0.04 | +0.07 | +0.05 |
| TCS | +0.01 | +0.10 | −0.01 |
| INFY | **+0.13** | **−0.24** | +0.05 |

For 4 of 5 stocks, attention has **no statistically significant link** to
returns in any direction. Only INFY shows a significant *predictive*
correlation — and it is **negative** (attention spikes precede slightly weaker
returns, a weak contrarian effect explaining < 6% of variance), which with
multiple-comparison caution is not a tradeable edge. Net: the news/attention
trend does not forecast NSE returns — consistent with the rest of this repo.

### Fundamentals & earnings surprises

`fundamentals.py` tests whether company financials drive price outcomes, via
yfinance data:

**Earnings-surprise event study** — for each quarterly earnings date, the price
reaction (close-before → close-after) vs the EPS surprise %, pooled across all
5 stocks (n = 120 events):

| Relationship | Pearson r | p-value |
|---|---|---|
| EPS surprise → 1-day reaction | **+0.18** | 0.055 (marginal) |
| EPS surprise → 3-day reaction | +0.13 | 0.15 |

There is a weak, *directionally sensible* effect — bigger EPS beats produce
bigger up-moves — but it explains only ~3% of the reaction variance and is only
borderline significant. Every individual stock shows the same positive sign
(r = 0.12–0.32), none individually significant. This is the **clearest
fundamental→price link in the whole repo**, and it is still small.

**Cross-section (n = 5)** — current valuation/quality vs realized 2y return:
correlations are large in magnitude but statistically meaningless at n = 5, and
several have *counterintuitive* signs (e.g. ROE vs return r = −0.77: the
high-ROE IT names TCS and INFY fell hardest as the sector de-rated). Takeaway:
strong fundamentals did **not** translate into strong returns over this window —
sector/regime moves dominated.

## Run it

```bash
pip install -r requirements.txt
python run_experiment.py     # deep RL on daily data (PPO+A2C x seeds x stocks)
python run_intraday.py       # deep RL on 1h data + validation-based selection
python news_trend.py         # Google Trends attention vs returns correlation
python fundamentals.py       # earnings-surprise event study + valuation cross-section
```

Market data is downloaded from Yahoo Finance and cached under `data/`
(gitignored). First run fetches; later runs reuse the cache.

## Files

| File | Purpose |
|---|---|
| `data.py` | Data download (cached, backoff), technical-indicator features, splits, normalization |
| `trading_env.py` | Gymnasium trading env (long/flat/short, costs, Differential Sharpe reward) + evaluation |
| `run_experiment.py` | Daily DRL training + out-of-sample evaluation across seeds |
| `run_intraday.py` | 1h DRL training with validation-based model selection |
| `news_trend.py` | Google Trends attention vs returns correlation (lead/lag) |
| `fundamentals.py` | Earnings-surprise event study + valuation cross-section |
| `results/` | Saved metrics (JSON) |

## Disclaimer

For **education and research only**. This is not financial advice. Past
performance does not guarantee future results. Nothing here is a profitable
trading system — and the whole point is that it honestly tells you so.
