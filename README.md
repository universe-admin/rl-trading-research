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

## Run it

```bash
pip install -r requirements.txt
python run_experiment.py     # deep RL on daily data (PPO+A2C x seeds x stocks)
python run_intraday.py       # deep RL on 1h data + validation-based selection
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
| `results/` | Saved metrics (JSON) |

## Disclaimer

For **education and research only**. This is not financial advice. Past
performance does not guarantee future results. Nothing here is a profitable
trading system — and the whole point is that it honestly tells you so.
