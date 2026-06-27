"""Train PPO & A2C DRL agents on NSE stocks; report OUT-OF-SAMPLE profitability.

Rigour:
  * chronological train/val/test split (no shuffling, no lookahead)
  * normalization fit on TRAIN only
  * metrics reported on TEST only (data the agent never trained on)
  * multiple random seeds -> report mean +/- std (shows luck vs skill)
  * transaction costs on every position change
"""
import os
import sys
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import data as D
from trading_env import TradingEnv, evaluate

from stable_baselines3 import PPO, A2C

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

TICKERS = ["RELIANCE.NS", "ICICIBANK.NS", "LT.NS", "TCS.NS", "INFY.NS"]
ALGOS = {"PPO": PPO, "A2C": A2C}
SEEDS = [0, 1, 2]
TIMESTEPS = 40000
COST = 0.001


def train_one(AlgoCls, train_df, mu, sd, seed):
    env = TradingEnv(train_df, mu, sd, cost=COST)
    model = AlgoCls("MlpPolicy", env, seed=seed, verbose=0,
                    policy_kwargs=dict(net_arch=[64, 64]))
    model.learn(total_timesteps=TIMESTEPS, progress_bar=False)
    return model


def main():
    all_results = {}
    for ticker in TICKERS:
        print(f"\n=== {ticker} ===", flush=True)
        try:
            df = D.add_features(D.fetch(ticker))
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP (data error): {e}", flush=True)
            continue
        tr, va, te = D.chrono_split(df)
        mu, sd = D.normalizer(tr)
        print(f"  rows: train={len(tr)} val={len(va)} test={len(te)} "
              f"(test {te['Date'].iloc[0].date()} -> {te['Date'].iloc[-1].date()})",
              flush=True)

        ticker_res = {}
        for algo_name, AlgoCls in ALGOS.items():
            test_metrics = []
            for seed in SEEDS:
                model = train_one(AlgoCls, tr, mu, sd, seed)
                m = evaluate(model, te, mu, sd, cost=COST)
                test_metrics.append(m)
                print(f"  {algo_name} seed{seed}: "
                      f"ret={m['total_return_pct']:+.1f}% "
                      f"sharpe={m['sharpe']:.2f} "
                      f"vsB&H={m['vs_buy_hold_pct']:+.1f}% "
                      f"(L{m['pct_long']}/S{m['pct_short']}/F{m['pct_flat']})",
                      flush=True)
            # aggregate across seeds
            keys = ["total_return_pct", "sharpe", "max_drawdown_pct", "vs_buy_hold_pct"]
            agg = {k: {"mean": round(float(np.mean([x[k] for x in test_metrics])), 2),
                       "std": round(float(np.std([x[k] for x in test_metrics])), 2)}
                   for k in keys}
            agg["buy_hold_pct"] = test_metrics[0]["buy_hold_pct"]
            ticker_res[algo_name] = {"aggregate": agg, "per_seed": test_metrics}
        all_results[ticker] = ticker_res

    with open(os.path.join(RESULTS, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # summary table
    print("\n\n================ OUT-OF-SAMPLE SUMMARY (TEST SET) ================", flush=True)
    print(f"{'Ticker':<14}{'Algo':<6}{'Ret%(mean±std)':<20}{'Sharpe':<9}"
          f"{'Buy&Hold%':<11}{'vs B&H%':<10}", flush=True)
    print("-" * 70, flush=True)
    for ticker, res in all_results.items():
        for algo_name, r in res.items():
            a = r["aggregate"]
            ret = f"{a['total_return_pct']['mean']:+.1f}±{a['total_return_pct']['std']:.1f}"
            print(f"{ticker:<14}{algo_name:<6}{ret:<20}"
                  f"{a['sharpe']['mean']:<9.2f}{a['buy_hold_pct']:<11.1f}"
                  f"{a['vs_buy_hold_pct']['mean']:<+10.1f}", flush=True)
    print("\nDONE. Full results -> results/results.json", flush=True)


if __name__ == "__main__":
    main()
