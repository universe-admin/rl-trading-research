"""Intraday (1h) DRL with validation-based model selection.

Upgraded rigour vs the daily run:
  * 1h bars (5k+ samples, finer resolution where short-term edges may exist)
  * train multiple seeds, SELECT the best model on the VALIDATION set,
    then report that single model on the TEST set (no test peeking)
  * Sharpe annualized for intraday bar frequency
"""
import os
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
TIMESTEPS = 50000
COST = 0.001
ANN = 1638              # ~ NSE 1h bars per year (252d * 6.5h)
PERIOD, INTERVAL = "730d", "60m"


def train_one(AlgoCls, train_df, mu, sd, seed):
    env = TradingEnv(train_df, mu, sd, cost=COST)
    model = AlgoCls("MlpPolicy", env, seed=seed, verbose=0,
                    policy_kwargs=dict(net_arch=[64, 64]))
    model.learn(total_timesteps=TIMESTEPS, progress_bar=False)
    return model


def main():
    all_results = {}
    for ticker in TICKERS:
        print(f"\n=== {ticker} (1h) ===", flush=True)
        try:
            raw = D.fetch(ticker, period=PERIOD, interval=INTERVAL)
            df = D.add_features(raw)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP (data error): {e}", flush=True)
            continue
        tr, va, te = D.chrono_split(df)
        mu, sd = D.normalizer(tr)
        print(f"  bars: train={len(tr)} val={len(va)} test={len(te)} "
              f"(test {te['Date'].iloc[0]} -> {te['Date'].iloc[-1]})", flush=True)

        ticker_res = {}
        for algo_name, AlgoCls in ALGOS.items():
            candidates = []
            for seed in SEEDS:
                model = train_one(AlgoCls, tr, mu, sd, seed)
                val_m = evaluate(model, va, mu, sd, cost=COST, ann=ANN)
                candidates.append((seed, val_m["sharpe"], model))
                print(f"  {algo_name} seed{seed}: VAL sharpe={val_m['sharpe']:.2f} "
                      f"ret={val_m['total_return_pct']:+.1f}%", flush=True)
            # SELECT best on validation Sharpe, then evaluate on TEST
            best_seed, best_val_sharpe, best_model = max(candidates, key=lambda x: x[1])
            test_m = evaluate(best_model, te, mu, sd, cost=COST, ann=ANN)
            print(f"  -> selected seed{best_seed} (val sharpe {best_val_sharpe:.2f}) | "
                  f"TEST ret={test_m['total_return_pct']:+.1f}% sharpe={test_m['sharpe']:.2f} "
                  f"vsB&H={test_m['vs_buy_hold_pct']:+.1f}% "
                  f"(L{test_m['pct_long']}/S{test_m['pct_short']}/F{test_m['pct_flat']})",
                  flush=True)
            ticker_res[algo_name] = {"selected_seed": best_seed,
                                     "val_sharpe": best_val_sharpe,
                                     "test": test_m}
        all_results[ticker] = ticker_res

    with open(os.path.join(RESULTS, "results_intraday.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print("\n\n======== INTRADAY OUT-OF-SAMPLE SUMMARY (val-selected model) ========", flush=True)
    print(f"{'Ticker':<14}{'Algo':<6}{'TestRet%':<10}{'Sharpe':<9}"
          f"{'Buy&Hold%':<11}{'vs B&H%':<10}", flush=True)
    print("-" * 60, flush=True)
    for ticker, res in all_results.items():
        for algo_name, r in res.items():
            t = r["test"]
            print(f"{ticker:<14}{algo_name:<6}{t['total_return_pct']:<+10.1f}"
                  f"{t['sharpe']:<9.2f}{t['buy_hold_pct']:<11.1f}"
                  f"{t['vs_buy_hold_pct']:<+10.1f}", flush=True)
    print("\nDONE. Full results -> results/results_intraday.json", flush=True)


if __name__ == "__main__":
    main()
