"""Complete the one ticker (INFY) the interrupted intraday run missed."""
import warnings, os
warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"] = "ignore"
import data as D
from trading_env import evaluate
from run_intraday import train_one, SEEDS, ALGOS, COST, ANN, PERIOD, INTERVAL

df = D.add_features(D.fetch("INFY.NS", period=PERIOD, interval=INTERVAL))
tr, va, te = D.chrono_split(df); mu, sd = D.normalizer(tr)
print(f"=== INFY.NS (1h) === test={len(te)} bars", flush=True)
for algo_name, AlgoCls in ALGOS.items():
    cand = []
    for seed in SEEDS:
        m = train_one(AlgoCls, tr, mu, sd, seed)
        vm = evaluate(m, va, mu, sd, cost=COST, ann=ANN)
        cand.append((seed, vm["sharpe"], m))
        print(f"  {algo_name} seed{seed}: VAL sharpe={vm['sharpe']:.2f} ret={vm['total_return_pct']:+.1f}%", flush=True)
    bs, bvs, bm = max(cand, key=lambda x: x[1])
    tm = evaluate(bm, te, mu, sd, cost=COST, ann=ANN)
    print(f"  -> selected seed{bs} (val sharpe {bvs:.2f}) | TEST ret={tm['total_return_pct']:+.1f}% "
          f"sharpe={tm['sharpe']:.2f} vsB&H={tm['vs_buy_hold_pct']:+.1f}% "
          f"(L{tm['pct_long']}/S{tm['pct_short']}/F{tm['pct_flat']}) | B&H={tm['buy_hold_pct']:+.1f}%", flush=True)
