"""Single-asset trading environment with Differential Sharpe Ratio reward.

Action space (Discrete 3): 0 -> short (-1), 1 -> flat (0), 2 -> long (+1).
The agent CAN open short positions, so downside is tradeable.

Differential Sharpe Ratio (Moody & Saffell, 1998): an online/incremental
form of the Sharpe ratio. Maximizing the cumulative DSR maximizes the
risk-adjusted return rather than raw return.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from data import FEATURE_COLS


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df, mu, sd, cost=0.001, eta=0.01, reward="dsr"):
        super().__init__()
        self.prices = df["Close"].values.astype(np.float64)
        feats = df[FEATURE_COLS].values.astype(np.float32)
        self.feats = np.clip((feats - mu) / sd, -5, 5).astype(np.float32)
        self.n = len(df)
        self.cost = cost          # per unit position change (0.1% = round-trip-ish)
        self.eta = eta            # DSR EMA adaptation rate
        self.reward_kind = reward

        self.action_space = spaces.Discrete(3)
        obs_dim = len(FEATURE_COLS) + 1  # features + current position
        self.observation_space = spaces.Box(-5.0, 5.0, (obs_dim,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.position = 0.0
        self.A = 0.0   # DSR: EMA of returns
        self.B = 0.0   # DSR: EMA of squared returns
        self.equity = 1.0
        return self._obs(), {}

    def _obs(self):
        return np.concatenate([self.feats[self.t], [np.float32(self.position)]])

    def _dsr(self, r):
        """Incremental differential Sharpe ratio for realized return r."""
        dA = r - self.A
        dB = r * r - self.B
        denom = (self.B - self.A * self.A) ** 1.5
        d = 0.0 if denom < 1e-12 else (self.B * dA - 0.5 * self.A * dB) / denom
        self.A += self.eta * dA
        self.B += self.eta * dB
        return float(d)

    def step(self, action):
        target = {0: -1.0, 1: 0.0, 2: 1.0}[int(action)]
        trade_cost = self.cost * abs(target - self.position)
        self.position = target

        # realize return over t -> t+1 (decision uses info up to t only)
        nxt = min(self.t + 1, self.n - 1)
        asset_ret = (self.prices[nxt] / self.prices[self.t]) - 1.0
        r = self.position * asset_ret - trade_cost
        self.equity *= (1.0 + r)

        reward = self._dsr(r) if self.reward_kind == "dsr" else r

        self.t += 1
        terminated = self.t >= self.n - 1
        return self._obs(), float(reward), terminated, False, {"ret": r}


def evaluate(model, df, mu, sd, cost=0.001, ann=252):
    """Run a trained model deterministically; return metrics + buy&hold."""
    env = TradingEnv(df, mu, sd, cost=cost)
    obs, _ = env.reset()
    rets, positions = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        rets.append(info["ret"])
        positions.append(env.position)
    rets = np.array(rets)
    return _metrics(rets, df["Close"].values, positions, ann=ann)


def _metrics(rets, prices, positions, ann=252):
    eq = np.cumprod(1 + rets)
    total = eq[-1] - 1 if len(eq) else 0.0
    sharpe = (rets.mean() / rets.std() * np.sqrt(ann)) if rets.std() > 1e-12 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min() if len(eq) else 0.0
    bh = prices[-1] / prices[0] - 1
    pos = np.array(positions)
    return {
        "total_return_pct": round(100 * total, 2),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown_pct": round(100 * float(mdd), 2),
        "buy_hold_pct": round(100 * float(bh), 2),
        "vs_buy_hold_pct": round(100 * (total - bh), 2),
        "n_steps": int(len(rets)),
        "pct_long": round(100 * float((pos > 0).mean()), 1),
        "pct_short": round(100 * float((pos < 0).mean()), 1),
        "pct_flat": round(100 * float((pos == 0).mean()), 1),
    }
