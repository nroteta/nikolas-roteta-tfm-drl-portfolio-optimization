import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


def _rolling_zscore(arr: np.ndarray, window: int = 63) -> np.ndarray:
    df   = pd.DataFrame(arr)
    mean = df.rolling(window, min_periods=5).mean()
    std  = df.rolling(window, min_periods=5).std() + 1e-9
    return ((df - mean) / std).fillna(0).values.astype(np.float32)


class PortfolioEnvBBVA(gym.Env):
    def __init__(
        self,
        prices: pd.DataFrame,
        transaction_cost: float = 0.001,
        turnover_penalty: float = 0.01,
        sharpe_window: int = 30,
        episode_length: int = 252,
        risk_free_rate: float = 0.05,
    ):
        super().__init__()
 
        self.prices           = prices.values.astype(np.float32)
        self.dates            = prices.index
        self.tickers          = list(prices.columns)
        self.n                = len(self.tickers)
        self.transaction_cost = transaction_cost
        self.turnover_penalty = turnover_penalty
        self.sharpe_window    = sharpe_window
        self.episode_length   = episode_length
        self.window_max       = 50
        
        self.rf_daily = (1 + risk_free_rate) ** (1/252) - 1
 
        ret        = prices.pct_change()
        vol7_raw  = ret.rolling(7).std().values
        vol30_raw = ret.rolling(30).std().values
        ret3_raw  = prices.pct_change(3).values
        ret7_raw  = prices.pct_change(7).values
        ret15_raw = prices.pct_change(15).values
        gain       = ret.clip(lower=0).rolling(14).mean()
        loss       = (-ret.clip(upper=0)).rolling(14).mean()
        rs         = gain / (loss + 1e-9)
        rsi_raw   = (100 - 100 / (1 + rs)).fillna(50).values / 100.0
        ma50_raw  = (prices / prices.rolling(50).mean() - 1).fillna(0).values

        self.vol7  = _rolling_zscore(vol7_raw)
        self.vol30 = _rolling_zscore(vol30_raw)
        self.ret3  = _rolling_zscore(ret3_raw)
        self.ret7  = _rolling_zscore(ret7_raw)
        self.ret15 = _rolling_zscore(ret15_raw)
        self.rsi   = _rolling_zscore(rsi_raw)
        self.ma50  = _rolling_zscore(ma50_raw)
        
 
        obs_dim = 7 * self.n + self.n
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(-np.inf, np.inf, shape=(self.n,),   dtype=np.float32)
 
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        window_max = self.window_max
        max_start = len(self.prices) - self.episode_length - 2
        self._t       = int(self.np_random.integers(window_max, max(window_max + 1, max_start)))
        self._t_start = self._t
        self._weights = np.ones(self.n, dtype=np.float32) / self.n
        self._value   = 1.0
        self._returns = []
        return self._obs(), {}
 
    def step(self, action):
        w_new    = self._softmax(action)
        turnover = float(np.abs(w_new - self._weights).sum())
 
        r_assets = self.prices[self._t + 1] / self.prices[self._t] - 1
        ret      = float(np.dot(w_new, r_assets)) - self.transaction_cost * turnover
        self._value *= (1 + ret)
 
        drifted       = w_new * (1 + r_assets)
        self._weights = (drifted / drifted.sum()).astype(np.float32)
 
        self._returns.append(ret)
        if len(self._returns) > self.sharpe_window:
            self._returns.pop(0)
 
        reward = self._reward(ret, turnover)
        self._t += 1
 

        steps_done = self._t - self._t_start
        terminated = (steps_done >= self.episode_length) or (self._t >= len(self.prices) - 1)
 
        info = {"value": self._value, "weights": self._weights.copy(), "turnover": turnover}
        return self._obs(), float(reward), terminated, False, info
 
    def _obs(self):
        t = self._t
        signals = np.concatenate([
            self.vol7[t], self.vol30[t],
            self.ret3[t], self.ret7[t], self.ret15[t],
            self.rsi[t], self.ma50[t],
            self._weights,
        ])
        return np.nan_to_num(np.clip(signals, -5, 5), nan=0.0).astype(np.float32)
 
    def _reward(self, ret, turnover):
        buf = np.array(self._returns)
        if len(buf) < 5:
            return ret 
        excess_returns = buf - self.rf_daily
        base_reward = float(excess_returns.mean() / (excess_returns.std() + 1e-9)) * np.sqrt(252)
        
        return base_reward - self.turnover_penalty * turnover
 
    @staticmethod
    def _softmax(x):
        e = np.exp(x - x.max())
        return (e / e.sum()).astype(np.float32)
 