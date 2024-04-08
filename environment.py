import datetime
import glob
import os
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import click
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from backtest import LIMIT_POSITIONS, MarketSimulator

warnings.filterwarnings("error")


def basic_reward_function():
    return np.log(
        history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    )


def dynamic_feature_last_position_taken(history):
    return history["position", -1]


def dynamic_feature_real_position(history):
    return history["real_position", -1]


class TradingEnv(gym.Env):

    def __init__(
        self,
        market_simulator: MarketSimulator,
        positions: list = [0],
        trading_fees=0,
        borrow_interest_rate=0,
        portfolio_initial_value=1000,
        max_episode_duration="max",
        verbose=1,
        name="Stock",
    ):
        self.market_simulator = market_simulator
        df = market_simulator.df
        self.max_episode_duration = max_episode_duration
        self.name = name
        self.verbose = verbose

        self.positions = positions
        self.trading_fees = trading_fees
        self.borrow_interest_rate = borrow_interest_rate
        self.portfolio_initial_value = float(portfolio_initial_value)

        self._set_df(df)

        self.action_space = spaces.Box(low=-1, high=1)

    def _set_df(self, df):
        df = df.copy()
        self._features_columns = []
        for i in range(1, 4):
            for action in ["ask", "bid"]:
                self._features_columns.append(f"{action}_price_{i}")
                self._features_columns.append(f"{action}_volume_{i}")

        self.df = df
        self._obs_array = np.array(self.df[self._features_columns], dtype=np.float32)

    def _get_ticker(self, delta=0):
        return self.df.iloc[self._idx + delta]

    def _get_obs(self):
        return {"market": self._obs_array[self._idx]}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._step = 0
        self._position = 0
        self._limit_orders = {}

        self._idx = 0
        if self.max_episode_duration != "max":
            self._idx = np.random.randint(
                low=self._idx, high=len(self.df) - self.max_episode_duration - self._idx
            )
        self.market_simulator._reset()
        return self._get_obs()

    def render(self):
        pass

    def _take_action(self, position):
        if position != self._position:
            # TODO: create a trade object
            self.market_simulator.compute_trades()

    def reward_function(self):
        return list(self.market_simulator.player_pnl.values())[-1]

    def step(self, position_index=None):
        if position_index is not None:
            self._take_action(self.positions[position_index])
        self.market_simulator.store_position()
        # TODO: update to multiple stocks
        self._position = list(self.market_simulator.player_position.values())[-1]

        self._idx += 1
        self._step += 1

        done, truncated = False, False

        if self._idx >= len(self.df) - 1:
            truncated = True
        if (
            isinstance(self.max_episode_duration, int)
            and self._step >= self.max_episode_duration - 1
        ):
            truncated = True

        if not done:
            reward = self.reward_function()

        if done or truncated:
            self.market_simulator.print()
        return (
            self._get_obs(),
            reward,
            done,
            truncated,
            self.market_simulator.player_pnl,
        )
