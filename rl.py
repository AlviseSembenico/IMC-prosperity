import os
import sys

import click
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from backtest import MarketSimulator
from environment import TradingEnv


@click.command()
@click.option("--round", default=1, help="Round to backtest")
@click.option("--steps", default=sys.maxsize, help="Number of steps to test against")
def main(round: int, steps: int):
    for file in os.listdir(f"data/round{round}"):
        day = file.split(".")[0]
        simulator = MarketSimulator(round, day)
    register(
        id="TradingEnvIMC",
        entry_point="environment:TradingEnv",
        disable_env_checker=True,
        kwargs={"market_simulator": simulator},
    )
    env = gym.make("TradingEnvIMC", simulator)
    # Run an episode until it ends :
    done, truncated = False, False
    observation, info = env.reset()
    while not done and not truncated:
        # Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
        position_index = (
            env.action_space.sample()
        )  # At every timestep, pick a random position index from your position list (=[-1, 0, 1])
        observation, reward, done, truncated, info = env.step(position_index)


if __name__ == "__main__":
    main()
