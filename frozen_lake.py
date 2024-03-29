# Standard library imports
from pathlib import Path
from typing import NamedTuple
import time

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Gymnasium imports
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


class Params(NamedTuple):
    total_episodes: int
    learning_rate: float
    discount_rate: float  # AKA "gamma" AKA "discount_factor"
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared env
    seed: int  # Define a seed so that we get reproducible results
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    proba_frozen: float  # Probability that a tile is frozen
    savefig_folder: Path  # Root folder where plots are saved


class QLearning:
    def __init__(self, learning_rate, discount_rate, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr*[R(s,a) + discount_rate * max(Q(s',a') - Q(s, a)]"""
        delta = (
            reward
            + self.discount_rate * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )

        return self.qtable[state, action] + self.learning_rate * delta

    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size))


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, action_space, state, qtable):
        """Choose an action `a` in the current world state (s)."""

        # p(epsilon) for exploration
        if rng.uniform(0, 1) < self.epsilon:
            action = action_space.sample()

        # p(1 - epsilon) for exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # If all actions are the same for this state we choose a random one
            # (otherwise `np.argmax()` would always take the first one)
            if np.all(qtable[state, :]) == qtable[state, 0]:
                action = action_space.sample()
            else:
                action = np.argmax(qtable[state, :])
        return action


def run_env():
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []

    # Run several times to account for stochasticity
    for run in range(params.n_runs):
        learner.reset_qtable()  # Reset the Q-table between runs
        for episode in tqdm(
            episodes,
            desc=f'Run {run}/{params.n_runs} - Episodes',
            leave=False
        ):
            state = env.reset(seed=params.seed)[0]  # Reset the env
            step = 0
            done = False
            total_rewards = 0
            while not done:
                action = explorer.choose_action(
                    action_space=env.action_space,
                    state=state,
                    qtable=learner.qtable
                )

                # Log all states and actions
                all_states.append(state)
                all_actions.append(action)

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, terminated, truncated, _ = env.step(action)
                # print(f'action_took={action} action_took_in_state={state} new_state={new_state} reward={reward} terminated={terminated}')

                done = terminated or truncated

                learner.qtable[state, action] = learner.update(
                    state, action, reward, new_state
                )
                # print(learner.qtable)
                # print('-----------------')
                # time.sleep(0.1)

                total_rewards += reward
                step += 1

                # Our new state is state
                state = new_state

            # Log all rewards and steps
            rewards[episode, run] = total_rewards
            steps[episode, run] = step
        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions


def postprocess(episodes, params, rewards, steps, map_size):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(),
            "Steps": steps.flatten()
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])

    return res, st


def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-Value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


def plot_q_values_map(qtable, env, map_size):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"}
    ).set(title="Learned Q-values \nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    img_title = f'frozenlake_q_values_{map_size}x{map_size}.png'
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


def plot_states_actions_distribution(states, actions, map_size):
    """Plot the distributions of states and actions."""
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=False)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    img_title = f'frozenlake_states_actions_distrib_{map_size}x{map_size}.png'
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


def plot_steps_and_rewards(rewards_df, steps_df):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(
        data=rewards_df, x="Episodes", y="cum_rewards", hue="map_size", ax=ax[0]
    )
    ax[0].set(ylabel="Cumulated rewards")

    sns.lineplot(data=steps_df, x="Episodes", y="Steps", hue="map_size", ax=ax[1])
    ax[1].set(ylabel="Averaged steps number")

    for axi in ax:
        axi.legend(title="map size")
    fig.tight_layout()
    img_title = "frozenlake_steps_and_rewards.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    # Activate seaborn's "default" theme
    sns.set_theme()

    params = Params(
        total_episodes=2000,
        learning_rate=0.8,
        discount_rate=0.95,
        epsilon=0.1,
        map_size=5,
        seed=123,
        n_runs=1,
        action_size=None,
        state_size=None,
        proba_frozen=0.9,
        savefig_folder=Path("./frozen_lake_savefig/")
    )

    # Set the seed for `rng.uniform(0, 1)`
    rng = np.random.default_rng(params.seed)

    # Create the figure folder if it doesn't exist
    params.savefig_folder.mkdir(parents=True, exist_ok=True)

    # map_sizes = [4, 7, 9, 11]
    map_sizes = [4]
    res_all = pd.DataFrame()
    st_all = pd.DataFrame()

    for map_size in map_sizes:
        env = gym.make(
            "FrozenLake-v1",
            is_slippery=False,
            render_mode="rgb_array",
            desc=generate_random_map(
                size=map_size, p=params.proba_frozen, seed=params.seed
            )
        )

        params = params._replace(action_size=env.action_space.n)
        params = params._replace(state_size=env.observation_space.n)
        env.action_space.seed(params.seed)
        learner = QLearning(
            learning_rate=params.learning_rate,
            discount_rate=params.discount_rate,
            state_size=params.state_size,
            action_size=params.action_size
        )
        explorer = EpsilonGreedy(
            epsilon=params.epsilon
        )

        print(f'Map size: {map_size}x{map_size}')
        rewards, steps, episodes, qtables, all_states, all_actions = run_env()

        # Save the results in dataframes
        res, st = postprocess(episodes, params, rewards, steps, map_size)
        res_all = pd.concat([res_all, res])
        st_all = pd.concat([st_all, st])
        qtable = qtables.mean(axis=0)  # Average the Q-table between runs

        # Print the averaged Q-table for this map_size
        mw = 12  # Minimum width
        dp = 3  # Decimal places
        print(f'AVERAGED Q-TABLE FOR {map_size}x{map_size}')
        print(f'{"LEFT":^{mw}}{"DOWN":^{mw}}{"RIGHT":^{mw}}{"UP":^{mw}}')
        for row in qtable:
            for i in range(params.action_size):
                print('{:.{}f}'.format(row[i], dp).center(mw), end='')
            else:
                print()

        # Plot 1: Distribution of states to action
        print('preparing plot 1...')
        plot_states_actions_distribution(all_states, all_actions, map_size)

        # Plot 2: Cumulative rewards & Avg steps number
        print('preparing plot 2...')
        plot_q_values_map(qtable, env, map_size)

        env.close()

    # Plot 3: Plot 3: Last frame & Averaged Q-Matrix represented as arrows
    print('preparing plot 3...')
    plot_steps_and_rewards(res_all, st_all)
