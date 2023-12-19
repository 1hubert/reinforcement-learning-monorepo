from __future__ import annotations

from collections import defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics


class BlackjackAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95
    ):
        """Initialize a RL agent with an empty dict of state-action values (q_values), a learning rate and an epsilon.
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, int]) -> int:
        """
        Returns the best action with probability (1-epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # With p(epsilon) return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # With p(1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, int],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, int]
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(
            self.final_epsilon, self.epsilon - self.epsilon_decay
        )

def create_grids(agent, usable_ace=False):
    """Create value and policy grid given an agent."""
    # Convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # player count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11)
    )

    # Create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count])
    )
    value_grid = player_count, dealer_count, value

    # Create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count])
    )

    return value_grid, policy_grid

def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # Create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # Plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap='viridis',
        edgecolor='none'
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ['A'] + list(range(2, 11)))
    ax1.set_title(f'state values: {title}')
    ax1.set_xlabel('Player sum')
    ax1.set_ylabel('Dealer showing')
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 20)

    # Plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f'Policy: {title}')
    ax2.set_xlabel('Player sum')
    ax2.set_ylabel('Dealer showing')
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(['A'] + list(range(2, 11)), fontsize=12)

    # Add a legend
    legend_elements = [
        Patch(facecolor='lightgreen', edgecolor='black', label='Hit'),
        Patch(facecolor='grey', edgecolor='black', label='Stick')
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))

    return fig


if __name__ == '__main__':
    env = gym.make("Blackjack-v1", sab=True, render_mode='rgb_array')

    # Reset the env to get the first observation
    done = False
    observation, _ = env.reset()

    # hyperparameters
    learning_rate = 0.01
    n_episodes = 100_000
    initial_epsilon = 1.0
    epsilon_decay = initial_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    agent = BlackjackAgent(
        learning_rate=learning_rate,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon
    )

    env = RecordEpisodeStatistics(env, deque_size=n_episodes)
    for episode in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        done = False

        # Play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Update the agent
            agent.update(obs, action, reward, terminated, next_obs)

            # terminated - the agent was killed
            # truncated - the agent went out of bounds
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

    # Visualizing the training
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Compute and assign a rolling average of the data to provide a smoother graph
    axs[0].set_title("Episode rewards")
    reward_moving_average = (
        np.convolve(
            np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

    axs[1].set_title("Episode lengths")
    length_moving_average = (
        np.convolve(
            np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)

    axs[2].set_title("Training error")
    training_error_moving_average = (
        np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="valid")
        / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    plt.tight_layout()
    plt.show()

    # state values & policy with usable ace (ace counts as 11)
    value_grid, policy_grid = create_grids(agent, usable_ace=True)
    fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
    plt.show()

    # state values & policy without usable ace (ace counts as 1)
    value_grid, policy_grid = create_grids(agent, usable_ace=False)
    fig2 = create_plots(value_grid, policy_grid, title="Without usable ace")
    plt.show()

    # It's good practice to force any used resources to close
    env.close()

    # Print some MA500-related stuff for clarity on "deficit data points on X-axis"
    print(f'length of env.return_queue: {len(env.return_queue)}')
    print(f'length of reward_moving_average: {len(reward_moving_average)}\n')
    print(f'length of env.length_queue: {len(env.length_queue)}')
    print(f'length of length_moving_average: {len(length_moving_average)}\n')
    print(f'length of agent.training_error: {len(agent.training_error)}')
    print(f'length of training_error_moving_average: {len(training_error_moving_average)}')
