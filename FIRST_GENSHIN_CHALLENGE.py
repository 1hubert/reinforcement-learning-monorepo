"""
Vaporize Reactions Triggered: 0/15
DMG Dealt to Monsters: 0/14000

--------------------
Things I'll need to implement at some point:
- a way to track is_q_available

--------------------
Potential problems we may come accross:
- what do we do about having diferent e/q abilities (and special abilities like neuvillette ca) for many different characters? hard-code everything? Should every character be a different agent, inheriting from an agent which only can move?
- what if the greedily-chosen action is not available (e/q/switch_character)? --> i guess that you take the state-action pair with the second-highest value

--------------------
Potential approaches:
- seconds / half-seconds as state
- rotation-based approach: estimate how much time an optimal rotation would take, then use (1, 2, ..., rotation_max_time) as state (cycle through it multiple times within one episode)
- visual-input-based approach: learn to recognize enemies, walk up to them and beat them up
- proximity-based approach: hack a way to get some data about x-y-z coordinates of your character and the enemies, lower the lowest proximity and beat the enemy up, rinse and repeat
- if i decided to go with `damage_done`+`reactions_done` when calculating reward, i could kind of simulate what i do: maximize `reactions_done` first, then maximize the remaining `damage_done` until you win. voila

APPROACH 1:
- Use seconds (1, 2, ..., last_second) as state
- Limit agent to char2 only
- focus on `damage_done`, ignore `reactions_done`

potentially next approach:
- Use seconds (1, 2, ..., last_second) as state
- Give agent the ability to change characters, but only as their very first action in an episode
- focus on `damage_done`, ignore `reactions_done`

i just realized that not all people have their windows bar on top so i'd have to:
a) subtract 40 on all Y axes
b) have a variable like WINDOWS_BAR_BOTTOM=True based on which I'd adjust y values.
"""
from time import sleep, perf_counter
import random
import logging
import multiprocessing

import numpy as np
from PIL import ImageGrab
import cv2
from paddleocr import PaddleOCR
import pyautogui

SAVE_FILENAME = 'next_episode+qtable.npy'

LOADING_SCREEN = (502, 338)
EXIT_DOMAIN_BUTTON = (380, 380)
SELECT_TRIAL_VAPORIZE = (152, 128)
START_DOMAIN = (570, 527)

ocr = PaddleOCR(
    lang='en',
    rec_char_dict_path='./allowed_chars.txt'
)

class GenshinEnvironment:
    def __init__(self):
        self.actions = {
            0: self.charged_attack,
            1: self.use_e,
            2: self.switch_characters,
            # 0: self.move_forward,
            # 1: self.move_backward,
            # 2: self.move_left,
            # 3: self.move_right,
            # 4: self.basic_attack,
            # 7: self.use_q,
        }
        self.stats = {
            # Barbara
            0: {
                'e_cooldown': 32,
                'e_casttime': 0.5
            },
            # Xiangling
            1: {
                'e_cooldown': 12,
                'e_casttime': 0.5
            }
        }

    def move_forward(self):
        logging.debug('action: moving forward')
        with pyautogui.hold('w'):
            sleep(1)

    def move_backward(self):
        logging.debug('action: moving backward')
        with pyautogui.hold('s'):
            sleep(1)

    def move_left(self):
        logging.debug('action: moving left')
        with pyautogui.hold('a'):
            sleep(1)

    def move_right(self):
        logging.debug('action: moving right')
        with pyautogui.hold('d'):
            sleep(1)

    def basic_attack(self):
        logging.debug('action: using basic attack')
        pyautogui.click()

    def charged_attack(self):
        logging.debug(f'action: using charged attack (char={self.current_char})')
        pyautogui.mouseDown()
        sleep(0.25)
        pyautogui.mouseUp()

    def use_e(self):
        logging.debug(f'action: using e (char={self.current_char})')
        pyautogui.hotkey('e')

        # Do nothing during cast time
        sleep(self.stats[self.current_char]['e_casttime'])

        # Keep track of cooldown
        self.next_usage_times[f'use_e{self.current_char}'] = (
            perf_counter()
            + self.stats[self.current_char]['e_cooldown']
        )

    def switch_characters(self):
        if self.current_char == 0:
            logging.debug('action: switching chars 0->1')
            pyautogui.hotkey('2')
            self.current_char = 1
        else:
            logging.debug('action: switching chars 1->0')
            pyautogui.hotkey('1')
            self.current_char = 0

        # Keep track of cooldown (3s so that there's no switch spam)
        self.next_usage_times[self.switch_characters] = perf_counter() + 3

    def is_action_on_cooldown(self, action: int) -> bool:
        if action == 1:  # e
            next_usage_time = self.next_usage_times.get(
                f'use_e{self.current_char}', 0)
        else:
            action_f = self.actions[action]
            next_usage_time = self.next_usage_times.get(action_f, 0)
        return next_usage_time > perf_counter()

    def ready_actions(self) -> list:
        """Return ready actions in a list of indeces."""
        return [a for a in self.actions if not self.is_action_on_cooldown(a)]

    def random_action(self) -> int:
        """Return the key of a random ready action."""
        ready_actions = [a for a in self.actions if not self.is_action_on_cooldown(a)]
        return random.choice(ready_actions)

    def reset(self, first_episode=False):
        """
        Basically quit domain, re-enter, run up the stairs and click 'f' near the key to start a new episode with same initial environment.

        Unfotunetely, this might be the way to do it in the MVP as this is the easiest way to guarantee reproducibility of mapping actions to results.

        About the first episode:
        - The human preparation required is to find the reaction domain on a map and click 'teleport'.
        - to start the first episode, run this function with `first_episode` keyword argument set to True.
        """
        def wait_until_loading_screen_gone(delay):
            sleep(delay)
            loading_screen_color = pyautogui.pixel(*LOADING_SCREEN)
            while pyautogui.pixel(*LOADING_SCREEN) == loading_screen_color:
                sleep(0.5)

        if first_episode:
            # Run up to the domain
            with pyautogui.hold('w'):
                sleep(2 + (random.random() / 5))
        else:
            # Terminate child process and reclaim resources
            self.update_terminal_states_process.terminate()
            self.update_terminal_states_process.join()

            # Quit domain
            pyautogui.hotkey('esc')
            sleep(random.random())
            pyautogui.click(*EXIT_DOMAIN_BUTTON)

            # Wait until the open world loads
            wait_until_loading_screen_gone(4)

            # Run up to the domain
            with pyautogui.hold('s'):
                sleep(2.4 + (random.random() / 5))

        # Open domain and run it
        pyautogui.hotkey('f')
        sleep(6.5)
        pyautogui.click(*SELECT_TRIAL_VAPORIZE)
        pyautogui.click(*START_DOMAIN)

        # Wait until domain loads
        wait_until_loading_screen_gone(1)

        # Click through the reaction tutorial
        for _ in range(3):
            sleep(1.1 + random.random())
            pyautogui.click(*LOADING_SCREEN)
        sleep(1 + random.random() / 10)

        # Run up to the key (this cannot be randomized)
        w_time_1 = random.uniform(2, 2.2)
        w_time_2 = 7 - w_time_1
        with pyautogui.hold('w'):
            sleep(w_time_1)
            pyautogui.keyDown('shiftleft')
            pyautogui.keyUp('shiftleft')
            sleep(w_time_2)
        sleep(0.1)

        # Reset relevant attributes
        self.current_char = 0
        self.next_usage_times = {
            'use_e0': 0,
            'use_e1': 0,
            self.switch_characters: 0
        }

        # Start a daemon process checking if a terminal state has been encountered
        self.death_event = multiprocessing.Event()
        self.victory_event = multiprocessing.Event()
        self.update_terminal_states_process = multiprocessing.Process(
            target=update_terminal_states,
            args=(self.death_event, self.victory_event)
        )
        self.update_terminal_states_process.start()

        # Start a daemon process updating a shared value reactions done
        self.reactions_done = multiprocessing.Value('i', 0)
        self.extract_reactions_done_process = multiprocessing.Process(
            target=extract_reactions_done,
            args=[self.reactions_done]
        )
        self.extract_reactions_done_process.start()

        # Start the domain!
        pyautogui.hotkey('f')

        # Start counting time
        self.start_time = perf_counter()

        # Return first state: (current character, time elapsed)
        return (self.current_char, 0)

    def step(self, action, discrete_time_elapsed):
        """
        Execute given action and return (next_state, reward, terminated, truncated).
        """
        logging.debug(f'reactions done before: {self.reactions_done.value}')
        reactions_done_before = self.reactions_done.value

        # Execute action
        self.actions[action]()

        # Wait until next discrete time block
        next_discrete_time = discrete_time_elapsed + 2
        while perf_counter() < self.start_time + next_discrete_time:
            sleep(0.001)

        # Check if reactions_done increased
        reactions_done_increased = self.reactions_done.value > reactions_done_before
        logging.debug(f'reactions done after: {self.reactions_done.value}')

        reward = 1 if reactions_done_increased else 0
        terminated = self.death_event.is_set() or self.victory_event.is_set()
        time_elapsed = perf_counter() - self.start_time

        return (
            (self.current_char, next_discrete_time),
            reward,
            terminated,
            time_elapsed > 90
        )

def update_terminal_states(death_event, victory_event):
    """Check if a terminal state has been encountered in a loop. Set a matching event and break in case a state has been recognized."""
    # Healthbar-related
    LEFT_END_HEALTHBAR = (271, 528)
    HEALTHBAR_GREEN = (150, 215, 34)
    HEALTHBAR_RED = (255, 90, 90)

    # "Challenge Completed"-related
    LETTER_C_1 = (182, 279)
    LETTER_C_2 = (320, 279)
    LETTER_D = (454, 278)
    WHITE = (255, 255, 255)

    healthbar_color = HEALTHBAR_GREEN
    while True:
        # Check healthbar
        if not pyautogui.pixelMatchesColor(*LEFT_END_HEALTHBAR, healthbar_color, tolerance=41):
            if healthbar_color == HEALTHBAR_RED:
                death_event.set()
                break

            healthbar_color = HEALTHBAR_RED

        # Check victory
        if (pyautogui.pixelMatchesColor(
            *LETTER_C_1, WHITE) and
            pyautogui.pixelMatchesColor(
            *LETTER_C_2, WHITE) and
            pyautogui.pixelMatchesColor(
            *LETTER_D, WHITE)):

            victory_event.set()
            break

def extract_reactions_done(shared_value, show_image=False):
    while True:
        # Grab image
        image = ImageGrab.grab(bbox=(135, 165, 154, 178))  # ltrb

        # Turn image into a numpy array
        image = np.array(image)

        # Show image
        if show_image:
            cv2.imshow('window', image)

        # Get text
        result = ocr.ocr(image, det=False, cls=False)[0][0]

        # # Get first number out of str like '0 15'
        result = result.split(' ')[0]
        try:
            result = int(result)
            if shared_value.value < result <= 15:
                with shared_value.get_lock():
                    shared_value.value = result
        except ValueError:  # Not a a valid int
            pass

class GenshinAgent:
    def __init__(self, learning_rate, discount_rate, initial_epsilon, epsilon_decay, final_epsilon, qtable):
        """Initialize a RL agent with an empty dict of state-action values (q_values), a learning rate and an epsilon.
        """
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.qtable = qtable

    def get_action(self, state) -> int:
        """
        Returns the best action with probability (1-epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # With p(epsilon) return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.random_action()

        # With p(1 - epsilon) act greedily (exploit)
        else:
            if not np.any(self.qtable[state]):  # if empty row
                return env.random_action()
            else:
                a = np.argsort(self.qtable[state])[-1]
                ready_actions = env.ready_actions()
                i = -2
                while a not in ready_actions:
                    a = np.argsort(self.qtable[state])[i]
                    i -= 1
                return a


    def update(self, state, action, reward, new_state, terminated):
        """Update Q(s,a):= Q(s,a) + lr*[R(s,a) + discount_rate * max(Q(s',a') - Q(s, a)]"""
        future_q_value = (not terminated) * np.max(self.qtable[new_state])

        delta = (
            reward
            + self.discount_rate * future_q_value
            - self.qtable[state][action]
        )

        print('--------------------------------------------')
        print(self.qtable)

        self.qtable[state][action] = (
            self.qtable[state][action] + self.learning_rate * delta
        )

    def decay_epsilon(self):
        self.epsilon = max(
            self.final_epsilon, self.epsilon - self.epsilon_decay
        )


def process_image(image):
    """Dumb the original image down using OpenCV (Open Source Computer Vision Library).

    :image: A numpy array with pixels in BGR.
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image


def extract_damage_done(show_image=False):
    # Grab image
    image = ImageGrab.grab(bbox=(110, 180, 180, 188))

    # Turn image into a numpy array
    image = np.array(image)

    # Show image
    if show_image:
        cv2.imshow('window', image)

    # Get text
    result = ocr.ocr(image, det=False, cls=False)[0][0]

    # Get first number out of str like '11023 14000'
    result = result.split(' ')[0]

    try:
        if 0 <= int(result) <= 14000:
            return result
        else:
            return False
    except ValueError:  # Not a a valid int
        return False


def main():
    while True:
        # Extract damage done from image
        print(extract_reactions_done(show_image=True))

        # ord(q) == 113
        if cv2.waitKey(1) == 113:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s] %(asctime)s - %(message)s',
        force=True
    )

    # hyerparameters
    learning_rate = 0.8
    discount_rate = 0.95
    n_episodes = 100
    initial_epsilon = 1.0
    epsilon_decay = initial_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    # sizes
    character_count = 2  # 0 - barbara, 1 - xg
    time_steps = 90  # 90s
    action_size = 3  # CA / E / switch

    # Try loading qmatrix and next_episode from file,
    # if the file doesn't exist, use defualt values
    try:
        loaded = np.load(SAVE_FILENAME, allow_pickle=True)
        next_episode = loaded[0]
        qtable = loaded[1]
        logging.info(f'resuming training... next episode: {next_episode}')
    except FileNotFoundError:
        next_episode = 0
        qtable = np.zeros((character_count, time_steps, action_size))
        logging.info('beginning a new training...')

    env = GenshinEnvironment()
    agent = GenshinAgent(
        learning_rate=learning_rate,
        discount_rate=discount_rate,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        qtable=qtable
    )

    # -----------------------------------------------------

    pyautogui.click(*LOADING_SCREEN)

    for episode in range(next_episode, n_episodes):
        if episode == next_episode:
            state = env.reset(first_episode=True)  # t=0s
        else:
            state = env.reset()  # t=0s

        done = False

        # Play one episode
        while True:
            action = agent.get_action(state)
            # state[0] is current character, state[1] is discreticized time elapsed
            next_state, reward, terminated, truncated = env.step(action, state[1])

            # terminated - either xiangling or barbara is dead OR we won
            # truncated - the trial has run out of time
            if truncated:
                logging.info('episode truncated..')
                break

            agent.update(state, action, reward, next_state, terminated)

            if terminated:
                logging.info('episode terminated..')
                break

            state = next_state

        agent.decay_epsilon()

        # Save training to file
        logging.info(f'saving training as of episode {episode} to {SAVE_FILENAME}...')
        obj = np.empty(2, dtype='object')
        obj[0] = episode + 1
        obj[1] = agent.qtable
        np.save(SAVE_FILENAME, obj)
        logging.info('training saved!')
