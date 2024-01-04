"""
przeciwnicy w tych reaction tutorialach są (potwierdzone) w tych samych miejscach więc mogłoby się okazać, że tyle wystarczy by pobić mój osobisty wynik przynajmniej w jednej malutkiej części genshina (i mało przydatnej do zautomatyzowania), ale i tak byłby to zauważalny postęp ku "generalnym ai do zastąpienia ludzkich graczy".

co do śledzonych liczników:
- są dwa: reactions_done i dmg_done
- na początku można przetestować śledzenie obydwu osobno w dwóch różnych treningach
- później można spróbować zrobić np. jakieś równanie typu:
    reactions_done% * dmg_done%
- może co każde 20 sekund gdzie nie podniósł maksymalizowanej wartości dostaje -1 punkt? albo co wartość cooldownu e  xg + 2s. w teamie z barbarą i xg używanie e na obydwu postaciach nie jest jedyną opcją by maksymalizować reward, więc nie powinienem oczekiwac że bot to zrobi.

Vaporize Reactions Triggered: 0/15
DMG Dealt to Monsters: 0/14000

--------------------
Things I'll need to implement at some point:
- a way to track is_q_available
- a way to track is_terminated (is the agent dead)

--------------------
Potential problems we may come accross:
- what do we do about having diferent e/q abilities (and special abilities like neuvillette ca) for many different characters? hard-code everything? Should every character be a different agent, inheriting from an agent which only can move?
- what state to use?
- what reward to use?
- what if the greedily-chosen action is not available (e/q/switch_character)? --> i guess that you take the state-action pair with the second-highest value

--------------------
Potential approaches:
- seconds / half-seconds as state
- rotation-based approach: estimate how much time an optimal rotation would take, then use (1, 2, ..., rotation_max_time) as state (cycle through it multiple times within one episode)
- visual-input-based approach: learn to recognize enemies, walk up to them and beat them up
- proximity-based approach: hack a way to get some data about x-y-z coordinates of your character and the enemies, lower the lowest proximity and beat the enemy up, rinse and repeat
- maybe using things like `damage_done` or/combined with `reactions_done`... suboptimal? maybe just let's just give the agent +1 when we win, -1 when we die/run out of time and 0 on non-terminal states ????????? that'd make sense.
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
b) have a variable like WINDOWS_BAR_BOTTOM=True based on which I'd adjust y values. the disadvantage of this is if the var would be in the main python file i'd i couldn't just `git add .` before every commit
"""
from time import sleep, perf_counter
import random
import logging

import numpy as np
from PIL import ImageGrab
import cv2
from paddleocr import PaddleOCR
import pyautogui

LOADING_SCREEN = (502, 338)
EXIT_DOMAIN_BUTTON = (380, 380)
SELECT_TRIAL_VAPORIZE = (152, 128)
START_DOMAIN = (570, 527)


class GenshinEnvironment:
    def __init__(self):
        self.actions = {
            0: self.move_forward,
            1: self.move_backward,
            2: self.move_left,
            3: self.move_right,
            4: self.basic_attack,
            5: self.charged_attack,
            6: self.use_e,
            # 7: self.use_q,
            # 8: self.switch_characters,
        }
        self.stats = {
            # Barbara
            1: {
                'e_cooldown': 32,
                'e_casttime': 0.5
            },
            # Xiangling
            2: {
                'e_cooldown': 12,
                'e_casttime': 0.5
            }
        }

        # These attributes will be reseted each episode
        self.start_time = None
        self.current_char = 1
        self.next_usage_times = {
            self.use_e: 0,
            self.switch_characters: 0
        }
        self.character_dead = False

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
        logging.debug('action: using charged attack')
        pyautogui.mouseDown()
        sleep(0.24)
        pyautogui.mouseUp()

        # Do nothing during cast time
        sleep(0.76)

    def use_e(self):
        logging.debug('action: using e')
        pyautogui.hotkey('e')

        # Do nothing during cast time
        sleep(self.stats[self.current_char]['e_casttime'])

        # Keep track of cooldown
        self.next_usage_times[self.use_e] = perf_counter() + self.stats[self.current_char]['e_cooldown']

    def switch_characters(self):
        if self.current_char == 1:
            logging.debug('action: switching chars 1->2')
            pyautogui.hotkey('2')
            self.current_char = 2
        else:
            logging.debug('action: switching chars 2->1')
            pyautogui.hotkey('1')
            self.current_char = 1

        # Keep track of cooldown (1s)
        self.next_usage_times[self.switch_characters] = perf_counter() + 1

    def is_action_on_cooldown(self, action: int) -> bool:
        action_f = self.actions[action]
        next_usage_time = self.next_usage_times.get(action_f, 0)
        return next_usage_time > perf_counter()

    def random_action(self) -> int:
        """Return the key of a random ready action."""
        ready_actions = [a for a in self.actions if not self.is_action_on_cooldown(a)]
        return random.choice(ready_actions)

    def test_all_actions(self):
        self.move_forward()
        self.move_backward()
        self.move_left()
        self.move_right()
        self.use_e()
        self.use_q()
        self.switch_characters()
        self.basic_attack()
        self.charged_attack()

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
        sleep(4)
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
        self.current_char = 1
        self.next_usage_times = {
            self.use_e: 0,
            self.switch_characters: 0
        }
        self.character_dead = False

        # TODO: Start an asynchronous proces of `update_terminal_states`

        # Start the domain!
        pyautogui.hotkey('f')

        # Start counting time
        self.start_time = perf_counter()

        # Return time spent in the trial
        return 0

    def step(self, action):
        """
        Execute given action and return (next_state, reward, terminated, truncated).

        each of those 4 individually is problematic.

        solutions:
        1) next_state
            perf_counter()
        2) reward
            a synchronous function reading `damage_done` and saving it to some global variable:
                option 1) store everything in one huge np.array
                option 2) only store the newest state and overrite it constantly
            then, read from that variable here
        3) terminated
            - i guess one "solution" is to just hardcode it to always be False until I find a reliable way to track it??
            - i guess that whenever last x `damage_done` readings have an error both chars could be considered dead, but i want terminated=True whenever a SINGLE char dies, not all of them.
            - seems like it needs an extra async function updating some global var like `TERMINATED`
        4) truncated
            - well, i guess i could have a GenshinEnv variable called `start_time` and just each time `step` is called check if 90s have passed since then or nah
        """
        # Execute action
        self.actions[action]()

        VICTORY_ROYALE = False  # TODO: write an async func for that??
        reward = 1 if VICTORY_ROYALE else -1
        terminated = self.character_dead or VICTORY_ROYALE
        next_state = round(perf_counter() - self.start_time)

        return (
            next_state,
            reward,
            terminated,
            next_state > 89
        )

    def update_terminal_states(self):
        """Update `VICTORY_ROYALE` and `CHARACTER_DEAD`.

        approach 1: ss+ocr
        one function for:
        - ss
        - cutting relevant parts
            - health bar
            - part of the screen that will say "you won"
        - ocr on health bar
        - ocr on "you won"

        approach 2: pyautogui.pixel
        - potential locations for CHAR_DEAD
            - green pixel to the left of health bar
            - gray icon of xiangling/barbara
        - potential location for VICTORY_ROYALE
            - maybe 2 pixels each on one button (exit / try again)
        """
        LEFT_END_HEALTHBAR = (271, 528)
        HEALTHBAR_GREEN = (150, 215, 34)
        HEALTHBAR_RED = (255, 90, 90)

        healthbar_color = HEALTHBAR_GREEN
        while True:
            if not pyautogui.pixelMatchesColor(*LEFT_END_HEALTHBAR, healthbar_color, tolerance=41):
                if healthbar_color == HEALTHBAR_RED:
                    self.character_dead = True
                    logging.debug(f'character dead')
                    break

                healthbar_color = HEALTHBAR_RED


class GenshinAgent:
    def __init__(self, state_size, action_size, learning_rate, discount_rate, initial_epsilon, epsilon_decay, final_epsilon):
        """Initialize a RL agent with an empty dict of state-action values (q_values), a learning rate and an epsilon.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.qtable = np.zeros((self.state_size, self.action_size))

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
            return int(np.argmax(self.qtable[state]))

    def update(self, state, action, reward, new_state, terminated):
        """Update Q(s,a):= Q(s,a) + lr*[R(s,a) + discount_rate * max(Q(s',a') - Q(s, a)]"""
        future_q_value = (not terminated) * np.max(self.qtable[new_state])

        delta = (
            reward
            + self.discount_rate * future_q_value
            - self.qtable[state, action]
        )

        # print('--------------------------------------------')
        # print(self.qtable)

        self.qtable[state, action] = (
            self.qtable[state, action] + self.learning_rate * delta
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


def extract_reactions_done(show_image=False):
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
        if 0 <= int(result) <= 15:
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
        format='[%(levelname)s] %(asctime)s - %(message)s'
    )

    ocr = PaddleOCR(
        lang='en',
        rec_char_dict_path='./allowed_chars.txt'
    )

    # hyerparameters
    learning_rate = 0.01
    discount_rate = 0.95
    n_episodes = 10
    initial_epsilon = 1.0
    epsilon_decay = initial_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    # sizes
    state_size = 90  # 90s?
    action_size = 7  # no switch character / q

    # -----------------------------------------------------

    pyautogui.click(*LOADING_SCREEN)

    env = GenshinEnvironment()
    env.update_terminal_states()
    exit()
    agent = GenshinAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        discount_rate=discount_rate,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon
    )

    for episode in range(n_episodes):
        if episode == 0:  # resume_info.next_episode
            state = env.reset(first_episode=True)  # t=0s
        else:
            state = env.reset()  # t=0s

        # Change character to Xiangling
        # env.switch_characters()

        done = False

        # Play one episode
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated = env.step(action)

            agent.update(state, action, reward, next_state, terminated)

            # terminated - either xiangling or barbara is dead OR we won
            # truncated - the trial has run out of time
            done = terminated or truncated
            state = next_state

        agent.decay_epsilon()
