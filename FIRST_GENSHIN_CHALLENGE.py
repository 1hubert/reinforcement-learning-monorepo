"""
- jak jest ta domena w mondstadt z "test runami reakcji", są w niej liczniki reakcji / damage'a. generalnie w genshinie nie ma za dużo takich miejsc z widocznymi cały czas licznikami. mógłbym zbudować skrypt minimalizujący czas przechodzenia tego "reaction tutoriala" z następującym inputem/outputem:
	- input
		- licznik reakcji w aktualnym epizodzie
	- output
		- akcja: zresetuj epizod
		- a w s d
		- e q
		- LPM

przeciwnicy w tych reaction tutorialach są (potwierdzone) w tych samych miejscach więc mogłoby się okazać, że tyle wystarczy by pobić mój wynik przynajmniej w jednej malutkiej części genshina (i mało przydatnej do zautomatyzowania), ale i tak byłby to zauważalny postęp ku "generalnym ai do zastąpienia ludzkich graczy".

domena którą będę się zajmował to pierwsza domena z reakcjami gdzie testujemy vaporize z barbarą i xiangling.

co do śledzonych liczników:
- są dwa: reactions_done i dmg_done
- na początku można przetestować śledzenie obydwu osobno w dwóch różnych treningach
- później można spróbować zrobić np. jakieś równanie typu:
    reactions_done% * dmg_done%
- może co każde 20 sekund gdzie nie podniósł maksymalizowanej wartości dostaje -1 punkt? albo co większą wartość cooldownu e + kilka sekund, bo chciałbym żeby na początku naucył się po prostu używania e na obydwu postaciach

Vaporize Reactions Triggered: 0/15
DMG Dealt to Monsters: 0/14000

barbara - postać 1 (default)
xiangling - postać 2
"""
import time
import random

import numpy as np
from PIL import ImageGrab
import cv2
from paddleocr import PaddleOCR
import pyautogui

LOADING_SCREEN = (502, 338)
EXIT_DOMAIN_BUTTON = (380, 380)
SELECT_TRIAL_VAPORIZE = (152, 128)
START_DOMAIN = (570, 527)


class GenshinAgent:
    def __init__(self):
        self.current_char = 1
        self.actions = [
            self.move_forward,
            self.move_backward,
            self.move_left,
            self.move_right,
            self.use_e,
            # self.switch_characters,
            self.basic_attack,
            self.charged_attack
        ]
        self.next_usage_times = {
            self.use_e: 0,
            self.switch_characters: 0
        }
        self.stats = {
            1: {
                'e_cooldown': 32,
                'e_casttime': 0.5
            },
            2: {
                'e_cooldown': 12,
                'e_casttime': 0.5
            }
        }

    def move_forward(self):
        print('action: moving forward')
        with pyautogui.hold('w'):
            time.sleep(1)

    def move_backward(self):
        print('action: moving backward')
        with pyautogui.hold('s'):
            time.sleep(1)

    def move_left(self):
        print('action: moving left')
        with pyautogui.hold('a'):
            time.sleep(1)

    def move_right(self):
        print('action: moving right')
        with pyautogui.hold('d'):
            time.sleep(1)

    def use_e(self):
        print('action: using e')
        pyautogui.hotkey('e')

        # Do nothing during cast time
        time.sleep(self.stats[self.current_char]['e_casttime'])

        # Keep track of cooldown
        self.next_usage_times[self.use_e] = time.perf_counter() + self.stats[self.current_char]['e_cooldown']

    def switch_characters(self):
        if self.current_char == 1:
            print('action: switching chars 1->2')
            pyautogui.hotkey('2')
            self.current_char = 2
        else:
            print('action: switching chars 2->1')
            pyautogui.hotkey('1')
            self.current_char = 1

        # Keep track of cooldown (1s)
        self.next_usage_times[self.switch_characters] = time.perf_counter() + 1

    def basic_attack(self):
        print('action: using basic attack')
        pyautogui.click()

    def charged_attack(self):
        print('action: using charged attack')
        pyautogui.mouseDown()
        time.sleep(0.24)
        pyautogui.mouseUp()

    def action_on_cooldown(self, action):
        next_usage_time = self.next_usage_times.get(action, 0)
        return  next_usage_time > time.perf_counter()

    def get_ready_action(self):
        ready_actions = [a for a in self.actions if not self.action_on_cooldown(a)]
        if len(ready_actions) > 0:
            return random.choice(ready_actions)

    def take_action(self):
        action = self.get_ready_action()
        if action:
            action()
        else:
            print("All abilities on cooldown")

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

def hard_reset_env(first_episode=False):
    """
    Basically quit domain, re-enter, run up the stairs and click 'f' near the key to start a new episode with same initial environment.

    Unfotunetely, this might be the way to do it in the MVP as this is the easiest way to guarantee reproducibility of mapping actions to results.

    About the first episode:
    - The human preparation required is to find the reaction domain on a map and click 'teleport'.
    - to start the first episode, run this function with `first_episode` keyword argument set to True.
    """
    def wait_until_loading_screen_gone(delay):
        time.sleep(delay)
        loading_screen_color = pyautogui.pixel(*LOADING_SCREEN)
        while pyautogui.pixel(*LOADING_SCREEN) == loading_screen_color:
            time.sleep(0.5)

    if first_episode == False:
        # Quit domain
        pyautogui.hotkey('esc')
        time.sleep(random.random())
        pyautogui.click(*EXIT_DOMAIN_BUTTON)

        # Wait until the open world loads
        wait_until_loading_screen_gone(4)

        # Run up to the domain
        with pyautogui.hold('s'):
            time.sleep(2.4 + (random.random() / 5))
    else:
        # Run up to the domain
        with pyautogui.hold('w'):
            time.sleep(2 + (random.random() / 5))

    # Open domain and run it
    pyautogui.hotkey('f')
    time.sleep(4)
    pyautogui.click(*SELECT_TRIAL_VAPORIZE)
    pyautogui.click(*START_DOMAIN)

    # Wait until domain loads
    wait_until_loading_screen_gone(1)

    # Click through the reaction tutorial
    for _ in range(3):
        time.sleep(1.1 + random.random())
        pyautogui.click(*LOADING_SCREEN)
    time.sleep(0.1 + random.random() / 5)

    # Run up to the key (this cannot be randomized)
    w_time_1 = random.uniform(0.5, 6)
    w_time_2 = 7.4 - w_time_1
    with pyautogui.hold('w'):
        time.sleep(w_time_1)
        pyautogui.keyDown('shiftleft')
        pyautogui.keyUp('shiftleft')
        time.sleep(w_time_2)

    # Start the domain!
    pyautogui.hotkey('f')


def process_image(image):
    """Dumb the original image down using OpenCV (Open Source Computer Vision Library).

    :image: A numpy array with pixels in BGR.
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image


def extract_damage_done(image):
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
        # Grab image
        screen = ImageGrab.grab(bbox=(110, 180, 180, 188))

        # Process image
        # image = process_image(np.array(screen))

        # Show image
        # cv2.imshow('window', image)

        # Extract damage done from image
        print(extract_damage_done(np.array(screen)))

        # ord(q) == 113
        # if cv2.waitKey(1) == 113:
        #     cv2.destroyAllWindows()
        #     break


if __name__ == '__main__':
    # ocr = PaddleOCR(
    #     lang='en',
    #     rec_char_dict_path='./allowed_chars.txt'
    # )
    # main()

    # Focus on genshin's window
    pyautogui.click(*LOADING_SCREEN)
    time.sleep(1)

    # hard_reset_env(first_episode=True)

    agent = GenshinAgent()
    while True:
        agent.take_action()
