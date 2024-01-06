"""How do I make the `check_color` method run in the background?"""

import time
import multiprocessing

import pyautogui

def check_color(event):
    LOC = (50, 50)
    WHITE = (255, 255, 255)

    while True:
        time.sleep(0.2)
        if pyautogui.pixelMatchesColor(*LOC, WHITE):
            print('White detected, event is now True')
            event.set()
            break
        else:
            print('not detected')

class PixelChecker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.event = multiprocessing.Event()

        self.color_check_process = multiprocessing.Process(
            target=check_color,
            args=(self.event,)
        )
        self.color_check_process.start()

    def step(self):
        if self.event.is_set():
            print("step: white detected!")
            print('step: resetting in 3s......')
            time.sleep(3)
            print('resetting!')
            self.reset()

        else:
            print('step: white not detected yet')


if __name__ == '__main__':
    pc = PixelChecker()
    while True:
        time.sleep(0.5)
        pc.step()
