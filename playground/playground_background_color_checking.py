"""How do I make the `check_color` method run in the background?"""

import time
import multiprocessing
import queue

import pyautogui

def check_color(queue):
    LOC = (50, 50)
    WHITE = (255, 255, 255)

    while True:
        time.sleep(0.2)
        if pyautogui.pixelMatchesColor(*LOC, WHITE):
            print('White detected, putting True to queue')
            queue.put(True)
            break
        else:
            print('not detected')

class PixelChecker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.queue = multiprocessing.Queue()

        self.color_check_process = multiprocessing.Process(
            target=check_color,
            args=(self.queue,)
        )
        self.color_check_process.start()

    def step(self):
        try:
            if self.queue.get(block=False) is True:
                print('step: White detected!')
        except queue.Empty:
            print('step: white not detected')

if __name__ == '__main__':
    pc = PixelChecker()
    while True:
        time.sleep(0.5)
        pc.step()
