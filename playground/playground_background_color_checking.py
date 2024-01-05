"""How do I make the `check_color` method run in the background?"""

import time
import threading

from playground_victory_detection import check_victory

class GenshinEnv:
    def run_left(self):
        print('running left...')

    @staticmethod
    def check_color():
        while True:
            print('checking color')
            time.sleep(0.2)

class TestThreading:
    def __init__(self, interval=1):
        self.interval = interval

        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start()

    def run(self):
        while True:
            # More statements comes here
            print(f'{time.time()}: Running in the background')

            time.sleep(self.interval)

if __name__ == '__main__':
    tr = TestThreading()
    time.sleep(1)
    print(f'{time.time()}: First output')

    time.sleep(2)
    print(f'{time.time()}: Second output')

    # env = GenshinEnv()
    # env.check_color()
    # env.run_left()
    # env.run_left()
    # env.run_left()
    # env.run_left()
    # env.run_left()
