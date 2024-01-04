"""How do I make the `check_color` method run in the background?"""

from time import sleep

class GenshinEnv:
    def run_left(self):
        print('running left...')

    @staticmethod
    def check_color():
        while True:
            print('checking color')
            sleep(0.2)


if __name__ == '__main__':
    env = GenshinEnv()
    env.check_color()
    env.run_left()
    env.run_left()
    env.run_left()
    env.run_left()
    env.run_left()
