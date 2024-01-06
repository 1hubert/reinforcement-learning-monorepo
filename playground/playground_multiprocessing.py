import random
import multiprocessing
import threading
import queue


import multiprocessing
import threading
import queue

class PixelChecker:
    def __init__(self):
        self.white_detected = False
        self.queue = queue.Queue()
        self.color_check_lock = threading.RLock()
        self.process_lock = threading.RLock()
        self.color_check_process = None

    def start_color_check(self):
        # Create a background process for color checking
        self.process_lock.acquire()
        try:
            # Create a unique lock object for the process
            self.process_lock_instance = threading.RLock()
        finally:
            self.process_lock.release()

        self.color_check_process = multiprocessing.Process(target=self.check_color, args=(self.queue, self.process_lock_instance))
        self.color_check_process.daemon = True  # Set daemon to True to make the process exit when the main process exits
        self.color_check_process.start()

        active_children = multiprocessing.active_children()
        if self.color_check_process not in active_children:
            print("Child process already terminated")
        else:
            print("Child process is running")

    def check_color(self, queue, process_lock_instance):
        while True:
            with process_lock_instance:
                with self.color_check_lock:
                    if random.randint(1, 99)>96:
                        print('found!')
                        queue.put(True)
                    else:
                        print('not foudn ')
    def do_other_stuff(self):
        # Other methods that rely on self.white_detected
        while True:
            if self.queue.get() is True:
                # White detected, proceed with other tasks
                break

            # Perform other tasks



if __name__ == '__main__':
    pc = PixelChecker()
    pc.start_color_check()
