import time
import multiprocessing
from paddleocr import PaddleOCR
import numpy as np
import cv2
from PIL import ImageGrab


ocr = PaddleOCR(
    lang='en',
    rec_char_dict_path='./allowed_chars.txt'
)

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

class PixelChecker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.reactions_done = multiprocessing.Value('i', 0)

        self.extract_reactions_done_process = multiprocessing.Process(
            target=extract_reactions_done,
            args=[self.reactions_done]
        )
        self.extract_reactions_done_process.start()

    def step(self):
        print(self.reactions_done.value)


if __name__ == '__main__':

    pc = PixelChecker()
    while True:
        time.sleep(0.5)
        pc.step()
