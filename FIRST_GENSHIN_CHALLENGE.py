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

reset będzie trochę zajmował:
- quit domain
- wait a second + until a certain location near middle stops being white
- hold 's' for ~2s -> click 'f'
- click right bottom corner to start
- wait a second + until a certain location near middle stops being white
- hold 'w' for some predefined amount of time
- click 'f' to start a domain (a new episode)
(unfotunetely this might be the way to do it in the beginning as this is the easiest way to guarantee reprodubility of actions -> results)

zamiast wybierać postacie klawiszami 1, 2, bot będzie miał akcję 'change char'.
obwódka wokół numerku aktywnej postaci != RGB(255, 255, 255)
obwódka wokół numerku nieaktywnej postaci == RGB(255, 255, 255)

domena którą będę się zajmował to pierwsza domena z reakcjami gdzie testujemy vaporize z barbarą i xiangling.

co do śledzonych liczników:
- są dwa: reactions_done i dmg_done
- na początku można przetestować śledzenie obydwu osobno w dwóch różnych treningach
- później można spróbować zrobić np. jakieś równanie typu:
    reactions_done% * dmg_done%
- może co każde 20 sekund gdzie nie podniósł maksymalizowanej wartości dostaje -1 punkt? albo co większą wartość cooldownu e + kilka sekund, bo chciałbym żeby na początku naucył się po prostu używania e na obydwu postaciach

Vaporize Reactions Triggered: 0/15
DMG Dealt to Monsters: 0/14000
"""
import time

import numpy as np
from PIL import ImageGrab
import cv2
from paddleocr import PaddleOCR

def process_image(image):
    """Dumb the original image down using OpenCV (Open Source Computer Vision Library).

    :image: A numpy array with pixels in BGR.
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def extract_damage_done(image):
    result = ocr.ocr(image, det=False, cls=False)[0][0]
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
    ocr = PaddleOCR(
        lang='en',
        rec_char_dict_path='./allowed_chars.txt'
    )
    main()
