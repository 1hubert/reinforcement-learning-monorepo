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
"""
import time
from threading import Thread

import numpy as np
from PIL import ImageGrab
import cv2
exit_key_pressed = False

def main():
    last_time = time.perf_counter()

    while True:
        screen = ImageGrab.grab(bbox=(0, 70, 640, 550))

        print(f'Loop took {time.perf_counter() - last_time} seconds')
        last_time = time.perf_counter()

        cv2.imshow('window', cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2RGB))

        # ord(q) == 113
        if cv2.waitKey(1) == 113:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
