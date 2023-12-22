import pyautogui, time, random


time.sleep(1)
w_time_1 = random.uniform(0.5, 6)
w_time_2 = 8.9 - w_time_1
with pyautogui.hold('w'):
    time.sleep(w_time_1)
    pyautogui.keyDown('shiftleft')
    pyautogui.keyUp('shiftleft')
    time.sleep(w_time_2)

