import pyautogui

def check_victory():
    LETTER_C_1 = (182, 279)
    LETTER_C_2 = (320, 279)
    LETTER_D = (454, 278)
    WHITE = (255, 255, 255)

    while True:
        if (pyautogui.pixelMatchesColor(
            *LETTER_C_1, WHITE) and
            pyautogui.pixelMatchesColor(
            *LETTER_C_2, WHITE) and
            pyautogui.pixelMatchesColor(
            *LETTER_D, WHITE)):

            print('VICTORY DETECTED')
            break
        else:
            print('not detected')

if __name__ == '__main__':
    check_victory()
