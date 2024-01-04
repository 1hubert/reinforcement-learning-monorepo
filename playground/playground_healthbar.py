import pyautogui

CHAR_HEALTH = 'green' # 'green'/'dead'/'dead'

GREEN_TRESHOLD = 150
LEFT_END_HEALTHBAR = (271, 528)
HEALTHBAR_GREEN = (150, 215, 34)
HEALTHBAR_RED = (255, 90, 90)



def if_i_had_to_choose_one():
    x_list = []
    y_list = []
    for x in range(LEFT_END_HEALTHBAR[0]-10, LEFT_END_HEALTHBAR[0]+3):
        for y in range(LEFT_END_HEALTHBAR[1]-5, LEFT_END_HEALTHBAR[1]+3):
            c = pyautogui.pixel(x, y)
            if c[1] >= GREEN_TRESHOLD:
                x_list.append(x)
                y_list.append(y)
                print(f'rgb: {c}\tx: {x}\ty: {y}')

    print(f'avg x: {sum(x_list) / len(x_list)}')
    print(f'avg y: {sum(y_list) / len(y_list)}')

    # the chosen one:
    # rgb: (150, 215, 34)     x: 271  y: 527

if __name__ == '__main__':
    healthbar_color = HEALTHBAR_GREEN
    while True:
        print(CHAR_HEALTH)
        c = pyautogui.pixel(*LEFT_END_HEALTHBAR)

        if not pyautogui.pixelMatchesColor(*LEFT_END_HEALTHBAR, healthbar_color, tolerance=41):
            if CHAR_HEALTH == 'red':
                CHAR_HEALTH = 'dead'
                print('we dead :(')
                print(f'c: {c}')
                break

            healthbar_color = HEALTHBAR_RED
            CHAR_HEALTH = 'red'


    # if_i_had_to_choose_one()
