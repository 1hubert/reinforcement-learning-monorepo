"""Save a np.array to disc as a file and load the file later.
Experimenting to find a best way to store a Q-Matrix and load it later.

I also need to save the current episode, so maybe I could do smth like:
np.array([current_episode, q_matrix])

Also, I Need a way to kind of "soft-quit" my script:
- listen to key presses anywhere, similar how `genshin-dialogue-autoskip` is doing
- when i click `f12`, soft-quit the script:
- wait for the current episode to end, then store everything useful in a file for later. there would be a way to later "resume" the training.
"""
import numpy as np

q_matrix = np.zeros((90, 7))
q_matrix[0, 1] = 420
q_matrix[1, 2] = 420
q_matrix[2, 5] = 420
q_matrix[3,:] = 1

def save(last_episode, q_matrix):
    obj = np.empty(2, dtype='object')
    obj[0] = last_episode + 1
    obj[1] = q_matrix
    np.save('./next_episode+qmatrix.npy', obj)
    print('saved!')

def load_and_print():
    loaded = np.load('./next_episode+qmatrix.npy', allow_pickle=True)
    print(f'next_episode: {loaded[0]}')
    print(f'next_episode type: {type(loaded[0])}')
    print()
    print(f'qmatrix: {loaded[1]}')
    print(f'qmatrix type: {type(loaded[1])}')

# save(10, q_matrix)
load_and_print()
