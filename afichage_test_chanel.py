import numpy as np
import pandas as pd


def one_hot_encode(number):
    """convertire un nbr (0-4)vecteur 4 bit on hote encode ."""
    one_hot = np.zeros(4, dtype=int)
    if number > 0:
        one_hot[number - 1] = 1
    return one_hot


def prepare_4_channel_grid(data_row):
    """le chanel a 4"""
    grid = np.array(data_row).reshape(4, 4)  # Reshape the flat list into a 4x4 grid
    channels = np.zeros((4, 4, 4), dtype=int)  # Prepare a 4x4x4 tensor (4 channels)

    for i in range(4):
        for j in range(4):
            channels[:, i, j] = one_hot_encode(grid[i, j])

    return channels


def afichage_chanel(sudoku_grid):
    """jaffiche la grille et les 4 shanel."""
    la_grille_encodeer = prepare_4_channel_grid(sudoku_grid)

    print("Original Sudoku grille (4x4):")
    print(np.array(sudoku_grid).reshape(4, 4), "\n")

    print("Channel 1 (presence des 1 ):")
    print(la_grille_encodeer[0], "\n")
    print("Channel 2 (presence de 2):")
    print(la_grille_encodeer[1], "\n")
    print("Channel 3 (presence de 3):")
    print(la_grille_encodeer[2], "\n")
    print("Channel 4 (presence de  4 ):")
    print(la_grille_encodeer[3], "\n")


# un exemple de sudoko
sudoku_example = [2, 3, 1, 4, 1, 4, 2, 3, 3, 1, 4, 2, 4, 2, 3, 1]
afichage_chanel(sudoku_example)
