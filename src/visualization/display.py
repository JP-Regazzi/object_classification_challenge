import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

def display_predictions(img_list, guesses, top_k=10):
    """
    Display predictions for the first few images.

    Parameters:
        img_list (list): List of test image paths.
        guesses (dict): Dictionary of predicted guesses.
        top_k (int): Number of top guesses to display.
    """
    for i, img_path in enumerate(img_list):
        if i >= 2:  # Limitar a exibição
            break
        file_name = img_path.split("\\")[-1]
        img_id = file_name.split(".")[0]
        guess_list = guesses[img_id]

        fig, axs = plt.subplots(2, 6, figsize=(20, 5))
        axs[0, 0].imshow(np.swapaxes(mpimg.imread(img_path), 0, 1))
        for j in range(1, top_k + 1):
            axs[int(j / 6), int(j % 6)].imshow(mpimg.imread(guess_list[j - 1]))
        plt.show()
