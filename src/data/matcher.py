import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import matplotlib.pyplot as plt
from evaluation.metrics import compute_mse, compute_mae


def load_cv2_image(path):
    """
    Load an image using OpenCV and convert it to RGB format.
    """
    img_cv2 = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return img_rgb

def match_images(imagesA, imagesB, embeddingsA, embeddingsB):
    """
    Match each image in B with the best match from A based on cosine similarity.
    """
    npA = embeddingsA.numpy()
    npB = embeddingsB.numpy()

    best_matches = []
    for b_idx in range(len(imagesB)):
        b_emb = npB[b_idx].reshape(1, -1)        # shape: (1, 29184)
        sim_scores = cosine_similarity(b_emb, npA)[0]  # shape: (num_imagesA,)
        max_idx = np.argmax(sim_scores)
        max_score = sim_scores[max_idx]
        best_matches.append((b_idx, max_idx, max_score))

    return best_matches

def match_with_metrics(imagesA, imagesB, embeddingsA, embeddingsB):
    """
    Match images with additional metrics (Cosine Similarity, MSE, MAE).
    """
    npA = embeddingsA.numpy()
    npB = embeddingsB.numpy()

    best_matches_with_metrics = []
    for b_idx in range(len(imagesB)):
        b_emb = npB[b_idx].reshape(1, -1)  # shape: (1, 29184)
        sim_scores = cosine_similarity(b_emb, npA)[0]  # shape: (num_imagesA,)

        max_idx = np.argmax(sim_scores)  # Best match index
        max_score = sim_scores[max_idx]  # Cosine similarity score
        
        # Calculate MSE and MAE for the best match
        mse = compute_mse(npB[b_idx], npA[max_idx])
        mae = compute_mae(npB[b_idx], npA[max_idx])

        # Append the results
        best_matches_with_metrics.append((b_idx, max_idx, max_score, mse, mae))

    return best_matches_with_metrics

def compute_similarity(client_embeddings, reference_embeddings):
    """
    Compute cosine similarity between client and reference embeddings.

    Parameters:
        client_embeddings (torch.Tensor): Tensor of client embeddings.
        reference_embeddings (torch.Tensor): Tensor of reference embeddings.

    Returns:
        torch.Tensor: Matrix of cosine similarities.
    """
    return torch.mm(client_embeddings, reference_embeddings.t())

def calculate_accuracy(answers, guesses):
    """
    Calculate accuracy based on provided answers and guesses.

    Parameters:
        answers (dict): Ground-truth answers.
        guesses (dict): Predicted guesses.

    Returns:
        float: Accuracy percentage.
    """
    accuracy = 0
    nb_guess = 0
    for answer_key in answers.keys():
        found = 0
        if answer_key in guesses.keys():
            nb_guess += 1
            for value_guess in guesses[answer_key]:
                for value_answ in answers[answer_key]:
                    if value_answ in value_guess and not found:
                        accuracy += 1
                        found = 1
    return accuracy / nb_guess * 100
