import numpy as np

def compute_mse(embedding1, embedding2):
    """
    Calculate Mean Squared Error (MSE) between two embeddings.
    """
    return np.mean((embedding1 - embedding2) ** 2)

def compute_mae(embedding1, embedding2):
    """
    Calculate Mean Absolute Error (MAE) between two embeddings.
    """
    return np.mean(np.abs(embedding1 - embedding2))