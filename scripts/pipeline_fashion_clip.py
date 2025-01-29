import os
import torch
from src.models.models import load_clip_model
from src.data.preprocess import load_image_paths, preprocess_image
from src.data.embeddings import generate_embeddings
from src.data.matcher import compute_similarity, calculate_accuracy
from src.visualization.display import display_predictions

# Load model
model, processor = load_clip_model()

# Load images
reference_images = load_image_paths(os.path.join("data", "DAM"))
test_images = load_image_paths(os.path.join("data", "test_image_headmind"))

# Generate embeddings
reference_features = generate_embeddings(reference_images, model, processor)
test_features = generate_embeddings(test_images, model, processor)

# Compute similarities
reference_tensor = torch.stack(reference_features)
test_tensor = torch.stack(test_features)
cosine_similarities = compute_similarity(test_tensor, reference_tensor)

# Find best matches
closest_indices = torch.argsort(cosine_similarities, dim=1, descending=True)[:, :11]

# Display predictions
guesses = {f"test_{i}": [reference_images[idx] for idx in indices] for i, indices in enumerate(closest_indices)}
display_predictions(test_images, guesses)
