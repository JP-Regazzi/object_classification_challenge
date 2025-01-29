import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from src.models.models import load_models
from src.data.preprocess import get_transforms
from src.data.embeddings import extract_hybrid_embeddings
from src.data.matcher import match_images, match_with_metrics

# Load models and layers
resnet_model, vgg_model, inception_model, resnet_layer, vgg_layer, inception_layer = load_models()

# Get image transforms
transform_224, transform_299 = get_transforms()

# Define image folders
folderA = "./data/DAM_white_background"
folderB = "./data/preprocessed_test"

# List images
imagesA = [os.path.join(folderA, fn) for fn in os.listdir(folderA) if fn.lower().endswith(('.png', '.jpg', '.jpeg'))]
imagesB = [os.path.join(folderB, fn) for fn in os.listdir(folderB) if fn.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Extract embeddings
embeddingsA = [extract_hybrid_embeddings(path, resnet_model, vgg_model, inception_model, resnet_layer, vgg_layer, inception_layer, transform_224, transform_299).unsqueeze(0) for path in imagesA]
embeddingsA = torch.cat(embeddingsA, dim=0)

embeddingsB = [extract_hybrid_embeddings(path, resnet_model, vgg_model, inception_model, resnet_layer, vgg_layer, inception_layer, transform_224, transform_299).unsqueeze(0) for path in imagesB]
embeddingsB = torch.cat(embeddingsB, dim=0)

# Match images
best_matches = match_images(imagesA, imagesB, embeddingsA, embeddingsB)

# Optional: Match with metrics (MSE, MAE)
best_matches_with_metrics = match_with_metrics(imagesA, imagesB, embeddingsA, embeddingsB)

# Display results
for b_idx, a_idx, score in best_matches[:5]:
    print(f"[B: {imagesB[b_idx]}] -> [A: {imagesA[a_idx]}], score={score:.4f}")
