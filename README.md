# One-Shot Learning for Fashion Item Classification and Retrieval

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Approach and Methodology](#approach-and-methodology)
- [Pipeline Workflow](#pipeline-workflow)
- [Results](#results)
- [Usage](#usage)

## Project Overview
This project focuses on a challenging **one-shot learning problem** in fashion item classification and retrieval. The objective is to match a single reference image per class from a **DAM training dataset** with diverse test images, despite differences in backgrounds, contexts, and real-world complexity. 

## Dataset Description
The dataset comprises two components:
1. **DAM Training Dataset**:
   - **Characteristics**: Limited to one image per class, mostly featuring uniform white backgrounds.
   - **Challenges**: Sparse data with minimal intra-class variation.

2. **Test Image Dataset**:
   - **Characteristics**: Contains images with diverse backgrounds, including different locations and lighting conditions.
   - **Challenges**: High variability, mimicking real-world scenarios.

This setup creates a **one-shot learning challenge**, where learning is performed using just one example per class.

## Approach and Methodology

### Preprocessing
- **Background Removal**: Standardized image conditions by removing complex backgrounds.
- **Category Classification**: Segmentation of images into high-level categories (e.g., shoes, bags, clothes).

### Feature Extraction and Data Augmentation
- Utilized **CNN-based models (e.g., ResNet variants)** for initial feature extraction.
- Incorporated **FashionCLIP**, a CLIP-based model fine-tuned for semantic understanding of fashion items.

### Model
The final pipeline integrates the **FashionCLIP model** to extract embeddings that capture both visual and semantic features of the items. This model ranks test images based on **cosine similarity** to the reference images.

### Evaluation
- **Manual Matching**: Each test image was manually matched to corresponding DAM reference images for ground truth.
- **Retrieval Metrics**:
  - **Top-10 Accuracy**: A retrieval is correct if the ground truth reference is in the top 10 predictions.
  - **Cosine Similarity Score**: Evaluates how well the predicted embeddings align with the ground truth embeddings.

## Pipeline Workflow
1. Extract embeddings from reference and test images using **FashionCLIP**.
2. Compute pairwise cosine similarity between reference and test image embeddings.
3. Rank test images based on similarity and retrieve the top 10 closest matches for each test image.
4. Evaluate performance using the established metrics.

## Results
to complete

## Usage
### Requirements
- Python 3.9+
- Required Python Libraries:
  - `torch`
  - `transformers`
  - `Pillow`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `tq

### Steps to Run
1. **Preprocessing**:
   - Run the script to remove backgrounds and categorize images:
     ```bash
     python preprocess_images.py
     ```

2. **Feature Extraction**:
   - Extract embeddings for both training and test datasets using the **FashionCLIP model**:
     ```bash
     python extract_embeddings.py --model FashionCLIP --dataset <dataset_path>
     ```

3. **Similarity Retrieval**:
   - Compute cosine similarities and retrieve the top-10 most similar matches:
     ```bash
     python retrieve_matches.py --embeddings <embeddings_path> --output <output_path>
     ```

4. **Evaluation**:
   - Evaluate the performance of the model using the ground truth matches:
     ```bash
     python evaluate.py --results <retrieved_results_path> --ground_truth <ground_truth_path>
     ```


---

Thank you for exploring our project! 
