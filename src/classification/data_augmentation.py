from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from typing import Callable, List, Dict, Any, Union
import argparse
from pathlib import Path
import pandas as pd



class DataAugmentation:
    """
    Modular class to perform data augmentation on images.
    """
    def __init__(self, augmentations: List[Dict[str, Any]], seed: int = 42):
        """
        Initialize the augmentation processor with a list of augmentations.

        Args:
            augmentations (List[Dict[str, Any]]): List of augmentations with function and arguments.
            seed (int): Random seed for deterministic augmentations.
        """
        self.augmentations = augmentations
        self.seed = seed
        random.seed(self.seed)

    def augment(self, image: Image.Image) -> List[Image.Image]:
        """
        Apply all augmentations to the image and return augmented images.

        Args:
            image (Image.Image): Input image.

        Returns:
            List[Image.Image]: List of augmented images.
        """
        augmented_images = []
        for augmentation in self.augmentations:
            func = augmentation["func"]
            args = augmentation.get("args", ())
            kwargs = augmentation.get("kwargs", {})
            augmented_images.extend(func(image, *args, **kwargs))
        return augmented_images


def rotate_image(image: Image.Image, angles=[-30, -15, 15, 30]) -> List[Image.Image]:
    """Rotate image by specified angles."""
    rotated_images = []
    for angle in angles:
        rotated = image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor = (255,255,255))
        # rotated = fill_background(rotated)
        rotated_images.append(rotated)
    return rotated_images


def translate_image(image: Image.Image, shifts=[(-50, -50), (50, 50), (-60, 30), (30, -60)]) -> List[Image.Image]:
    """Translate image by specified shifts."""
    translated_images = []
    for x_shift, y_shift in shifts:
        translation_matrix = (1, 0, x_shift, 0, 1, y_shift)
        translated = image.transform(
            image.size, Image.AFFINE, translation_matrix, resample=Image.BICUBIC, fillcolor = (255,255,255)
        )
        translated_images.append(translated)
    return translated_images



def scale_image(image: Image.Image, scales=[0.6, 1.4]) -> List[Image.Image]:
    """Scale image by specified factors."""
    scaled_images = []
    for scale in scales:
        new_size = (int(image.width * scale), int(image.height * scale))
        scaled = image.resize(new_size, resample=Image.BICUBIC)

        # Paste the scaled image onto a blank canvas of original size
        canvas = Image.new("RGB", image.size, (255, 255, 255))
        offset = (
            (canvas.width - scaled.width) // 2,
            (canvas.height - scaled.height) // 2
        )
        canvas.paste(scaled, offset)
        scaled_images.append(canvas)
    return scaled_images




def perspective_transform(image: Image.Image) -> List[Image.Image]:
    """Apply random perspective transformations."""
    width, height = image.size
    shift = 50
    original_points = np.float32([
        [0, 0], [width, 0], [0, height], [width, height]
    ])
    transformed_images = []
    for _ in range(5):
        transformed_points = np.float32([
            [np.random.randint(0, shift), np.random.randint(0, shift)],
            [width - np.random.randint(0, shift), np.random.randint(0, shift)],
            [np.random.randint(0, shift), height - np.random.randint(0, shift)],
            [width - np.random.randint(0, shift), height - np.random.randint(0, shift)]
        ])
        coeffs = np.linalg.inv(
            np.vstack([transformed_points.T, [1, 1, 1, 1]]) @ np.linalg.pinv(
                np.vstack([original_points.T, [1, 1, 1, 1]])
            )
        ).flatten()
        transformed = image.transform(image.size, Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC, fillcolor = (255,255,255))
        transformed_images.append(transformed)

    return transformed_images

def adjust_brightness(image: Image.Image, factors=[0.7, 1.5]) -> List[Image.Image]:
    """Adjust brightness of the image by specified factors."""
    brightness_images = []
    for factor in factors:
        enhancer = ImageEnhance.Brightness(image)
        brightness_images.append(enhancer.enhance(factor))
    return brightness_images


def adjust_contrast(image: Image.Image, factors=[0.7, 1.5]) -> List[Image.Image]:
    """Adjust contrast of the image by specified factors."""
    contrast_images = []
    for factor in factors:
        enhancer = ImageEnhance.Contrast(image)
        contrast_images.append(enhancer.enhance(factor))
    return contrast_images


def adjust_saturation(image: Image.Image, factors=[0.7, 1.5]) -> List[Image.Image]:
    """Adjust saturation of the image by specified factors."""
    saturation_images = []
    for factor in factors:
        enhancer = ImageEnhance.Color(image)
        saturation_images.append(enhancer.enhance(factor))
    return saturation_images


def add_noise(image: Image.Image, noise_levels=[10, 30]) -> List[Image.Image]:
    """Add random noise to the image."""
    noise_images = []
    np_image = np.array(image)
    for noise_level in noise_levels:
        noise = np.random.randint(-noise_level, noise_level, np_image.shape, dtype=np.int16)
        noisy_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
        noise_images.append(Image.fromarray(noisy_image))
    return noise_images


def process_images_with_augmentation(
    input_folder: str,
    output_folder: str,
    augmenter: DataAugmentation,
    csv_file: str,
    output_csv: str
):
    """
    Apply data augmentation to all images in a folder and save them, updating the CSV.

    Args:
        input_folder (str): Path to the input directory.
        output_folder (str): Path to the output directory.
        augmenter (DataAugmentation): Instance of the data augmenter.
        csv_file (str): Path to the input CSV file with image names and classes.
        output_csv (str): Path to save the updated CSV file.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Create the output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the CSV
    df = pd.read_csv(csv_file)

    # Ensure the CSV has the correct columns
    if not all(col in df.columns for col in ["image", "class"]):
        raise ValueError("The CSV file must contain 'image' and 'class' columns.")

    augmented_data = []

    # Iterate through all rows in the CSV
    for _, row in df.iterrows():
        image_name = row["image"] + ".jpeg"
        image_class = row["class"]

        image_file = input_path / image_name

        if image_file.exists():
            try:
                # Load the image
                image = Image.open(image_file)
                image.save(output_path / image_name)
                # Apply augmentations
                augmented_images = augmenter.augment(image)

                # Save each augmented image
                for i, augmented_image in enumerate(augmented_images):
                    augmented_name = f"{image_file.stem}_aug_{i}{image_file.suffix}"
                    output_file = output_path / augmented_name
                    augmented_image.save(output_file)

                    # Append the augmented image and class to the list
                    augmented_data.append({"image": augmented_name, "class": image_class})
                augmented_data.append({"image": image_name, "class": image_class})

                print(f"Processed and saved augmentations for: {image_name}")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        else:
            print(f"Image file not found: {image_file}")

    # Create a DataFrame for the augmented data
    augmented_df = pd.DataFrame(augmented_data)

    # Save the updated CSV
    augmented_df.to_csv(output_csv, index=False)



if __name__ == "__main__":
    # Configure argparse for command-line arguments
    parser = argparse.ArgumentParser(description="Modular data augmentation for images.")
    parser.add_argument("input_folder", type=str, help="Path to the input directory.")
    parser.add_argument("output_folder", type=str, help="Path to the output directory.")
    parser.add_argument("--rotate", type=str, help="Angles for rotation (e.g., -30,-15,15,30).")
    parser.add_argument("--contrast", type=str, help="Contrast factors (e.g., 0.5,1.5,2.0).")
    parser.add_argument("--saturation", type=str, help="Saturation factors (e.g., 0.5,1.5,2.0).")
    parser.add_argument("--scale", type=str, help="Scaling factors (e.g., 0.2,0.5,2.0).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random operations (default: 42).")

    args = parser.parse_args()

    # Build the list of augmentations based on the provided arguments
    augmentations = []

    if args.rotate:
        angles = list(map(int, args.rotate.split(",")))
        augmentations.append({"func": rotate_image, "args": (angles,), "kwargs": {}})

    if args.contrast:
        factors = list(map(float, args.contrast.split(",")))
        augmentations.append({"func": adjust_contrast, "args": (factors,), "kwargs": {}})

    if args.saturation:
        factors = list(map(float, args.saturation.split(",")))
        augmentations.append({"func": adjust_saturation, "args": (factors,), "kwargs": {}})

    if args.scale:
        scales = list(map(float, args.scale.split(",")))
        augmentations.append({"func": scale_image, "args": (scales,), "kwargs": {}})

    # Create the augmenter with a fixed seed for deterministic behavior
    augmenter = DataAugmentation(augmentations, seed=args.seed)

    # Process images with augmentations
    process_images_with_augmentation(args.input_folder, args.output_folder, augmenter)
