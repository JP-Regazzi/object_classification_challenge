import torchvision.transforms as transforms
import os
from PIL import Image

def get_transforms():
    """
    Return the necessary image transforms for 224x224 and 299x299 input sizes.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])

    transform_224 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    transform_299 = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        normalize
    ])

    return transform_224, transform_299


def load_image_paths(folder_path):
    """
    Load all image paths from a folder.

    Parameters:
        folder_path (str): Path to the folder containing images.

    Returns:
        list: List of image file paths.
    """
    return [os.path.join(folder_path, img) for img in os.listdir(folder_path)]

def preprocess_image(image_path, processor):
    """
    Preprocess an image for the CLIP model.

    Parameters:
        image_path (str): Path to the image file.
        processor (CLIPProcessor): Processor for the CLIP model.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", padding=True)
    return inputs