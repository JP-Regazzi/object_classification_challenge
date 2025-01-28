import torchvision.transforms as transforms

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
