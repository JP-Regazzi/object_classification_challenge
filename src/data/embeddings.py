import torch
from PIL import Image
from tqdm import tqdm
from preprocess import preprocess_image

def extract_hybrid_embeddings(image_path, resnet_model, vgg_model, inception_model, resnet_layer, vgg_layer, inception_layer, transform_224, transform_299):
    """
    Extract hybrid embeddings using three models: ResNet, VGG, and Inception.
    """
    resnet_emb = torch.zeros(2048)
    vgg_emb = torch.zeros(25088)
    inception_emb = torch.zeros(2048)

    img = Image.open(image_path).convert('RGB')

    input_224 = transform_224(img).unsqueeze(0)
    input_299 = transform_299(img).unsqueeze(0)

    def resnet_hook(module, inp, out):
        resnet_emb.copy_(out.data.view(-1))
    def vgg_hook(module, inp, out):
        vgg_emb.copy_(out.data.view(-1))
    def inception_hook(module, inp, out):
        inception_emb.copy_(out.data.view(-1))

    handle_rn = resnet_layer.register_forward_hook(resnet_hook)
    handle_vgg = vgg_layer.register_forward_hook(vgg_hook)
    handle_in = inception_layer.register_forward_hook(inception_hook)

    with torch.no_grad():
        _ = resnet_model(input_224)
        _ = vgg_model(input_224)
        _ = inception_model(input_299)

    handle_rn.remove()
    handle_vgg.remove()
    handle_in.remove()

    return torch.cat([resnet_emb, vgg_emb, inception_emb])


def generate_embeddings(image_paths, model, processor):
    """
    Generate embeddings for a list of images using the CLIP model.

    Parameters:
        image_paths (list): List of image file paths.
        model (CLIPModel): Pre-trained CLIP model.
        processor (CLIPProcessor): Processor for the CLIP model.

    Returns:
        list: List of normalized embeddings.
    """
    features = []
    for img_path in tqdm(image_paths):
        inputs = preprocess_image(img_path, processor)
        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
            embedding = embedding / embedding.norm(p=2, dim=-1)
            features.append(embedding.flatten())
    return features
