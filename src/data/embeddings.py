import torch
from PIL import Image

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
