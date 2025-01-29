import torch
import torchvision.models as models
from transformers import CLIPProcessor, CLIPModel

def load_models():
    """
    Load the pre-trained models (ResNet50, VGG16, InceptionV3) and return their layers.
    """
    resnet_model = models.resnet50(pretrained=True)
    vgg_model = models.vgg16(pretrained=True)
    inception_model = models.inception_v3(pretrained=True)
    inception_model.aux_logits = False
    inception_model.eval()

    resnet_layer = resnet_model._modules.get('avgpool')
    vgg_layer = vgg_model._modules.get('avgpool')
    inception_layer = inception_model._modules.get('avgpool')

    return resnet_model, vgg_model, inception_model, resnet_layer, vgg_layer, inception_layer


def load_clip_model(model_name="patrickjohncyh/fashion-clip"):
    """
    Load the CLIP model and processor.
    
    Parameters:
        model_name (str): Name of the pre-trained model.

    Returns:
        model (CLIPModel): Pre-trained CLIP model.
        processor (CLIPProcessor): Processor for the model.
    """
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return model, processor
