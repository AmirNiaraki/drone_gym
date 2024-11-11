import sys

import torch

sys.path.append("pytorch-retinanet")
from retinanet import model


class ModelConfig:
    def __init__(
        self,
        supervised=True,
        depth=50,
        num_classes=1,
        dim_out=32,
        model_path="/home/aniaraki/projects/drone_gym/weights/best_anomaly.pt",
    ):
        self.supervised = supervised
        self.depth = depth
        self.num_classes = num_classes
        self.dim_out = dim_out
        self.model_path = model_path


def load_retinanet(config=None):
    """
    Load and configure the RetinaNet model
    Args:
        config (ModelConfig, optional): Configuration for the model.
                                      If None, uses default settings.
    Returns:
        model: Configured RetinaNet model
    """
    if config is None:
        config = ModelConfig()

    retinanet = None
    # Initialize model
    if config.depth == 18:
        retinanet = model.resnet18(num_classes=config.num_classes, pretrained=True)
    if config.depth == 50:
        retinanet = model.resnet50(num_classes=config.num_classes, pretrained=True)
    if config.depth == 101:
        retinanet = model.resnet101(num_classes=config.num_classes, pretrained=True)

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load weights
    if device.type == "cuda":
        print("Loading to cuda")
        print(config.model_path)
        retinanet.load_state_dict(torch.load(config.model_path), strict=False)
        retinanet = torch.nn.DataParallel(retinanet).cuda()
        print('device', device)
    else:
        print("Loading to cpu")
        dictionary_weights = torch.load(config.model_path, map_location=device)
        retinanet.load_state_dict(dictionary_weights, strict=False)
        retinanet = torch.nn.DataParallel(retinanet)
        print("Done loading model")

    # Set model to evaluation mode
    retinanet.training = False
    retinanet.module.freeze_bn()
    retinanet.eval()

    return retinanet


# Example usage
if __name__ == "__main__":
    # Create custom config if needed
    custom_config = ModelConfig(model_path="/home/aniaraki/projects/drone_gym/weights/best_anomaly.pt")

    # Load model with custom config
    model = load_retinanet(custom_config)
