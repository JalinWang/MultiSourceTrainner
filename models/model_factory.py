from models.classifier import Classifier
from torchvision import models


def setup_model(config, return_hidden=False):
    
    name = config.model.name

    backbone = models.get_model(
        name,
        # weights="DEFAULT", # if enable this, it won't allow to change num_classes directly
        num_classes=config.model.hidden_dim
    )

    return Classifier(
        backbone, 
        hidden_dim=config.model.hidden_dim,
        num_classes=config.dataset.num_classes,
        return_hidden=return_hidden
    )
