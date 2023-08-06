from torchvision import models


def setup_model(name, config):
    if name in models.__dict__:
        fn = models.__dict__[name]
    else:
        raise RuntimeError(f"Unknown model name {name}")

    return fn(num_classes=config.dataset.num_classes)
