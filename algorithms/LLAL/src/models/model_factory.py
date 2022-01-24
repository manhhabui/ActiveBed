
  
import algorithms.LLAL.src.models.resnet as resnet
from algorithms.LLAL.src.models.LossNet import LossNet


nets_map = {"resnet18": resnet.ResNet18, "lossnet": LossNet}


def get_model(name):
    if name not in nets_map:
        raise ValueError("Name of model unknown %s" % name)

    def get_model_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_model_fn