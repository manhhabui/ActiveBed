
  
from algorithms.LLALOD.src.models.SSD300 import SSD300
from algorithms.LLALOD.src.models.LossNet import LossNet

nets_map = {"SSD300": SSD300, "lossnet": LossNet}

def get_model(name):
    if name not in nets_map:
        raise ValueError("Name of model unknown %s" % name)

    def get_model_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_model_fn