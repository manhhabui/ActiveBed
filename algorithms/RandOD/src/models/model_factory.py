
  
from algorithms.RandOD.src.models.SSD300 import SSD300

nets_map = {"SSD300": SSD300}

def get_model(name):
    if name not in nets_map:
        raise ValueError("Name of model unknown %s" % name)

    def get_model_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_model_fn