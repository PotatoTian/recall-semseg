import json
from ptsemseg.loader.cityscapes_loader import cityscapesLoader
from ptsemseg.loader.synthia_loader import synthiaLoader

def get_loader(name,**kargs):
    """get_loader

    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "synthia": synthiaLoader,
    }[name]
