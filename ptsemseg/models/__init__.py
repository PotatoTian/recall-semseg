import copy
import torchvision.models as model
from ptsemseg.models.deeplab import DeepLab

def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")
    
    if name == "DeepLab":
        model = model(backbone= param_dict['backbone'], output_stride=16, n_classes=n_classes, sync_bn=True, freeze_bn=False)
    else:
        model = model(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "DeepLab": DeepLab,
        }[name]
    except:
        raise ("Model {} not available".format(name))
