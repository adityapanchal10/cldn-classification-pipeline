import importlib


def build_model(cfg):

    module_name = cfg["model"]["module"]
    class_name = cfg["model"]["class_name"]

    module = importlib.import_module(module_name)

    model_class = getattr(module, class_name)

    return model_class