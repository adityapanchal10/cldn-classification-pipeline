import importlib


def build_model(cfg, instantiate=False, device=None):
    module_name = cfg["model"]["module"]
    class_name = cfg["model"]["class_name"]
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    if instantiate:
        init_args = cfg.get("model", {}).get("init_args", {})
        model = model_class(**init_args) if init_args else model_class()
        if device is not None:
            model.to(device)
        return model
    return model_class