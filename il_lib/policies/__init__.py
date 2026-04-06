from importlib import import_module

__all__ = [
    "ACT",
    "BC_RNN",
    "DiffusionPolicy",
    "ResidualPolicy",
    "SimpleResidualPolicy",
    "WBVIMA",
]

_IMPORT_MAP = {
    "ACT": (".act_policy", "ACT"),
    "BC_RNN": (".bcrnn_policy", "BC_RNN"),
    "DiffusionPolicy": (".diffusion_policy", "DiffusionPolicy"),
    "ResidualPolicy": (".residual_policy", "ResidualPolicy"),
    "SimpleResidualPolicy": (".simple_residual_policy", "SimpleResidualPolicy"),
    "WBVIMA": (".wbvima_policy", "WBVIMA"),
}


def __getattr__(name):
    if name not in _IMPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _IMPORT_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
