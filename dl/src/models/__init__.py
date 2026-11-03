"""Model registry. Each module exposes NAME, KIND, n_trials, default_params, fixed_trials,
suggest_params, adapt_to_size and build."""
from types import ModuleType

from . import ft_transformer, mlp, modernnca, tabm, tabnet

REGISTRY: dict[str, ModuleType] = {"ft_transformer": ft_transformer, "tabnet": tabnet, "mlp": mlp, "tabm": tabm,
                                   "modernnca": modernnca}


def get_model_module(name: str) -> ModuleType:
    if name not in REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choose from {list(REGISTRY)}")
    return REGISTRY[name]
