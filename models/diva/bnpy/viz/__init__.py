"""Load bnpy's optional visualization helpers only when explicitly requested."""
from importlib import import_module


def __getattr__(name):
    if name.startswith('_'):
        raise AttributeError(name)
    try:
        module = import_module(f'{__name__}.{name}')
    except ModuleNotFoundError as exc:
        if exc.name == f'{__name__}.{name}':
            raise AttributeError(name) from exc
        raise
    globals()[name] = module
    return module
