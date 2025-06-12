import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import importlib

from models import NeuralODE, NeuralMetriplecticODE, GFINNODE

def _qualname(obj):
    """Return 'package.subpackage.module:ClassName'."""
    return f"{obj.__module__}:{obj.__qualname__}"

def _resolve_qualname(path):
    """Inverse of _qualname -> the actual Python object."""
    modname, qual = path.split(":")
    mod = importlib.import_module(modname)
    out = mod
    for attr in qual.split("."):
        out = getattr(out, attr)
    return out