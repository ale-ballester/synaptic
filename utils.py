import os
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import importlib

from jax.nn.initializers import glorot_uniform

def make_dir(path):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)

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

def xavier_uniform_reinit(model, *, key):
    glorot = glorot_uniform()

    leaves, treedef = jax.tree_util.tree_flatten(model)
    keys_flat = jax.random.split(key, len(leaves))
    keys_tree = jax.tree_util.tree_unflatten(treedef, keys_flat)

    def _maybe_reinit(p, k):
        if isinstance(p, jnp.ndarray) and p.ndim == 2:
            return glorot(k, p.shape, p.dtype)
        if isinstance(p, jnp.ndarray) and p.ndim == 1:
            return jnp.zeros_like(p)
        return p

    return jax.tree_util.tree_map(_maybe_reinit, model, keys_tree)

def _to_numpy(x):
    # move off device + ensure plain NumPy array (not masked, not jax array)
    return np.asarray(jax.device_get(x))

def count_parameters(model):
    # eqx.is_array marks any JAX array (trainable or not).
    # eqx.is_inexact_array marks specifically inexact dtypes (float/complex),
    # which are the ones you normally optimize.
    leaves = jax.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    return sum(jnp.size(leaf) for leaf in leaves)