from typing import Callable
import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp

class MLPScalarField(eqx.Module):
    func: eqx.nn.MLP

    def __init__(self, dim, width, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = eqx.nn.MLP(
            in_size=dim,
            out_size=1,
            width_size=width,
            depth=depth,
            activation=jnn.tanh,
            key=key,
        )
        
    def __call__(self, t, y, args):
        return self.func(y)[0]

class MLPVectorField(eqx.Module):
    func: eqx.nn.MLP

    def __init__(self, dim, width, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = eqx.nn.MLP(
            in_size=dim,
            out_size=dim,
            width_size=width,
            depth=depth,
            activation=jnn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.func(y)

class NeuralODE(eqx.Module):
    vector_field: eqx.Module
    dt: float

    def __init__(self, vector_field, dim, width, depth, dt, *, key, **kwargs):
        super().__init__(**kwargs)
        self.vector_field = vector_field
        self.dt = dt

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.vector_field),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=self.dt,
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys

from jax.nn.initializers import glorot_uniform

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