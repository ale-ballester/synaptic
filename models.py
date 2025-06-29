import json
import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp

from utils import _qualname, _resolve_qualname

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
    dim: int = eqx.static_field()
    width: int = eqx.static_field()
    depth: int = eqx.static_field()

    def __init__(self, dim, width, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.width = width
        self.depth = depth
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
    dt: float = eqx.static_field()
    vf_kwargs: dict = eqx.static_field()

    def __init__(self, vector_field_cls, dim, width, depth, dt, *, key, **kwargs):
        super().__init__()
        self.vector_field = vector_field_cls(dim=dim, width=width, depth=depth, key=key, **kwargs)
        self.vf_kwargs = kwargs # Extra arguments for the vector field
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
    
    def save_model(self, filename):
        with open(filename, "wb") as f:
            hyperparams = {
                "vector_field_cls": _qualname(self.vector_field.__class__),
                "dim": self.vector_field.dim,
                "width": self.vector_field.width,
                "depth": self.vector_field.depth,
                "dt": self.dt,
                "kwargs": self.vf_kwargs,
            }
            hyperparam_str = json.dumps(hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self)
    
    @classmethod
    def load_model(cls, filename):
        def make_model(vector_field_cls, dim, width, depth, dt, key, kwargs):
            return cls(vector_field_cls, dim, width, depth, dt, key=key, kwargs=kwargs)
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = make_model(key=jax.random.PRNGKey(0), **hyperparams)
            return eqx.tree_deserialise_leaves(f, model)

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

class GFINNComponent(eqx.Module):
    T: eqx.nn.MLP
    G: eqx.nn.MLP
    S: jax.Array
    dim: int = eqx.static_field()
    K: int = eqx.static_field()
    which_one: str = eqx.static_field()

    def __init__(self, dim, width, depth, K, which_one, *, key, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.K = K
        self.which_one = which_one
        key1, key2, key3 = jax.random.split(key, num=3)
        self.T = eqx.nn.MLP(
            in_size=dim,
            out_size=K**2,
            width_size=width,
            depth=depth,
            activation=jnn.tanh,
            key=key1,
        )
        key1, subkey1 = jax.random.split(key1)
        self.T = xavier_uniform_reinit(self.T, key=subkey1)
        self.G = eqx.nn.MLP(
            in_size=dim,
            out_size=1,
            width_size=width,
            depth=depth,
            use_final_bias=False,
            activation=jnn.tanh,
            key=key2,
        )
        key2, subkey2 = jax.random.split(key2)
        self.G = xavier_uniform_reinit(self.G, key=subkey2)
        self.S = 0.1*jax.random.normal(key=key3,shape=(self.K, self.dim, self.dim))
    
    def __call__(self, x):
        T = self.T(x).reshape(self.K, self.K)
        if self.which_one == "M":
            B = T@jnp.moveaxis(T,-1,-2)
        elif self.which_one == "L":
            B = T - jnp.moveaxis(T,-1,-2)
        else:
            raise ValueError("which_one must be either 'M' or 'L'")
        gradG = jax.grad(lambda x: self.G(x).squeeze())(x).reshape([-1,self.dim])
        gradG1 = jnp.expand_dims(gradG,-2)
        Q = []
        for i in range(self.K):
            S = jnp.triu(self.S[i], k=1)
            S = S - jnp.moveaxis(S,-1,-2)
            Q.append(gradG1@S)
        Q = jnp.concatenate(Q, axis=-2).squeeze()
        A = jnp.moveaxis(Q,-1,-2) @ B @ Q
        return A, gradG

class GFINN(MLPVectorField):
    LgradS: GFINNComponent
    MgradE: GFINNComponent

    def __init__(self, dim, width, depth, *, K, key, **kwargs):
        key, subkey = jax.random.split(key, num=2)
        super().__init__(dim=dim, width=width, depth=depth, key=subkey, **kwargs)
        key1, key2 = jax.random.split(key, num=2)
        self.LgradS = GFINNComponent(dim, width, depth, K, "L", key=key1)
        self.MgradE = GFINNComponent(dim, width, depth, K, "M", key=key2)
    
    def __call__(self, t, x, args):
        L, dS = self.LgradS(x)
        M, dE = self.MgradE(x)
        dE = jnp.expand_dims(dE,1)
        dS = jnp.expand_dims(dS,1)
        return -(dE @ L).squeeze() + (dS @ M).squeeze() 

