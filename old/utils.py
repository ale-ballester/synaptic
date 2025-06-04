import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import json

from models import NeuralODE, NeuralMetriplecticODE, GFINNODE

class Args:
    dim:int
    width:int
    depth:int
    dt:float
    K:int
    #L_ib: bool
    #M_ib: bool
    #F_ib: bool
    #system_params: list
    #trainset_stats: list
    #checkpoint: str

def make_model(dim, width, depth, dt, key):
    return NeuralMetriplecticODE(dim, width, depth, dt, key=key)

def load_model(filename):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = make_model(key=jax.random.PRNGKey(0), **hyperparams)
        return eqx.tree_deserialise_leaves(f, model)


def save_model(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams.__dict__)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)