import json
import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp

from utils import _qualname, _resolve_qualname, xavier_uniform_reinit

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

class GFINNComponent(eqx.Module):
    T: eqx.nn.MLP
    G: eqx.nn.MLP
    S: jax.Array
    dim: int = eqx.static_field()
    K: int = eqx.static_field()
    which_one: str = eqx.static_field()
    T_dim: int = eqx.static_field()
    triu_indices_T: jax.Array = eqx.static_field()
    triu_indices_S: jax.Array = eqx.static_field()

    def __init__(self, dim, width, depth, K, which_one, *, key, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.K = K
        self.which_one = which_one
        key1, key2, key3 = jax.random.split(key, num=3)
        self.T_dim = K*(K-1) // 2 if which_one == "L" else K*(K+1) // 2
        self.T = eqx.nn.MLP(
            in_size=dim,
            out_size=self.T_dim,
            width_size=width,
            depth=depth,
            activation=jnn.tanh,
            key=key1,
        )
        self.triu_indices_T = jnp.triu_indices(K, k=1) if which_one == "L" else jnp.triu_indices(K, k=0)
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
        self.S = jax.random.normal(key=key3,shape=(self.K, self.dim*(self.dim-1)//2))
        self.triu_indices_S = jnp.triu_indices(self.dim, k=1)
    
    def __call__(self, x):
        T = jnp.zeros((self.K, self.K))
        T_vals = self.T(x)
        T = T.at[self.triu_indices_T].set(T_vals)
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
            S = jnp.zeros((self.dim, self.dim))
            S = S.at[self.triu_indices_S].set(self.S[i])
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
        self.func = None
    
    def energy(self, x):
        return self.MgradE.G(x)
    
    def entropy(self, x):
        return self.LgradS.G(x)
    
    def __call__(self, t, x, args):
        L, dS = self.LgradS(x)
        M, dE = self.MgradE(x)
        #dE = jnp.array([x[0], x[1], 1])
        #L = jnp.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        #dS = jnp.array([0, 0, 1])
        #M = jnp.array([[0, 0, 0], [0, 0, -2*0.2*1*x[1]], [0, -2*0.2*1*x[1], 2*0.2*1*x[1]**2]])
        dE = jnp.expand_dims(dE,1)
        dS = jnp.expand_dims(dS,1)
        return -(dE @ L).squeeze() + (dS @ M).squeeze()

class BasicParam(MLPVectorField):
    L: eqx.nn.MLP
    M: eqx.nn.MLP
    E: eqx.nn.MLP
    S: eqx.nn.MLP
    triu_indices_L: jax.Array = eqx.static_field()
    triu_indices_M: jax.Array = eqx.static_field()

    def __init__(self, dim, width, depth, *, key, **kwargs):
        key, subkey = jax.random.split(key, num=2)
        super().__init__(dim=dim, width=width, depth=depth, key=subkey, **kwargs)
        key1, key2, key3, key4 = jax.random.split(key, num=4)
        L_dim = dim * (dim - 1) // 2
        self.L = eqx.nn.MLP(
            in_size=dim,
            out_size=L_dim,
            width_size=width,
            depth=depth,
            activation=jnn.tanh,
            key=key1,
        )
        M_dim = dim * (dim + 1) // 2
        self.M = eqx.nn.MLP(
            in_size=dim,
            out_size=M_dim,
            width_size=width,
            depth=depth,
            activation=jnn.tanh,
            key=key2,
        )

        self.triu_indices_L = jnp.triu_indices(dim, k=1)
        self.triu_indices_M = jnp.triu_indices(dim, k=0)
        self.E = eqx.nn.MLP(
            in_size=dim,
            out_size=1,
            width_size=width,
            depth=depth,
            activation=jnn.tanh,
            key=key3,
        )
        self.S = eqx.nn.MLP(
            in_size=dim,
            out_size=1,
            width_size=width,
            depth=depth,
            activation=jnn.tanh,
            key=key3,
        )
        self.func = None
    
    def energy(self, x):
        return self.E(x)
    
    def entropy(self, x):
        return self.S(x)
    
    def __call__(self, t, x, args):
        L_vals = self.L(x)
        M_vals = self.M(x)
        L = jnp.zeros((self.dim, self.dim))
        L = L.at[self.triu_indices_L].set(L_vals)
        L = L - L.T
        M = jnp.zeros((self.dim, self.dim))
        M = M.at[self.triu_indices_M].set(M_vals)
        M = M.T@M
        dE = jax.grad(lambda x: self.E(x).squeeze())(x).reshape([-1,self.dim])
        dS = jax.grad(lambda x: self.S(x).squeeze())(x).reshape([-1,self.dim])
        return -(dE @ L).squeeze() + (dS @ M).squeeze()

class NMS(MLPVectorField):
    dim: int
    D: int
    C2: int
    poisson_A: eqx.Module
    friction_C: eqx.Module
    friction_B: eqx.Module
    energy: eqx.Module
    entropy: eqx.Module
    poisson_A_idx: tuple

    def __init__(self, dim, width, depth, *, lE, lS, lA, lB, lD, nE, nS, nA, nB, nD, D, C2, key, **kwargs):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        super().__init__(dim=dim, width=0, depth=0, key=k5, **kwargs)

        self.D = D
        self.C2 = C2

        self.poisson_A = eqx.nn.MLP(
            in_size=dim,
            out_size=dim * (dim - 1) // 2,
            width_size=nA,
            depth=lA,
            use_final_bias=False,
            activation=jnn.tanh,
            key=k1,
        )
        self.poisson_A_idx = jnp.tril_indices(dim, -1)

        self.friction_C = eqx.nn.MLP(
            in_size=dim,
            out_size=D*C2,
            width_size=nD,
            depth=lD,
            use_final_bias=False,
            activation=jnn.tanh,
            key=k1,
        )
        self.friction_B = eqx.nn.MLP(
            in_size=dim,
            out_size=D*dim,
            width_size=nB,
            depth=lB,
            use_final_bias=False,
            activation=jnn.tanh,
            key=k2,
        )

        self.energy = eqx.nn.MLP(
            in_size=dim,
            out_size=1,
            width_size=nE,
            depth=lE,
            use_final_bias=False,
            activation=jnn.tanh,
            key=k3,
        )
        self.entropy = eqx.nn.MLP(
            in_size=dim,
            out_size=1,
            width_size=nS,
            depth=lS,
            use_final_bias=False,
            activation=jnn.tanh,
            key=k4,
        )

    def poisson_product(self, y, dE, dS):
        bdim = y.shape[0]
        A_flat = self.poisson_A(y)
        Amat = jnp.zeros((bdim, self.dim, self.dim))
        Amat = Amat.at[:, self.poisson_A_idx[0], self.poisson_A_idx[1]].set(A_flat)
        A = Amat - jnp.swapaxes(Amat, 1, 2)

        AdE = jnp.einsum('bij,bj->bi', A, dE)
        AdS = jnp.einsum('bij,bj->bi', A, dS)

        dE_AdS = jnp.sum(dE * AdS, axis=-1, keepdims=True)
        dE_dS = jnp.sum(dE * dS, axis=-1, keepdims=True)
        dS_sq = jnp.sum(dS ** 2, axis=-1, keepdims=True)

        correction = (dE_AdS * dS - dE_dS * AdS) / dS_sq
        LdE = AdE + correction
        return LdE

    def friction_product(self, y, dE, dS):
        bdim = y.shape[0]

        C = self.friction_C(y).reshape(bdim, self.D, self.C2)
        B = self.friction_B(y).reshape(bdim, self.D, self.dim)
        Dmat = jnp.matmul(C, jnp.swapaxes(C, 1, 2))  # (bdim, D, D)

        BdotdE = jnp.einsum('bij,bj->bi', B, dE)
        BdotdS = jnp.einsum('bij,bj->bi', B, dS)
        dE_sq = jnp.sum(dE ** 2, axis=-1, keepdims=True)

        BdE = B - BdotdE[:, :, None] * dE[:, None, :] / dE_sq[:, None, :]

        dEdS = jnp.sum(dE * dS, axis=-1, keepdims=True)
        BdEdS = BdotdS - BdotdE * dEdS.squeeze(-1) / dE_sq.squeeze(-1)

        MdS = jnp.einsum('bik,bkl,bi->bl', jnp.swapaxes(BdE, 1, 2), Dmat, BdEdS)
        return MdS

    def get_penalty(self, y, dE, dS):
        LdS = self.poisson_product(y, dS, dS)
        MdE = self.friction_product(y, dE, dE)
        return LdS, MdE

    def __call__(self, t, y):
        dE = jax.vmap(jax.grad(lambda y_: self.energy(y_.reshape(1, -1)).sum()))(y)
        dS = jax.vmap(jax.grad(lambda y_: self.entropy(y_.reshape(1, -1)).sum()))(y)

        LdE = self.poisson_product(y, dE, dS)
        MdS = self.friction_product(y, dE, dS)

        return LdE + MdS
