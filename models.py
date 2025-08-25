import json
import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils import _qualname, _resolve_qualname, xavier_uniform_reinit, _to_numpy

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
    
    def plot(
            self,
            ranges,
            *,
            t: float = 0.0,
            grid_size: int = 25,
            vary: tuple = (0, 1),
            fixed=None,
            kind: str = "quiver",          # "quiver" or "stream"
            stream_density: float = 1.2,
            figsize=(6, 5),
            quiver_scale=None,             # e.g., 50; None lets matplotlib choose
            normalize: bool = True,        # normalize arrows for readability
            title: str = None,
        ):
        """
        Visualize the (unbatched) vector field f(t, y) implemented by self.vector_field.

        Parameters
        ----------
        ranges : tuple
            For dim == 1: (x_min, x_max)
            For dim >= 2: ((x_min, x_max), (y_min, y_max)) for the two varying coordinates.
            (Coordinates are chosen via `vary` when dim > 2.)
        t : float
            Time at which to evaluate the autonomous field; kept for generality.
        grid_size : int
            Number of points per axis.
        vary : tuple(int, int)
            Indices of the two coordinates to vary when dim >= 2.
        fixed : dict or list[dict] or None
            When dim > 2, specify values for the other coordinates.
            - dict: one cross-section (e.g., {2: 0.0, 3: 1.0}).
            - list of dict: multiple cross-sections -> creates subplots.
            If None, all other coords are set to 0.0.
        kind : {"quiver", "stream"}
            Plot type for 2D slices.
        stream_density : float
            Density parameter for streamplot.
        figsize : tuple
            Figure size for a single plot. Multiple cross-sections scale rows.
        quiver_scale : float or None
            Matplotlib quiver scale; None lets matplotlib choose.
        normalize : bool
            If True, normalize vectors to unit length for readability (direction field).
        title : str or None
            Title for the plot(s).
        """

        dim = int(self.dim)

        # Utility: evaluate f(t, y) over a set of points
        def f_single(y):
            return self.func(y)

        # ---------- 1D ----------
        if dim == 1:
            x_min, x_max = ranges
            xs = jnp.linspace(x_min, x_max, grid_size)
            ys = jax.vmap(f_single)(xs)  # shape (grid_size,) since dim=1
            xs_np = jnp.asarray(xs)
            ys_np = jnp.asarray(ys)

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.plot(xs_np, ys_np, lw=1.5)
            # arrows along the x-axis to show direction
            if normalize:
                mag = jnp.maximum(jnp.abs(ys_np), 1e-12)
                u = jnp.sign(ys_np)  # direction only
            else:
                u = ys_np
            ax.quiver(xs_np, jnp.zeros_like(xs_np), u, jnp.zeros_like(xs_np),
                      angles='xy', scale_units='xy', scale=quiver_scale if quiver_scale else 1.0,
                      width=0.003)
            ax.axhline(0, color='k', lw=0.8, alpha=0.6)
            ax.set_xlabel("y")
            ax.set_ylabel("f(y)")
            if title:
                ax.set_title(title)
            plt.show()
            return

        # ---------- 2D or higher ----------
        (x_min, x_max), (y_min, y_max) = ranges
        i, j = vary
        assert 0 <= i < dim and 0 <= j < dim and i != j, "`vary` must be two distinct valid indices."

        # Normalize `fixed` into a list of dicts (for multiple cross-sections)
        if fixed is None:
            fixed_list = [dict()]
        elif isinstance(fixed, dict):
            fixed_list = [fixed]
        else:
            # assume iterable of dicts
            fixed_list = list(fixed)

        # Fill missing fixed coordinates with zeros
        def complete_fixed_dict(fd):
            fd = dict(fd)  # copy
            for k in range(dim):
                if k not in (i, j) and k not in fd:
                    fd[k] = 0.0
            return fd

        fixed_list = [complete_fixed_dict(fd) for fd in fixed_list]

        # Build grid for the two varying coordinates
        Xi = jnp.linspace(x_min, x_max, grid_size)
        Xj = jnp.linspace(y_min, y_max, grid_size)
        XI, XJ = jnp.meshgrid(Xi, Xj, indexing='xy')  # (G, G)

        # Prepare figure
        nplots = len(fixed_list)
        nrows = nplots
        figsize_total = (figsize[0], figsize[1] * nrows)
        fig, axes = plt.subplots(nrows, 1, figsize=figsize_total, squeeze=False)
        axes = axes[:, 0]

        # For each cross-section, evaluate and plot
        for ax, fd in zip(axes, fixed_list):
            # Make full grid of y points: shape (G*G, dim)
            def make_point(xi, xj):
                y = jnp.zeros((dim,), dtype=XI.dtype)
                y = y.at[i].set(xi)
                y = y.at[j].set(xj)
                # set fixed coords
                for k, v in fd.items():
                    if k != i and k != j:
                        y = y.at[k].set(v)
                return y

            # Vectorized over grid
            pts = jax.vmap(
                lambda xi_row, xj_row: jax.vmap(make_point)(xi_row, xj_row)
            )(XI, XJ)  # (G, G, dim)

            # Flatten to (G*G, dim), evaluate, reshape back
            pts_flat = pts.reshape((-1, dim))
            f_out_flat = jax.vmap(f_single)(pts_flat)           # (G*G, dim)
            f_out = f_out_flat.reshape((grid_size, grid_size, dim))

            Ui = jnp.asarray(f_out[..., i])
            Vj = jnp.asarray(f_out[..., j])
            X = jnp.asarray(XI)
            Y = jnp.asarray(XJ)

            # optional: sanitize NaNs/Infs to avoid masked arrays being created
            Ui_plot = jnp.nan_to_num(Ui, nan=0.0, posinf=0.0, neginf=0.0)
            Vj_plot = jnp.nan_to_num(Vj, nan=0.0, posinf=0.0, neginf=0.0)

            # convert to NumPy for matplotlib
            X = _to_numpy(XI)
            Y = _to_numpy(XJ)
            U = _to_numpy(Ui_plot)
            V = _to_numpy(Vj_plot)

            # (sometimes streamplot wants contiguous arrays)
            #X = np.ascontiguousarray(X)
            #Y = np.ascontiguousarray(Y)
            #U = np.ascontiguousarray(U)
            #V = np.ascontiguousarray(V)

            if normalize:
                mag = jnp.sqrt(Ui**2 + Vj**2)
                mag = jnp.maximum(mag, 1e-12)
                Ui_plot = Ui / mag
                Vj_plot = Vj / mag
            else:
                Ui_plot, Vj_plot = Ui, Vj

            if kind == "stream":
                strm = ax.streamplot(
                    X, Y, U, V,
                    density=stream_density, linewidth=1.2, arrowsize=1.2
                )
            else:
                ax.quiver(
                    X, Y, U, V,
                    angles='xy', scale_units='xy',
                    scale=quiver_scale if quiver_scale else None, width=0.0025
                )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel(f"y[{i}]")
            ax.set_ylabel(f"y[{j}]")
            fd_str = ", ".join([f"y[{k}]={v:g}" for k, v in sorted(fd.items()) if k not in (i, j)])
            subtitle = f"Cross-section over (y[{i}], y[{j}])"
            if fd_str:
                subtitle += f" with {fd_str}"
            ax.set_title(subtitle)

        if title:
            fig.suptitle(title, y=0.99)
        fig.tight_layout()
        plt.show()

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
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())

            # resolve class from saved string
            vfc_path = hyperparams.pop("vector_field_cls")
            vector_field_cls = _resolve_qualname(vfc_path)

            dim   = hyperparams["dim"]
            width = hyperparams["width"]
            depth = hyperparams["depth"]
            dt    = hyperparams["dt"]
            vf_kwargs = hyperparams.get("kwargs", {})  # may be empty dict

            # build skeleton with identical hyperparams
            model = cls(
                vector_field_cls=vector_field_cls,
                dim=dim,
                width=width,
                depth=depth,
                dt=dt,
                key=jax.random.PRNGKey(0),
                **vf_kwargs,                    # <- pass kwargs properly
            )

            # load parameters into that structure
            return eqx.tree_deserialise_leaves(f, model)

"""
A basic parametrization that does not preserve degeneracy conditions exactly.
"""

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
    
    def L(self, x):
        L_vals = self.L(x)
        L = jnp.zeros((self.dim, self.dim))
        L = L.at[self.triu_indices_L].set(L_vals)
        return L - L.T
    
    def M(self, x):
        M_vals = self.M(x)
        M = jnp.zeros((self.dim, self.dim))
        M = M.at[self.triu_indices_M].set(M_vals)
        return M.T@M
    
    def get_terms(self, x):
        L = self.L(x)
        M = self.M(x)
        dE = jax.grad(lambda x: self.E(x).squeeze())(x).reshape([-1,self.dim])
        dS = jax.grad(lambda x: self.S(x).squeeze())(x).reshape([-1,self.dim])
        return -(dE @ L).squeeze(),(dS @ M).squeeze(),dE,dS
    
    def get_penalty(self, x):
        L = self.L(x)
        M = self.M(x)
        dE = jax.grad(lambda x: self.E(x).squeeze())(x).reshape([-1,self.dim])
        dS = jax.grad(lambda x: self.S(x).squeeze())(x).reshape([-1,self.dim])
        return -(dS @ L).squeeze(),(dE @ M).squeeze()
    
    def __call__(self, t, x, args):
        L,M,dE,dS = self.metriplectic(x)
        return -(dE @ L).squeeze() + (dS @ M).squeeze()

"""
GFINN parametrization.
"""

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
    
    def L(self, x):
        L, _ = self.LgradS(x)
        return L

    def M(self, x):
        M, _ = self.MgradE(x)
        return M
    
    def get_terms(self, x):
        L, dS = self.LgradS(x)
        M, dE = self.MgradE(x)
        dE = jnp.expand_dims(dE,1)
        dS = jnp.expand_dims(dS,1)
        return -(dE @ L).squeeze(),(dS @ M).squeeze(),dE,dS
    
    def get_penalty(self, x):
        L, dS = self.LgradS(x)
        M, dE = self.MgradE(x)
        dE = jnp.expand_dims(dE,1)
        dS = jnp.expand_dims(dS,1)
        return -(dS @ L).squeeze(),(dE @ M).squeeze()
    
    def __call__(self, t, x, args):
        L, M, dE, dS = self.metriplectic(x)
        return -(dE @ L).squeeze() + (dS @ M).squeeze()

"""
Gruber's NMS model
"""

class NMS(MLPVectorField):
    dim: int
    eps: float
    D: int
    C2: int
    poisson_A: eqx.Module
    friction_C: eqx.Module
    friction_B: eqx.Module
    E_MLP: eqx.Module
    S_MLP: eqx.Module
    poisson_A_idx: tuple

    def __init__(self, dim, width, depth, *, lE, lS, lA, lB, lD, nE, nS, nA, nB, nD, D, C2, key, **kwargs):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        super().__init__(dim=dim, width=0, depth=0, key=k6, **kwargs)

        self.eps = 1e-6  # Small value to avoid division by zero

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
            key=k2,
        )
        self.friction_B = eqx.nn.MLP(
            in_size=dim,
            out_size=D*dim,
            width_size=nB,
            depth=lB,
            use_final_bias=False,
            activation=jnn.tanh,
            key=k3,
        )

        self.E_MLP = eqx.nn.MLP(
            in_size=dim,
            out_size=1,
            width_size=nE,
            depth=lE,
            use_final_bias=False,
            activation=jnn.tanh,
            key=k4,
        )
        self.S_MLP = eqx.nn.MLP(
            in_size=dim,
            out_size=1,
            width_size=nS,
            depth=lS,
            use_final_bias=False,
            activation=jnn.tanh,
            key=k5,
        )

    def _build_skew_A(self, y):
        """Return A(y) as a skew-symmetric (dim, dim) matrix from packed lower-tri entries."""
        A_flat = self.poisson_A(y)                                  # (dim*(dim-1)//2,)
        A = jnp.zeros((self.dim, self.dim))
        A = A.at[self.poisson_A_idx[0], self.poisson_A_idx[1]].set(A_flat)
        A = A - A.T                                                 # antisymmetrize
        return A                                                    # (dim, dim)

    def poisson_product(self, y, dE, dS):
        """
        Compute L(y) dE where
        L = A - ((A dS) ⊗ dS - (dE·dS) A dS ⊗ ???) / ||dS||^2
        Implemented as: L dE = A dE + ((dE·A dS) dS - (dE·dS) A dS)/||dS||^2
        Shapes: y,(dE,dS) ∈ R^dim → output ∈ R^dim
        """
        A = self._build_skew_A(y)               # (dim, dim)
        AdE = A @ dE                             # (dim,)
        AdS = A @ dS                             # (dim,)

        dE_AdS = jnp.dot(dE, AdS)               # scalar
        dE_dS  = jnp.dot(dE, dS)                # scalar
        dS_sq  = jnp.dot(dS, dS)                # scalar

        denom  = dS_sq + self.eps
        correction = (dE_AdS * dS - dE_dS * AdS) / denom
        return AdE + correction                  # (dim,)

    def friction_product(self, y, dE, dS):
        """
        Compute M(y) dS with M = P_E^⊥ B D B^T P_E^⊥, D = C C^T ⪰ 0.
        Shapes: y,(dE,dS) ∈ R^dim → output ∈ R^dim
        """
        # Unpack networks
        B = self.friction_B(y).reshape(self.D, self.dim)      # (D, dim)
        C = self.friction_C(y).reshape(self.D, self.C2)       # (D, C2)

        # D = C C^T  (PSD)
        Dmat = C @ C.T                                        # (D, D)

        # Project each row b_s off of dE → v_s = P_E^⊥ b_s
        dE_sq = jnp.dot(dE, dE) + self.eps                    # scalar
        # For each row s: v_s = b_s - ((b_s·dE)/||dE||^2) dE
        b_dot_dE = B @ dE                                     # (D,)
        V = B - (b_dot_dE[:, None] / dE_sq) * dE[None, :]     # (D, dim)

        # v_t · dS
        v_dot_dS = V @ dS                                     # (D,)

        # alpha_s = sum_t D_{s t} (v_t · dS) = (D @ v_dot_dS)_s
        alpha = Dmat @ v_dot_dS                               # (D,)

        # MdS = sum_s alpha_s v_s  = V^T @ alpha
        MdS = V.T @ alpha                                     # (dim,)
        return MdS
    
    # Scalar energies per sample; grads in R^dim
    def energy(self, x):   # x: (dim,)
        return self.E_MLP(x).squeeze()
    
    def entropy(self, x):
        return self.S_MLP(x).squeeze()
    
    def get_terms(self, y):
        """
        Returns the terms L, M, dE, dS for the metriplectic form.
        Shapes: y ∈ R^dim → (L ∈ R^(dim, dim), M ∈ R^(dim, dim), dE ∈ R^dim, dS ∈ R^dim)
        """
        dE = jax.grad(self.energy)(y)         # (dim,)
        dS = jax.grad(self.entropy)(y)        # (dim,)
        
        LdE = self.poisson_product(y, dE, dS)   # (dim,)
        MdS = self.friction_product(y, dE, dS)  # (dim,)
        return LdE, MdS, dE, dS

    def get_penalty(self, y):
        """
        Degeneracy penalties (should be ~0 if constraints are satisfied):
          L∇S = 0 and M∇E = 0.
        """
        dE = jax.grad(self.energy)(y)         # (dim,)
        dS = jax.grad(self.entropy)(y)        # (dim,)
        
        LdS = self.poisson_product(y, dS, dS)  # (dim,)
        MdE = self.friction_product(y, dE, dE) # (dim,)
        return LdS, MdE

    def __call__(self, t, y, args):
        """
        y: (dim,) → returns (dim,)
        """
        LdE, MdS, _, _ = self.get_terms(y)  # (dim,), (dim,), (dim,), (dim,)
        return (LdE + MdS).squeeze()

"""
Our own model.
"""

class MLPMatrix(eqx.Module):
    n_in: int = eqx.static_field()
    symmetric: bool = eqx.static_field()
    antisymmetric: bool = eqx.static_field()
    unitary: bool = eqx.static_field()
    skew_block_diag: bool = eqx.static_field()
    diagonal: bool = eqx.static_field()
    psd: bool = eqx.static_field()
    kernel: int = eqx.static_field()
    model: eqx.Module

    def __init__(self, n_in: int, width: int, depth:int, symmetric=False, psd=False, antisymmetric=False, unitary=False, skew_block_diag=False, diagonal=False, kernel=0, *, key):
        self.n_in = n_in
        self.symmetric = symmetric # Symmetric? Here n_out should be (n_in-1)n_in/2
        self.antisymmetric = antisymmetric # Antiymmetric? Here n_out should be (n_in-1)n_in/2
        self.unitary = unitary
        self.skew_block_diag = skew_block_diag
        self.diagonal = diagonal # Takes precedence over others. Here n_out should be n_in
        self.psd = psd # Only for symmetric matrices (or diagonal)
        self.kernel = kernel
        self.model = 0
        if self.diagonal:
            self.model = eqx.nn.MLP(
                            in_size=n_in,
                            out_size=n_in,
                            width_size=width,
                            depth=depth,
                            use_final_bias=False,
                            activation=jnn.tanh,
                            key=key,
                        )
        elif self.skew_block_diag:
            self.model = eqx.nn.MLP(
                            in_size=n_in,
                            out_size=n_in//2-kernel,
                            width_size=width,
                            depth=depth,
                            use_final_bias=False,
                            activation=jnn.tanh,
                            key=key,
                        )
        elif self.symmetric or self.antisymmetric or self.unitary:
            self.model = eqx.nn.MLP(
                            in_size=n_in,
                            out_size=int((n_in-1)*n_in/2),
                            width_size=width,
                            depth=depth,
                            use_final_bias=False,
                            activation=jnn.tanh,
                            key=key,
                        )
    
    def __call__(self, x: jax.Array):
        x = self.model(x)
        if self.diagonal:
            A = jnp.zeros((self.n_in,self.n_in))
            # Make diagonal elements positive (kills 0 elements!)
            #if self.psd: x = jax.nn.softplus(x)
            if self.psd: x = x**2
            i_diag = jnp.pad(jnp.arange(self.n_in-self.kernel), (0, self.kernel))
            j_diag = jnp.pad(jnp.arange(self.n_in-self.kernel), (0, self.kernel))
            A = A.at[i_diag, j_diag].set(x)
            x = A
        elif self.symmetric:
            A = jnp.zeros((self.n_in,self.n_in))
            # Get the indices of the strictly upper-triangular elements
            i_upper, j_upper = jnp.triu_indices(self.n_in, k=1)
            # Assign the given array values to these indices
            A = A.at[..., i_upper, j_upper].set(x)
            # Make the matrix symmetric and psd if needed
            if self.psd:
                # Make diagonal elements positive (kills 0 elements!)
                #diag = jnp.diag(A)
                #A = A - jnp.diag(diag) + jnp.diag(jax.nn.softplus(diag))
                A = A.T@A
            else:
                A = A + A.T
            x = A
        elif self.antisymmetric or self.unitary:
            A = jnp.zeros((self.n_in,self.n_in))
            # Get the indices of the strictly upper-triangular elements
            i_upper, j_upper = jnp.triu_indices(self.n_in, k=1)
            # Assign the given array values to these indices
            A = A.at[i_upper, j_upper].set(x)
            # Make the matrix antisymmetric
            A = A - A.T
            if self.unitary: A = jax.scipy.linalg.expm(A)
            x = A
        elif self.skew_block_diag:
            # Make sparse in the future
            m = self.n_in//2 - self.kernel
            A = jnp.zeros((self.n_in,self.n_in))
            # For each i, we place a block at (2i,2i+1) = mu and (2i+1,2i) = -mu
            # Create index arrays:
            rows_upper = 2*jnp.arange(m)
            cols_upper = 2*jnp.arange(m)+1
            rows_lower = 2*jnp.arange(m)+1
            cols_lower = 2*jnp.arange(m)
            # Assign values
            A = A.at[rows_upper, cols_upper].set(x) #  [0,mu] block
            A = A.at[rows_lower, cols_lower].set(-x) # [-mu,0] block
            x = A
        return x

class FEMS(MLPVectorField):
    L_func: eqx.Module
    M_func: eqx.Module
    F_func: eqx.Module
    F_MLP: eqx.nn.MLP
    Q_MLP: MLPMatrix
    Sigma_MLP: MLPMatrix
    U_MLP: MLPMatrix
    Lambda_MLP: MLPMatrix

    def __init__(self, dim:int, width: int, depth:int, *, L_func=None, M_func=None, F_func=None, key, **kwargs):
        key, subkey = jax.random.split(key, num=2)
        super().__init__(dim=dim, width=width, depth=depth, key=subkey, **kwargs)
        key1,key2,key3,key4,key5 = jax.random.split(key, 5)
        ### Take into account the model might learn additional 0 eigenvalues? We cannot incorporate those in the E and S evaluations
        #self.dim_null_L = dim_null_L
        #self.dim_null_M = dim_null_M
        self.L_func = L_func
        self.M_func = M_func
        self.F_func = F_func
        if L_func is None:
            self.Q_MLP = MLPMatrix(dim, width, depth, unitary=True, key=key1)
            self.Sigma_MLP = MLPMatrix(dim, width, depth, skew_block_diag=True, key=key2)
        if M_func is None:
            self.U_MLP = MLPMatrix(dim,width, depth, unitary=True, key=key3)
            self.Lambda_MLP = MLPMatrix(dim, width, depth, diagonal=True, psd=True, key=key4)
        if F_func is None:
            self.F_MLP = eqx.nn.MLP(
                            in_size=dim,
                            out_size=1,
                            width_size=width,
                            depth=depth,
                            use_final_bias=False,
                            activation=jnn.tanh,
                            key=key,
                        )
        #self.E_MLP = MLP(dim_null_M,n_hidden,1,rngs=rngs)
        #self.S_MLP = MLP(dim_null_L,n_hidden,1,rngs=rngs)

    def energy(self,x):
        return jnp.squeeze(self.F_MLP(x))
    
    def entropy(self,x):
        return jnp.squeeze(self.F_MLP(x))
    
    def free_energy(self,x):
        return jnp.squeeze(self.F_MLP(x))
    
    def get_terms(self, x):
        if self.L_func is None:
            Q = self.Q_MLP(x)
            Sigma = self.Sigma_MLP(x)
            L = Q@Sigma@Q.T
        else:
            L = self.L_func(x)
        if self.M_func is None:
            U = self.U_MLP(x)
            Lambda = self.Lambda_MLP(x)
            M = U@Lambda@U.T
        else:
            M = self.M_func(x)
        if self.F_func is None:
            gradF = jax.grad(self.free_energy, argnums=0)(x)
        else:
            gradF = jax.grad(lambda x: jnp.squeeze(self.F_func(x)), argnums=0)(x)
        return L@gradF, M@gradF, gradF, gradF
    
    def get_penalty(self, x):
        LdF,MdF,gradF,_ = self.get_terms(x)
        return LdF, MdF
    
    def __call__(self, t, x, args):
        L,M,gradF,_ = self.get_terms(x)

        #self.energy(x,U)
        #gradE = jax.grad(self.energy, argnums=0)(x, U)
        #gradS = jax.grad(self.entropy, argnums=0)(x, Q)

        #poisson = L@gradE
        #metric = M@gradS
        metriplectic = (L+M)@gradF #poisson + metric

        #jax.debug.print("jax.debug.print(metriplectic) -> {x}", x=metriplectic)

        return metriplectic