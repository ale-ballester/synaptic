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
    energy: eqx.Module
    entropy: eqx.Module
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

        self.energy = eqx.nn.MLP(
            in_size=dim,
            out_size=1,
            width_size=nE,
            depth=lE,
            use_final_bias=False,
            activation=jnn.tanh,
            key=k4,
        )
        self.entropy = eqx.nn.MLP(
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

    def get_penalty(self, y, dE, dS):
        """
        Degeneracy penalties (should be ~0 if constraints are satisfied):
          L∇S = 0 and M∇E = 0.
        """
        LdS = self.poisson_product(y, dS, dS)  # (dim,)
        MdE = self.friction_product(y, dE, dE) # (dim,)
        return LdS, MdE

    def __call__(self, t, y, args):
        """
        y: (dim,) → returns (dim,)
        """
        # Scalar energies per sample; grads in R^dim
        def energy_scalar(x):   # x: (dim,)
            return self.energy(x).squeeze()
        def entropy_scalar(x):
            return self.entropy(x).squeeze()

        dE = jax.grad(energy_scalar)(y)         # (dim,)
        dS = jax.grad(entropy_scalar)(y)        # (dim,)

        LdE = self.poisson_product(y, dE, dS)   # (dim,)
        MdS = self.friction_product(y, dE, dS)  # (dim,)
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
    
    def __call__(self, t, x, args):
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

        #self.energy(x,U)
        #gradE = jax.grad(self.energy, argnums=0)(x, U)
        #gradS = jax.grad(self.entropy, argnums=0)(x, Q)

        #poisson = L@gradE
        #metric = M@gradS
        metriplectic = (L+M)@gradF #poisson + metric

        #jax.debug.print("jax.debug.print(metriplectic) -> {x}", x=metriplectic)

        return metriplectic