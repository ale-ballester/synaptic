import numpy as np
if not hasattr(np, "issctype"):
    np.issctype = lambda t: issubclass(t, np.generic)
from diffrax import diffeqsolve, Tsit5
import jax
import jax.numpy as jnp

from models import NeuralODE, MLPVectorField
from system import SystemTrajectoryGenerator, MetriplecticSystem
from train import Trainer

jax.config.update("jax_enable_x64", True)

def SackurTetrode(S,Nkb,c_hat,V):
        return jnp.power(jnp.exp(S/Nkb)/(c_hat*V),2.0/3.0)

def M_gas(x, args):
    q, p, S1, S2 = x
    Nkb, m, alpha, L, c_hat, A_char = args
    
    V1 = A_char * q
    V2 = A_char * (2.0 * L - q)
    E1 = SackurTetrode(S1, Nkb, c_hat, V1)
    E2 = SackurTetrode(S2, Nkb, c_hat, V2)
    
    factor = 9.0 * (Nkb**2) * alpha / 4.0
    
    M = jnp.zeros((4, 4), dtype=x.dtype)
    M = M.at[2, 2].set(factor / (E1**2))
    M = M.at[2, 3].set(-factor / (E1 * E2))
    M = M.at[3, 2].set(-factor / (E1 * E2))
    M = M.at[3, 3].set(factor / (E2**2))
    
    return M

def E_gas(x, args):
    q, p, S1, S2 = x
    Nkb, m, alpha, L_, c_hat, A_char = args
    
    E_kin = p**2 / (2 * m)
    
    V1 = A_char * q
    V2 = A_char * (2.0 * L_ - q)
    
    E1 = SackurTetrode(S1, Nkb, c_hat, V1)
    E2 = SackurTetrode(S2, Nkb, c_hat, V2)
    
    return E_kin + E1 + E2

def S_gas(x, args):
    q, p, S1, S2 = x
    return S1 + S2

def L_gas(x, args):
    return jnp.array([
        [0.,  1., 0., 0.],
        [-1., 0., 0., 0.],
        [0.,  0., 0., 0.],
        [0.,  0., 0., 0.],
    ], dtype=x.dtype)

def gradE_gas(x, args):
    return jax.grad(lambda x_: E_gas(x_, args))(x)

def gradS_gas(x, args):
    return jax.grad(lambda x_: S_gas(x_, args))(x)

def two_gas_container(t, y, args):
    L = L_gas(y, args)
    M = M_gas(y, args)
    grad_E = gradE_gas(y, args)
    grad_S = gradS_gas(y, args)
    return L@grad_E + M@grad_S

Nkb = 1 # Number of particles and Boltzmann constant (characteristic unit of entropy)
m = 1 # Mass of wall
alpha = 8 # 
length = 1 # Distance from origin to the middle of the container (equilibrium wall position)
c_hat = 1 # Energy normalization
A_char = 1 # Average cross-sectional area
args = (Nkb, m, alpha, length, c_hat, A_char)

dim = 4
system = MetriplecticSystem([L_gas,M_gas,E_gas,S_gas], dim, Tsit5(), ["q", "p", "S1", "S2"], args)


key = jax.random.PRNGKey(42)
model = NeuralODE(MLPVectorField, system.dim, 64, 2, 0.01, key=key)
trainer = Trainer(system, model=model)
model, train_losses, valid_losses = trainer.train(N=4, 
                                                  N_valid=4, 
                                                  n_epochs=10, 
                                                  bs=2, 
                                                  bs_valid=2, 
                                                  mins=[0.2,-1,1,1], 
                                                  maxs=[1.8,1,3,3], 
                                                  ts_train=jnp.linspace(0, 10, 10000), 
                                                  dt=1e-3,
                                                  time_windows=[1, 10],
                                                  nrand=[100, 200],
                                                  save_every=100,
                                                  seed=42,
                                                  print_status=True)