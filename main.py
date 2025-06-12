import numpy as np
if not hasattr(np, "issctype"):
    np.issctype = lambda t: issubclass(t, np.generic)
from diffrax import diffeqsolve, Tsit5
import jax
import jax.numpy as jnp

from models import NeuralODE, NeuralMetriplecticODE, GFINNODE
from system import SystemTrajectoryGenerator, MetriplecticSystem
from train import train_loop

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


system = MetriplecticSystem([L_gas,M_gas,E_gas,S_gas], 4, Tsit5(), ["q", "p", "S1", "S2"], args)

model, train_losses, valid_losses = train_loop(system,
            y0_mins=[0.2, -2, 1, 1], # TODO: Play around with these values
            y0_maxs=[1.8, 2, 3, 3],
            model_class=GFINNODE,
            dataset_size=256,
            batch_size=64,
            n_epochs=10,
            valid_size=128,
            lr=3e-3,
            width=32,
            depth=5,
            K=3,
            t0=0.0,
            dt=1e-3,
            t1=[8.0], # This is approximate, the actual time will be given by the closest element in jnp.linspace(t0, max(t1), n_train)
            t_val=10.0,
            n_train=2000,
            n_valid=50,
            save_name="two_gas_container")