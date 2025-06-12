import time
import numpy as np

from typing import Callable, List, Union, Iterator

import diffrax
from diffrax import diffeqsolve, Tsit5, Kvaerno5, ODETerm, SaveAt, PIDController
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping

jax.config.update("jax_enable_x64", True)

class SystemTrajectoryGenerator():
    def __init__(self, vector_field, dim, solver, variables, args, trainset_stats=None):
        self.vector_field = vector_field
        self.dim = dim
        self.solver = solver
        self.variables = variables
        self.args = args
        self.final_time = 0.0
        self.trajectories = np.array([])
    
    def solve(self, y0, ts, dt, rtol=1e-8, atol=1e-8):
        t0 = ts[0]
        t1 = ts[-1]
        saveat = SaveAt(ts=ts)
        term = ODETerm(self.vector_field)
        sol = diffeqsolve(term, self.solver, t0, t1, dt, y0, args=self.args, saveat=saveat, stepsize_controller=PIDController(rtol=rtol, atol=atol))
        return sol.ts, sol.ys
    
    def generate_trajectories(self, y0s, t0, t1, dt):
        trajectories = []
        ts0 = jnp.linspace(t0,t1,int((t1-t0)/dt))
        for y0 in y0s:
            ts, ys = self.solve(y0, ts0, dt)
            trajectories.append(ys)
        self.trajectories = np.array(trajectories)
        self.final_time = t1
        return self.trajectories
    
    def plot_trajectories(self):
        n_state = self.trajectories.shape[2]
        n_trajectories = self.trajectories.shape[0]
        times = np.linspace(0.0, self.final_time, self.trajectories.shape[1])
        fig, axs = plt.subplots(n_state, 1, figsize=(10, 5*n_state))
        for i in range(n_state):
            for j in range(n_trajectories):
                axs[i].plot(times, self.trajectories[j,:,i])
            axs[i].set_title(self.variables[i])
        plt.show()

    def generate_training_data(self, ts, y0s, dt):
        training = []
        for y0 in y0s:
            ts, ys = self.solve(y0, ts, dt)
            training.append(ys)
        return ts,np.array(training)
    
    def random_initial_conditions(self, n_samples, mins, maxs, key):
        initial_conditions = jax.random.uniform(key, (n_samples,self.dim)) * (maxs - mins) + mins
        return initial_conditions
        
    def trajectories_from_random_ics(self, N, mins, maxs, ts, dt, seed=0):
        key = jax.random.PRNGKey(seed)
        y0s = self.random_initial_conditions(N, jnp.array(mins), jnp.array(maxs), key)
        ts, training = self.generate_training_data(ts, y0s, dt)
        return ts,training