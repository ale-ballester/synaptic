import time
import numpy as np

from typing import Callable, List, Union, Iterator

import diffrax
from diffrax import diffeqsolve, Tsit5, Kvaerno5, ODETerm, SaveAt, PIDController
import equinox as eqx
import optax
import jax
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float
import jax.tree_util as jtu

from models import NeuralODE, NeuralMetriplecticODE, GFINNODE

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from utils import Args, save_model

class Trainer():
    def __init__(self, system, 
                       model_class=None, 
                       model=None,
                       vector_field=None,
                       depth=2,
                       width=64,
                       model_args=(),
                       loss="L2",
                       save_dir="/tmp/", 
                       save_name="model_checkpoint"):
        self.system = system
        if model_class is not None and model is None:
            if vector_field is None:
                raise ValueError("If model_class is provided, vector_field must also be provided.")
            model = model_class(vector_field, system.dim, width, depth, *model_args)
        elif model_class is None and model is not None:
            self.model = model
        else:
            raise ValueError("Either model_class or model must be provided, but not both.")
        if loss == "L2":
            self.loss = self.L2_loss
        else:
            self.loss = loss # Same signature as L2_loss, but different implementation
        self.save_dir = save_dir
        self.save_name = save_name
    
    def L2_loss(self, model, ti, yi):
        """
        Computes the L2 loss between the model predictions and the true values.
        Args:
            model: The model to evaluate.
            ti: The time points at which to evaluate the model.
            yi: The true values at the time points, with shape (batch_size, time_steps, dim).
        Returns:
            The L2 loss value.
        """
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:,0,:])
        loss = ((yi - y_pred) ** 2).mean(axis=(1,2)).mean()
        return loss

    def train(self, *args, **kwargs):
        grad_loss = eqx.filter_value_and_grad(self.loss)
        

def train_loop(system,
                y0_mins,
                y0_maxs,
                model_class=None,
                model=None,
                dataset_size=1024,
                batch_size=4,
                n_epochs=200,
                valid_size=10,
                lr=1e-4,
                width=64,
                depth=2,
                K=1,
                t0=0,
                dt=1e-3,
                t1=2,
                t_val=4,
                seed=5678,
                n_train=10,
                n_valid=10,
                print_status=True,
                save_every=100,
                save_dir="/Users/aballester3/Projects/Metriplectic/metriplectic/diffrax-example/",
                save_name="model_checkpoint"):
    
    save_dir = save_dir + save_name + "/"

    # We have a longer validation time horizon so our search prioritizes hyperparameters with long-term good performance
    # We have two loss functiosn, one for training and one for validation
    # This is to avoid long compile times for the validation loss function (since ts is bigger, it may generate a bigger graph)

    if np.isscalar(t1): t1 = np.array([t1])
    else: t1 = np.array(t1)
    N_time_schedules = t1.shape[0]

    ts_train = jnp.linspace(t0, np.max(t1), n_train)
    print(f"Training time schedule: {ts_train}")
    ts_valid = jnp.linspace(t0, t_val, n_valid)

    @eqx.filter_value_and_grad
    @eqx.debug.assert_max_traces(max_traces=t1.shape[0])
    def grad_loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:,0,:])
        loss = ((yi - y_pred) ** 2).mean(axis=(1,2)).mean()
        return loss

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=t1.shape[0])
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    
    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def val_loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:,0,:])
        return ((yi - y_pred) ** 2).mean(axis=(1,2)).mean()
    
    key = jax.random.PRNGKey(seed) # TODO: Check initialization
    train_key, valid_key, model_key, loader_key = jax.random.split(key, 4)

    system.populate_training_data(dataset_size, y0_mins, y0_maxs, ts_train, dt, train_key, save=True, save_name=save_dir)
    system.populate_validation_data(valid_size, y0_mins, y0_maxs, ts_valid, dt, valid_key)
    system_dim = system.dim

    args = Args()
    
    #args.system_params = system.args
    #args.trainset_stats = [system.mins, system.maxs]
    model = model_class(system_dim, width, depth, dt, K, key=model_key)
    #print(model(np.zeros(system_dim,), np.linspace(0, 1, 100)).shape)
    args.dim = system_dim
    args.width = width
    args.depth = depth
    args.dt = dt
    args.K = K
    
    # Print model and system evaluations
    if print_status:
        x = jnp.ones(system_dim)
        print("Model evaluation:")
        print(model.func(0, x, None))

        if model_class == GFINNODE:
            print("Checking GFINNODE properties...")
            y = jax.random.normal(jax.random.PRNGKey(0), (system_dim,))

            L, gradS = model.func.LgradS(y)
            M, gradE = model.func.MgradE(y)

            assert jnp.allclose(L @ gradS.T, 0.0, atol=1e-6)
            assert jnp.allclose(M @ gradE.T, 0.0, atol=1e-6)
            assert jnp.allclose(L + L.T, 0.0, atol=1e-6)
            assert jnp.allclose(M - M.T, 0.0, atol=1e-6)
            eigvals = jnp.linalg.eigvalsh(M)
            assert (eigvals >= -1e-9).all()
            print("GFINNODE properties satisfied.")
        key = jax.random.PRNGKey(0)
        ys  = jax.random.normal(key, (512, system_dim))
        vf  = jax.vmap(model.func, in_axes=(None, 0, None))(0.0, ys, None)
        print("mean ‖f̂‖₂:", jnp.mean(jnp.linalg.norm(vf, axis=1)))
        print("System evaluation:")
        print(system.vector_field(0, system.denormalize_data(x), None))

    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    steps_per_epoch = int(dataset_size / batch_size)
    steps_valid = max([int(valid_size / batch_size),1])
    print(f"Steps per epoch: {steps_per_epoch}, Steps valid: {steps_valid}")

    train_losses = []
    valid_losses = []

    if print_status:
        print("Training model...")

    for epoch in range(n_epochs):
        if print_status:
            print("--------------------")
            print(f"Epoch: {epoch}")
        if epoch % (n_epochs // N_time_schedules) == 0 and epoch // (n_epochs // N_time_schedules) < N_time_schedules:
            t_now = t1[epoch // (n_epochs // N_time_schedules)]
            print(f"Training for time {t_now}")
            n_t = round((t_now - ts_train[0]) / (ts_train[-1] - ts_train[0]) * len(ts_train))
            ts = ts_train[:n_t]
            if print_status:
                print(f"Training for {n_t} timesteps, time window {t0} to {ts[-1]}")
        train_loss_epoch = 0
        valid_loss_epoch = 0
        loader_key, train_loader_key = jax.random.split(loader_key)
        for step, yi in zip(range(steps_per_epoch),system.dataloader(batch_size, time_clip=n_t, key=train_loader_key)):
            start = time.time()
            #ids = jnp.asarray(yi[:, 0, 0])  # e.g. first state component
            #print(f"step {step} ids:", ids[:5])   # print first 5 ids
            """
            for i in range(batch_size):
                plt.plot(yi[i,:, 0])
            plt.show()
            for i in range(batch_size):
                plt.plot(yi[i,:, 1])
            plt.show()
            for i in range(batch_size):
                plt.plot(yi[i,:, 2])
            plt.show()
            for i in range(batch_size):
                plt.plot(yi[i,:, 3])
            plt.show()
            """
            """
            print(
                f"Parameters of all model layers:\n"
                f"{jtu.tree_leaves(model)}\n"
            )
            """
            loss, model, opt_state = make_step(ts, yi, model, opt_state)
            train_loss_epoch += loss
            end = time.time()
        train_loss_epoch /= steps_per_epoch
        train_losses.append(train_loss_epoch)
        loader_key, valid_loader_key = jax.random.split(loader_key)
        for step, yi in zip(range(steps_valid),system.dataloader(batch_size, key=valid_loader_key, train=False)):
            loss = val_loss(model, ts_valid, yi)
            valid_loss_epoch += loss
        valid_loss_epoch /= steps_valid
        valid_losses.append(valid_loss_epoch)
        if print_status: print(f"Train loss: {train_loss_epoch}, Valid loss: {valid_loss_epoch}")
        if epoch % save_every == 0 and epoch > 0 and epoch < n_epochs-1:
            print(f"Saving model at epoch {epoch}")
            checkpoint_name = save_dir+f"model_checkpoint_{epoch}"
            save_model(checkpoint_name, args, model)

    print("Training complete.")
    checkpoint_name = save_dir+"model_checkpoint_final"
    print(f"Saving model at {checkpoint_name}")
    save_model(checkpoint_name, args, model)

    return model, train_losses, valid_losses