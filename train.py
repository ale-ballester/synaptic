import time
import numpy as np

import matplotlib.pyplot as plt

import equinox as eqx
import optax
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from dataloader import DataLoader

class Trainer():
    def __init__(self, system, 
                       model_class=None, 
                       model=None,
                       vector_field=None,
                       depth=2,
                       width=64,
                       dt=0.01, # For NeuralODEs, the time step for the ODE solver
                       model_kwargs=(),
                       loss="L2",
                       lr=1e-4,
                       optim=None,
                       save_dir="model/", 
                       save_name="model_checkpoint",
                       seed=0):
        self.system = system
        if model_class is not None and model is None:
            if vector_field is None:
                raise ValueError("If model_class is provided, vector_field must also be provided.")
            key = jax.random.PRNGKey(seed)
            model = model_class(vector_field, system.dim, width, depth, dt, key=key, **model_kwargs)
        elif model_class is None and model is not None:
            self.model = model
        else:
            raise ValueError("Either model_class or model must be provided, but not both.")
        if loss == "L2":
            self.loss = self.L2_loss
        else:
            self.loss = loss # Same signature as L2_loss, but different implementation
        self.grad_loss = eqx.filter_value_and_grad(self.loss)
        self.lr = lr
        if optim is None:
            self.optim = optax.adam(lr)
        else:
            self.optim = optim(lr) # if you want to modify other parameters, prepass through a lambda (TODO: ugly, fix later)

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
    
    def make_step(self, ti, yi, model, opt_state):
        loss, grads = self.grad_loss(model, ti, yi)
        updates, opt_state = self.optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    
    def create_dataloader(self, N, mins, maxs, ts, dt, seed=0):
        ts, data = self.system.trajectories_from_random_ics(N, mins, maxs, ts, dt, seed=seed)
        dataloader = DataLoader(ts, data)
        return dataloader

    def train(self, N, N_valid, n_epochs, bs, bs_valid, mins, maxs, ts_train, dt, time_windows=None, nrand=None, save_every=100, seed=0, print_status=True, save_plots=False):
        make_step = eqx.filter_jit(self.make_step)
        dl_train = self.create_dataloader(N, mins, maxs, ts_train, dt, seed=seed)
        dl_valid = self.create_dataloader(N_valid, mins, maxs, ts_train, dt, seed=seed+1)

        opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))

        steps_per_epoch = int(N / bs)
        steps_valid = max([int(N_valid / bs_valid),1])

        N_time_schedules = len(time_windows) if time_windows is not None else 1
        schedule = [(None, None)]
        if time_windows is not None:
            if nrand is None:
                nrand = [None] * N_time_schedules
            schedule = list(zip(time_windows, nrand))

        train_losses = []
        valid_losses = []

        loader_key = jax.random.PRNGKey(seed)

        for epoch in range(n_epochs):
            index = epoch // (n_epochs // N_time_schedules)
            if epoch % (n_epochs // N_time_schedules) == 0 and index < N_time_schedules:
                t_now, nrand_now = schedule[index]
                n_t = round((t_now - ts_train[0]) / (ts_train[-1] - ts_train[0]) * (len(ts_train)-1))
                if print_status:
                    print("--------------------\n")
                    print(f"Training until time {ts_train[n_t]}, with {nrand_now} samples.\n")
            if print_status:
                print("--------------------")
                print(f"Epoch: {epoch}")
            train_loss_epoch = 0
            valid_loss_epoch = 0
            loader_key, train_loader_key = jax.random.split(loader_key)
            for step, batch in zip(range(steps_per_epoch),dl_train(bs, key=train_loader_key,n1=n_t+1,nrand=nrand_now)):
                start = time.time()
                ts, yi = batch
                loss, self.model, opt_state = make_step(ts, yi, self.model, opt_state)
                y_pred = jax.vmap(self.model, in_axes=(None, 0))(ts, yi[:,0,:])
                train_loss_epoch += loss
                end = time.time()
            train_loss_epoch /= steps_per_epoch
            train_losses.append(train_loss_epoch)
            loader_key, valid_loader_key = jax.random.split(loader_key)
            #for step, yi in zip(range(steps_valid),dl_valid(bs, key=train_loader_key)):
                ### TODO: Implement validation loss function
                #loss = val_loss(model, ts_valid, yi)
            #    valid_loss_epoch += loss
            #valid_loss_epoch /= steps_valid
            #valid_losses.append(valid_loss_epoch)
            if print_status: print(f"Train loss: {train_loss_epoch}, Valid loss: {valid_loss_epoch}")
            if epoch % save_every == 0 and epoch > 0 and epoch < n_epochs-1:
                if print_status: print(f"Saving model at epoch {epoch}")
                checkpoint_name = self.save_dir+self.save_name+f"_{epoch}"
                self.model.save_model(checkpoint_name)
                if save_plots:
                    plt.figure(figsize=(10, 5))
                    plt.plot(ts, yi[0,:,0], label='True Trajectory')
                    plt.plot(ts, y_pred[0,:,0], label='Predicted Trajectory')
                    plt.title(f'Epoch {epoch} - Training Loss: {loss:.4f}')
                    plt.xlabel('Time')
                    plt.ylabel('Value')
                    plt.legend()
                    plt.savefig(f"{self.save_dir}png/epoch_{epoch}_loss_{train_loss_epoch:.4f}.png")
                    plt.close()

        if print_status: print("Training complete.")
        checkpoint_name = self.save_dir+self.save_name+"_final"
        if print_status: print(f"Saving model at {checkpoint_name}")
        self.model.save_model(checkpoint_name)
        if save_plots:
            plt.figure(figsize=(10, 5))
            plt.plot(ts, yi[0,:,0], label='True Trajectory')
            plt.plot(ts, y_pred[0,:,0], label='Predicted Trajectory')
            plt.title(f'Epoch {epoch} - Training Loss: {loss:.4f}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig(f"{self.save_dir}png/final_loss_{train_loss_epoch:.4f}.png")
            plt.close()

        return self.model, train_losses, valid_losses