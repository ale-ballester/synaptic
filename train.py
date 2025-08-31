import time
import numpy as np

import matplotlib.pyplot as plt

import equinox as eqx
import optax
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from dataloader import DataLoader
from utils import make_dir

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
        self.nobs = None # If not None, should be a slice or list of indices of observed variables
        if optim is None:
            self.optim = optax.adam(lr)
        else:
            self.optim = optim(lr) # if you want to modify other parameters, prepass through a lambda (TODO: ugly, fix later)

        self.save_dir = save_dir
        self.save_name = save_name
    
    def L2_loss(self, model, ti, yi, alpha=0.0, beta=0.0, eps=1e-8):
        """
        Computes the L2 loss between the model predictions and the true values.
        Args:
            model: The model to evaluate.
            ti: The time points at which to evaluate the model.
            yi: The true values at the time points, with shape (batch_size, time_steps, dim).
        Returns:
            The L2 loss value.
        """
        if self.nobs is None: self.nobs = slice(None, None, None)
        print("yi shape in loss: ", yi.shape)
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:,0,:])
        loss = ((yi[:,:,self.nobs] - y_pred[:,:,self.nobs]) ** 2).mean(axis=(1,2)).mean()
        # loss = jnp.mean(jnp.mean(jnp.linalg.norm(y_pred - yi, axis=-1),axis=0),axis=0)

        if alpha > 0.0 or beta > 0.0:
            # Regularization term to avoid zero gradients for E and S
            dE = jax.vmap(jax.vmap(jax.grad(model.vector_field.energy), in_axes=0), in_axes=0)(y_pred)
            dS = jax.vmap(jax.vmap(jax.grad(model.vector_field.entropy), in_axes=0), in_axes=0)(y_pred)
            nE = jnp.linalg.norm(dE, axis=-1)
            nS = jnp.linalg.norm(dS, axis=-1)
            loss += jnp.mean(-alpha*(jnp.log(nE+eps)+jnp.log(nS+eps)) + beta*(nE**2+nS**2))
        return loss
    
    def make_step(self, ti, yi, model, opt_state):
        loss, grads = self.grad_loss(model, ti, yi)
        updates, opt_state = self.optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    
    def create_dataloader(self, N, mins, maxs, ts, dt, seed=0):
        ts, data = self.system.trajectories_from_random_ics(N, mins, maxs, ts, dt, seed=seed)
        dataloader = DataLoader(ts, data, nobs=self.nobs)
        return dataloader

    def train(self, N, N_valid, n_epochs, bs, bs_valid, mins, maxs, ts_train, dt, time_windows=None, nrand=None, nobs=None, save_every=100, seed=0, print_status=True, save_plots=False, diagnostics=False):
        make_step = eqx.filter_jit(self.make_step)

        self.nobs = nobs

        dl_train = self.create_dataloader(N, mins, maxs, ts_train, dt, seed=seed)
        dl_valid = self.create_dataloader(N_valid, mins, maxs, ts_train, dt, seed=seed+1)

        make_dir(self.save_dir)
        make_dir(self.save_dir + "png")

        opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))

        steps_per_epoch = max([int(N / bs),1])
        steps_valid = max([int(N_valid / bs_valid),1])

        N_time_schedules = len(time_windows) if time_windows is not None else 1
        schedule = [(None, None)]
        if time_windows is not None:
            if nrand is None:
                nrand = [None] * N_time_schedules
            elif len(nrand) != N_time_schedules:
                raise ValueError("nrand must have the same length as time_windows.")
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
                    print(f"Training until time {ts_train[n_t]} ({n_t} total samples), with {nrand_now} samples.\n")
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
                    self.plot_training(ts, yi, y_pred, epoch, train_loss_epoch, diagnostics=diagnostics)

        if print_status: print("Training complete.")
        checkpoint_name = self.save_dir+self.save_name+"_final"
        if print_status: print(f"Saving model at {checkpoint_name}")
        self.model.save_model(checkpoint_name)
        if save_plots:
            self.plot_training(ts, yi, y_pred, "final", train_loss_epoch, diagnostics=diagnostics)

        return self.model, train_losses, valid_losses
    
    def plot_training(self, ts, yi, y_pred, epoch, train_loss_epoch, dims=None, diagnostics=False):
        E_vals = eqx.filter_vmap(self.model.vector_field.energy)(y_pred[0])
        S_vals = eqx.filter_vmap(self.model.vector_field.entropy)(y_pred[0])

        # -------------------------------
        # 2. Figure 1 – state trajectories
        # -------------------------------
        fig_states, ax_states = plt.subplots(figsize=(10, 5))

        for i, var in enumerate(self.system.variables):
            ax_states.plot(ts, yi[0, :, i],           label=f"True {var}")
            ax_states.plot(ts, y_pred[0, :, i], "--", label=f"Predicted {var}")

        ax_states.set_xlabel("Time")
        ax_states.set_ylabel("State")
        ax_states.grid(which="both")
        ax_states.legend(loc="best")
        fig_states.suptitle(f"Epoch {epoch} – Training Loss: {train_loss_epoch:.4f}")
        fig_states.tight_layout()

        fname_states = f"{self.save_dir}png/epoch_{epoch}_states_loss_{train_loss_epoch:.4f}.png"
        fig_states.savefig(fname_states, dpi=150)
        plt.close(fig_states)         # no display

        # ----------------------------------------------
        # 3. Figure 2 – energy (E) and entropy (S) traces
        # WARNING: Depends on the model having energy and entropy methods (move elsewhere?)
        # ----------------------------------------------
        fig_es, ax_es = plt.subplots(figsize=(10, 3))

        ax_es.plot(ts, E_vals, "k-",  label="Energy E")
        ax_es.plot(ts, S_vals, "r--", label="Entropy S")
        ax_es.set_xlabel("Time")
        ax_es.set_ylabel("Value")
        ax_es.grid(which="both")
        ax_es.legend(loc="best")
        fig_es.suptitle(f"Epoch {epoch} – Energy & Entropy")
        fig_es.tight_layout()

        fname_es = f"{self.save_dir}png/epoch_{epoch}_ES_loss_{train_loss_epoch:.4f}.png"
        fig_es.savefig(fname_es, dpi=150)
        plt.close(fig_es)

        # ----------------------------------------------
        # 4. Figure 3 – Diagnostics (if applicable)
        # WARNING: Depends on the model having energy and entropy methods (move elsewhere?)
        # ----------------------------------------------
        if diagnostics:
            # Plot norms of nablaE, nablaS, LnablaE, LnablaS, MnablaE, MnablaS throughout state space
            # Use methods get_terms and get_penalty from model.vector_field
            LdE, MdS, dE, dS = eqx.filter_vmap(self.model.vector_field.get_terms)(y_pred[0])
            LdS, MdE = eqx.filter_vmap(self.model.vector_field.get_penalty)(y_pred[0])

            # Create a figure with two subplots
            fig_diag, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            # Plot the norms of the gradients
            ax1.plot(ts, jnp.linalg.norm(dE, axis=1), label="||∇E||", color="blue")
            ax1.plot(ts, jnp.linalg.norm(dS, axis=1), label="||∇S||", color="orange")
            ax1.plot(ts, jnp.linalg.norm(LdE, axis=1), label="||L∇E||", color="green")
            ax1.plot(ts, jnp.linalg.norm(MdS, axis=1), label="||M∇S||", color="red")
            ax1.plot(ts, jnp.linalg.norm(LdE+MdS, axis=1), label="||L∇E + M∇S||", color="black")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Norm")
            ax1.set_title("Norms of Gradients")
            ax1.grid(which="both")
            ax1.legend()

            # Plot the norms of the L and M terms
            ax2.plot(ts, jnp.linalg.norm(LdS, axis=1), label="||L∇E||", color="green")
            ax2.plot(ts, jnp.linalg.norm(MdE, axis=1), label="||M∇S||", color="red")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Norm")
            ax2.set_yscale("log")
            ax2.set_title("Degeneracy Conditions")
            ax2.grid(which="both")
            ax2.legend()

            fig_diag.suptitle(f"Epoch {epoch} - Diagnostics")
            fig_diag.tight_layout()
            fname_diag = f"{self.save_dir}png/epoch_{epoch}_diagnostics_loss_{train_loss_epoch:.4f}.png"
            fig_diag.savefig(fname_diag, dpi=150)
            plt.close(fig_diag)
            
    
    """
    def generate_heatmap(grid_size, box, t_max, data_length, odeint, model, system, device, traj=False, vec=False):
        # === Grid for initial conditions ===
        x_vals = jnp.linspace(box[0,0], box[0,1], grid_size).to(device)
        y_vals = jnp.linspace(box[1,0], box[1,1], grid_size).to(device)
        X, Y = jnp.meshgrid(x_vals, y_vals, indexing='ij')

        x0_flat = X.flatten()
        y0_flat = Y.flatten()
        z0 = jnp.stack([x0_flat, y0_flat], dim=1)  # shape (2500, 2)

        # === Simulate ===
        t_test = jnp.linspace(0., t_max, data_length).to(device)

        true_y = system.generate_trajectories(y0s, t_test[0], t_test[-1], t_test[1]-t_test[0])  # (N, T, D)
        pred_y = jax.vmap(model, in_axes=(None, 0))(t_test, y0s[:,:])     # (N, T, D)

        # === Compute MSE per trajectory ===
        squared_error = (pred_y - true_y)**2
        mse_per_traj = squared_error.mean(dim=(1, 2))  # (N,)
        mse_grid = mse_per_traj.reshape(grid_size, grid_size).cpu().numpy()

        # === Plot heatmap ===
        plt.figure(figsize=(7, 6))
        plt.contourf(X.cpu(), Y.cpu(), mse_grid, levels=50, cmap='viridis')
        plt.colorbar(label='Trajectory MSE')
        plt.xlabel('$x_0$')
        plt.ylabel('$y_0$')
        plt.title('Prediction Error Heatmap with Trajectories')

        if traj:
            # === Overlay a few trajectories ===
            indices = [
                int(grid_size * grid_size * 0.5 + grid_size * 0.5),  # center
                int(grid_size * grid_size * 0.2 + grid_size * 0.2),  # top-left
                int(grid_size * grid_size * 0.8 + grid_size * 0.8),  # bottom-right
                int(grid_size * grid_size * 0.2 + grid_size * 0.8),  # top-right
                int(grid_size * grid_size * 0.8 + grid_size * 0.2),  # bottom-left
            ]

            colors = ['r', 'g', 'b', 'c', 'm']
            counter = 0
            for idx in indices:
                # Extract trajectories
                true_traj = true_y[:, idx, :].cpu().numpy()
                pred_traj = pred_y[:, idx, :].cpu().numpy()

                # Plot: solid = true, dashed = predicted
                plt.plot(true_traj[:, 0], true_traj[:, 1], colors[counter]+'-', linewidth=1.5)
                plt.plot(pred_traj[:, 0], pred_traj[:, 1], colors[counter]+'--', linewidth=1.5)
                counter += 1

            # === Finalize plot ===
            from matplotlib.lines import Line2D

            custom_lines = [Line2D([0], [0], color="black", ls="-", lw=1),
                            Line2D([0], [0], color="black", ls="--", lw=1)]
            plt.legend(custom_lines, ['True Trajectory', 'Predicted Trajectory'], loc='upper left')
        if vec:
            # === Compute vector field difference ===
            vf_true = system(0, z0)       # shape (2500, 2)
            vf_pred = model(0, z0)           # shape (2500, 2)

            vf_diff = (vf_pred - vf_true).cpu().numpy()  # shape (2500, 2)
            U = vf_diff[:, 0].reshape(grid_size, grid_size)
            V = vf_diff[:, 1].reshape(grid_size, grid_size)

            # Normalize for arrow direction only (optional)
            mag = 1 #np.sqrt(U**2 + V**2) + 1e-8
            U_norm = U / mag
            V_norm = V / mag

            # Plot vector field difference
            plt.quiver(X.cpu(), Y.cpu(), U_norm, V_norm, angles='xy', scale_units='xy', scale=20, color='white', width=0.002)

        # === Finalize plot ===
        plt.xlim(box[0,0], box[0,1])
        plt.ylim(box[1,0], box[1,1])
        plt.tight_layout()
        plt.show()

        return X.cpu().numpy(), Y.cpu().numpy(), mse_grid
    """