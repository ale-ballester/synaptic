import jax
import jax.numpy as jnp

class DataLoader:
    def __init__(self, ts, data, nobs=None):
        self.ts = ts
        self.data = data
        self.n_times = len(ts)
        self.nobs = nobs
        if nobs is not None:
            dim = data.shape[-1]
            self.data = self.data[:,:,nobs]
            len_nobs = len(range(*nobs.indices(dim)))
            nentropy = dim-len_nobs
            if nentropy > 0:
                print(f"DataLoader: Ignoring {nentropy} entropy dimensions.")
                virtual_entropy = jnp.linspace(0, 1, self.n_times)
                virtual_entropy = jnp.broadcast_to(virtual_entropy[None, :, None], (data.shape[0], self.n_times, nentropy))
                self.data = jnp.concatenate([self.data, virtual_entropy], axis=-1)
                print(f"DataLoader: Added {nentropy} virtual entropy dimensions.")
    
    def __call__(self, batch_size, key, ind_times=None, n0=None, n1=None, nskip=None, nobs=None, nrand=None):
        key_perm, key_times = jax.random.split(key, 2)
        dataset_size = self.data.shape[0]
        indices = jnp.arange(dataset_size)
        if nobs is None: nobs = slice(None, None, None)
        if nrand is None:
            if ind_times is None: ind_times = slice(n0, n1, nskip)
        while True:
            perm = jax.random.permutation(key, indices)
            (key_perm,) = jax.random.split(key_perm, 1)
            start = 0
            end = batch_size
            while end < dataset_size:
                if nrand is not None:
                    if n0 is None:
                        key_times,key_n0 = jax.random.split(key_times, 2)
                        n0 = jax.random.choice(key_n0, jnp.arange(self.n_times-n1)) if n1 < self.n_times else 0
                    else:
                        (key_times,) = jax.random.split(key_times, 1)
                    ind_times = jnp.sort(jax.random.choice(key_times, jnp.arange(n0,n0+n1), shape=(nrand,), replace=False))
                batch_perm = perm[start:end]
                subset = self.data[batch_perm,:,:]
                subset = subset[:,ind_times,:]
                subset = subset[:,:,nobs]
                yield self.ts[ind_times], subset
                start = end
                end = start + batch_size