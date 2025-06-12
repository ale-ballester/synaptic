import jax
import jax.numpy as jnp

class DataLoader:
    def __init__(self, ts, data):
        self.ts = ts
        self.data = data
    
    def __call__(self, batch_size, key, ntimes=None, n0=0, n1=None, nskip=None, nobs=None, nrand=None):
        dataset_size = self.data.shape[0]
        indices = jnp.arange(dataset_size)
        if nobs is None: nobs = slice(None, None, None)
        if nrand is None:
            if ntimes is None: ntimes = slice(n0, n1, nskip)
        while True:
            perm = jax.random.permutation(key, indices)
            (key,) = jax.random.split(key, 1)
            start = 0
            end = batch_size
            while end < dataset_size:
                key, key_times = jax.random.split(key, 2)
                if nrand is not None:
                    ntimes = jax.random.choice(key_times, self.ts.shape[0], shape=(nrand,), replace=False)
                batch_perm = perm[start:end]
                yield self.ts[ntimes], self.data[batch_perm,ntimes,nobs]
                start = end
                end = start + batch_size