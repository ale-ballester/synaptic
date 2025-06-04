import jax
import jax.numpy as jnp

class DataLoader:
    def __init__(self, ts, data, split=0.8, batch_size=32, shuffle=True):
        self.ts = ts
        self.data = data
        self.split = split
    
    def __call__(self, batch_size, key, n0=None, n1=None, nskip=None, nobs=None):
        nslice = slice(n0, n1, nskip)
        if nobs is None: nobs = slice(None, None, None)
        dataset_size = self.data.shape[0]
        indices = jnp.arange(dataset_size)
        while True:
            perm = jax.random.permutation(key, indices)
            (key,) = jax.random.split(key, 1)
            start = 0
            end = batch_size
            while end < dataset_size:
                batch_perm = perm[start:end]
                yield self.ts[nslice], self.data[batch_perm,nslice,nobs]
                start = end
                end = start + batch_size