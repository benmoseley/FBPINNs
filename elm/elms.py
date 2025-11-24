"""
Defines ELM networks with linear bases suitable for linear solves.
"""


from jax import random
import jax.numpy as jnp
import jax

from fbpinns.networks import Network


class ELM(Network):
    "Extreme learning machine"

    @staticmethod
    def init_params(key, layer_sizes, weight_scale):
        keys = random.split(key, len(layer_sizes)-1)
        params = [ELM._random_layer_params(k, m, n, weight_scale)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        static_params = {"basis_hidden": params[:-1]}
        p = jnp.concatenate([params[-1][0], params[-1][1].reshape(-1,1)], axis=1)
        trainable_params = {"basis_coeffs": p}
        return static_params, trainable_params

    @staticmethod
    def _random_layer_params(key, m, n, weight_scale):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        v = weight_scale*jnp.sqrt(1/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        return w,b

    @staticmethod
    def network_fn(params, x):
        x = ELM.basis_fn(params, x)
        a = params["trainable"]["network"]["subdomain"]["basis_coeffs"]
        x = jnp.dot(a, x)# (layer_sizes[-1],)
        return x

    @staticmethod
    def basis_fn(params, x):
        for w, b in params["static"]["network"]["subdomain"]["basis_hidden"]:
            x = jnp.dot(w, x) + b
            x = jnp.tanh(x)# (layer_sizes[-2],)
        x = jnp.concatenate([jnp.array([1]),
                             x])
        return x

class ELM_sigmoid(ELM):

    @staticmethod
    def network_fn(params, x):
        x = ELM_sigmoid.basis_fn(params, x)
        a = params["trainable"]["network"]["subdomain"]["basis_coeffs"]
        x = jnp.dot(a, x)# (layer_sizes[-1],)
        return x

    @staticmethod
    def basis_fn(params, x):
        for w, b in params["static"]["network"]["subdomain"]["basis_hidden"]:
            x = jnp.dot(w, x) + b
            x = jax.nn.sigmoid(x)# (layer_sizes[-2],)
        x = jnp.concatenate([jnp.array([1]),
                             x])
        return x

class ELM_SIREN(Network):

    @staticmethod
    def init_params(key, layer_sizes, weight_scale):
        keys = random.split(key, len(layer_sizes)-1)
        params = [ELM_SIREN._random_layer_params(k, m, n, weight_scale)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        static_params = {"basis_hidden": params[:-1]}
        p = jnp.concatenate([params[-1][0], params[-1][1].reshape(-1,1)], axis=1)
        trainable_params = {"basis_coeffs": p}
        return static_params, trainable_params

    @staticmethod
    def _random_layer_params(key, m, n, weight_scale):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        v = weight_scale*jnp.sqrt(6/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        return w,b

    @staticmethod
    def network_fn(params, x):
        x = ELM_SIREN.basis_fn(params, x)
        a = params["trainable"]["network"]["subdomain"]["basis_coeffs"]
        x = jnp.dot(a, x)# (layer_sizes[-1],)
        return x

    @staticmethod
    def basis_fn(params, x):
        for w, b in params["static"]["network"]["subdomain"]["basis_hidden"]:
            x = jnp.dot(w, x) + b
            x = jnp.sin(x)# (layer_sizes[-2],)
        x = jnp.concatenate([jnp.array([1]),
                             x])
        return x

