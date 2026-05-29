"""
Defines 2D L-shaped problem domain and decomposition used in our paper:
ELM-FBPINNs: An Efficient Multilevel Random Feature Method
"""

import jax
import jax.numpy as jnp
import numpy as np

from fbpinns.domains import Domain, RectangularDomainND
from fbpinns.decompositions import MultilevelRectangularDecompositionND
from fbpinns import networks


class LDomain2D(Domain):
    "Defines a L-shaped 2D domain"

    @staticmethod
    def init_params(xmin, xmax):

        assert xmin.shape == xmax.shape == (2,)

        static_params = {
            "xd":2,
            "xmin":jnp.array(xmin),
            "xmax":jnp.array(xmax),
            }
        return static_params, {}

    @staticmethod
    def sample_rectangle(all_params, key, sampler, batch_shape):

        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]

        # sample full box and get filter for top right
        assert len(batch_shape) == 2
        x_batch = RectangularDomainND._rectangle_samplerND(key, sampler, xmin, xmax, batch_shape)
        xh = xmin[0] + (xmax[0]-xmin[0])/2.
        yh = xmin[1] + (xmax[1]-xmin[1])/2.
        tr = (x_batch[:,0] > xh) & (x_batch[:,1] > yh)

        return x_batch, tr

    @staticmethod
    def sample_interior(all_params, key, sampler, batch_shape):

        # filter out top right
        x_batch, tr = LDomain2D.sample_rectangle(all_params, key, sampler, batch_shape)
        x_batch = x_batch[~tr]
        return x_batch

    @staticmethod
    def sample_boundaries(all_params, key, sampler, batch_shapes):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]

        assert len(batch_shapes) == 1
        batch_shape = batch_shapes[0]
        assert len(batch_shape) == 2
        b0 = batch_shape[0]//2 + batch_shape[0]%2
        b1 = batch_shape[1]//2 + batch_shape[1]%2
        xh = xmin[0] + (xmax[0]-xmin[0])/2.
        yh = xmin[1] + (xmax[1]-xmin[1])/2.
        x_batches = []
        # y lines
        for x0, mi, ma, b in [
                (xmin[0], xmin[1], xmax[1], batch_shape[1]),
                (xh, yh, xmax[1], b1),
                (xmax[0], xmin[1], yh, b1)
                ]:
                key, subkey = jax.random.split(key)
                x_batches.append(jnp.concatenate([
                    x0*jnp.ones((b,1)),
                    RectangularDomainND._rectangle_samplerND(subkey, sampler, jnp.array((mi,)), jnp.array((ma,)), (b,)),
                    ], axis=1)
                    )
        # x lines
        for y0, mi, ma, b in [
                (xmin[1], xmin[0], xmax[0], batch_shape[0]),
                (yh, xh, xmax[0], b0),
                (xmax[1], xmin[0], xh, b0),
                ]:
                key, subkey = jax.random.split(key)
                x_batches.append(jnp.concatenate([
                    RectangularDomainND._rectangle_samplerND(subkey, sampler, jnp.array((mi,)), jnp.array((ma,)), (b,)),
                    y0*jnp.ones((b,1)),
                    ], axis=1)
                    )
        return x_batches

    @staticmethod
    def norm_fn(all_params, x):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        mu, sd = (xmax+xmin)/2, (xmax-xmin)/2
        x = networks.norm(mu, sd, x)
        return x


class MultilevelLDecomposition2D(MultilevelRectangularDecompositionND):

    def init_params(xmin, xmax, subdomain_xss, subdomain_wss, unnorm):
        """Creates multiscale hyperrectangular subdomains initialised on a regular grid
        with subdomain centers subdomain_xs and widths subdomain_ws.
        """

        assert xmin.shape == xmax.shape == (2,)

        static_params, trainable_params = MultilevelRectangularDecompositionND.init_params(subdomain_xss, subdomain_wss, unnorm)

        m = static_params["m"]
        xd = static_params["xd"]
        params = static_params["subdomain"]["params"]
        pou = static_params["subdomain"]["pou"]
        assert xd == 2
        x_batch = (params[0]+params[1])/2

        # filter out top right
        xh = xmin[0] + (xmax[0]-xmin[0])/2.
        yh = xmin[1] + (xmax[1]-xmin[1])/2.
        tr = (x_batch[:,0] > xh) & (x_batch[:,1] > yh)
        x_batch = x_batch[~tr]

        m = len(x_batch)
        params = jax.tree_util.tree_map(lambda x: x[~tr], params)
        pou = jax.tree_util.tree_map(lambda x: x[~tr], pou)

        static_params = {
            "m":m,
            "xd":xd,
            "subdomain":{"params":params,
                         "pou":pou},
            }

        return static_params, trainable_params

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    key = jax.random.PRNGKey(0)

    # test L-shaped domain

    domain = LDomain2D
    sampler = "grid"

    xmin, xmax = jnp.array([0.1,-1]), jnp.array([1,2])
    batch_shape = (10,10)
    batch_shapes = ((10,10),)

    ps_ = domain.init_params(xmin, xmax)
    all_params = {"static":{"domain":ps_[0]}, "trainable":{"domain":ps_[1]}}
    x_batch = domain.sample_interior(all_params, key, sampler, batch_shape)
    x_batches = domain.sample_boundaries(all_params, key, sampler, batch_shapes)

    plt.figure()
    plt.scatter(x_batch[:,0], x_batch[:,1])
    for x_batch in x_batches:
        print(x_batch.shape)
        plt.scatter(x_batch[:,0], x_batch[:,1])
    plt.show()

    # test L-shaped decomposition

    decomposition = MultilevelLDecomposition2D

    subdomain_xss = [[np.linspace(-3,3,4), np.linspace(-2,2,3)],
                     [np.linspace(-3,3,10), np.linspace(-2,2,10)],
                     ]
    subdomain_wss = [[3*np.ones(4), 2.2*np.ones(3)],
                     [1*np.ones(10), 1*np.ones(10)],
                     ]

    ps_ = decomposition.init_params(np.array([-3,-2]), np.array([3,2]), subdomain_xss, subdomain_wss, (0,1))
    all_params = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}}
    m = all_params["static"]["decomposition"]["m"]
    active = np.ones(m)

    decomposition.plot(all_params, active=active, show_norm=True, show_window=True)

