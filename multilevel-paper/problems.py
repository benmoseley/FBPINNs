"""
Defines all of the problems used in our paper:
Multilevel domain decomposition-based architectures for physics-informed neural networks
https://arxiv.org/abs/2306.05486
"""

import numpy as np
import jax.numpy as jnp

from fbpinns.problems import Problem
from fbpinns.traditional_solutions.helmholtz_solver import helmholtz_solver


class Laplace1D_quadratic(Problem):

    @staticmethod
    def init_params(sd=0.2, adaptive_weights=()):
        static_params = {
            "dims":(1,1),
            "sd":sd,
            }
        if adaptive_weights:
            trainable_params = {
                "adaptive_weights": [jnp.ones(batch_shape, dtype=float) for batch_shape in adaptive_weights]
                }
        else:
            trainable_params = {}
        return static_params, trainable_params

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(0,0)),
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        tanh, sd = jnp.tanh, all_params["static"]["problem"]["sd"]
        x = x_batch[:,0:1]
        u = tanh((x-0)/sd)*tanh((1-x)/sd)*u
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        _, uxx = constraints[0]
        f = 8
        phys = f + uxx
        phys2 = phys**2
        if "problem" in all_params["trainable"] and "adaptive_weights" in all_params["trainable"]["problem"]:
            phys2 = all_params["trainable"]["problem"]["adaptive_weights"][0].reshape(-1,1)*phys2
        return jnp.mean(phys2)

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        x = x_batch[:,0:1]
        u = 4*x*(1-x)
        return u


class Laplace2D_quadratic(Problem):

    @staticmethod
    def init_params(sd=0.2, adaptive_weights=()):
        static_params = {
            "dims":(1,2),
            "sd":sd,
            }
        if adaptive_weights:
            trainable_params = {
                "adaptive_weights": [jnp.ones(batch_shape, dtype=float) for batch_shape in adaptive_weights]
                }
        else:
            trainable_params = {}
        return static_params, trainable_params

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(0,0)),
            (0,(1,1)),
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        tanh, sd = jnp.tanh, all_params["static"]["problem"]["sd"]
        x, y = x_batch[:,0:1], x_batch[:,1:2]
        u = tanh((x-0)/sd)*tanh((1-x)/sd)*tanh((y-0)/sd)*tanh((1-y)/sd)*u
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        x_batch,uxx,uyy = constraints[0]
        x, y = x_batch[:,0:1], x_batch[:,1:2]
        f = 32*(x*(1-x)+y*(1-y))
        phys = f + uxx + uyy
        phys2 = phys**2
        if "problem" in all_params["trainable"] and "adaptive_weights" in all_params["trainable"]["problem"]:
            phys2 = all_params["trainable"]["problem"]["adaptive_weights"][0].reshape(-1,1)*phys2
        return jnp.mean(phys2)

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        x, y = x_batch[:,0:1], x_batch[:,1:2]
        u = 16*(x*(1-x)*y*(1-y))
        return u


class Laplace2D_multiscale(Problem):

    @staticmethod
    def init_params(sd=0.2, omegas=[2,4,8,16], adaptive_weights=()):
        static_params = {
            "dims":(1,2),
            "sd":sd,
            "omegas":omegas,
            "ns":len(omegas),
            }
        if adaptive_weights:
            trainable_params = {
                "adaptive_weights": [jnp.ones(batch_shape, dtype=float) for batch_shape in adaptive_weights]
                }
        else:
            trainable_params = {}
        return static_params, trainable_params

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(0,0)),
            (0,(1,1)),
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        tanh, sd = jnp.tanh, all_params["static"]["problem"]["sd"]
        x, y = x_batch[:,0:1], x_batch[:,1:2]
        u = tanh((x-0)/sd)*tanh((1-x)/sd)*tanh((y-0)/sd)*tanh((1-y)/sd)*u
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        x_batch,uxx,uyy = constraints[0]
        x, y = x_batch[:,0:1], x_batch[:,1:2]
        params = all_params["static"]["problem"]
        sin, pi, omegas, ns = jnp.sin, jnp.pi, params["omegas"], params["ns"]
        f = (2/ns)*jnp.sum(jnp.stack([((omega*pi)**2)*sin(omega*pi*x)*sin(omega*pi*y) for omega in omegas], axis=0), axis=0)
        phys = f + uxx + uyy
        phys2 = phys**2
        if "problem" in all_params["trainable"] and "adaptive_weights" in all_params["trainable"]["problem"]:
            phys2 = all_params["trainable"]["problem"]["adaptive_weights"][0].reshape(-1,1)*phys2
        return jnp.mean(phys2)

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        x, y = x_batch[:,0:1], x_batch[:,1:2]
        params = all_params["static"]["problem"]
        sin, pi, omegas, ns = jnp.sin, jnp.pi, params["omegas"], params["ns"]
        u = (1/ns)*jnp.sum(jnp.stack([sin(omega*pi*x)*sin(omega*pi*y) for omega in omegas], axis=0), axis=0)
        return u


class Helmholtz2D(Problem):

    @staticmethod
    def init_params(c=1, w=1, sd=0.2, adaptive_weights=()):
        if c == "marmousi":
            c = jnp.array(np.load("marmousi_crop_sm.npy")[:,::-1])
            c = (c/jnp.median(c), Helmholtz2D._c_discrete, "discrete")
        else:
            c = (c, Helmholtz2D._c_scalar, "scalar")
        static_params = {
            "dims":(1,2),
            "sd":sd,
            "c":c,
            "w":w,
            }
        if adaptive_weights:
            trainable_params = {
                "adaptive_weights": [jnp.ones(batch_shape, dtype=float) for batch_shape in adaptive_weights]
                }
        else:
            trainable_params = {}
        return static_params, trainable_params

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(),),
            (0,(0,0)),
            (0,(1,1)),
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        tanh, w = jnp.tanh, all_params["static"]["problem"]["w"]
        sd = 1/w
        x, y = x_batch[:,0:1], x_batch[:,1:2]
        u = tanh((x-0)/sd)*tanh((1-x)/sd)*tanh((y-0)/sd)*tanh((1-y)/sd)*u
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        x_batch,u,uxx,uyy = constraints[0]
        x, y = x_batch[:,0:1], x_batch[:,1:2]
        params = all_params["static"]["problem"]
        c, w, sd, exp, sqrt, pi = params["c"][1](all_params, x_batch), params["w"], params["sd"], jnp.exp, jnp.sqrt, jnp.pi

        f = (1/sqrt(((2*pi)**2)*((sd**2)**2)))*exp(-0.5*(((x-0.5)/sd)**2 + ((y-0.5)/sd)**2))
        phys = (uxx + uyy) + ((w/c)**2)*u - f
        phys2 = phys**2
        if "problem" in all_params["trainable"] and "adaptive_weights" in all_params["trainable"]["problem"]:
            phys2 = all_params["trainable"]["problem"]["adaptive_weights"][0].reshape(-1,1)*phys2
        return jnp.mean(phys2)

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        params = all_params["static"]["problem"]
        c, w, sd = params["c"], params["w"], params["sd"]
        (xmin, ymin), (xmax, ymax) = np.array(x_batch.min(0)), np.array(x_batch.max(0))
        # run FD simulation (in numpy)
        nx, ny = batch_shape
        x,dx = np.linspace(xmin,xmax,nx,retstep=True)
        y,dy = np.linspace(ymin,ymax,ny,retstep=True)
        if c[2] == "discrete":
            c = np.array(c[0])
            assert c.shape == batch_shape
        else:
            c = c[0]*np.ones((nx, ny))
        x,y = np.meshgrid(x,y,indexing="ij")
        exp, sqrt, pi = np.exp, np.sqrt, np.pi
        f = (1/sqrt(((2*pi)**2)*((sd**2)**2)))*exp(-0.5*(((x-0.5)/sd)**2 + ((y-0.5)/sd)**2))
        u = helmholtz_solver(nx, ny, dx, dy, w, c, f)
        assert u.shape == batch_shape
        u = u.reshape((-1,1))
        return u

    @staticmethod
    def _c_discrete(all_params, x_batch):
        c = all_params["static"]["problem"]["c"][0]
        nx, ny = c.shape
        dx, dy = 1/(nx-1), 1/(ny-1)# TODO: assumes c spans domain (0,1), (0,1) (!)
        ix = jnp.floor_divide(x_batch[:,0], dx).astype(int)# TODO: assumes c spans domain (0,1), (0,1) (!)
        iy = jnp.floor_divide(x_batch[:,1], dy).astype(int)
        return c[ix, iy].reshape((-1,1))

    @staticmethod
    def _c_scalar(all_params, x_batch):
        return all_params["static"]["problem"]["c"][0]


