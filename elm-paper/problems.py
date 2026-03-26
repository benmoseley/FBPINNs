"""
Defines all of the problems used in our paper:
Local Feature Filtering for Scalable and Well-Conditioned Domain-Decomposed Random Feature Methods
https://arxiv.org/abs/2506.17626
"""

import jax.numpy as jnp
import jax

from fbpinns.problems import HarmonicOscillator1D, HarmonicOscillator1DHardBC, WaveEquationConstantVelocity3D, Problem
from fbpinns.util.logger import logger


class HarmonicOscillatorELM1D(HarmonicOscillator1D):

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),
            (0,(0,)),
            (0,(0,0))
        )

        # boundary losses
        x_batch_boundary = jnp.array([0.]).reshape((1,1))
        u_boundary = jnp.array([1.]).reshape((1,1))
        ut_boundary = jnp.array([0.]).reshape((1,1))
        required_ujs_boundary = (
            (0,()),
            (0,(0,)),
        )

        return [[x_batch_phys, required_ujs_phys],
                [x_batch_boundary, u_boundary, required_ujs_boundary[:1]],
                [x_batch_boundary, ut_boundary, required_ujs_boundary[1:]]
                ]

    @staticmethod
    def constraining_fn(all_params, x_batch, u, part=None):
        left = u
        if part == "left":
            return left
        right = 0
        return left + right

    @staticmethod
    def loss_fn(all_params, constraints):

        mu, k = all_params["static"]["problem"]["mu"], all_params["static"]["problem"]["k"]

        # physics residual
        x_batch, u, ut, utt = constraints[0]
        phys = utt + mu*ut + k*u
        f = jnp.zeros_like(x_batch)

        # boundary residual
        (x1, uc, u), (x2, utc, ut) = constraints[1], constraints[2]
        if len(uc):
            b1 = jnp.sqrt(1e6)*u
            g1 = jnp.sqrt(1e6)*uc
            b2 = jnp.sqrt(1e2)*ut
            g2 = jnp.sqrt(1e2)*utc
        else:
            b1 = jnp.zeros_like(u)
            g1 = jnp.zeros_like(x1)
            b2 = jnp.zeros_like(ut)
            g2 = jnp.zeros_like(x2)

        logger.debug("residual")
        logger.debug((phys.shape, f.shape, b1.shape, g1.shape, b2.shape, g2.shape))

        return [[phys, f], [b1, g1], [b2, g2]]


class HarmonicOscillatorELM1DHardBC(HarmonicOscillator1DHardBC):

    @staticmethod
    def constraining_fn(all_params, x_batch, u, part=None):

        sd = all_params["static"]["problem"]["sd"]
        x, tanh = x_batch[:,0:1], jnp.tanh

        left = (tanh(x/sd)**2) * u
        if part == "left":
            return left
        right = 1
        return left + right

    @staticmethod
    def loss_fn(all_params, constraints):

        mu, k = all_params["static"]["problem"]["mu"], all_params["static"]["problem"]["k"]

        # physics residual
        x_batch, u, ut, utt = constraints[0]
        phys = utt + mu*ut + k*u
        f = jnp.zeros_like(x_batch)

        logger.debug("residual")
        logger.debug((phys.shape, f.shape))

        return [[phys, f]]


class WaveEquationConstantVelocityELM3D(WaveEquationConstantVelocity3D):

    @staticmethod
    def constraining_fn(all_params, x_batch, u, part=None):
        params = all_params["static"]["problem"]
        c0, source = params["c0"], params["source"]
        x, t = x_batch[:,0:2], x_batch[:,2:3]
        tanh, exp = jax.nn.tanh, jnp.exp

        t1 = source[:,2].min()/c0
        bt = tanh(2.5*t/t1)**2
        x1 = source[:,2].min()
        bx = tanh(2.5*(x[:,0:1]+1)/x1)*\
             tanh(2.5*(1-x[:,0:1])/x1)*\
             tanh(2.5*(x[:,1:2]+1)/x1)*\
             tanh(2.5*(1-x[:,1:2])/x1)
        left = bx*bt*u
        if part == "left":
            return left
        p = jnp.expand_dims(source, axis=1)# (k, 1, 4)
        x = jnp.expand_dims(x, axis=0)# (1, n, 2)
        f = (p[:,:,3:4]*exp(-0.5 * ((x-p[:,:,0:2])**2).sum(2, keepdims=True)/(p[:,:,2:3]**2))).sum(0)# (n, 1)
        f = exp(-0.5*(1.5*t/t1)**2) * f
        right = bx*f
        return left + right

    @staticmethod
    def loss_fn(all_params, constraints):
        c_fn = all_params["static"]["problem"]["c_fn"]
        x_batch, uxx, uyy, utt = constraints[0]
        phys = (uxx + uyy) - (1/c_fn(all_params, x_batch)**2)*utt
        f = jnp.zeros_like(x_batch[:,0:1])
        return [[phys, f]]



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


class LaplaceELM2D_multiscale(Laplace2D_multiscale):

    @staticmethod
    def constraining_fn(all_params, x_batch, u, part=None):
        tanh, sd = jnp.tanh, all_params["static"]["problem"]["sd"]
        x, y = x_batch[:,0:1], x_batch[:,1:2]
        left = tanh((x-0)/sd)*tanh((1-x)/sd)*tanh((y-0)/sd)*tanh((1-y)/sd)*u
        if part == "left":
            return left
        return left

    @staticmethod
    def loss_fn(all_params, constraints):
        x_batch,uxx,uyy = constraints[0]
        x, y = x_batch[:,0:1], x_batch[:,1:2]
        params = all_params["static"]["problem"]
        sin, pi, omegas, ns = jnp.sin, jnp.pi, params["omegas"], params["ns"]
        f = -(2/ns)*jnp.sum(jnp.stack([((omega*pi)**2)*sin(omega*pi*x)*sin(omega*pi*y) for omega in omegas], axis=0), axis=0)
        phys = uxx + uyy
        return [[phys, f]]

