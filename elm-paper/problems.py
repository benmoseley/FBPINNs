"""
Defines all of the problems used in our paper:
ELM-FBPINNs: An Efficient Multilevel Random Feature Method
"""

import jax.numpy as jnp

from fbpinns.problems import Problem


class HarmonicOscillator1D(Problem):

    @staticmethod
    def init_params(d=2, w0=20):

        mu, k = 2*d, w0**2

        static_params = {
            "dims":(1,1),
            "d":d,
            "w0":w0,
            "mu":mu,
            "k":k,
            }

        return static_params, {}

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
    def loss_fn(all_params, constraints):

        mu, k = all_params["static"]["problem"]["mu"], all_params["static"]["problem"]["k"]

        # physics residual
        x_batch, u, ut, utt = constraints[0]
        phys = utt + mu*ut + k*u

        # boundary residual
        (_, uc, u), (_, utc, ut) = constraints[1], constraints[2]
        b1 = k*(u - uc)
        b2 = jnp.sqrt(k)*(ut - utc)

        return jnp.mean(phys**2) + jnp.mean(b1**2) + jnp.mean(b2**2)

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):

        d, w0 = all_params["static"]["problem"]["d"], all_params["static"]["problem"]["w0"]

        w = jnp.sqrt(w0**2-d**2)
        phi = jnp.arctan(-d/w)
        A = 1/(2*jnp.cos(phi))
        cos = jnp.cos(phi + w * x_batch)
        exp = jnp.exp(-d * x_batch)
        u = exp * 2 * A * cos

        return u

class HarmonicOscillatorELM1D(HarmonicOscillator1D):

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
            b1 = k*u
            g1 = k*uc
            b2 = jnp.sqrt(k)*ut
            g2 = jnp.sqrt(k)*utc
        else:
            b1 = jnp.zeros_like(u)
            g1 = jnp.zeros_like(x1)
            b2 = jnp.zeros_like(ut)
            g2 = jnp.zeros_like(x2)

        return [[phys, f], [b1, g1], [b2, g2]]


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
        f = -(2/ns)*jnp.sum(jnp.stack([((omega*pi)**2)*sin(omega*pi*x)*sin(omega*pi*y) for omega in omegas], axis=0), axis=0)
        phys = uxx + uyy - f
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
        right = 0
        return left + right

    @staticmethod
    def loss_fn(all_params, constraints):
        x_batch,uxx,uyy = constraints[0]
        x, y = x_batch[:,0:1], x_batch[:,1:2]
        params = all_params["static"]["problem"]
        sin, pi, omegas, ns = jnp.sin, jnp.pi, params["omegas"], params["ns"]
        f = -(2/ns)*jnp.sum(jnp.stack([((omega*pi)**2)*sin(omega*pi*x)*sin(omega*pi*y) for omega in omegas], axis=0), axis=0)
        phys = uxx + uyy
        return [[phys, f]]



class Helmholtz2D(Problem):

    @staticmethod
    def init_params(k=1, omega=1):
        static_params = {
            "dims":(1,2),
            "k":k,
            "omega":omega,
            }
        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(),),
            (0,(0,0)),
            (0,(1,1)),
        )

        # boundary loss
        x_batchs_boundary = domain.sample_boundaries(all_params, key, sampler, batch_shapes[1])
        x_batch_boundary = jnp.concatenate(x_batchs_boundary, axis=0)
        u_boundary = HelmholtzELM2D.exact_solution(all_params, x_batch_boundary)
        required_ujs_boundary = (
            (0,()),
        )
        return [[x_batch_phys, required_ujs_phys], [x_batch_boundary, u_boundary, required_ujs_boundary]]

    @staticmethod
    def loss_fn(all_params, constraints):

        # physics residual
        x_batch,u,uxx,uyy = constraints[0]
        x, y = x_batch[:,0:1], x_batch[:,1:2]
        params = all_params["static"]["problem"]
        k, omega = params["k"], params["omega"]

        a = jnp.pi*omega
        u_exact = jnp.sin(a*x)*jnp.sin(a*2*y)+jnp.sin(a*3*x*y)
        f = (-5*a**2*jnp.sin(a*x)*jnp.sin(2*a*y)
             -9*a**2*(x**2+y**2)*jnp.sin(3*a*x*y)
             +k**2*u_exact)
        phys = (uxx + uyy) + (k**2)*u - f

        # boundary residual
        _, uc, u = constraints[1]
        b1 = 5*(omega**2)*(u - uc)

        return jnp.mean(phys**2) + jnp.mean(b1**2)

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        omega = all_params["static"]["problem"]["omega"]
        x, y = x_batch[:,0:1], x_batch[:,1:2]
        a = jnp.pi*omega
        u = jnp.sin(a*x)*jnp.sin(a*2*y)+jnp.sin(a*3*x*y)
        return u

class HelmholtzELM2D(Helmholtz2D):

    @staticmethod
    def constraining_fn(all_params, x_batch, u, part=None):
        left = u
        if part == "left":
            return left
        right = 0
        return left + right

    @staticmethod
    def loss_fn(all_params, constraints):

        # physics residual
        x_batch,u,uxx,uyy = constraints[0]
        x, y = x_batch[:,0:1], x_batch[:,1:2]
        params = all_params["static"]["problem"]
        k, omega = params["k"], params["omega"]

        a = jnp.pi*omega
        u_exact = jnp.sin(a*x)*jnp.sin(a*2*y)+jnp.sin(a*3*x*y)
        f = (-5*a**2*jnp.sin(a*x)*jnp.sin(2*a*y)
             -9*a**2*(x**2+y**2)*jnp.sin(3*a*x*y)
             +k**2*u_exact)
        phys = (uxx + uyy) + (k**2)*u

        # boundary residual
        x, uc, u = constraints[1]
        if len(uc):
            b1 = 5*(omega**2)*u
            g1 = 5*(omega**2)*uc
        else:
            b1 = jnp.zeros_like(u)
            g1 = jnp.zeros_like(x)

        return [[phys, f], [b1, g1]]




