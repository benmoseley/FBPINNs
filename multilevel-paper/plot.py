"""
Defines some helper functions for plotting
"""

import jax.numpy as jnp

from fbpinns.analysis import load_model
from fbpinns.analysis import FBPINN_solution as FBPINN_solution_
from fbpinns.analysis import PINN_solution as PINN_solution_


def load_FBPINN(tag, problem, network, l, w, h, p, n, lr, seed):
    run = f"FBPINN_{tag}_{problem.__name__}_{network.__name__}_{l}-levels_{w}-overlap_{h}-layers_{p}-hidden_{n[0]}-n_{lr}-lr-{seed}"
    c, model = load_model(run, rootdir="results/")
    i,t,l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]
    return c, model, i, t, l1n

def load_PINN(tag, problem, network, h, p, n, lr, seed):
    run = f"PINN_{tag}_{problem.__name__}_{network.__name__}_{h}-layers_{p}-hidden_{n[0]}-n_{lr}-lr-{seed}"
    c, model = load_model(run, rootdir="results/")
    i,t,l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]
    return c, model, i, t, l1n

def load_SelfAdaptivePINN(tag, problem, network, h, p, n, lr, seed):
    run = f"SelfAdaptivePINN_{tag}_{problem.__name__}_{network.__name__}_{h}-layers_{p}-hidden_{n[0]}-n_{lr}-lr-{seed}"
    c, model = load_model(run, rootdir="results/")
    i,t,l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]
    return c, model, i, t, l1n

def exact_solution(c, model):
    all_params, domain, problem = model[1], c.domain, c.problem
    x_batch = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=c.n_test)
    u_exact = problem.exact_solution(all_params, x_batch, batch_shape=c.n_test)
    return u_exact.reshape(c.n_test)

def FBPINN_solution(c, model):
    all_params, domain = model[1], c.domain
    x_batch = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=c.n_test)
    active = jnp.ones((all_params["static"]["decomposition"]["m"]))
    u_test = FBPINN_solution_(c, all_params, active, x_batch)
    return u_test.reshape(c.n_test)

def PINN_solution(c, model):
    all_params, domain = model[1], c.domain
    x_batch = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=c.n_test)
    u_test = PINN_solution_(c, all_params, x_batch)
    return u_test.reshape(c.n_test)