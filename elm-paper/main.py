"""
Defines and runs all the experiments in our paper:
ELM-FBPINNs: An Efficient Multilevel Random Feature Method
"""

import copy
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import optax

from fbpinns.domains import RectangularDomainND
from fbpinns.decompositions import MultilevelRectangularDecompositionND
from fbpinns.constants import Constants, get_subdomain_ws
from fbpinns.trainers import FBPINNTrainer, PINNTrainer
from fbpinns.networks import FCN
from fbpinns.schedulers import AllActiveSchedulerND
from fbpinns.util.sbatch import apply_config_deltas, submit
from fbpinns.util.logger import logger, FileLogging

from elm.elms import ELM
from elm.trainers import ELMFBPINNTrainer
from elm.optimisers import LinearSolver
from elm.solvers import sps_lsqr

from domains import LDomain2D, MultilevelLDecomposition2D
from problems import (HarmonicOscillator1D, HarmonicOscillatorELM1D,
                      Laplace2D_multiscale, LaplaceELM2D_multiscale,
                      Helmholtz2D, HelmholtzELM2D)


def run_ELMFBPINN():
    run = f"ELMFBPINN_{tag}_{problem.__name__}_{network.__name__}_{l}-levels_{w}-overlap_{h}-layers_{C}-hidden_{ns[0][0]}-n_{optimiser.__name__}-{optimiser_kwargs['system']}-{optimiser_kwargs['solver'].__name__}-{seed}"
    c = Constants(
        run=run,
        domain=domain,
        domain_init_kwargs=domain_init_kwargs,
        problem=problem,
        problem_init_kwargs=problem_init_kwargs,
        decomposition=decomposition,
        decomposition_init_kwargs=decomposition_init_kwargs,
        network=network,
        network_init_kwargs=network_init_kwargs,
        n_steps=n_steps,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        ns=ns,
        n_test=n_test,
        optimiser=optimiser,
        optimiser_kwargs=optimiser_kwargs,
        seed=seed,
        test_freq=test_freq,
        model_save_freq=model_save_freq,
        show_figures=False,
        )
    return c, "ELMFBPINN"


def run_FBPINN():
    run = f"FBPINN_{tag}_{problem.__name__}_{network.__name__}_{l}-levels_{w}-overlap_{h}-layers_{C}-hidden_{ns[0][0]}-n_{lr}-lr-{seed}"
    c = Constants(
        run=run,
        domain=domain,
        domain_init_kwargs=domain_init_kwargs,
        problem=problem,
        problem_init_kwargs=problem_init_kwargs,
        decomposition=decomposition,
        decomposition_init_kwargs=decomposition_init_kwargs,
        network=network,
        network_init_kwargs=network_init_kwargs,
        n_steps=n_steps,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        ns=ns,
        n_test=n_test,
        optimiser=optimiser,
        optimiser_kwargs=optimiser_kwargs,
        seed=seed,
        test_freq=test_freq,
        model_save_freq=model_save_freq,
        show_figures=False,
        )
    return c, "FBPINN"


def run_PINN():
    run = f"PINN_{tag}_{problem.__name__}_{network.__name__}_{h}-layers_{C}-hidden_{ns[0][0]}-n_{lr}-lr-{seed}"
    c = Constants(
        run=run,
        domain=domain,
        domain_init_kwargs=domain_init_kwargs,
        problem=problem,
        problem_init_kwargs=problem_init_kwargs,
        decomposition=decomposition,
        decomposition_init_kwargs=decomposition_init_kwargs,
        network=network,
        network_init_kwargs=network_init_kwargs,
        n_steps=n_steps,
        ns=ns,
        n_test=n_test,
        optimiser=optimiser,
        optimiser_kwargs=optimiser_kwargs,
        seed=seed,
        test_freq=test_freq,
        model_save_freq=model_save_freq,
        show_figures=False,
        )
    return c, "PINN"

runs=[]



n_seeds = 5
# note first save results run saves intermediate tensors and does warm start compilation
seed_save_results = list(zip([n_seeds,]+list(range(n_seeds)),[True,]+[False]*n_seeds))


optimisers = [
(optax.adam,
 dict(
    ),
 run_PINN),
(optax.adam,
 dict(
    ),
run_FBPINN),
(LinearSolver,
 dict(system="least-squares",
      solver=sps_lsqr,
      solver_kwargs=dict(atol=0, btol=0, damp=0., conlim=0, iter_lim=15000, force_iter_lim=True),# need to specify iterlim otherwise set based on n
      ),
    run_ELMFBPINN),
]

weight_scales = [0.125, 0.25, 0.5, 1, 2, 4, 8]




## 1D harmonic oscillator

J_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 32, 48, 64, 92, 128]
C_list = [2, 4, 8, 16, 32, 64]
levels = [1, 2, 3, 4]

p0 = dict(# base model params
    optimiser=optimisers[-1],# default optimiser
    network=ELM,# network
    h=1,# number of hidden layers
    C=12,# number of hidden units
    weight_scale=1,# weight scaling for network init
    w0=80,# oscillator frequency
    w=2.9,# subdomain overlap
    J=20,# number of subdomains
    level=1,# number of levels for multilevel decomposition
    delta_name="p0",
    )

deltas = (
    [dict(optimiser=v, delta_name="optimisers") for v in optimisers] +
    [dict(weight_scale=v, delta_name="weight_scaling")
         for v in weight_scales] +
    [dict(J=v1, optimiser=v2, delta_name="n_subdomains")
         for v2 in optimisers[1:] for v1 in J_list] +
    [dict(C=v1, optimiser=v2, delta_name="n_hidden_units")
         for v2 in optimisers for v1 in C_list] +
    [dict(w0=v1, J=v2, optimiser=v3, level=v4, delta_name="n_subdomains_multiscale")
         for v1, v2, v3, v4 in [

            # important: all scale n with J*C to keep M well-determined

            (20, 10, optimisers[1], 1),# weak scaling with J
            (40, 20, optimisers[1], 1),
            (80, 40, optimisers[1], 1),
            (160, 80, optimisers[1], 1),
            (320, 160, optimisers[1], 1),

            (20, 10, optimisers[2], 1),# weak scaling with J
            (40, 20, optimisers[2], 1),
            (80, 40, optimisers[2], 1),
            (160, 80, optimisers[2], 1),
            (320, 160, optimisers[2], 1),

            (20, 10, optimisers[2], 3),# weak scaling with J
            (40, 20, optimisers[2], 3),
            (80, 40, optimisers[2], 3),
            (160, 80, optimisers[2], 3),
            (320, 160, optimisers[2], 3),
    ]] +
    [dict(level=v1, optimiser=v2, delta_name="decomposition_level")
     for v1 in levels for v2 in optimisers[1:]]
)

for config in apply_config_deltas(p0, deltas):
    (optimiser,optimiser_kwargs,run_fn) = config["optimiser"]
    network = config["network"]
    h,C,weight_scale = config["h"], config["C"], config["weight_scale"]
    w0,w,J = config["w0"], config["w"], config["J"]
    level = config["level"]
    delta_name = config["delta_name"]

    if run_fn == run_ELMFBPINN:
        problem=HarmonicOscillatorELM1D
    else:
        problem=HarmonicOscillator1D
    problem_init_kwargs=dict(
        d=2, w0=w0,
    )
    domain=RectangularDomainND
    domain_init_kwargs=dict(xmin=np.array([0.]),
                            xmax=np.array([1.]),)
    decomposition=MultilevelRectangularDecompositionND
    if level > 1:
        l = [1] + [J // 2**k for k in range(level-2, 0, -1)] + [J]
        subdomain_xss = [[np.array([0.5])]] + [[np.linspace(0,1,n_)] for n_ in l[1:]]
        subdomain_wss = [[np.array([w*1.])]] + [get_subdomain_ws(xs, w) for xs in subdomain_xss[1:]]
    else:
        l = [J]
        subdomain_xss = [[np.linspace(0,1,J)]]
        subdomain_wss = [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss]
    decomposition_init_kwargs = dict(
                subdomain_xss=subdomain_xss,
                subdomain_wss=subdomain_wss,
                unnorm=(0., 1.),
                )
    ns = (((C+1)*J,),)
    n_test = (50*J,)
    if run_fn != run_ELMFBPINN:
        network = FCN
    if run_fn == run_PINN:
        if delta_name != "n_hidden_units":
            C = 64
        h = 2
    layer_sizes = [1,] + [C,]*h + [1,]
    if run_fn == run_ELMFBPINN:
        network_init_kwargs = dict(layer_sizes=layer_sizes, weight_scale=weight_scale)
    else:
        network_init_kwargs = dict(layer_sizes=layer_sizes)
    scheduler = AllActiveSchedulerND
    scheduler_kwargs = dict()
    if run_fn == run_ELMFBPINN:
        tag = f"{delta_name}-{w0}-w0_{weight_scale}-R"
    else:
        lr = 1e-3
        optimiser_kwargs["learning_rate"] = lr
        tag = f"{delta_name}-{w0}-w0"
    if run_fn == run_ELMFBPINN:
        test_freq = 100
    else:
        test_freq = 1000
    model_save_freq = 5000
    n_steps = 15000
    o_ = optimiser_kwargs
    runs_ = []
    for seed, save_results in seed_save_results:
        if run_fn == run_ELMFBPINN:
            optimiser_kwargs = copy.deepcopy(o_)
            optimiser_kwargs["save_results"] = save_results
        runs_.append(run_fn())
    runs.append(runs_)





## 2D laplace

J_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
C_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
levels = [1, 2, 3, 4]

omegas = [4, 8, 12, 16, 20, 24, 28]

p0 = dict(# base model params
    optimiser=optimisers[-1],# default optimiser
    network=ELM,# network
    h=1,# number of hidden layers
    C=16,# number of hidden units
    weight_scale=1,# weight scaling for network init
    n_omegas=4,# oscillator frequency
    w=2.9,# subdomain overlap
    J=16,# number of subdomains
    level=1,# number of levels for multilevel decomposition
    delta_name="p0",
    )

deltas = (
    [dict(optimiser=v, delta_name="optimisers") for v in optimisers] +
    [dict(weight_scale=v, delta_name="weight_scaling")
         for v in weight_scales] +
    [dict(J=v1, optimiser=v2, delta_name="n_subdomains")
         for v2 in optimisers[1:] for v1 in J_list] +
    [dict(C=v1, optimiser=v2, delta_name="n_hidden_units")
         for v2 in optimisers for v1 in C_list] +
    [dict(n_omegas=v1, J=v2, optimiser=v3, level=v4, delta_name="n_subdomains_multiscale")
         for v1, v2, v3, v4 in [

            # important: all scale n with J*C to keep M well-determined

            (1, 4, optimisers[1], 1),# weak scaling with J
            (2, 8, optimisers[1], 1),
            (3, 12, optimisers[1], 1),
            (4, 16, optimisers[1], 1),
            (5, 20, optimisers[1], 1),
            (6, 24, optimisers[1], 1),
            (7, 28, optimisers[1], 1),

            (1, 4, optimisers[2], 1),# weak scaling with J
            (2, 8, optimisers[2], 1),
            (3, 12, optimisers[2], 1),
            (4, 16, optimisers[2], 1),
            (5, 20, optimisers[2], 1),
            (6, 24, optimisers[2], 1),
            (7, 28, optimisers[2], 1),

            (1, 4, optimisers[2], 3),# weak scaling with J
            (2, 8, optimisers[2], 3),
            (3, 12, optimisers[2], 3),
            (4, 16, optimisers[2], 3),
            (5, 20, optimisers[2], 3),
            (6, 24, optimisers[2], 3),
            (7, 28, optimisers[2], 3),
    ]] +
    [dict(level=v1, optimiser=v2, delta_name="decomposition_level")
     for v1 in levels for v2 in optimisers[1:]]
)

for config in apply_config_deltas(p0, deltas):
    (optimiser,optimiser_kwargs,run_fn) = config["optimiser"]
    network = config["network"]
    h,C,weight_scale = config["h"], config["C"], config["weight_scale"]
    n_omegas,w,J = config["n_omegas"], config["w"], config["J"]
    level = config["level"]
    delta_name = config["delta_name"]

    if run_fn == run_ELMFBPINN:
        problem=LaplaceELM2D_multiscale
    else:
        problem=Laplace2D_multiscale
    problem_init_kwargs=dict(omegas=omegas[:n_omegas], sd=1/omegas[n_omegas-1])
    domain=RectangularDomainND
    domain_init_kwargs=dict(xmin=np.array([0.,0.]),
                            xmax=np.array([1.,1.]),)
    decomposition=MultilevelRectangularDecompositionND
    if level > 1:
        l = [1] + [J // 2**k for k in range(level-2, 0, -1)] + [J]
        subdomain_xss = [[np.array([0.5]),np.array([0.5])]] + [[np.linspace(0,1,n_),np.linspace(0,1,n_)] for n_ in l[1:]]
        subdomain_wss = [[np.array([w*1.]),np.array([w*1.])]] + [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss[1:]]
    else:
        l = [J]
        subdomain_xss = [[np.linspace(0,1,J), np.linspace(0,1,J)]]
        subdomain_wss = [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss]
    decomposition_init_kwargs = dict(
                subdomain_xss=subdomain_xss,
                subdomain_wss=subdomain_wss,
                unnorm=(0., 0.75),
                )
    ns = ((int(np.ceil(np.sqrt((C+1))))*J, int(np.ceil(np.sqrt((C+1))))*J),)
    n_test = (int(np.ceil(np.sqrt(30)))*J, int(np.ceil(np.sqrt(30)))*J)
    if run_fn != run_ELMFBPINN:
        network = FCN
    if run_fn == run_PINN:
        if delta_name != "n_hidden_units":
            C = 64
        h = 2
    layer_sizes = [2,] + [C,]*h + [1,]
    if run_fn == run_ELMFBPINN:
        network_init_kwargs = dict(layer_sizes=layer_sizes, weight_scale=weight_scale)
    else:
        network_init_kwargs = dict(layer_sizes=layer_sizes)
    scheduler = AllActiveSchedulerND
    scheduler_kwargs = dict()
    if run_fn == run_ELMFBPINN:
        tag = f"{delta_name}-{n_omegas}-n_omegas_{weight_scale}-R"
    else:
        lr = 1e-3
        optimiser_kwargs["learning_rate"] = lr
        tag = f"{delta_name}-{n_omegas}-n_omegas"
    if run_fn == run_ELMFBPINN:
        test_freq = 100
    else:
        test_freq = 1000
    model_save_freq = 5000
    n_steps = 15000
    o_ = optimiser_kwargs
    runs_ = []
    for seed, save_results in seed_save_results:
        if run_fn == run_ELMFBPINN:
            optimiser_kwargs = copy.deepcopy(o_)
            optimiser_kwargs["save_results"] = save_results
        runs_.append(run_fn())
    runs.append(runs_)







## 2D helmholtz

J_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
C_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
levels = [1, 2, 3, 4]

p0 = dict(# base model params
    optimiser=optimisers[-1],# default optimiser
    network=ELM,# network
    h=1,# number of hidden layers
    C=16,# number of hidden units
    weight_scale=1,# weight scaling for network init
    omega=8,# oscillator frequency
    w=2.9,# subdomain overlap
    J=16,# number of subdomains
    level=1,# number of levels for multilevel decomposition
    delta_name="p0",
    )

deltas = (
    [dict(optimiser=v, delta_name="optimisers") for v in optimisers] +
    [dict(weight_scale=v, delta_name="weight_scaling")
         for v in weight_scales] +
    [dict(J=v1, optimiser=v2, delta_name="n_subdomains")
         for v2 in optimisers[1:] for v1 in J_list] +
    [dict(C=v1, optimiser=v2, delta_name="n_hidden_units")
         for v2 in optimisers for v1 in C_list] +
    [dict(omega=v1, J=v2, optimiser=v3, level=v4, delta_name="n_subdomains_multiscale")
         for v1, v2, v3, v4 in [

            # important: all scale n with J*C to keep M well-determined

            (4, 8, optimisers[1], 1),# weak scaling with J
            (6, 12, optimisers[1], 1),
            (8, 16, optimisers[1], 1),
            (10, 20, optimisers[1], 1),
            (12, 24, optimisers[1], 1),
            (14, 28, optimisers[1], 1),
            (16, 32, optimisers[1], 1),

            (4, 8, optimisers[2], 1),# weak scaling with J
            (6, 12, optimisers[2], 1),
            (8, 16, optimisers[2], 1),
            (10, 20, optimisers[2], 1),
            (12, 24, optimisers[2], 1),
            (14, 28, optimisers[2], 1),
            (16, 32, optimisers[2], 1),

            (4, 8, optimisers[2], 3),# weak scaling with J
            (6, 12, optimisers[2], 3),
            (8, 16, optimisers[2], 3),
            (10, 20, optimisers[2], 3),
            (12, 24, optimisers[2], 3),
            (14, 28, optimisers[2], 3),
            (16, 32, optimisers[2], 3),
    ]] +
    [dict(level=v1, optimiser=v2, delta_name="decomposition_level")
     for v1 in levels for v2 in optimisers[1:]]
)

for config in apply_config_deltas(p0, deltas):
    (optimiser,optimiser_kwargs,run_fn) = config["optimiser"]
    network = config["network"]
    h,C,weight_scale = config["h"], config["C"], config["weight_scale"]
    omega,w,J = config["omega"], config["w"], config["J"]
    level = config["level"]
    delta_name = config["delta_name"]

    if run_fn == run_ELMFBPINN:
        problem=HelmholtzELM2D
    else:
        problem=Helmholtz2D
    problem_init_kwargs=dict(k=1, omega=omega)
    domain=LDomain2D
    domain_init_kwargs=dict(xmin=np.array([0.,0.]),
                            xmax=np.array([1.,1.]),)
    decomposition=MultilevelLDecomposition2D
    if level > 1:
        l = [1] + [J // 2**k for k in range(level-2, 0, -1)] + [J]
        subdomain_xss = [[np.array([0.5]),np.array([0.5])]] + [[np.linspace(0,1,n_),np.linspace(0,1,n_)] for n_ in l[1:]]
        subdomain_wss = [[np.array([w*1.]),np.array([w*1.])]] + [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss[1:]]
    else:
        l = [J]
        subdomain_xss = [[np.linspace(0,1,J), np.linspace(0,1,J)]]
        subdomain_wss = [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss]
    decomposition_init_kwargs = dict(
                xmin=np.array([0.,0.]),
                xmax=np.array([1.,1.]),
                subdomain_xss=subdomain_xss,
                subdomain_wss=subdomain_wss,
                unnorm=(0., 1.),
                )
    n = (int(np.ceil(np.sqrt((C+1))))*J, int(np.ceil(np.sqrt((C+1))))*J)
    ns = (n, (n,))
    n_test = (int(np.ceil(np.sqrt(30)))*J, int(np.ceil(np.sqrt(30)))*J)
    if run_fn != run_ELMFBPINN:
        network = FCN
    if run_fn == run_PINN:
        if delta_name != "n_hidden_units":
            C = 64
        h = 2
    layer_sizes = [2,] + [C,]*h + [1,]
    if run_fn == run_ELMFBPINN:
        network_init_kwargs = dict(layer_sizes=layer_sizes, weight_scale=weight_scale)
    else:
        network_init_kwargs = dict(layer_sizes=layer_sizes)
    scheduler = AllActiveSchedulerND
    scheduler_kwargs = dict()
    if run_fn == run_ELMFBPINN:
        tag = f"{delta_name}-{omega}-omega_{weight_scale}-R"
    else:
        lr = 1e-3
        optimiser_kwargs["learning_rate"] = lr
        tag = f"{delta_name}-{omega}-omega"
    if run_fn == run_ELMFBPINN:
        test_freq = 100
    else:
        test_freq = 1000
    model_save_freq = 5000
    n_steps = 15000
    o_ = optimiser_kwargs
    runs_ = []
    for seed, save_results in seed_save_results:
        if run_fn == run_ELMFBPINN:
            optimiser_kwargs = copy.deepcopy(o_)
            optimiser_kwargs["save_results"] = save_results
        runs_.append(run_fn())
    runs.append(runs_)



# print all runs
flatten = lambda runs: [run for seed_runs in runs for run in seed_runs]
for i,(c,_) in enumerate(flatten(runs)):
    print(i, c.run)

# check all experiments uniquely named
unique = []
for i,(c,_) in enumerate(flatten(runs)):
    if c.run in unique:
        raise Exception(f"duplicate run: {i} {c.run}")
    unique.append(c.run)

# run all runs
if 0:
    trainers = dict(FBPINN=FBPINNTrainer,
                    PINN=PINNTrainer,
                    ELMFBPINN=ELMFBPINNTrainer)
    start = time.time()
    for i,(c,k) in enumerate(runs):
        trainer = trainers[k]
        run = trainer(c)
        with FileLogging(f"{c.summary_out_dir}log.txt"):
            logger.info(f"Running run {i+1} of {len(runs)}..")
            logger.info(c)
            run.train()
    print(f"Total runtime: {(time.time()-start)/(60*60):.2f} hours")
else:
    submit(runs)
    pass



