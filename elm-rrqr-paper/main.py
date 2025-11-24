"""
Defines and runs all the experiments in our paper:
Local Feature Filtering for Scalable and Well-Conditioned Domain-Decomposed Random Feature Methods
https://arxiv.org/abs/2506.17626
"""

import copy
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import optax

from fbpinns.problems import HarmonicOscillator1D, WaveEquationConstantVelocity3D
from fbpinns.domains import RectangularDomainND
from fbpinns.decompositions import RectangularDecompositionND, MultilevelRectangularDecompositionND
from fbpinns.constants import Constants, get_subdomain_ws
from fbpinns.trainers import FBPINNTrainer, PINNTrainer
from fbpinns.networks import FCN
from fbpinns.schedulers import AllActiveSchedulerND
from fbpinns.util.sbatch import apply_config_deltas, submit
from fbpinns.util.logger import logger, FileLogging

from elm.elms import ELM, ELM_sigmoid, ELM_SIREN
from elm.trainers import ELMFBPINNTrainer
from elm.optimisers import LinearSolver, BlockRRQRLinearSolver, AdditiveSchwarzLinearSolver
from elm.solvers import sps_cg, sps_lsqr

from problems import HarmonicOscillatorELM1D, LaplaceELM2D_multiscale, Laplace2D_multiscale, WaveEquationConstantVelocityELM3D


def run_ELMFBPINN():
    run = f"ELMFBPINN_{tag}_{problem.__name__}_{network.__name__}_{l}-levels_{w}-overlap_{h}-layers_{p}-hidden_{n[0]}-n_{optimiser.__name__}-{optimiser_kwargs['system']}-{optimiser_kwargs['solver'].__name__}-{seed}"
    c = Constants(
        run=run,
        domain=domain,
        domain_init_kwargs=domain_init_kwargs,
        problem=problem,
        problem_init_kwargs=problem_init_kwargs,
        decomposition=MultilevelRectangularDecompositionND,
        decomposition_init_kwargs = dict(
                    subdomain_xss=subdomain_xss,
                    subdomain_wss=subdomain_wss,
                    unnorm=unnorm,
                    ),
        network=network,
        network_init_kwargs=network_init_kwargs,
        n_steps=n_steps,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        ns=(n,),
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
    run = f"FBPINN_{tag}_{problem.__name__}_{network.__name__}_{l}-levels_{w}-overlap_{h}-layers_{p}-hidden_{n[0]}-n_{lr}-lr-{seed}"
    c = Constants(
        run=run,
        domain=domain,
        domain_init_kwargs=domain_init_kwargs,
        problem=problem,
        problem_init_kwargs=problem_init_kwargs,
        decomposition=MultilevelRectangularDecompositionND,
        decomposition_init_kwargs = dict(
                    subdomain_xss=subdomain_xss,
                    subdomain_wss=subdomain_wss,
                    unnorm=unnorm,
                    ),
        network=network,
        network_init_kwargs=network_init_kwargs,
        n_steps=n_steps,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        ns=(n,),
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
    run = f"PINN_{tag}_{problem.__name__}_{network.__name__}_{h}-layers_{p}-hidden_{n[0]}-n_{lr}-lr-{seed}"
    c = Constants(
        run=run,
        domain=domain,
        domain_init_kwargs=domain_init_kwargs,
        problem=problem,
        problem_init_kwargs=problem_init_kwargs,
        decomposition=RectangularDecompositionND,
        decomposition_init_kwargs = dict(
                    subdomain_xs=subdomain_xss[-1],
                    subdomain_ws=subdomain_wss[-1],
                    unnorm=unnorm,
                    ),
        network=network,
        network_init_kwargs=network_init_kwargs,
        n_steps=n_steps,
        ns=(n,),
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
      solver_kwargs=dict(atol=0, btol=0, damp=0., conlim=0, iter_lim=10000, force_iter_lim=True),# need to specify iterlim otherwise set based on n
      ),
    run_ELMFBPINN),
(LinearSolver,
 dict(system="normal",
      solver=sps_cg,
      solver_kwargs=dict(rtol=0, atol=0, maxiter=10000),# need to specify maxiter otherwise set based on n
      ),
    run_ELMFBPINN),
(AdditiveSchwarzLinearSolver,
 dict(system="normal",
      solver=sps_cg,
      solver_kwargs=dict(rtol=0, atol=0, maxiter=10000),# need to specify maxiter otherwise set based on n
      ),
    run_ELMFBPINN),
(BlockRRQRLinearSolver,
 dict(system="least-squares",
      solver=sps_lsqr,
      solver_kwargs=dict(atol=0, btol=0, damp=0., conlim=0, iter_lim=10000, force_iter_lim=True),# need to specify iterlim otherwise set based on n
      ),
    run_ELMFBPINN),
]
sigmas = [1e-10, 1e-6, 1e-4, 1e-2]
networks = [ELM, ELM_sigmoid, ELM_SIREN]
overlaps = [1.9, 1.3]
weight_scales = [0.125, 0.25, 0.5, 1, 2, 4, 8]




## 1D harmonic oscillator


depths = [(3, 8), (5, 8)]
p0 = dict(
    optimiser=optimisers[-1],
    sigma=1e-8,
    network=ELM,
    h=1,
    p=8,
    weight_scale=1,
    w0=60,
    w=2.9,
    m=20,
    )
deltas = (
    [dict(optimiser=v) for v in optimisers] +
    [dict(sigma=v) for v in sigmas] +
    [dict(network=v) for v in networks] +
    [dict(h=v1, p=v2) for v1,v2 in depths] +
    [dict(w=v) for v in overlaps] +
    [dict(weight_scale=v1, network=v2) for v1 in weight_scales for v2 in networks] +
    [dict(w0=v1, w=v2, m=v3, p=v4, optimiser=v5) for v5 in optimisers[2:] for v1,v2,v3,v4 in [

        # important: all scale n with J*C to keep M well-determined

        (60, 2.9*0.5, 10, 8),# strong, w-m
        (60, 2.9*1, 20, 8),
        (60, 2.9*2, 40, 8),
        (60, 2.9*4, 80, 8),
        (60, 2.9*8, 160, 8),

        (60, 2.9, 10, 8),# strong, m
        (60, 2.9, 20, 8),
        (60, 2.9, 40, 8),
        (60, 2.9, 80, 8),
        (60, 2.9, 160, 8),

        (60, 2.9, 20, 4),# strong, p
        (60, 2.9, 20, 8),
        (60, 2.9, 20, 16),
        (60, 2.9, 20, 32),
        (60, 2.9, 20, 64),

        (30, 2.9, 10, 8),# weak, m
        (60, 2.9, 20, 8),
        (120, 2.9, 40, 8),
        (240, 2.9, 80, 8),
        (580, 2.9, 160, 8),
       ]]
    )

for config in apply_config_deltas(p0, deltas):
    (optimiser,optimiser_kwargs,run_fn) = config["optimiser"]
    sigma = config["sigma"]
    network = config["network"]
    h,p,weight_scale = config["h"], config["p"], config["weight_scale"]
    w0,w,m = config["w0"], config["w"], config["m"]

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
    l = 1
    subdomain_xss=[[np.linspace(0,1, m)]]
    subdomain_wss=[get_subdomain_ws(subdomain_xss[0], w)]
    unnorm=(0.,1.)
    n = ((p+1)*m,)
    n_test = (50*m,)
    if run_fn != run_ELMFBPINN:
        network = FCN
    if run_fn == run_PINN:
        h,p = 2, 64
    layer_sizes = [1,] + [p,]*h + [1,]
    if run_fn == run_ELMFBPINN:
        network_init_kwargs = dict(layer_sizes=layer_sizes, weight_scale=weight_scale)
    else:
        network_init_kwargs = dict(layer_sizes=layer_sizes)
    scheduler = AllActiveSchedulerND
    scheduler_kwargs = dict()
    if run_fn == run_ELMFBPINN:
        optimiser_kwargs["sigma"] = sigma
        tag = f"main-{w0}-w0_{m}-m_sigma-{sigma}_ws-{weight_scale}"
    else:
        lr = 1e-3
        optimiser_kwargs["learning_rate"] = lr
        tag = f"main-{w0}-w0_{m}-m"
    if run_fn == run_ELMFBPINN:
        test_freq = 100
    else:
        test_freq = 5000
    model_save_freq = 5000
    n_steps = 10000
    o_ = optimiser_kwargs
    runs_ = []
    for seed, save_results in seed_save_results:
        if run_fn == run_ELMFBPINN:
            optimiser_kwargs = copy.deepcopy(o_)
            optimiser_kwargs["save_results"] = save_results
        runs_.append(run_fn())
    runs.append(runs_)


## 1D harmonic oscillator (appendix)

p0 = dict(
    optimiser=optimisers[-1],
    sigma=0,
    network=ELM,
    h=1,
    p=8,
    weight_scale=1,
    w0=60,
    w=2.9,
    m=20,
    )
deltas = [dict(w=v1, p=v2) for v1,v2 in [
        (2.9*0.5, 8),# delta-varying
        (2.9*1, 8),
        (2.9*2, 8),
        (2.9*4, 8),
        (2.9*8, 8),

        (2.9*1, 4),# channel-varying
        (2.9*1, 8),
        (2.9*1, 16),
        (2.9*1, 32),
        (2.9*1, 64),
       ]]
for config in apply_config_deltas(p0, deltas):
    (optimiser,optimiser_kwargs,run_fn) = config["optimiser"]
    sigma = config["sigma"]
    network = config["network"]
    h,p,weight_scale = config["h"], config["p"], config["weight_scale"]
    w0,w,m = config["w0"], config["w"], config["m"]

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
    l = 1
    subdomain_xss=[[np.linspace(0,1, m)]]
    subdomain_wss=[get_subdomain_ws(subdomain_xss[0], w)]
    unnorm=(0.,1.)

    n = (100*m,)
    n_test = (200*m,)

    if run_fn != run_ELMFBPINN:
        network = FCN
    if run_fn == run_PINN:
        h,p = 2, 64
    layer_sizes = [1,] + [p,]*h + [1,]
    if run_fn == run_ELMFBPINN:
        network_init_kwargs = dict(layer_sizes=layer_sizes, weight_scale=weight_scale)
    else:
        network_init_kwargs = dict(layer_sizes=layer_sizes)
    scheduler = AllActiveSchedulerND
    scheduler_kwargs = dict()
    if run_fn == run_ELMFBPINN:
        optimiser_kwargs["sigma"] = sigma
        tag = f"main-{w0}-w0_{m}-m_sigma-{sigma}_ws-{weight_scale}"
    else:
        lr = 1e-3
        optimiser_kwargs["learning_rate"] = lr
        tag = f"main-{w0}-w0_{m}-m"
    if run_fn == run_ELMFBPINN:
        test_freq = 100
    else:
        test_freq = 5000
    model_save_freq = 5000
    n_steps = 10000
    o_ = optimiser_kwargs
    runs_ = []
    for seed, save_results in seed_save_results:
        if run_fn == run_ELMFBPINN:
            optimiser_kwargs = copy.deepcopy(o_)
            optimiser_kwargs["save_results"] = save_results
        runs_.append(run_fn())
    runs.append(runs_)



## 2D laplace

# original ns in multilevel paper:
#ns = [(10,10),(20,20),(40,40),(80,80),(160,160),(320,320)]
#n_test = (350,350)
# our max is m = 64, p = 16 (n = 320,320, n_test = 384, 384)

omegas = [2, 4, 8, 16, 32, 64]
depths = [(3, 16), (5, 16)]
p0 = dict(
    optimiser=optimisers[-1],
    sigma=1e-8,
    network=ELM,
    h=1,
    p=16,
    weight_scale=1,
    w0=3,
    w=2.9,
    m=16,# m = omegas
    )
deltas = (
    [dict(optimiser=v) for v in optimisers] +
    [dict(sigma=v) for v in sigmas] +
    [dict(network=v) for v in networks] +
    [dict(h=v1, p=v2) for v1,v2 in depths] +
    [dict(w=v) for v in overlaps] +
    [dict(weight_scale=v1, network=v2) for v1 in weight_scales for v2 in networks] +
    [dict(w0=v1, w=v2, m=v3, p=v4, optimiser=v5) for v5 in optimisers[2:] for v1,v2,v3,v4 in [

        # important: all scale n with J*C to keep M well-determined

        (3, 2.9*0.5, 8, 16),# strong, w-m
        (3, 2.9*1, 16, 16),
        (3, 2.9*2, 32, 16),

        (3, 2.9, 4, 16),# strong, m
        (3, 2.9, 8, 16),
        (3, 2.9, 16, 16),
        (3, 2.9, 32, 16),
        (3, 2.9, 64, 16),

        (3, 2.9, 16, 4),# strong, p
        (3, 2.9, 16, 8),
        (3, 2.9, 16, 16),
        (3, 2.9, 16, 32),
        (3, 2.9, 16, 64),

        (0, 2.9, 2, 16),# weak, m
        (1, 2.9, 4, 16),
        (2, 2.9, 8, 16),
        (3, 2.9, 16, 16),
        (4, 2.9, 32, 16),
        (5, 2.9, 64, 16),
       ]]
    )

for config in apply_config_deltas(p0, deltas):
    (optimiser,optimiser_kwargs,run_fn) = config["optimiser"]
    sigma = config["sigma"]
    network = config["network"]
    h,p,weight_scale = config["h"], config["p"], config["weight_scale"]
    w0,w,m = config["w0"], config["w"], config["m"]

    if run_fn == run_ELMFBPINN:
        problem=LaplaceELM2D_multiscale
    else:
        problem=Laplace2D_multiscale
    problem_init_kwargs=dict(omegas=omegas[:w0+1], sd=1/omegas[w0])
    domain=RectangularDomainND
    domain_init_kwargs=dict(xmin=np.array([0.,0.]),
                            xmax=np.array([1.,1.]),)
    l = 1
    subdomain_xss = [[np.linspace(0,1,m), np.linspace(0,1,m)]]
    subdomain_wss=[get_subdomain_ws(subdomain_xss[0], w)]
    unnorm=(0., 0.75)
    n = (int(np.ceil(np.sqrt((p+1))))*m, int(np.ceil(np.sqrt((p+1))))*m)
    n_test = (int(np.ceil(np.sqrt(30)))*m, int(np.ceil(np.sqrt(30)))*m)
    if run_fn != run_ELMFBPINN:
        network = FCN
    if run_fn == run_PINN:
        h,p = 2, 128
    layer_sizes = [2,] + [p,]*h + [1,]
    if run_fn == run_ELMFBPINN:
        network_init_kwargs = dict(layer_sizes=layer_sizes, weight_scale=weight_scale)
    else:
        network_init_kwargs = dict(layer_sizes=layer_sizes)
    scheduler = AllActiveSchedulerND
    scheduler_kwargs = dict()
    if run_fn == run_ELMFBPINN:
        optimiser_kwargs["sigma"] = sigma
        tag = f"main-{w0}-w0_{m}-m_sigma-{sigma}_ws-{weight_scale}"
    else:
        lr = 1e-3
        optimiser_kwargs["learning_rate"] = lr
        tag = f"main-{w0}-w0_{m}-m"
    if run_fn == run_ELMFBPINN:
        test_freq = 100
    else:
        test_freq = 5000
    model_save_freq = 5000
    n_steps = 10000
    o_ = optimiser_kwargs
    runs_ = []
    for seed, save_results in seed_save_results:
        if run_fn == run_ELMFBPINN:
            optimiser_kwargs = copy.deepcopy(o_)
            optimiser_kwargs["save_results"] = save_results
        runs_.append(run_fn())
    runs.append(runs_)




## 2+1D wave equation

# original tests:
#ns=((30,30,30),),
#n_test=(100,100,5),
# our max is m = 16, p = 8

depths = [(3, 8), (5, 8)]
p0 = dict(
    optimiser=optimisers[-1],
    sigma=1e-8,
    network=ELM,
    h=1,
    p=8,
    weight_scale=1,
    w0=0.2,
    w=2.9,
    m=8,
    )
deltas = (
    [dict(optimiser=v) for v in optimisers] +
    [dict(sigma=v) for v in sigmas] +
    [dict(network=v) for v in networks] +
    [dict(h=v1, p=v2) for v1,v2 in depths] +
    [dict(w=v) for v in overlaps] +
    [dict(weight_scale=v1, network=v2) for v1 in weight_scales for v2 in networks] +
    [dict(w0=v1, w=v2, m=v3, p=v4, optimiser=v5) for v5 in optimisers[2:] for v1,v2,v3,v4 in [

        # important: all scale n with J*C to keep M well-determined

        (0.2, 2.9*0.5, 4, 8),# strong, w-m
        (0.2, 2.9*1, 8, 8),
        (0.2, 2.9*2, 16, 8),

        (0.2, 2.9, 4, 8),# strong, m
        (0.2, 2.9, 8, 8),
        (0.2, 2.9, 16, 8),

        (0.2, 2.9, 8, 4),# strong, p
        (0.2, 2.9, 8, 8),
        (0.2, 2.9, 8, 16),
        (0.2, 2.9, 8, 32),

        (0.4, 2.9, 4, 8),# weak, m
        (0.2, 2.9, 8, 8),
        (0.1, 2.9, 16, 8),
       ]]
    )

for config in apply_config_deltas(p0, deltas):
    (optimiser,optimiser_kwargs,run_fn) = config["optimiser"]
    sigma = config["sigma"]
    network = config["network"]
    h,p,weight_scale = config["h"], config["p"], config["weight_scale"]
    w0,w,m = config["w0"], config["w"], config["m"]

    if run_fn == run_ELMFBPINN:
        problem=WaveEquationConstantVelocityELM3D
    else:
        problem=WaveEquationConstantVelocity3D
    problem_init_kwargs=dict(
        c0=1,
        source=np.array([[0., 0., w0, 1.]]),# gaussian source x,y, width, amplitude
    )
    domain=RectangularDomainND
    domain_init_kwargs=dict(xmin=np.array([-1,-1,0]),
                            xmax=np.array([1,1,0.5]))
    l = 1
    subdomain_xss = [[np.linspace(-1,1,m), np.linspace(-1,1,m), np.linspace(0,0.5,m)]]
    subdomain_wss=[get_subdomain_ws(subdomain_xss[0], w)]
    unnorm=(0.,1)
    n = (int(np.ceil(np.pow((p+1),1/3)))*m,
         int(np.ceil(np.pow((p+1),1/3)))*m,
         int(np.ceil(np.pow((p+1),1/3)))*m)
    n_test = (int(np.ceil(np.pow(30,1/3)))*m,
              int(np.ceil(np.pow(30,1/3)))*m,
              10)
    if run_fn != run_ELMFBPINN:
        network = FCN
    if run_fn == run_PINN:
        h,p = 2, 128
    layer_sizes = [3,] + [p,]*h + [1,]
    if run_fn == run_ELMFBPINN:
        network_init_kwargs = dict(layer_sizes=layer_sizes, weight_scale=weight_scale)
    else:
        network_init_kwargs = dict(layer_sizes=layer_sizes)
    scheduler = AllActiveSchedulerND
    scheduler_kwargs = dict()
    if run_fn == run_ELMFBPINN:
        optimiser_kwargs["sigma"] = sigma
        tag = f"main-{w0}-w0_{m}-m_sigma-{sigma}_ws-{weight_scale}"
    else:
        lr = 1e-3
        optimiser_kwargs["learning_rate"] = lr
        tag = f"main-{w0}-w0_{m}-m"
    if run_fn == run_ELMFBPINN:
        test_freq = 100
    else:
        test_freq = 5000
    model_save_freq = 5000
    n_steps = 10000
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

