"""
Defines and runs all the experiments in our paper:
Multilevel domain decomposition-based architectures for physics-informed neural networks
https://arxiv.org/abs/2306.05486
"""

import numpy as np

from fbpinns.constants import Constants, get_subdomain_ws
from fbpinns.domains import RectangularDomainND
from fbpinns.decompositions import MultilevelRectangularDecompositionND, RectangularDecompositionND
from fbpinns.networks import FCN, SIREN
from fbpinns.util.sbatch import submit

from problems import Laplace1D_quadratic, Laplace2D_quadratic, Laplace2D_multiscale, Helmholtz2D


def pscan(p0, *pss):
    "scan from fixed point"
    assert len(p0) == len(pss)
    return [list(p0[:ip] + [p] + p0[ip+1:]) for ip, ps in enumerate(pss) for p in ps]


def run_FBPINN():
    run = f"FBPINN_{tag}_{problem.__name__}_{network.__name__}_{l}-levels_{w}-overlap_{h}-layers_{p}-hidden_{n[0]}-n"
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
        network_init_kwargs=dict(
            layer_sizes=layer_sizes,
            ),
        n_steps=n_steps,
        ns=(n,),
        n_test=n_test,
        test_freq=test_freq,
        model_save_freq=model_save_freq,
        )
    return c, "FBPINN"


def run_PINN():
    run = f"PINN_{tag}_{problem.__name__}_{network.__name__}_{h}-layers_{p}-hidden_{n[0]}-n"
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
        network_init_kwargs=dict(
            layer_sizes=layer_sizes,
            ),
        n_steps=n_steps,
        ns=(n,),
        n_test=n_test,
        test_freq=test_freq,
        model_save_freq=model_save_freq,
        )
    return c, "PINN"


runs = []


test_freq=500
model_save_freq=10000

network = FCN

## TEST 1: simple ablation tests
tag = "ablation"

# Laplace1D_quadratic ablation
domain=RectangularDomainND
domain_init_kwargs=dict(xmin=np.array([0.,]),
                        xmax=np.array([1.,]),)
problem=Laplace1D_quadratic
problem_init_kwargs=dict()
unnorm=(0.5, 0.375)# unnorm

n_steps=20000# number training steps
n=(80,)# number training points
n_test=(350,)# number test points

h=1

ls=[2, 3, 4, 5]# number of levels
ws=[1.1, 1.5, 1.9, 2.3, 2.7]# overlap width
ps=[2, 4, 8, 16, 32]# number of hidden units
p0=[3, 1.9, 16]
for l_,w,p in pscan(p0, ls, ws, ps):

    # multilevel scaling
    l = [2**i for i in range(l_)]
    subdomain_xss = [[np.array([0.5]),]] + [[np.linspace(0,1,n_),] for n_ in l[1:]]
    subdomain_wss = [[np.array([w*1.]),]] + [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss[1:]]
    layer_sizes = [1,] + [p,]*h + [1,]
    runs.append(run_FBPINN())

# Laplace2D_quadratic ablation
domain=RectangularDomainND
domain_init_kwargs=dict(xmin=np.array([0.,0]),
                        xmax=np.array([1.,1.]),)
problem=Laplace2D_quadratic

n=(80,80)# number training points
n_test=(350,350)# number test points

for l_,w,p in pscan(p0, ls, ws, ps):

    # multilevel scaling
    l = [2**i for i in range(l_)]
    subdomain_xss = [[np.array([0.5]),np.array([0.5])]] + [[np.linspace(0,1,n_),np.linspace(0,1,n_)] for n_ in l[1:]]
    subdomain_wss = [[np.array([w*1.]),np.array([w*1.])]] + [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss[1:]]
    layer_sizes = [2,] + [p,]*h + [1,]
    runs.append(run_FBPINN())

    # subdomain scaling
    l = [2**(l_-1)]
    subdomain_xss = [[np.linspace(0,1,n_),np.linspace(0,1,n_)] for n_ in l]
    subdomain_wss = [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss]
    layer_sizes = [2,] + [p,]*h + [1,]
    runs.append(run_FBPINN())

h,p=3,64
layer_sizes = [2,] + [p,]*h + [1,]
runs.append(run_PINN())


## TEST 2: strong / weak scaling tests

# Laplace2D_multiscale strong scaling
tag = "strong"

problem=Laplace2D_multiscale
omegas=[2, 4, 8, 16, 32, 64]
unnorm=(0., 0.75)# unnorm

n_steps=30000# number training steps
n_test=(350,350)# number test points

h=1
w=1.9
p=16

# increase levels and collocation points
ls=[2, 3, 4, 5, 6, 7]# number of levels
ns=[(10,10),(20,20),(40,40),(80,80),(160,160),(320,320)]# number of collocation points
for il,(l_,n) in enumerate(zip(ls, ns)):
    problem_init_kwargs=dict(omegas=omegas, sd=1/omegas[-1])

    # scaling points

    # multilevel scaling
    l = [2**i for i in range(l_)]
    subdomain_xss = [[np.array([0.5]),np.array([0.5])]] + [[np.linspace(0,1,n_),np.linspace(0,1,n_)] for n_ in l[1:]]
    subdomain_wss = [[np.array([w*1.]),np.array([w*1.])]] + [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss[1:]]
    layer_sizes = [2,] + [p,]*h + [1,]
    runs.append(run_FBPINN())

    # fixed points
    n = (320, 320)

    # multilevel scaling
    for b in [2,4,8,16]:
        l = [b**i for i in range(l_)]
        if max(l) <= 64:
            subdomain_xss = [[np.array([0.5]),np.array([0.5])]] + [[np.linspace(0,1,n_),np.linspace(0,1,n_)] for n_ in l[1:]]
            subdomain_wss = [[np.array([w*1.]),np.array([w*1.])]] + [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss[1:]]
            layer_sizes = [2,] + [p,]*h + [1,]
            runs.append(run_FBPINN())

    # subdomain scaling
    l = [2**(l_-1)]
    subdomain_xss = [[np.linspace(0,1,n_),np.linspace(0,1,n_)] for n_ in l]
    subdomain_wss = [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss]
    layer_sizes = [2,] + [p,]*h + [1,]
    runs.append(run_FBPINN())

h,p=5,256
layer_sizes = [2,] + [p,]*h + [1,]
runs.append(run_PINN())

# Laplace2D_multiscale weak scaling
tag = "weak"

# add omegas to problem, whilst increasing levels and collocation points
for il,(l_,n) in enumerate(zip(ls, ns)):
    problem_init_kwargs=dict(omegas=omegas[:il+1], sd=1/omegas[il])
    h,p=1,16

    # multilevel scaling
    l = [2**i for i in range(l_)]
    subdomain_xss = [[np.array([0.5]),np.array([0.5])]] + [[np.linspace(0,1,n_),np.linspace(0,1,n_)] for n_ in l[1:]]
    subdomain_wss = [[np.array([w*1.]),np.array([w*1.])]] + [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss[1:]]
    layer_sizes = [2,] + [p,]*h + [1,]
    runs.append(run_FBPINN())

    h,p=5,256
    layer_sizes = [2,] + [p,]*h + [1,]
    runs.append(run_PINN())



# TEST 4: helmholtz tests

test_freq=1000

tag = "ablation"

problem=Helmholtz2D
problem_init_kwargs = dict(c=1, w=2*np.pi/(0.2), sd=0.05)
unnorm=(0.00, 0.07)# unnorm

n_steps=150000# number training steps
n=(160,160)# number training points
n_test=(320,320)# number test points

h=1

ls=[2, 3, 4, 5]# number of levels
ws=[1.1, 1.5, 1.9, 2.3, 2.7]# overlap width
ps=[2, 4, 8, 16, 32]# number of hidden units
p0=[4, 1.9, 16]
for l_,w,p in pscan(p0, ls, ws, ps):

    # multilevel scaling
    l = [2**i for i in range(l_)]
    subdomain_xss = [[np.array([0.5]),np.array([0.5])]] + [[np.linspace(0,1,n_),np.linspace(0,1,n_)] for n_ in l[1:]]
    subdomain_wss = [[np.array([w*1.]),np.array([w*1.])]] + [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss[1:]]
    layer_sizes = [2,] + [p,]*h + [1,]
    runs.append(run_FBPINN())

    # subdomain scaling
    l = [2**(l_-1)]
    subdomain_xss = [[np.linspace(0,1,n_),np.linspace(0,1,n_)] for n_ in l]
    subdomain_wss = [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss]
    layer_sizes = [2,] + [p,]*h + [1,]
    runs.append(run_FBPINN())

h,p=5,256
layer_sizes = [2,] + [p,]*h + [1,]
runs.append(run_PINN())

w=1.9

# add omegas to problem, whilst increasing levels and collocation points
ls=[2, 3, 4, 5, 6]# number of levels
ns=[(20,20),(40,40),(80,80),(160,160),(320,320)]# number of collocation points
unnorms=[(0.03, 0.02), (-0.01, 0.15), (0.00, 0.07), (-0.00, 0.05), (0.00, 0.04)]
for network in [FCN, SIREN]:
    for c in [1,]:
        tag = f"weak_{c}"
        for il,(l_,n,unnorm) in enumerate(zip(ls, ns, unnorms)):
            problem_init_kwargs=dict(c=c, w=((2**il)/4)*2*np.pi/(0.2), sd=(4/(2**il))*0.05)
            h,p = 1,16

            # multilevel scaling
            l = [2**i for i in range(l_)]
            subdomain_xss = [[np.array([0.5]),np.array([0.5])]] + [[np.linspace(0,1,n_),np.linspace(0,1,n_)] for n_ in l[1:]]
            subdomain_wss = [[np.array([w*1.]),np.array([w*1.])]] + [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss[1:]]
            layer_sizes = [2,] + [p,]*h + [1,]
            runs.append(run_FBPINN())

            h,p=5,256
            layer_sizes = [2,] + [p,]*h + [1,]
            runs.append(run_PINN())




if __name__ == "__main__":

        # parallel submit all runs
        submit(runs)

