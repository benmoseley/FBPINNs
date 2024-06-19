"""
Script for running test FourierFCN ablations
"""

import numpy as np

from fbpinns.constants import Constants, get_subdomain_ws
from fbpinns.domains import RectangularDomainND
from fbpinns.decompositions import MultilevelRectangularDecompositionND, RectangularDecompositionND
from fbpinns.networks import FCN, SIREN, FourierFCN
from fbpinns.util.sbatch import submit

from problems import Laplace1D_quadratic, Laplace2D_quadratic, Laplace2D_multiscale, Helmholtz2D


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
        optimiser_kwargs=dict(learning_rate=lr),
        seed=seed,
        test_freq=test_freq,
        model_save_freq=model_save_freq,
        )
    return c, "PINN"



runs = []


model_save_freq=10000
seed = 0


domain=RectangularDomainND
domain_init_kwargs=dict(xmin=np.array([0.,0]),
                        xmax=np.array([1.,1.]),)


lr = 1e-4
sds = [0.1, 0.2, 0.4, 0.8, 1, 1.25, 1.5, 2, 2.5, 5, 6, 8, 10]


# Laplace2D_multiscale strong scaling
test_freq=500

problem=Laplace2D_multiscale
omegas=[2, 4, 8, 16, 32, 64]
unnorm=(0., 0.75)# unnorm

n_steps=30000# number training steps
n = (320,320)# number training points
n_test=(350,350)# number test points

h=1
w=1.9
p=16

# increase levels, but not collocation points
ls=[2, 3, 4, 5, 6, 7]# number of levels

i = 5
l_ = 7
l = [2**i for i in range(l_)]
subdomain_xss = [[np.array([0.5]),np.array([0.5])]] + [[np.linspace(0,1,n_),np.linspace(0,1,n_)] for n_ in l[1:]]
subdomain_wss = [[np.array([w*1.]),np.array([w*1.])]] + [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss[1:]]
problem_init_kwargs=dict(omegas=omegas, sd=1/omegas[-1])

h,p=5,256
layer_sizes = [2,] + [p,]*h + [1,]
network = FourierFCN
for sd in sds:
    network_init_kwargs = dict(layer_sizes=layer_sizes, mu=0, sd=sd, n_features=p)
    tag = f"test_{sd}-sd_strong"
    runs.append(run_PINN())


# TEST 4: helmholtz tests

test_freq=1000

problem=Helmholtz2D
n_steps=150000# number training steps
n_test=(320,320)# number test points
w=1.9

# add omegas to problem, whilst increasing levels and collocation points
ls=[2, 3, 4, 5, 6]# number of levels
ns=[(20,20),(40,40),(80,80),(160,160),(320,320)]# number of collocation points
unnorms=[(0.03, 0.02), (-0.01, 0.15), (0.00, 0.07), (-0.00, 0.05), (0.00, 0.04)]
for c in [1,]:
    for il,(l_,n,unnorm) in enumerate(zip(ls, ns, unnorms)):
        problem_init_kwargs=dict(c=c, w=((2**il)/4)*2*np.pi/(0.2), sd=(4/(2**il))*0.05)

        # multilevel scaling
        l = [2**i for i in range(l_)]
        subdomain_xss = [[np.array([0.5]),np.array([0.5])]] + [[np.linspace(0,1,n_),np.linspace(0,1,n_)] for n_ in l[1:]]
        subdomain_wss = [[np.array([w*1.]),np.array([w*1.])]] + [get_subdomain_ws(subdomain_xs, w) for subdomain_xs in subdomain_xss[1:]]

        h,p=5,256
        layer_sizes = [2,] + [p,]*h + [1,]
        network = FourierFCN
        for sd in sds:
            network_init_kwargs = dict(layer_sizes=layer_sizes, mu=0, sd=sd, n_features=p)
            tag = f"test_{sd}-sd_weak_{c}"
            runs.append(run_PINN())




if __name__ == "__main__":

        # parallel submit all runs
        submit(runs)



