"""
Extra code for self-adaptive PINNs benchmark
"""

import time
from functools import partial

import jax
from jax import jit, value_and_grad
from jax import random
import optax
import numpy as np

import matplotlib.pyplot as plt

from fbpinns import networks
from fbpinns.util.logger import logger
from fbpinns.util.jax_util import total_size, str_tensor, partition, combine
from fbpinns.trainers import PINNTrainer

from fbpinns.trainers import _common_train_initialisation, PINN_loss



@partial(jit, static_argnums=(0, 4, 6, 7, 8))
def self_adaptive_PINN_update(optimiser_fn, active_opt_states,
                active_params, static_params_dynamic, static_params_static,
                constraints, model_fns, jmapss, loss_fn):
    # recombine static params
    static_params = combine(static_params_dynamic, static_params_static)
    # update step
    lossval, grads = value_and_grad(PINN_loss, argnums=0)(
        active_params, static_params, constraints, model_fns, jmapss, loss_fn)

    # FLIP THE SIGN OF GRAD WRT LOSS FUNCTION WEIGHTS (GRADIENT ASCENT)
    grads["problem"]["adaptive_weights"] = jax.tree_map(lambda x: -x,
                                                        grads["problem"]["adaptive_weights"])

    updates, active_opt_states = optimiser_fn(grads, active_opt_states, active_params)
    active_params = optax.apply_updates(active_params, updates)
    return lossval, active_opt_states, active_params



class SelfAdaptivePINNTrainer(PINNTrainer):
    "SelfAdaptive PINN model trainer class"

    def train(self):
        "Train model"

        c, writer = self.c, self.writer

        # generate root key
        key = random.PRNGKey(c.seed)
        np.random.seed(c.seed)

        # define all_params
        all_params = {"static":{},"trainable":{}}

        # initialise domain, problem and decomposition params
        domain, problem = c.domain, c.problem
        for tag, cl, kwargs in zip(["domain", "problem"], [domain, problem],
                                   [c.domain_init_kwargs, c.problem_init_kwargs]):
            ps_ = cl.init_params(**kwargs)
            if ps_[0]: all_params["static"][tag] = ps_[0]
            if ps_[1]: all_params["trainable"][tag] = ps_[1]
        assert (all_params["static"]["domain"]["xd"] ==\
                all_params["static"]["problem"]["dims"][1])

        # initialise network params
        network = c.network
        key, subkey = random.split(key)
        ps_ = network.init_params(key=subkey, **c.network_init_kwargs)
        if ps_[0]: all_params["static"]["network"] = {"subdomain": ps_[0]}# add subdomain key
        if ps_[1]: all_params["trainable"]["network"] = {"subdomain": ps_[1]}# add subdomain key
        logger.debug("all_params")
        logger.debug(jax.tree_map(lambda x: str_tensor(x), all_params))

        # define unnorm function
        mu_, sd_ = c.decomposition_init_kwargs["unnorm"]
        unnorm_fn = lambda u: networks.unnorm(mu_, sd_, u)
        model_fns = (domain.norm_fn, network.network_fn, unnorm_fn, problem.constraining_fn)

        # common initialisation
        (optimiser, all_opt_states, optimiser_fn, loss_fn, key,
        constraints_global, x_batch_global, _, _, jmapss,
        x_batch_test, u_exact) = _common_train_initialisation(c, key, all_params, problem, domain)

        # get implicit jitted update function
        active_params = all_params["trainable"]
        static_params = all_params["static"]
        active_opt_states = all_opt_states
        x_batch = x_batch_global
        constraints = constraints_global

        # AOT compile update function
        startc = time.time()
        logger.info(f"[i: {0}/{self.c.n_steps}] Compiling update step..")
        static_params_dynamic, static_params_static = partition(static_params)
        update = self_adaptive_PINN_update.lower(optimiser_fn, active_opt_states,
                                   active_params, static_params_dynamic, static_params_static,
                                   constraints, model_fns, jmapss, loss_fn).compile()
        logger.info(f"[i: {0}/{self.c.n_steps}] Compiling done ({time.time()-startc:.2f} s)")
        cost_ = update.cost_analysis()
        p,f = total_size(active_params["network"]), cost_[0]["flops"] if (cost_ and "flops" in cost_[0]) else 0
        logger.debug("p, f")
        logger.debug((p,f))

        # train loop
        pstep, fstep, u_test_losses = 0, 0, []
        start0, start1, report_time = time.time(), time.time(), 0.
        lossval = None
        for i in range(c.n_steps):

            if i == 0:
                # report initial model
                u_test_losses, start1, report_time = \
                self._report(i, pstep, fstep, u_test_losses, start0, start1, report_time,
                            u_exact, x_batch_test, all_params, all_opt_states, model_fns, problem,
                            active_opt_states, active_params,
                            x_batch,
                            lossval)

            # take a training step
            lossval, active_opt_states, active_params = update(active_opt_states,
                                       active_params, static_params_dynamic,
                                       constraints)# note compiled function only accepts dynamic arguments
            pstep, fstep = pstep+p, fstep+f

            # report
            u_test_losses, start1, report_time = \
            self._report(i + 1, pstep, fstep, u_test_losses, start0, start1, report_time,
                        u_exact, x_batch_test, all_params, all_opt_states, model_fns, problem,
                        active_opt_states, active_params,
                        x_batch,
                        lossval)
            if (i+1) % (c.test_freq * 5) == 0:
                f_ = plt.figure()
                plt.imshow(active_params["problem"]["adaptive_weights"][0])
                plt.colorbar()
                writer.add_figure("adaptive_weights", f_, i+1, close=False)
                if self.c.show_figures: plt.show()
                else: plt.close("all")

        # cleanup
        writer.close()
        logger.info(f"[i: {i+1}/{self.c.n_steps}] Training complete")

        # return trained parameters
        all_params["trainable"] = active_params
        all_opt_states = active_opt_states

        return all_params


