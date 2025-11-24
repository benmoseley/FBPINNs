"""
Defines trainer class for ELM-FBPINNs.
"""

import time

import jax
from fbpinns.util.logger import logger
if jax.config.read("jax_enable_x64") is not True:
    logger.warn("jax_enable_x64 is not enabled - precision may be insufficient for ELM matrix inversions")
from jax import vmap
import jax.numpy as jnp
import numpy as np

from fbpinns.trainers import _get_ujs, FBPINNTrainer, tree_map_dicts, FBPINN_model
from fbpinns.util.jax_util import str_tensor, combine, partition


def FBPINN_model_inner_ELM(params, x, norm_fn, basis_fn, unnorm_fn, window_fn):
    x_norm = norm_fn(params, x)# normalise
    u_raw = basis_fn(params, x_norm)# network BASIS function
    mu, sd = params["static"]["decomposition"]["subdomain"]["params"][5]
    u = u_raw*sd# note, mu is dealt with in FBPINN_forward_ELM
    w = window_fn(params, x)# window
    return u*w, w, u_raw

def FBPINN_model_ELM(all_params, x_batch, takes, model_fns, verbose=True):
    "Defines FBPINN model"

    norm_fn, basis_fn, unnorm_fn, window_fn, constraining_fn = model_fns
    m_take, n_take, p_take, np_take, npou = takes

    # take x_batch
    x_take = x_batch[n_take]# (s, xd)
    log_ = logger.info if verbose else logger.debug
    log_("x_batch")
    log_(str_tensor(x_batch))# (n, xd)
    log_("x_take")
    log_(str_tensor(x_take))

    # take subdomain params
    d = all_params
    all_params_take = {t_k: {cl_k: {k: jax.tree_util.tree_map(lambda p:p[m_take], d[t_k][cl_k][k]) if k=="subdomain" else d[t_k][cl_k][k]
        for k in d[t_k][cl_k]}
        for cl_k in d[t_k]}
        for t_k in ["static", "trainable"]}
    f = {t_k: {cl_k: {k: jax.tree_util.tree_map(lambda p: 0, d[t_k][cl_k][k]) if k=="subdomain" else jax.tree_util.tree_map(lambda p: None, d[t_k][cl_k][k])
        for k in d[t_k][cl_k]}
        for cl_k in d[t_k]}
        for t_k in ["static", "trainable"]}
    logger.debug("all_params")
    logger.debug(jax.tree_util.tree_map(lambda x: str_tensor(x), all_params))
    logger.debug("all_params_take")
    logger.debug(jax.tree_util.tree_map(lambda x: str_tensor(x), all_params_take))
    logger.debug("vmap f")
    logger.debug(f)

    # batch over parameters and points
    us, ws, _ = vmap(FBPINN_model_inner_ELM, in_axes=(f,0,None,None,None,None))(all_params_take, x_take, norm_fn, basis_fn, unnorm_fn, window_fn)# (s, ud)
    logger.debug("us")
    logger.debug(str_tensor(us))

    # apply POU
    wp = jax.ops.segment_sum(ws, p_take, indices_are_sorted=False, num_segments=len(np_take))# sum weights onto unique point-pou list (_, 1)
    logger.debug("wp")
    logger.debug(str_tensor(wp))
    wp = wp[p_take]# scatter the weight back to each point in n_take (s, 1)
    logger.debug(str_tensor(wp))
    us = (us/wp)/npou# apply POU
    logger.debug("us")
    logger.debug(str_tensor(us))

    # apply left part of constraining operator. note, right part is dealt with in FBPINN_forward_ELM
    us = constraining_fn(all_params, x_take, us, part="left")# (s, ud)
    logger.debug(str_tensor(us))

    # reshape (add dimension)# !small ~hack so that _get_ujs indexes gets second dim
    us = us.reshape((us.shape[0], 1, us.shape[1]))# (s, 1, C)
    logger.debug(str_tensor(us))

    return us,

def FBPINN_forward_ELM(all_params, x_batch, takes, model_fns, jmaps):
    "Computes gradients of FBPINN model"

    # compute operator terms on left part of u
    def u_left(x_batch):
        return FBPINN_model_ELM(all_params, x_batch, takes, model_fns)[0], ()
    left = _get_ujs(x_batch, jmaps, u_left)

    # compute operator terms on right part of u
    mu = all_params["static"]["decomposition"]["subdomain"]["params"][5][0,0]
    npou, constraining_fn = takes[-1], model_fns[-1]
    def u_right(x_batch):
        u_right = constraining_fn(all_params, x_batch, (mu/npou)*jnp.ones((x_batch.shape[0], 1)))
        return u_right, ()
    right = _get_ujs(x_batch, jmaps, u_right)

    return left, right

def FBPINN_linear_solve(optimiser_fn, active_opt_states,
                  active_params, fixed_params, static_params_dynamic, static_params_static,
                  takess, constraints, model_fns, jmapss, loss_fn,

                  all_test_inputs, test_freq, summary_out_dir,
                  ):
    # recombine static params
    static_params = combine(static_params_dynamic, static_params_static)

    # add fixed params to active, recombine all_params
    d, da = active_params, fixed_params
    trainable_params = {cl_k: {k: jax.tree_util.tree_map(lambda p1, p2:jnp.concatenate([p1,p2],0), d[cl_k][k], da[cl_k][k]) if k=="subdomain" else d[cl_k][k]
        for k in d[cl_k]}
        for cl_k in d}
    all_params = {"static":static_params, "trainable":trainable_params}

    # run FBPINN for each constraint, with shared params
    constraints_left, constraints_right  = [], []
    for takes, jmaps, constraint in zip(takess, jmapss, constraints):
        logger.debug("constraint")
        for c_ in constraint:
            logger.debug(str_tensor(c_))
        x_batch = constraint[0]
        left, right = FBPINN_forward_ELM(all_params, x_batch, takes, model_fns, jmaps)
        constraints_left.append(constraint+left)
        constraints_right.append(constraint+right)
    del takes, jmaps, constraint

    # compute left and right operator terms
    terms_left = loss_fn(all_params, constraints_left)
    terms_right = loss_fn(all_params, constraints_right)
    logger.info("terms left")
    for (t1,t2) in terms_left:
        logger.info((t1.shape, t2.shape))
    logger.info("terms right")
    for (t1,t2) in terms_right:
        logger.info((t1.shape, t2.shape))

    # get fixed parameters
    _a0 = all_params["trainable"]["network"]["subdomain"]["basis_coeffs"]
    J, C = _a0.shape[0], _a0.shape[2]# (J,1,C)
    J_active = active_params["network"]["subdomain"]["basis_coeffs"].shape[0]
    a0_fixed = _a0.reshape(J*C, 1)[J_active*C:]

    # build and solve system using optimiser
    logger.info("Solving linear system..")
    testing = (summary_out_dir, test_freq, test, active_params, all_test_inputs)
    a, info, Mall, fall = optimiser_fn["solver_fn"](
        terms_left, terms_right, takess, constraints_left, J, C, J_active, a0_fixed, optimiser_fn,
        testing,
        )
    if np.isnan(a).any() or np.isinf(a).any():
        logger.warn("a has nans / infs")
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)# deal with bad solves
    logger.info(f"Solver ran {info[0]} iterations")
    np.savez(f"{summary_out_dir}/info.npz", *info)
    #print(info[1])

    # update active params
    active_params["network"]["subdomain"]["basis_coeffs"] = a.reshape((J_active, 1, C))

    # compute final loss
    loss = test(info[0], Mall, fall, a, a0_fixed, J_active, C, active_params,
                *all_test_inputs)
    logger.info(f"test losses: {loss}")
    lossval = jnp.array(loss[1])

    return lossval, active_opt_states, active_params


def test(i, Mall, fall, a, a0_fixed, J_active, C, active_params,
         all_params, merge_active, test_inputs, x_batch_test, model_fns, u_exact, start0):
    "Jittable version of computing of test loss, same workflow as FBPINNTrainer"

    # TODO: actually make jittable! (see notes below)

    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)# deal with bad solves

    # get train loss
    aall = jnp.concatenate([a.reshape(-1,1), a0_fixed],0)
    loss = float(np.sum((Mall @ aall - fall)**2))

    # update active params
    active_params["network"]["subdomain"]["basis_coeffs"] = a.reshape((J_active, 1, C))


    # same as FBPINNTrainer

    # merge latest all_params / all_opt_states
    all_params["trainable"] = merge_active(active_params, all_params["trainable"])

    # get FBPINN solution using test data
    takes, all_ims, cut_all = test_inputs
    all_params_cut = {"static":cut_all(all_params["static"]),
                      "trainable":cut_all(all_params["trainable"])}
    u_test, wp_test_, us_test_, ws_test_, us_raw_test_ = FBPINN_model(all_params_cut, x_batch_test, takes, model_fns, verbose=False)

    # get losses over test data
    l1 = jnp.mean(jnp.abs(u_exact-u_test)).item()
    l1n = l1 / u_exact.std().item()

    return [i, time.time()-start0, loss, l1, l1n]


class ELMFBPINNTrainer(FBPINNTrainer):

    def _train_loop(self, scheduler,
                    all_params, all_opt_states,
                    x_batch_global, constraints_global, constraint_fs_global, constraint_offsets_global,
                    decomposition, problem, optimiser_fn, model_fns, jmapss, loss_fn,
                    u_exact, x_batch_test, test_inputs,
                    writer):

        model_fns_ELM = model_fns[:1]+(self.c.network.basis_fn,)+model_fns[2:]

        pstep, fstep, u_test_losses = 0, 0, []
        start0, start1, report_time = time.time(), time.time(), 0.
        merge_active, active_params, active_opt_states = None, None, None
        lossval = None
        for i,active_ in enumerate(scheduler):

            # update active
            if active_ is not None:
                active = active_

                # first merge latest all_params / all_opt_states
                if i != 0:
                    all_params["trainable"] = merge_active(active_params, all_params["trainable"])
                    all_opt_states = tree_map_dicts(merge_active, active_opt_states, all_opt_states)

                # then get new inputs to update step
                active, merge_active, active_opt_states, active_params, fixed_params, static_params, takess, constraints, x_batch = \
                     self._get_update_inputs(i, active, all_params, all_opt_states, x_batch_global, constraints_global, constraint_fs_global, constraint_offsets_global, decomposition, problem)

                # report initial model
                if i == 0:
                    u_test_losses, start1, report_time = \
                    self._report(i, pstep, fstep, u_test_losses, start0, start1, report_time,
                                u_exact, x_batch_test, test_inputs, all_params, all_opt_states, model_fns, problem, decomposition,
                                active, merge_active, active_opt_states, active_params, x_batch,
                                lossval)

                # run linear solve
                # Note: we design this function so that it supports linear solvers which are designed to be jittable at the global solve level (rather than single update step level)
                # and use internal callbacks at each update step
                # this allows us to more easily "plug and play" with e.g. lineax / scipy / of-the-shelf linear solvers
                # this means, if the solver supports callbacks, the reporting process needs to made jittable (specifically the test loss)
                # therefore we define a jittable test() function above
                # note this is different to FBPINNTrainer, which only jits the update step (not the reporting process)
                static_params_dynamic, static_params_static = partition(static_params)
                lossval, active_opt_states, active_params = FBPINN_linear_solve(optimiser_fn, active_opt_states,
                                  active_params, fixed_params, static_params_dynamic, static_params_static,
                                  takess, constraints, model_fns_ELM, jmapss, loss_fn,# uses same arguments as compiled update function in FBPINNTrainer

                                  # test inputs
                                  (all_params, merge_active, test_inputs, x_batch_test, model_fns, u_exact, start0), self.c.test_freq, self.c.summary_out_dir,# TODO: make these inputs jittable!
                                  # specifically, make merge_active, cut_all jittable (currently they constant fold all_ims/ active_ims/ fixed_ims)
                                  # and split out static arguments
                                  )

                p = f = 0
                pstep, fstep = pstep+p, fstep+f

                # report
                u_test_losses, start1, report_time = \
                self._report(i + 1, pstep, fstep, u_test_losses, start0, start1, report_time,
                            u_exact, x_batch_test, test_inputs, all_params, all_opt_states, model_fns, problem, decomposition,
                            active, merge_active, active_opt_states, active_params, x_batch,
                            lossval)

        # report
        u_test_losses, start1, report_time = \
        self._report(i + 1, pstep, fstep, u_test_losses, start0, start1, report_time,
                    u_exact, x_batch_test, test_inputs, all_params, all_opt_states, model_fns, problem, decomposition,
                    active, merge_active, active_opt_states, active_params, x_batch,
                    lossval)

        # cleanup
        writer.close()
        logger.info(f"[i: {i+1}/{self.c.n_steps}] Training complete")

        # return trained parameters
        all_params["trainable"] = merge_active(active_params, all_params["trainable"])
        all_opt_states = tree_map_dicts(merge_active, active_opt_states, all_opt_states)

        return all_params
