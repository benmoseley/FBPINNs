"""
Defines some helper functions for plotting
"""

import math
from IPython.display import Latex
import jax.numpy as jnp

from fbpinns.analysis import load_model
from fbpinns.analysis import FBPINN_solution as FBPINN_solution_
from fbpinns.analysis import PINN_solution as PINN_solution_


def load_ELMFBPINN(tag, problem, network, l, w, h, p, n, lr, seed, optimiser, optimiser_kwargs, rootdir="results/"):
    run = f"ELMFBPINN_{tag}_{problem.__name__}_{network.__name__}_{l}-levels_{w}-overlap_{h}-layers_{p}-hidden_{n[0]}-n_{optimiser.__name__}-{optimiser_kwargs['system']}-{optimiser_kwargs['solver'].__name__}-{seed}"
    c, model = load_model(run, rootdir=rootdir)
    i,t,l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]
    return c, model, i, t, l1n

def load_FBPINN(tag, problem, network, l, w, h, p, n, lr, seed, rootdir="results/"):
    run = f"FBPINN_{tag}_{problem.__name__}_{network.__name__}_{l}-levels_{w}-overlap_{h}-layers_{p}-hidden_{n[0]}-n_{lr}-lr-{seed}"
    c, model = load_model(run, rootdir=rootdir)
    i,t,l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]
    return c, model, i, t, l1n

def load_PINN(tag, problem, network, h, p, n, lr, seed, rootdir="results/"):
    run = f"PINN_{tag}_{problem.__name__}_{network.__name__}_{h}-layers_{p}-hidden_{n[0]}-n_{lr}-lr-{seed}"
    c, model = load_model(run, rootdir=rootdir)
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


def _convert_tabular_to_array(entry):
    """
    Convert a LaTeX tabular-style string (with inline math) to array-compatible form,
    wrapping plain text in \\text{...} and preserving math expressions.

    Args:
        entry (str): The input string, e.g., "some $math$ text"

    Returns:
        str: Transformed string suitable for use in LaTeX array environments
    """
    parts = entry.split("$")
    result = []
    for i, part in enumerate(parts):
        if part:
            if i % 2 == 0:
                result.append(r"\text{" + part + "}")
            else:
                result.append(part)
    result = "".join(result)
    return result

def make_latex_array(columns, rows):
    """
    Generate a minimal LaTeX array-based table for JupyterLab display.

    Args:
        columns (list of str): Column headers.
        rows (list of lists): Table rows.

    Returns:
        IPython.display.Latex: A renderable LaTeX array table.
    """
    table = "\\begin{array}{" + "c" * len(columns) + "}\n"
    table += " & ".join(_convert_tabular_to_array(str(c)) for c in columns) + " \\\\\n"
    table += "\\hline\n"
    for row in rows:
        table += " & ".join(_convert_tabular_to_array(str(val)) for val in row) + " \\\\\n"
    table += "\\end{array}\n"
    return Latex(table)

def make_latex_tabular(columns, rows):
    """
    Generate a minimal LaTeX table.

    Args:
        columns (list of str): Column headers.
        rows (list of lists): Table rows.

    Returns:
        str: LaTeX tabular environment.
    """
    table = "\\begin{tabular}{" + "c" * len(columns) + "}\n"
    table += "\\toprule\n"
    table += " & ".join(str(c) for c in columns) + " \\\\\n"
    table += "\\midrule\n"
    for row in rows:
        table += " & ".join(str(val) for val in row) + " \\\\\n"
    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    return table

def compact_sci(mean, std, precision=1, sci_thresh=1):
    "Pretty format mean +/- std in scientific notation"
    if mean == 0:
        exp = 0
    else:
        exp = int(math.floor(math.log10(abs(mean))))

    if abs(exp) < sci_thresh:
        # Use regular fixed-point formatting
        return f"{mean:.{precision}f}$\\pm${std:.{precision}f}"
    else:
        # Use compact scientific notation with shared exponent
        scale = 10 ** exp
        m_scaled = mean / scale
        s_scaled = std / scale
        return f"{m_scaled:.{precision}f}$\\pm${s_scaled:.{precision}f}e{exp:+03d}"