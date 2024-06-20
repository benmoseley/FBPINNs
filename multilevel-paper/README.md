# Multilevel domain decomposition-based architectures for physics-informed neural networks

This branch reproduces the results of our paper: [Multilevel domain decomposition-based architectures for physics-informed neural networks](https://doi.org/10.1016/j.cma.2024.117116).

## Reproducing our results

To reproduce our results, install the `fbpinns` library and then run `main.py`.

Once all the models have been trained using `main.py` (this script will save models in the `results/` folder), run `Paper plots.ipynb` to reproduce all of the figures in the paper.

## Description of files

`main.py` defines and trains all the FBPINNs and PINNs used in the paper.

`problems.py` defines all the problems studied in the paper.

`plot.py` and `Paper plots.ipynb` reproduces all of the figures shown in the paper.

`fourier_pinn_ablation.py` is an extra script which was used to select optimal hyperparameters for the PINNs with Fourier input features used in the paper.

## Using slurm

`main.py` is set up to submit each training run as an independent job using [slurm](https://slurm.schedmd.com/).

If you are running this script locally, or if your server does not use slurm, you will need to alter the last line of `main.py` appropriately before you can run the script.

If you are using slurm, you need to define the `sbatch` command which `main.py` uses to submit each slurm job, inside a `run.sh` file. An example `run.sh` file is provided for convenience, but you will likely need to alter this depending on your own slurm system.

## Further questions?

Please raise a GitHub [issue](https://github.com/benmoseley/FBPINNs/issues) or feel free to contact us.
