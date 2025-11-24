# Local Feature Filtering for Scalable and Well-Conditioned Domain-Decomposed Random Feature Methods

This branch reproduces the results of our paper: [Local Feature Filtering for Scalable and Well-Conditioned Domain-Decomposed Random Feature Methods](https://arxiv.org/abs/2506.17626).

## Reproducing our results

To reproduce our results, install the `fbpinns` library and then run `main.py`.

Once all the models have been trained using `main.py` (this script will save models in the `results/` folder), run `Paper plots.ipynb` to reproduce all of the figures in the paper.

## Description of files

`main.py` defines and trains all the FBPINNs and PINNs used in the paper.

`problems.py` defines all the problems studied in the paper.

`plot.py` and `Paper plots.ipynb` reproduces all of the figures shown in the paper.

## Using PBS

`main.py` is set up to submit each training run as an independent job using [PBS](https://en.wikipedia.org/wiki/Portable_Batch_System).

If you are running this script locally, or if your server does not use PBS, you will need to alter the last line of `main.py` appropriately before you can run the script.

If you are using PBS, you need to define the `qsub` command which `main.py` uses to submit each PBS job, inside a `run.sh` file. An example `run.sh` file is provided for convenience, but you will likely need to alter this depending on your own PBS system.

## Further questions?

Please raise a GitHub [issue](https://github.com/benmoseley/FBPINNs/issues) or feel free to contact us.
