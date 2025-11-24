"""
Generic helper functions for running parallel sbatch jobs
"""

import copy
import subprocess
import sys
import pickle

from fbpinns.trainers import FBPINNTrainer, PINNTrainer
from fbpinns.constants import Constants

from elm.trainers import ELMFBPINNTrainer


def apply_config_deltas(p0, deltas):
    "Returns a list of parameter configurations with deltas applied"

    configs = [copy.deepcopy({**p0, **delta}) for delta in deltas]+[p0,]# create new dictionary with parameter overrides
    # note important to deepcopy otherwise configs may share nested references

    # Check all configs have the same keys
    assert all(set(d.keys()) == set(configs[0].keys()) for d in configs), "configs have key mismatch"

    # Filter out unique config combinations
    unique, configs_filtered, c = [], [], 0
    for d in configs:
        s = str(sorted(d.items()))# serialised
        if s in unique:
            c +=1
        else:
            unique.append(s)
            configs_filtered.append(d)
    if c > 0:
        print(f"Warning: {c} duplicate config(s) detected and removed")

    return configs_filtered


def submit(runs):
    "submits parallel sbatch jobs"

    for i,runs_ in enumerate(runs):
        print(f"submitting job {i+1} of {len(runs)}..")
        args = []
        for c,trainer in runs_:
            # save constants
            c.get_outdirs()
            c.save_constants_file()
            args += [c.constants_file, trainer]
        # submit batch job
        o = subprocess.run(["bash", "run.sh", f"{i}"]+args,
                           check=False, capture_output=True)
        print(o.stdout.decode())
        print(o.stderr.decode())
        if o.returncode != 0:
            raise Exception(f"process returned non-zero exit code {o.returncode}")
    print("all jobs submitted")

def run():
    "target function for sbatch job"

    # parse arguments
    args = sys.argv[1:][1:]
    print(args)
    assert len(args) % 2 == 0

    # run batch of jobs
    for constants_file, trainer in zip(args[0::2], args[1::2]):

        # get trainer
        if trainer not in ["FBPINN", "PINN", "ELMFBPINN"]:
            raise Exception(f"ERRROR: unexpected trainer: {trainer}")
        trainer = {"FBPINN": FBPINNTrainer,
                   "PINN": PINNTrainer,
                   "ELMFBPINN": ELMFBPINNTrainer}[trainer]

        # get constants
        with open(constants_file, "rb") as f:
            kwargs = pickle.load(f)

        # create constants object
        c = Constants(**kwargs)
        c.show_figures = c.clear_output = False# make sure plots are not shown

        # create trainer, start training
        run = trainer(c)
        run.train()