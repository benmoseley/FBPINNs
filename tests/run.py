# for now, just run simple tests on 1D harmonic oscillator
# TODO: add more tests

from fbpinns.constants import Constants
from fbpinns.problems import HarmonicOscillator1D, HarmonicOscillator1DHardBC, HarmonicOscillator1DInverse
from fbpinns.util.logger import logger
from fbpinns.trainers import FBPINNTrainer, PINNTrainer

logger.setLevel("DEBUG")

for problem in [
        HarmonicOscillator1D, HarmonicOscillator1DHardBC, HarmonicOscillator1DInverse
        ]:

    c = Constants(
        run="test",
        problem=problem,
        show_figures=False,
        )

    run = FBPINNTrainer(c)
    all_params = run.train()

    run = PINNTrainer(c)
    all_params = run.train()