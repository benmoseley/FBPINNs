#!/bin/bash
mkdir -p logs

# note heredoc performs variable expansion, command substitution, and other parsing immediately, unless using 'EOF'
# so make sure you escape variables with \$ if you want late evaluation

# heredoc doesn't quote and expand input parameters like bash, so
# expand input arguments as quoted string suitable for heredoc
quoted_args=$(printf " \'"%s"\'" "$@")
quoted_args=${quoted_args# }

qsub <<EOF
#!/bin/bash
#PBS -l select=1:ncpus=1:mem=50gb:cpu_type=rome
#PBS -l walltime=24:0:0
#PBS -N main.$1
#PBS -j oe
#PBS -o logs/log-$1.txt

# path of directory job was submitted in
cd \$PBS_O_WORKDIR

# or for fast local storage
#cd \$TMPDIR
#rsync -aq \${PBS_O_WORKDIR} \${TMPDIR}
#rsync -auq \${TMPDIR} \${PBS_O_WORKDIR}

eval "\$(~/miniforge3/bin/conda shell.bash hook)"
conda activate jax

export PYTHONPATH="\$(realpath ../):\$PYTHONPATH"

python - $quoted_args <<EOFP
import jax
jax.config.update("jax_enable_x64", True)
from fbpinns.util.sbatch import run
run()
EOFP

EOF