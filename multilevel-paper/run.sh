#!/bin/bash
mkdir -p logs

sbatch <<EOF
#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH -n 8
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=logs/log-$1-%j.txt
#SBATCH --error=logs/log-$1-%j.txt
#SBATCH --job-name=main.$1

module load gcc/6.3.0 cuda cudnn
cd $PWD
python - <<EOFP "$2" "$3"
from fbpinns.util.sbatch import run
run()
EOFP

EOF
