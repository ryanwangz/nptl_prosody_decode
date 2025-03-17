#!/usr/bin/bash
#SBATCH --job-name=jn_start
#SBATCH --output=jn_start.%j.out
#SBATCH --error=jn_start.%j.err
#SBATCH --time=47:59:59
#SBATCH -p owners,henderj
#SBATCH -c 4
#SBATCH --mem=32GB

# load modules
module reset
ml load python/3.9.0 cudnn/8.6.0.163
ml gcc/10.1.0
ml ffmpeg
ml cmake
# ml py-scipystack
# ml py-scipystack math py-autograd py-pytorch cuda torch praat matlab

source $GROUP_HOME/rzwang/jupyter_env/bin/activate
jupyter lab --no-browser --port=8888 --notebook-dir=$GROUP_HOME/
