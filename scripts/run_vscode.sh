#!/usr/bin/bash
#SBATCH --job-name=vscode
#SBATCH --output=vscode.%j.out
#SBATCH --error=vscode.%j.err
#SBATCH --time=47:59:59
#SBATCH -p owners,henderj
#SBATCH -c 4
#SBATCH --mem=32GB

PORT=$1
if [ -z $PORT ]; then
    echo "Usage: sbatch vscode.sh <port>"
    exit 1
fi

# vscode works better with git>2
ml load system git

# Load your modules here
# ml load
ml load matlab
ml load python/3.9.0 cudnn/8.6.0.163
ml gcc/10.1.0
ml ffmpeg
ml cmake
#ml load anaconda

# source /home/users/sasidhar/vscodePy/bin/activate
source /home/groups/henderj/rzwang/vscode_env/bin/activate

# Add folder to MATLAB path
# matlab -batch "addpath('/oak/stanford/groups/henderj/sasidhar/nptlrig2'); savepath; exit;"

# Create a bin directory in your home directory
WORK_DIR=$HOME/bin
if [ ! -z $WORK_DIR ]; then
    mkdir $WORK_DIR
fi

# Download the latest version of vscode
if [ ! -f $WORK_DIR/code ]; then
    cd $WORK_DIR
    curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
    tar -xzf vscode_cli.tar.gz
    rm vscode_cli.tar.gz
fi

# Run vscode
cd $WORK_DIR
./code serve-web --port $PORT --without-connection-token
