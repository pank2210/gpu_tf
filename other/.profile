

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64

export RET_HOME=/data1/ret

export KAGGLE_CONFIG_DIR=$RET_HOME/kaggle/
export SRC=$RET_HOME/gpu_tf
export DATA=$RET_HOME/../data/

source $RET_HOME/../venv/bin/activate
set -o vi
