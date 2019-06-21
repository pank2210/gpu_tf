

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64

export RET_HOME=/disk1/data1/ret

export KAGGLE_CONFIG_DIR=$RET_HOME/kaggle/
export SRC=$RET_HOME/gpu_tf
export DATA=$RET_HOME/../data/

source $RET_HOME/venv/bin/activate

export XLA_FLAGS=--xla_hlo_profile
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit


set -o vi
