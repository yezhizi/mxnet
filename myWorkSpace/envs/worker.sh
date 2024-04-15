# 当前目录下的command.sh
source $(dirname $BASH_SOURCE)/command.sh

# export MXNET_LIBRARY_PATH=$(readlink -f $(dirname $0)/../build/libmxnet.so) 
source $(dirname $0)/envs.sh

export DMLC_ROLE=worker
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=9092
export DMLC_NUM_SERVER=1
export DMLC_NUM_WORKER=1
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
if [[ $0 != "$BASH_SOURCE" ]]; then
    echo "The environment variables are set for the worker."
    echo "DMLC_ROLE=$DMLC_ROLE"
    echo "DMLC_PS_ROOT_URI=$DMLC_PS_ROOT_URI"
    echo "DMLC_PS_ROOT_PORT=$DMLC_PS_ROOT_PORT"
    echo "DMLC_NUM_SERVER=$DMLC_NUM_SERVER"
    echo "DMLC_NUM_WORKER=$DMLC_NUM_WORKER"
else
    echo "The Worker is running."
    $COMMAND 
fi