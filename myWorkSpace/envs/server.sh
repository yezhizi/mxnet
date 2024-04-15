COMMAND=("python3" "-c" "import mxnet")

# export MXNET_LIBRARY_PATH=$(readlink -f $(dirname $0)/../build/libmxnet.so) 
source $(dirname $0)/envs.sh

export DMLC_ROLE=server
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=9092
export DMLC_NUM_SERVER=1
export DMLC_NUM_WORKER=1
if [[ $0 != "$BASH_SOURCE" ]]; then
    echo "The environment variables are set for the server."
    echo "DMLC_ROLE=$DMLC_ROLE"
    echo "DMLC_PS_ROOT_URI=$DMLC_PS_ROOT_URI"
    echo "DMLC_PS_ROOT_PORT=$DMLC_PS_ROOT_PORT"
    echo "DMLC_NUM_SERVER=$DMLC_NUM_SERVER"
    echo "DMLC_NUM_WORKER=$DMLC_NUM_WORKER"
else
    echo "The Server is running."
    "${COMMAND[@]}" 
fi