source $(dirname $BASH_SOURCE)/command.sh

export DMLC_ROLE=scheduler
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=9092
export DMLC_NUM_SERVER=2
export DMLC_NUM_WORKER=2
if [[ $0 != "$BASH_SOURCE" ]]; then
    echo "The environment variables are set for the scheduler."
    echo "DMLC_ROLE=$DMLC_ROLE"
    echo "DMLC_PS_ROOT_URI=$DMLC_PS_ROOT_URI"
    echo "DMLC_PS_ROOT_PORT=$DMLC_PS_ROOT_PORT"
    echo "DMLC_NUM_SERVER=$DMLC_NUM_SERVER"
    echo "DMLC_NUM_WORKER=$DMLC_NUM_WORKER"
else
    echo "The Scheduler is running."
    $COMMAND &
fi