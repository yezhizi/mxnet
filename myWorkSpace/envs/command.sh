# export COMMAND='python /home/deemo/mxnet/myWorkSpace/cifar10_dist.py'
export COMMAND="python3 $(dirname $(dirname $(readlink -f $0)))/test.py"