export COMMAND='python /home/deemo/mxnet/example/gluon/image_classification.py \
--model     alexnet \
--dataset   caltech101 \
--gpus      0  \
--epoch     1 \
--kvstore   dist-sync-device'
