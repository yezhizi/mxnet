#!/usr/bin/env python

import logging
import os
import sys
import random
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon, kv, nd
from mxnet.gluon.model_zoo import vision

# get command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--logfile', type=str, default='', help='log file')
args = parser.parse_args()
logfile = args.logfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if(logfile != ''):
    fh = logging.FileHandler(logfile)
    logger.addHandler(fh)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)

use_cuda = os.environ.get('USE_CUDA', '0') == '1'
mx.random.seed(42)
random.seed(42)

kvstore = kv.create('dist')

num_outputs = 10 # classify the images into one of the 10 digits

batch_size_per_gpu = 64
epochs = 5

num_gpus = mx.context.num_gpus()  # number of GPUs on this machine
ctx = [mx.gpu(i) for i in range(num_gpus)] if use_cuda else [mx.cpu()]
batch_size = batch_size_per_gpu * (len(ctx))

# We'll use cross entropy loss since we are doing multiclass classification
loss = gluon.loss.SoftmaxCrossEntropyLoss()

# Convert to float 32
# Having channel as the first dimension makes computation more efficient. Hence the (2,0,1) transpose.
# Dividing by 255 normalizes the input between 0 and 1
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2, 0, 1))/255, label.astype(np.float32)

class SplitSampler(gluon.data.sampler.Sampler):
    """ Split the dataset into `num_parts` parts and sample from the part with index `part_index`

    Parameters
    ----------
    length: int
      Number of examples in the dataset
    num_parts: int
      Partition the data into multiple parts
    part_index: int
      The index of the part to read from
    """
    def __init__(self, length, num_parts=1, part_index=0):
        # Compute the length of each partition
        self.part_len = length // num_parts
        # Compute the start index for this partition
        self.start = self.part_len * part_index
        # Compute the end index for this partition
        self.end = self.start + self.part_len

    def __iter__(self):
        # Extract examples between `start` and `end`, shuffle and return them.
        indices = list(range(self.start, self.end))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.part_len

# Evaluate accuracy of the given network using the given data
def evaluate_accuracy(data_iterator, network):
    """ Measure the accuracy of ResNet

    Parameters
    ----------
    data_iterator: Iter
      examples of dataset
    network:
      ResNet

    Returns
    ----------
    tuple of array element
    """
    acc = mx.metric.Accuracy()

    # Iterate through data and label
    for i, (data, label) in enumerate(data_iterator):

        # Get the data and label into the GPU
        data = data.as_in_context(ctx[0])
        label = label.as_in_context(ctx[0])

        # Get network's output which is a probability distribution
        # Apply argmax on the probability distribution to get network's classification.
        output = network(data)
        predictions = nd.argmax(output, axis=1)

        # Give network's prediction and the correct label to update the metric
        acc.update(preds=predictions, labels=label)

    # Return the accuracy
    return acc.get()[1]

# Train a batch using multiple GPUs
def train_batch(batch_list, context, network, gluon_trainer):
    """ Training with multiple GPUs
    Parameters
    ----------
    batch_list: List
      list of dataset
    context: List
      a list of all GPUs to be used for training
    network:
      ResNet
    gluon_trainer:
      rain module of gluon
    """
    
    # Run one forward and backward pass on multiple GPUs
    def forward_backward(network, data, label):

        # Ask autograd to remember the forward pass
        with autograd.record():
            # Compute the loss on all GPUs
            losses = [loss(network(X), Y) for X, Y in zip(data, label)]

        # Run the backward pass (calculate gradients) on all GPUs
        for l in losses:
            l.backward()

    # Split and load data into multiple GPUs
    data = batch_list[0]
    data = gluon.utils.split_and_load(data, context)
    
    num_samples = len(data)

    # Split and load label into multiple GPUs
    label = batch_list[1]
    label = gluon.utils.split_and_load(label, context)

    # Run the forward and backward pass
    forward_backward(network, data, label)
    
    params = list(network.collect_params().values())
    for idx, param in enumerate(params):
        if param.grad_req == 'null':
            continue
        kvstore.push(idx, param.grad() / num_samples, priority=-idx)
    
    for idx, param in enumerate(params):
        if param.grad_req == 'null':
            continue
        # temp = nd.zeros(param.shape, ctx=ctx[0])
        kvstore.pull(idx, param.list_grad(), priority=-idx)
        param.grad().wait_to_read()
        # temp.wait_to_read()
        # param.grad()[:] = temp
        # kvstore.pull(idx, param.grad(), priority=-idx)
    # nd.waitall()
            
    # Update the parameters
    # this_batch_size = batch_list[0].shape[0]
    gluon_trainer.step(kvstore.num_workers)
    # for idx, param in enumerate(params):
    #     logger.info ("Key: %d, Value: %f" % (idx, np.mean(param.data().asnumpy())))

# Load the training data
train_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=True, transform=transform), batch_size,
                                   sampler=SplitSampler(50000),last_batch='discard')

# Load the test data
test_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=False, transform=transform), batch_size,
                                  sampler=SplitSampler(500), shuffle=False)
# test_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=False, transform=transform), batch_size,
#                                   shuffle=False)

# Use ResNet from model zoo
net = vision.resnet18_v1()

# Initialize the parameters with Xavier initializer
net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
net(nd.ones((1,3,32,32), ctx=ctx[0]))

# SoftmaxCrossEntropy is the most common choice of loss function for multiclass classification
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# Use Adam optimizer. Ask trainer to use the distributor kv store.
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .001})


kvstore.notify_begin()

# Init parameters
params = list(net.collect_params().values())
for idx, param in enumerate(params):
    if param.grad_req == 'null':
        continue
    kvstore.init(idx, param.data(), priority=-idx)
    
    # we merge the `pull` into  `init` for initializing the parameters at first time
    # kvstore.pull(idx, param.data(), priority=-idx)
    param.data().wait_to_read()

# for idx, param in enumerate(params):
#     kvstore.test(idx, param.data())
#     param.data().wait_to_read()
#     print("Key: ", idx, "Value: ", np.mean(param.data().asnumpy()))

# for idx, param in enumerate(params):
#     # key mean
#     logger.info ("Key: %d, Value: %f" % (idx, np.mean(param.data().asnumpy())))

test_per_batch = 20

# Iterate through batches and run training using multiple GPUs
gloval_batch_num = 1

# Run as many epochs as required
for epoch in range(epochs):

    for batch in train_data:
        # logger.info("Batch %d" % batch_num)
        # Train the batch using multiple GPUs
        train_batch(batch, ctx, net, trainer)
        if test_per_batch > 0 and gloval_batch_num % test_per_batch == 0:
            test_accuracy = evaluate_accuracy(test_data, net)
            logger.info("Epoch %d: Batch %d: Test_acc %f" % (epoch, gloval_batch_num, test_accuracy))
        
        gloval_batch_num += 1
        kvstore.batch_end()
      

    # Print test accuracy after every epoch
    test_accuracy = evaluate_accuracy(test_data, net)
    logger.info("Epoch %d: Test_acc %f" % (epoch, test_accuracy))
