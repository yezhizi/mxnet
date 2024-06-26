#!/usr/bin/env python3

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=arguments-differ

# This test checks if dynamic loading of library into MXNet is successful
# and checks the end of end computation of custom operator

import os, ctypes
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd
from mxnet.base import _LIB, check_call, mx_uint, c_str, c_str_array, SymbolHandle

# load library
if (os.name=='posix'):
    path = os.path.abspath('libsubgraph_lib.so')
    mx.library.load(path)
elif (os.name=='nt'):
    path = os.path.abspath('libsubgraph_lib.dll')
    mx.library.load(path)

# example model, ops to be partitioned do not have args (use outputs from other ops as inputs)
a = mx.sym.var('a')
b = mx.sym.var('b')
c = a + b
d = mx.sym.exp(c)
sym = mx.sym.log(d)

# example model, ops to be partitioned have args
d2 = mx.sym.exp(a)
sym2 = mx.sym.log(d2)

def test(backend):
    args = {'a':mx.nd.ones((3,2)), 'b':mx.nd.ones((3,2))}
    ###############################################
    # Test with subgraph not consuming params
    ###############################################
    #execute in MXNet
    print('-------------------------------')
    print('Testing regular MXNet execution')
    exe = sym.bind(ctx=mx.cpu(), args=args)
    out = exe.forward()
    print(out)

    # with propogating shapes/types
    print('-------------------------------')
    print('Testing %s partitioning with shapes/types' % backend)
    print(sym.tojson())
    mysym2 = sym.optimize_for(backend, args, dedup_subgraph=True)
    print(mysym2.tojson())
    exe2 = mysym2.bind(ctx=mx.cpu(), args=args)
    out2 = exe2.forward()
    print(out2)

    # with propogating shapes/types, rejecting subgraph
    print('-------------------------------')
    print('Testing %s partitioning with shapes/types - rejecting subgraph' % backend)
    mysym2 = sym.optimize_for(backend, args, reject=True, dedup_subgraph=True)
    exe2 = mysym2.bind(ctx=mx.cpu(), args=args)
    out2 = exe2.forward()
    print(out2)

    # without propogating shapes/types
    print('-------------------------------')
    print('Testing %s partitioning without shapes/types' % backend)
    mysym3 = sym.optimize_for(backend, myOpt='yello', dedup_subgraph=True)
    exe3 = mysym3.bind(ctx=mx.cpu(), args=args)
    out3 = exe3.forward()
    print(out3)

    # Gluon Hybridize partitioning with shapes/types
    print('-------------------------------')
    print('Testing %s Gluon Hybridize partitioning with shapes/types' % backend)
    inputs = [a,b]
    sym_block = nn.SymbolBlock(sym, inputs)
    sym_block.initialize()
    sym_block.hybridize(backend=backend, dedup_subgraph=True)
    out2 = sym_block(mx.nd.ones((3,2)),mx.nd.ones((3,2)))
    print(out2)

    # Gluon Hybridize partitioning with shapes/types without inference
    print('-------------------------------')
    print('Testing %s Gluon Hybridize partitioning with shapes/types without inference' % backend)
    inputs = [a,b]
    sym_block2 = nn.SymbolBlock(sym, inputs)
    sym_block2.initialize()
    sym_block2.optimize_for(mx.nd.ones((3,2)), mx.nd.ones((3,2)), backend=backend,
                            dedup_subgraph=True)
    sym_block2.export('partitioned')

    # Test with additional input to subgraph op
    print('-------------------------------')
    print('Testing %s Gluon Hybridize partitioning with extra input' % backend)
    sym_block2.optimize_for(mx.nd.ones((3,2)), mx.nd.ones((3,2)), backend="addInputPass",
                            dedup_subgraph=True)
    out3 = sym_block2(mx.nd.ones((3,2)),mx.nd.ones((3,2)))
    print(out3)
    
    ###############################################
    # Test with subgraph directly consuming params
    ###############################################
    args = {'a':mx.nd.ones((3,2))}
    #execute in MXNet
    print('-------------------------------')
    print('Testing regular MXNet execution')
    exe5 = sym2.bind(ctx=mx.cpu(), args=args)
    out5 = exe5.forward()
    print(out5)

    # with propogating shapes/types
    print('-------------------------------')
    print('Testing %s partitioning with shapes/types' % backend)
    mysym6 = sym2.optimize_for(backend, args, reqArgs=True, dedup_subgraph=True)
    print(mysym6.tojson())
    exe6 = mysym6.bind(ctx=mx.cpu(), args=args)
    out6 = exe6.forward()
    print(out6)

    # without propogating shapes/types
    print('-------------------------------')
    print('Testing %s partitioning without shapes/types' % backend)
    mysym7 = sym2.optimize_for(backend, reqArgs=True, dedup_subgraph=True)
    exe7 = mysym7.bind(ctx=mx.cpu(), args=args)
    out7 = exe7.forward()
    print(out7)

    # Gluon Hybridize partitioning with shapes/types
    print('-------------------------------')
    print('Testing %s Gluon Hybridize partitioning with shapes/types' % backend)
    inputs = [a]
    sym2_block = nn.SymbolBlock(sym2, inputs)
    sym2_block.initialize()
    sym2_block.hybridize(backend=backend)
    out8 = sym2_block(mx.nd.ones((3,2)))
    print(out8)

test("myProp")
test("mySelect")
