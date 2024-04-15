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
"""A server node for the key value store."""
import ctypes
import sys
import pickle
import logging
from ..base import _LIB, check_call
from .base import create

__all__ = ['KVStoreController']

class KVStoreController(object):
    """The key-value store controller."""
    def __init__(self, kvstore):
        """Initialize a new KVStoreController.

        Parameters
        ----------
        kvstore : KVStore
        """
        self.kvstore = kvstore
        self.handle = kvstore.handle
        self.init_logginig = False

    def run(self):
        """Run the controller.
        """
        
        check_call(_LIB.MXKVStoreRunController(self.handle))

def _init_kvstore_controller_module():
    """Start scheduler."""
    is_controller = ctypes.c_int()
    check_call(_LIB.MXKVStoreIsSchedulerNode(ctypes.byref(is_controller)))
    if is_controller.value == 1:
        kvstore = create('dist')
        controller = KVStoreController(kvstore)
        controller.run()
        sys.exit()

_init_kvstore_controller_module()
