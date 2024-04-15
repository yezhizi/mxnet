/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef MXNET_KVSTORE_KVSTORE_DIST_CONTROLLER_H_
#define MXNET_KVSTORE_KVSTORE_DIST_CONTROLLER_H_
#include <mxnet/c_api.h>
#include <mxnet/kvstore.h>
#include <ps/ps.h>
#include <queue>
#include <string>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>
#include <future>
#include <vector>
#include "../profiler/profiler.h"
#include "../operator/tensor/elemwise_binary_op-inl.h"
#include "../operator/tensor/init_op.h"
#include "./kvstore_dist_server.h"

namespace mxnet {
namespace kvstore {

class KVStoreDistController {
 public:
  KVStoreDistController() {
    using namespace std::placeholders;
    ps_scheduler_ = new ps::KVServer<char>(0);
    static_cast<ps::SimpleApp*>(ps_scheduler_)
        ->set_request_handle(std::bind(&KVStoreDistController::RequestHandle, this, _1, _2));
    static_cast<ps::SimpleApp*>(ps_scheduler_)
        ->set_response_handle(std::bind(&KVStoreDistController::ResponseHandle, this, _1, _2));

    log_verbose_ = dmlc::GetEnv("MXNET_KVSTORE_DIST_ROW_SPARSE_VERBOSE", false);
  }

  ~KVStoreDistController() {
    profiler::Profiler::Get()->SetState(profiler::Profiler::ProfilerState(0));
    delete ps_scheduler_;
  }

  static bool isSignal(int head) {
    return head > ps::SignalBound;
  }

 private:

  struct ScalingNodes {
    std::vector<int> nodes;
    bool is_worker;
    int num_nodes = 0;
  };
  ScalingNodes scaling_nodes_;

  void RequestHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
    ps::kControllerSignal signal;
    int head = recved.head;
    const std::string& body = recved.body;
    if (isSignal(head)) {
      signal = static_cast<ps::kControllerSignal>(head);
      switch (signal) {
        case ps::kControllerSignal::kAddNodeSignal: {
          bool is_worker = body[0] == 'w';
          int node_id = std::stoi(body.substr(1));
          std::vector<int>& nodes = scaling_nodes_.nodes;
          if (!nodes.empty()) {
            CHECK(is_worker && scaling_nodes_.is_worker)
                << "Not Support different type of node scalling";
          }
          nodes.push_back(node_id);
          scaling_nodes_.is_worker = is_worker;
          break;
        }
        default: {
          LOG(WARNING) << "Controller UnknowSignal";
          break;
        }
      }
    } else {
      ControllerCommand command;
      command = static_cast<ControllerCommand>(head);
      switch (command) {
        case ControllerCommand::kNotifyPreFinished: {
          int sender = recved.sender;
          std::vector<int>& nodes = scaling_nodes_.nodes;
          if(std::find(nodes.begin(), nodes.end(), sender) == nodes.end()) {
            LOG(WARNING)<< "Controller Recieve Unknow Node id: "<< sender;
            break;
          }
          scaling_nodes_.num_nodes++;
          if(scaling_nodes_.num_nodes == nodes.size()) {
            LOG(INFO) << "Controller Recieve All PreFinished";
          }
          break;
        }
        default: {
          LOG(WARNING) << "Controller UnknowCommand";
          break;
        }
      }
    }
  }
  void ResponseHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
    LOG(INFO) << "Controller ResponseHanle";
  }

  ps::KVServer<char>* ps_scheduler_;

  // whether to LOG verbose information
  bool log_verbose_;
};

}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_KVSTORE_DIST_CONTROLLER_H_
