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

enum class ControllerCommand {
  kNotifyPreparationFinished = 11,

  kNodeScaleAnounce = 12,

  kTimestampUpdate = 13,
};

static bool isControllerCommand(int head) {
  return head > 10;
}
class NodeScalingInfo {
 public:
  NodeScalingInfo() : is_worker_(false), num_ready_nodes_(0), future_timestamp_(0){};

  ~NodeScalingInfo() = default;

  void Clear() {
    nodes_.clear();
    is_worker_ = false;
    num_ready_nodes_ = 0;
    future_timestamp_ = 0;
  }

  int GetFutureTimestamp() {
    return future_timestamp_;
  }
  const std::vector<int>& GetNodes() const {
    return nodes_;
  }

  const bool IsWorker() const {
    return is_worker_;
  }

  void Decode(const std::string& str) {
    std::istringstream is(str);
    std::string type;
    is >> type;
    is_worker_ = type == "w";
    is >> future_timestamp_;
    int node;
    while (is >> node) {
      nodes_.push_back(node);
    }
  }
  std::string DebugStrNodes(){
    std::string ret;
    for (auto node : nodes_) {
      ret += std::to_string(node);
      ret += ",";
    }
    return ret;
  }

  std::string Encode() const {
    std::string ret;
    // w 10 8,9,10  ==>  worker timestamp=10 nodes=8,9,10
    ret += is_worker_ ? "w " : "s ";
    ret += std::to_string(future_timestamp_);
    ret += " ";
    for (auto node : nodes_) {
      ret += std::to_string(node);
      ret += ",";
    }
    return ret;
  }

 private:
  void RegisterNode(bool is_worker, int node_id) {
    if (!nodes_.empty()) {
      CHECK(is_worker && is_worker_) << "Not Support different type of node scalling";
    }
    nodes_.push_back(node_id);
    is_worker_ = is_worker;
  }

  bool NodeReady(int node_id) {
    if (std::find(nodes_.begin(), nodes_.end(), node_id) != nodes_.end()) {
      num_ready_nodes_++;
      if (num_ready_nodes_ == nodes_.size())
        return true;
    } else {
      LOG(WARNING) << "NodeReady Recieve A unregister Node id: " << node_id;
    }
    return false;
  }
  void setFutureTimestamp(int future_timestamp) {
    this->future_timestamp_ = future_timestamp;
  }

  friend class KVStoreDistController;

  std::vector<int> nodes_;
  bool is_worker_;
  int num_ready_nodes_;
  int future_timestamp_;
};
class KVStoreDistController {
 public:
  KVStoreDistController() {
    using namespace std::placeholders;
    ps_scheduler_ = new ps::KVServer<char>(0);
    static_cast<ps::SimpleApp*>(ps_scheduler_)
        ->set_request_handle(std::bind(&KVStoreDistController::RequestHandle, this, _1, _2));
    static_cast<ps::SimpleApp*>(ps_scheduler_)
        ->set_response_handle(std::bind(&KVStoreDistController::ResponseHandle, this, _1, _2));

    this->timestamp_ = 0;
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
  NodeScalingInfo scaling_nodes_;

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
          scaling_nodes_.RegisterNode(is_worker, node_id);
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
        case ControllerCommand::kNotifyPreparationFinished: {
          
          int sender = recved.sender;
          if (scaling_nodes_.NodeReady(sender)) {
            LOG(INFO) << "All the scaling nodes are ready"; 
            // calculate the future timestamp
            int future_timestamp = calcFutTimestamp();
            scaling_nodes_.setFutureTimestamp(future_timestamp);
            // tell all the node the info
            std::string body = scaling_nodes_.Encode();
            // TODO： remember to hanle the case when scheduler send kNodeScaleAnounce, 
            // but the fut ts not reach, and there is another node join
            SendtoAllNodes(ControllerCommand::kNodeScaleAnounce, body);
          }
          // Response
          app->Response(recved);
          break;
        }
        case ControllerCommand::kTimestampUpdate: {
          this->timestamp_ = std::stoi(body);
          
          if (this->timestamp_ == scaling_nodes_.GetFutureTimestamp()) {
            LOG(INFO) << "Scheduler reach the timestamp " << this->timestamp_
                      << ", start to scale, id: " << scaling_nodes_.DebugStrNodes();
            // start to scale
            ps::Postoffice::Get()->UpdateScaleNodes(scaling_nodes_.GetNodes(), scaling_nodes_.IsWorker());
            scaling_nodes_.Clear();
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

  int calcFutTimestamp() {
    return timestamp_ + 5;
  }

  void SendtoAllNodes(ControllerCommand cmd, const std::string& body) {
    CHECK_NOTNULL(ps_scheduler_);
    int head = static_cast<int>(cmd);
    ps_scheduler_->Wait(ps_scheduler_->Request(head, body, ps::kServerGroup + ps::kWorkerGroup));
  }

  ps::KVServer<char>* ps_scheduler_;

  // whether to LOG verbose information
  bool log_verbose_;

  int timestamp_;
};

}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_KVSTORE_DIST_CONTROLLER_H_
