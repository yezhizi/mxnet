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

#include "./kvstore_dist_commons.h"
#include "../profiler/profiler.h"
#include "../operator/tensor/elemwise_binary_op-inl.h"
#include "../operator/tensor/init_op.h"

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
            LOG(INFO) << "All the scaling nodes are ready!";
            // calculate the future timestamp
            int future_timestamp = calcFutTimestamp();
            scaling_nodes_.setFutureTimestamp(future_timestamp);
            // tell all the node the info
            std::string body = scaling_nodes_.Encode();

            // TODO： remember to hanle the case when scheduler send kNodeScaleAnounce,
            // but the fut ts not reach, and there is another node join

            // let front thread exec it, so that it will not block the backend thread;
            SendtoAllNodes(ControllerCommand::kNodeScaleAnounce, body);
          }
          // Response
          app->Response(recved);
          break;
        }
        case ControllerCommand::kTimestampUpdate: {
          this->timestamp_ = std::stoi(body);

          if (this->timestamp_ == scaling_nodes_.GetFutureTimestamp()) {
            ProcessOnScaleClock();
          }
          break;
        }
        case ControllerCommand::kNotifyExit: {
          int sender = recved.sender;
          scaling_nodes_.NodeOut(sender, true);
          LOG(INFO) << "Node " << sender << " is going to exit";
          int expect = std::stoi(body);
          int timestamp = calcFutTimestamp(true, expect);
          scaling_nodes_.setFutureTimestamp(timestamp);
          SendtoAllNodes(ControllerCommand::kNodeScaleAnounce, scaling_nodes_.Encode());
          // Response
          app->Response(recved);
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

  int calcFutTimestamp(bool is_scale_in_ = false, int expect = 0) {
    if (is_scale_in_) {
      CHECK_GT(expect, 0);
      return expect + timestamp_;
    } else {
      return timestamp_ + 1;
    }
  }

  void SendtoAllNodes(ControllerCommand cmd, const std::string& body) {
    CHECK_NOTNULL(ps_scheduler_);
    int head = static_cast<int>(cmd);
    // do not wait! becase there is only one thread
    // wait will block reciving response ,which results in dead lock
    ps_scheduler_->Request(head, body, ps::kServerGroup + ps::kWorkerGroup);
  }

  void ProcessOnScaleClock() {
    LOG(INFO) << "Scheduler reach the timestamp " << this->timestamp_
              << ", start to scale, id: " << scaling_nodes_.DebugStrNodes();
    // start to scale
    bool is_worker = scaling_nodes_.IsWorker();
    bool is_scale_out = scaling_nodes_.isScaleOut();
    if(is_scale_out){
      //scale out
      ps::Postoffice::Get()->UpdateScaleNodes(scaling_nodes_.GetNodes(), is_worker);
    }else{
      //scale in
      //TODO: 没有处理当已经有scale node的情况下，有节点加入/退出的情况。
      // 现在只是简单广播时间戳，所有的节点仅在postoffice层更新。等待退出节点在van层发送delmsg消息后，scheduler广播del消息才更新van层。
      // 系统只支持处理完一个节点的加入和退出后，才能处理下一个节点的加入退出。一个节点的scale的时间为s级别，节点scale发生的时间间隔为min级别
      // 至少从现有的论文来看是这样的。
      ps::Postoffice::Get()->ScaleIn(scaling_nodes_.GetNodes(), is_worker);
    }
    scaling_nodes_.Clear();
  }

  ps::KVServer<char>* ps_scheduler_;

  // whether to LOG verbose information
  bool log_verbose_;

  int timestamp_;
};

}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_KVSTORE_DIST_CONTROLLER_H_
