/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "mcts_revamp/search.h"

#include <iostream>
#include <fstream>

#include "neural/encoder.h"

namespace lczero {

namespace {

  std::chrono::steady_clock::time_point start_comp_time_;
  std::chrono::steady_clock::time_point stop_comp_time_;

  const char* LOGFILENAME = "lc0.log";
  std::ofstream LOGFILE;

}  // namespace

const char* Search_revamp::kMiniBatchSizeStr = "Minibatch size for NN inference";
const char* Search_revamp::kMaxPrefetchBatchStr = "Max prefetch nodes, per NN call";
const char* Search_revamp::kCpuctStr = "Cpuct MCTS option";
const char* Search_revamp::kTemperatureStr = "Initial temperature";
const char* Search_revamp::kTempDecayMovesStr = "Moves with temperature decay";
const char* Search_revamp::kTemperatureVisitOffsetStr = "Temperature visit offset";
const char* Search_revamp::kNoiseStr = "Add Dirichlet noise at root node";
const char* Search_revamp::kVerboseStatsStr = "Display verbose move stats";
const char* Search_revamp::kAggressiveTimePruningStr =
    "Aversion to search if change unlikely";
const char* Search_revamp::kFpuReductionStr = "First Play Urgency Reduction";
const char* Search_revamp::kCacheHistoryLengthStr =
    "Length of history to include in cache";
const char* Search_revamp::kPolicySoftmaxTempStr = "Policy softmax temperature";
const char* Search_revamp::kAllowedNodeCollisionsStr =
    "Allowed node collisions, per batch";
const char* Search_revamp::kOutOfOrderEvalStr = "Out-of-order cache backpropagation";
const char* Search_revamp::kMultiPvStr = "MultiPV";




void Search_revamp::PopulateUciParams(OptionsParser* options) {
  // Here the "safe defaults" are listed.
  // Many of them are overridden with optimized defaults in engine.cc and
  // tournament.cc

  options->Add<IntOption>(kMiniBatchSizeStr, 1, 1024, "minibatch-size") = 1;
  options->Add<IntOption>(kMaxPrefetchBatchStr, 0, 1024, "max-prefetch") = 32;
  options->Add<FloatOption>(kCpuctStr, 0.0f, 100.0f, "cpuct") = 1.2f;
  options->Add<FloatOption>(kTemperatureStr, 0.0f, 100.0f, "temperature") =
      0.0f;
  options->Add<FloatOption>(kTemperatureVisitOffsetStr, -0.99999f, 1000.0f,
                            "temp-visit-offset") = 0.0f;
  options->Add<IntOption>(kTempDecayMovesStr, 0, 100, "tempdecay-moves") = 0;
  options->Add<BoolOption>(kNoiseStr, "noise", 'n') = false;
  options->Add<BoolOption>(kVerboseStatsStr, "verbose-move-stats") = false;
  options->Add<FloatOption>(kAggressiveTimePruningStr, 0.0f, 10.0f,
                            "futile-search-aversion") = 1.33f;
  options->Add<FloatOption>(kFpuReductionStr, -100.0f, 100.0f,
                            "fpu-reduction") = 0.0f;
  options->Add<IntOption>(kCacheHistoryLengthStr, 0, 7,
                          "cache-history-length") = 7;
  options->Add<FloatOption>(kPolicySoftmaxTempStr, 0.1f, 10.0f,
                            "policy-softmax-temp") = 1.0f;
  options->Add<IntOption>(kAllowedNodeCollisionsStr, 0, 1024,
                          "allowed-node-collisions") = 0;
  options->Add<BoolOption>(kOutOfOrderEvalStr, "out-of-order-eval") = false;
  options->Add<IntOption>(kMultiPvStr, 1, 500, "multipv") = 1;
}

Search_revamp::Search_revamp(const NodeTree_revamp& tree, Network* network,
               BestMoveInfo::Callback best_move_callback,
               ThinkingInfo::Callback info_callback, const SearchLimits_revamp& limits,
               const OptionsDict& options, NNCache* cache,
               SyzygyTablebase* syzygy_tb)
    :
      root_node_(tree.GetCurrentHead()),
      //~ cache_(cache),
      //~ syzygy_tb_(syzygy_tb),
      played_history_(tree.GetPositionHistory()),
      network_(network),
      limits_(limits),
      //~ start_time_(std::chrono::steady_clock::now()),
      //~ initial_visits_(root_node_->GetN()),
      //~ best_move_callback_(best_move_callback),
      //~ info_callback_(info_callback),
      kMiniBatchSize(options.Get<int>(kMiniBatchSizeStr))
      //~ kMaxPrefetchBatch(options.Get<int>(kMaxPrefetchBatchStr)),
      //~ kCpuct(options.Get<float>(kCpuctStr)),
      //~ kTemperature(options.Get<float>(kTemperatureStr)),
      //~ kTemperatureVisitOffset(options.Get<float>(kTemperatureVisitOffsetStr)),
      //~ kTempDecayMoves(options.Get<int>(kTempDecayMovesStr)),
      //~ kNoise(options.Get<bool>(kNoiseStr)),
      //~ kVerboseStats(options.Get<bool>(kVerboseStatsStr)),
      //~ kAggressiveTimePruning(options.Get<float>(kAggressiveTimePruningStr)),
      //~ kFpuReduction(options.Get<float>(kFpuReductionStr)),
      //~ kCacheHistoryLength(options.Get<int>(kCacheHistoryLengthStr))
      //~ kPolicySoftmaxTemp(options.Get<float>(kPolicySoftmaxTempStr)),
      //~ kAllowedNodeCollisions(options.Get<int>(kAllowedNodeCollisionsStr)),
      //~ kOutOfOrderEval(options.Get<bool>(kOutOfOrderEvalStr)),
      //~ kMultiPv(options.Get<int>(kMultiPvStr))
    {}


void Search_revamp::StartThreads(size_t how_many) {
  if (root_node_->GetNumEdges() > 0) {
    std::cerr << "Tree not empty, doing nothing\n";
    return;
  }

  std::cerr << "Letting " << how_many << " threads create " << limits_.visits << " nodes each\n";

  LOGFILE.open(LOGFILENAME);

  // create enough leaves so that each thread gets its own subtree
  int nleaf = 1;
  Node_revamp* current_node = root_node_;
  while (nleaf < (int)how_many) {
    current_node->ExtendNode(&played_history_);
    int nedges = current_node->GetNumEdges();
    if (nedges > 2) nedges = 2;
    nleaf += nedges - 1;
    current_node = current_node->GetNextLeaf(root_node_, &played_history_);
  }

  Mutex::Lock lock(threads_mutex_);
  for (int i = 0; i < (int)how_many; i++) {
    threads_.emplace_back([this, current_node]()
     {
      SearchWorker_revamp worker(this, current_node);
      worker.RunBlocking();
     }
    );
    if (i < (int)how_many - 1) {
      current_node = current_node->GetNextLeaf(root_node_, &played_history_);
    }
  }
}

//~ bool Search_revamp::IsSearchActive() const {
  //~ Mutex::Lock lock(counters_mutex_);
  //~ return !stop_;
//~ }

void Search_revamp::WatchdogThread() {
  //~ SearchWorker_revamp worker(this, root_node_);
  //~ worker.RunBlocking();

  //~ while (IsSearchActive()) {
    //~ {
      //~ using namespace std::chrono_literals;
      //~ constexpr auto kMaxWaitTime = 100ms;
      //~ constexpr auto kMinWaitTime = 1ms;
      //~ Mutex::Lock lock(counters_mutex_);
      //~ auto remaining_time = limits_.time_ms >= 0
                                //~ ? (limits_.time_ms - GetTimeSinceStart()) * 1ms
                                //~ : kMaxWaitTime;
      //~ if (remaining_time > kMaxWaitTime) remaining_time = kMaxWaitTime;
      //~ if (remaining_time < kMinWaitTime) remaining_time = kMinWaitTime;
      //~ // There is no real need to have max wait time, and sometimes it's fine
      //~ // to wait without timeout at all (e.g. in `go nodes` mode), but we
      //~ // still limit wait time for exotic cases like when pc goes to sleep
      //~ // mode during thinking.
      //~ // Minimum wait time is there to prevent busy wait and other thread
      //~ // starvation.
      //~ watchdog_cv_.wait_for(lock.get_raw(), remaining_time,
                            //~ [this]()
                                //~ NO_THREAD_SAFETY_ANALYSIS { return stop_; });
    //~ }
    //~ MaybeTriggerStop();
  //~ }
  //~ MaybeTriggerStop();
}


/*
void Search_revamp::RunBlocking(size_t threads) {
}

bool Search_revamp::IsSearchActive() const {
	return false;
}
*/

void Search_revamp::Stop() {
}

/*
void Search_revamp::Abort() {
}
*/

void Search_revamp::Wait() {
  Mutex::Lock lock(threads_mutex_);
  while (!threads_.empty()) {
    threads_.back().join();
    threads_.pop_back();
  }

  LOGFILE.close();
}

/*
float Search_revamp::GetBestEval() const {
	return 1.0;
}

std::pair<Move, Move> Search_revamp::GetBestMove() const {
	return {Move("d2d4", false), NULL};
}

*/


Search_revamp::~Search_revamp() {
//  Abort();
  Wait();
}


//////////////////////////////////////////////////////////////////////////////
// SearchWorker
//////////////////////////////////////////////////////////////////////////////


void SearchWorker_revamp::RunBlocking() {
  std::cerr << "Running thread for node " << worker_root_ << "\n";

  Node_revamp *current_node = worker_root_;
  int lim = search_->limits_.visits;

  const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
  int i = 0, ic = 0;
  Node_revamp** minibatch = new Node_revamp *[search_->kMiniBatchSize];
  int const MAXNEDGE = 100;
  float pval[MAXNEDGE];
  while (i < lim) {
    computation_ = search_->network_->NewComputation();

    for (int j = 0; j < search_->kMiniBatchSize;) {
      //~ if (current_node_ == nullptr) {
        //~ std::cerr << "current_node_ is null\n";
      //~ }
      current_node->ExtendNode(&history_);
      if (!current_node->IsTerminal()) {
        AddNodeToComputation(current_node);
        minibatch[j] = current_node;
        j++;
      }
      current_node = current_node->GetNextLeaf(worker_root_, &history_);
      i++;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(73)); // optimised for 1060
    LOGFILE << "RunNNComputation START ";
    start_comp_time_ = std::chrono::steady_clock::now();

    computation_->ComputeBlocking();
    ic += search_->kMiniBatchSize;

    stop_comp_time_ = std::chrono::steady_clock::now();
    auto duration = stop_comp_time_ - start_comp_time_;
    LOGFILE << "RunNNComputation STOP nanoseconds used: " << duration.count() << "; ";
    int idx_in_computation = search_->kMiniBatchSize;
    int duration_mu = duration.count();
    if(duration_mu > 0){
      float better_duration = duration_mu / 1000;
      float nps = 1000 * idx_in_computation / better_duration;
      LOGFILE << " nodes in last batch that were evaluated " << idx_in_computation << " nps " << 1000 * nps << "\n";
    }

    for (int j = 0; j < search_->kMiniBatchSize; j++) {
      Node_revamp* node = minibatch[j];
      node->SetQ(-computation_->GetQVal(j));  // should it be negated?
      float total = 0.0;
      int nedge = node->GetNumEdges();
      if (nedge > MAXNEDGE) {
        std::cerr << "Too many edges\n";
        nedge = MAXNEDGE;
      }
      for (int k = 0; k < nedge; k++) {
        float p = computation_->GetPVal(j, node->edges_[k].GetMove().as_nn_index());
        pval[k] = p;
        total += p;
      }
      float scale = total > 0.0f ? 1.0f / total : 0.0f;
      for (int k = 0; k < nedge; k++) {
        node->edges_[k].SetP(pval[k] * scale);
      }
    }
  }
  int64_t elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count();
  std::cerr << "Elapsed time when thread for node " << worker_root_ << " finished " << i << " nodes and " << ic << " computations: " << elapsed_time << "ms\n";

  delete [] minibatch;
}

void SearchWorker_revamp::AddNodeToComputation(Node_revamp* node) {
//  auto hash = history_.HashLast(search_->kCacheHistoryLength + 1);
  auto planes = EncodePositionForNN(history_, 8);
//  std::vector<uint16_t> moves;
//  int nedge = node->GetNumEdges();
//  for (int k = 0; k < nedge; k++) {
//    moves.emplace_back(node->edges_[k].GetMove().as_nn_index());
//  }
//  computation_->AddInput(hash, std::move(planes), std::move(moves));
  computation_->AddInput(std::move(planes));
}

}  // namespace lczero
