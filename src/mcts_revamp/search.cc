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
#include <math.h>

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
      cache_(cache),
      //~ syzygy_tb_(syzygy_tb),
      played_history_(tree.GetPositionHistory()),
      network_(network),
      limits_(limits),
      //~ start_time_(std::chrono::steady_clock::now()),
      //~ initial_visits_(root_node_->GetN()),
      //~ best_move_callback_(best_move_callback),
      //~ info_callback_(info_callback),
      kMiniBatchSize(options.Get<int>(kMiniBatchSizeStr)),
      //~ kMaxPrefetchBatch(options.Get<int>(kMaxPrefetchBatchStr)),
      //~ kCpuct(options.Get<float>(kCpuctStr)),
      //~ kTemperature(options.Get<float>(kTemperatureStr)),
      //~ kTemperatureVisitOffset(options.Get<float>(kTemperatureVisitOffsetStr)),
      //~ kTempDecayMoves(options.Get<int>(kTempDecayMovesStr)),
      //~ kNoise(options.Get<bool>(kNoiseStr)),
      //~ kVerboseStats(options.Get<bool>(kVerboseStatsStr)),
      //~ kAggressiveTimePruning(options.Get<float>(kAggressiveTimePruningStr)),
      //~ kFpuReduction(options.Get<float>(kFpuReductionStr)),
      kCacheHistoryLength(options.Get<int>(kCacheHistoryLengthStr))
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
     // worker.RunBlocking();
      worker.RunBlocking2();
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
    
    computation_ = std::make_unique<CachingComputation>(std::move(search_->network_->NewComputation()), search_->cache_);

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
        float p = computation_->GetPVal(j, (node->GetEdges())[k].GetMove().as_nn_index());
        pval[k] = p;
        total += p;
      }
      float scale = total > 0.0f ? 1.0f / total : 0.0f;
      for (int k = 0; k < nedge; k++) {
        (node->GetEdges())[k].SetP(pval[k] * scale);
      }
    }
  }
  int64_t elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count();
  std::cerr << "Elapsed time when thread for node " << worker_root_ << " finished " << i << " nodes and " << ic << " computations: " << elapsed_time << "ms\n";

  delete [] minibatch;
}

  std::vector<float> SearchWorker_revamp::q_to_prob(std::vector<float> Q, int d) {
    float max_q = *max_element(std::begin(Q), std::end(Q));
    std::vector<float> a;
    std::vector<float> b;
    float c = 0;
    std::vector<float> q_prob;
    for(int i = 0; i < Q.size(); i++){
      a[i] = (float)d/max_q  * Q[i];
      b[i] = exp(a[i]);
    }
    std::for_each(b.begin(), b.end(), [&] (float f) {
    c += f;
});
    for(int i = 0; i < Q.size(); i++){
      q_prob[i] = b[i]/c;
    }
    return(q_prob);
  }
  
// computes weights for the children based on average Qs (and possibly Ps) and, if there are unexpanded edges, a weight for the first unexpanded edge (the unexpanded with highest P)
// weights are >= 0, sum of weights is 1
// stored in weights_, idx corresponding to index in EdgeList
// for now, weights are simply normalized Ps
void SearchWorker_revamp::computeWeights(Node_revamp* node) {
  double sum = 0.0;
  int n = node->GetNumChildren() + 1;
  if (n > node->GetNumEdges()) n = node->GetNumEdges();

  int widx = weights_.size();

  for (int i = 0; i < n; i++) {
//    weights_.push_back(1.0);
    weights_.push_back((node->GetEdges())[i].GetP());
    sum += weights_[widx + i];
  }
  if (sum > 0.0) {
    float scale = (float)(1.0 / sum);
    for (int i = 0; i < n; i++) {
      weights_[widx + i] *= scale;
    }
  } else {
    float x = 1.0f / (float)n;
    for (int i = 0; i < n; i++) {
      weights_[widx + i] = x;
    }
  }
}

// returns number of nodes added in sub tree root at current_node
int SearchWorker_revamp::pickNodesToExtend(Node_revamp* current_node, int noof_nodes) {
  
  bool const DEBUG = false;
  
  int orig_noof_nodes = noof_nodes;
  
  int widx = weights_.size();
  computeWeights(current_node);
  int ntot = current_node->GetN() - 1;
  int ntotafter = ntot + noof_nodes;
  int npos = 0;
  double weightpos = 0.0;
  for (int i = 0; i < current_node->GetNumChildren(); i++) {
    int n = (current_node->GetEdges())[i].GetChild()->GetN();
    if ((double)ntotafter * (double)weights_[widx + i] - (double)n > 0.0) {
      npos += n;
      weightpos += (double)weights_[widx + i];
    } else {
      weights_[widx + i] = 0.0;
    }
  }
  
  
  if (DEBUG) {
    std::cerr << "q: " << noof_nodes << ", n: " << current_node->GetN() << ", nedge: " << current_node->GetNumEdges() << ", nchild: " << current_node->GetNumChildren() << "\n";
    for (int i = widx; i < weights_.size(); i++) {
      std::cerr << " " << weights_[i];
    }
    std::cerr << "\n";
    for (int i = 0; i < current_node->GetNumChildren(); i++) {
      std::cerr << " " << (current_node->GetEdges())[i].GetChild()->GetN();
    }
    std::cerr << "\n";
  }
  
  int nnewnodes = 0;
  if (weights_.size() - widx > current_node->GetNumChildren()) {  // there is an unexpanded edge to potentially extend
    int idx = current_node->GetNumChildren();
    double w = (double)weights_[widx + idx];
    int ai = round((double)(npos + noof_nodes) * w / (w + weightpos));
    if (ai >= 1) {

      if (DEBUG) std::cerr << "Creating child\n";

      (current_node->GetEdges())[idx].CreateChild(current_node, idx);
      nnewnodes++;
      Node_revamp* newchild = (current_node->GetEdges())[idx].GetChild();
      history_.Append((current_node->GetEdges())[idx].GetMove());
      newchild->ExtendNode(&history_);
      if (!newchild->IsTerminal()) {
        AddNodeToComputation2(newchild);
        minibatch_.push_back(newchild);

        if (DEBUG) std::cerr << "Adding child to batch\n";

        noof_nodes--;  // could alternatively be noof_nodes -= ai but that would mean more frequent under full batches
      }
      history_.Pop();
    }
    weights_.pop_back();
  }
  
  if (DEBUG) std::cerr << "weights_.size(): " << (weights_.size() - widx) << "\n";
  
  for (int i = 0; i < weights_.size() - widx; i++) {
    double w = (double)weights_[widx + i];
    if (w > 0.0) {
      int n = (current_node->GetEdges())[i].GetChild()->GetN();
      int ai = round((double)(npos + noof_nodes) * w / weightpos - (double)n);

      if (DEBUG) std::cerr << "Child " << i << ", ai: " << ai << "\n";
      
      //~ if (ai < 0) {
        //~ std::cerr << "ai: " << ai << "\n";
      //~ }

      if (ai >= 1) {
        history_.Append((current_node->GetEdges())[i].GetMove());

        if (DEBUG) std::cerr << "rec call\n";

        nnewnodes += pickNodesToExtend((current_node->GetEdges())[i].GetChild(), ai);
        history_.Pop();

        if (DEBUG) std::cerr << "return rec call\n";

        noof_nodes -= ai;  // could alternatively be result of pickNodesToExtend call but this would favor later edges
      }
      npos -= n;
      weightpos -= w;
    }
  }
  
  // noof_nodes unchanged if sub tree is exhausted (node has no edges (terminal) or all unexpanded descendants are terminal)
  // noof_nodes > 0 if not enough nodes were added to children or no children and new child is terminal

  for (int n = weights_.size() - widx; n > 0; n--) {
    weights_.pop_back();
  }

  if (weights_.size() != widx) {
    std::cerr << "weights_.size() != widx\n";
  }

  current_node->IncreaseN(nnewnodes);

  if (nnewnodes > orig_noof_nodes) {
    std::cerr << "new nodes: " << nnewnodes << ", should be: " << orig_noof_nodes << "\n";
  }

  return nnewnodes;
}

void SearchWorker_revamp::retrieveNNResult(Node_revamp* node, int batchidx) {
  node->SetQ(-computation2_->GetQVal(batchidx));  // should it be negated?

  float total = 0.0;
  int nedge = node->GetNumEdges();
  pvals_.clear();
  for (int k = 0; k < nedge; k++) {
    float p = computation2_->GetPVal(batchidx, (node->GetEdges())[k].GetMove().as_nn_index());
    if (p < 0.0) {
      std::cerr << "p value < 0\n";
      p = 0.0;
    }
    pvals_.push_back(p);
    total += p;
  }
  if (total > 0.0f) {
    float scale = 1.0f / total;
    for (int k = 0; k < nedge; k++) {
      (node->GetEdges())[k].SetP(pvals_[k] * scale);
    }
    node->SortEdgesByPValue();
  } else {
    float x = 1.0f / (float)nedge;
    for (int k = 0; k < nedge; k++) {
      (node->GetEdges())[k].SetP(x);
    }
  }
}

void SearchWorker_revamp::RunBlocking2() {
  std::cerr << "Running thread for node " << worker_root_ << "\n";
  const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

  worker_root_->ExtendNode(&history_);
  if (worker_root_->IsTerminal()) {
    std::cerr << "Root " << worker_root_ << " is terminal, nothing to do\n";
    return;
  }
  minibatch_.clear();
  computation2_ = search_->network_->NewComputation();
  AddNodeToComputation2(worker_root_);
  std::cerr << "Computing thread root ..";
  computation2_->ComputeBlocking();
  std::cerr << " done\n";
  retrieveNNResult(worker_root_, 0);

  int lim = search_->limits_.visits;

  int i = 1;

  while (i < lim) {
    minibatch_.clear();
    computation2_ = search_->network_->NewComputation();

    //~ std::cerr << "n: " << worker_root_->GetN() << "\n";

    pickNodesToExtend(worker_root_, search_->kMiniBatchSize);

    //~ std::cerr << "weights_.size(): " << weights_.size() << "\n";
    
    std::cerr << "Computing batch of size " << minibatch_.size() << " ..";

    computation2_->ComputeBlocking();
    
    std::cerr << " done\n";

    i += minibatch_.size();  // == computation2_->GetBatchSize()
    
    for (int j = 0; j < minibatch_.size(); j++) {
      retrieveNNResult(minibatch_[j], j);
    }
  }

  int64_t elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count();
  std::cerr << "Elapsed time when thread for node " << worker_root_ << " finished " << worker_root_->GetN() << " nodes and " << i << " computations: " << elapsed_time << "ms\n";

  std::cerr << "n: " << worker_root_->GetN() << "\n";

  float totp = 0.0;
  for (int i = 0; i < worker_root_->GetNumChildren(); i++) {
    std::cerr << " " << (worker_root_->GetEdges())[i].GetP();
    totp += (worker_root_->GetEdges())[i].GetP();
  }
  std::cerr << "\n";
  for (int i = 0; i < worker_root_->GetNumChildren(); i++) {
    std::cerr << " " << (worker_root_->GetEdges())[i].GetP() / totp;
  }
  std::cerr << "\n";
  for (int i = 0; i < worker_root_->GetNumChildren(); i++) {
    std::cerr << " " << (worker_root_->GetEdges())[i].GetChild()->GetN();
  }
  std::cerr << "\n";
  for (int i = 0; i < worker_root_->GetNumChildren(); i++) {
    std::cerr << " " << (float)(worker_root_->GetEdges())[i].GetChild()->GetN() / (float)(worker_root_->GetN() - 1);
  }
  std::cerr << "\n";
}

void SearchWorker_revamp::AddNodeToComputation(Node_revamp* node) {
 auto hash = history_.HashLast(search_->kCacheHistoryLength + 1);
 auto planes = EncodePositionForNN(history_, 8);
 std::vector<uint16_t> moves;
 int nedge = node->GetNumEdges();
 for (int k = 0; k < nedge; k++) {
   moves.emplace_back(node->GetEdges()[k].GetMove().as_nn_index());
 }
 computation_->AddInput(hash, std::move(planes), std::move(moves));
}

void SearchWorker_revamp::AddNodeToComputation2(Node_revamp* node) {
 // auto hash = history_.HashLast(search_->kCacheHistoryLength + 1);
  auto planes = EncodePositionForNN(history_, 8);
 // std::vector<uint16_t> moves;
 // int nedge = node->GetNumEdges();
 // for (int k = 0; k < nedge; k++) {
 //   moves.emplace_back(node->edges_[k].GetMove().as_nn_index());
 // }
  computation2_->AddInput(std::move(planes));
}

  
}  // namespace lczero
