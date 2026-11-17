// Monte Carlo tree search with chance nodes for a policy-value network evaluated outside (track 33,
// docs/specs/cheqqers_gpu_mcts.md). Additive: the alpha-beta search (search.h) does not use any of this.
//
// Tree: decision nodes (a position, its legal moves with the network's priors) and chance nodes (one per move tried:
// the move's measurement outcomes with their probabilities, get_move_outcomes). Values are kept from WHITE's point
// of view throughout (a multi-jump keeps the same side to move, so "negate at every ply" would be wrong).
//   decision node value V = (v_net + sum_a N_a Q_a) / (1 + sum_a N_a)    (Q_a = the value of move a's chance node)
//   chance node value     = sum_k p_k V(child_k)                           (expectation over all outcomes)
// A chance node is created with all its outcome positions; the non-terminal ones go to the network together, so the
// expectation is always over every outcome (no sampling). Selection at a decision node: PUCT with the priors;
// at a chance node: the outcome with the largest p_k / (1 + n_k) (visits in proportion to probability).
//
// Batching: collect() runs simulations until `max_leaves` positions wait for the network, with a virtual loss on
// every edge of every pending path (a pending edge counts as a lost visit for its mover); a move whose outcomes are
// still with the network is skipped by selection (a simulation is abandoned only if all of a node's moves are); it returns the positions'
// encodings (value_net.h encode_position) and legal moves. apply() takes the network's values (side to move's
// expected score, win - loss) and move priors and backs every pending simulation up. collect() may be called again
// before apply() (pipelining); apply() consumes the batches in order.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "engine.h"

namespace cq {

struct MctsOpts {
    double c_puct = 1.5;
    double fpu_reduction = 0.25;  // unvisited move: parent's value for its mover minus this
    double dirichlet_alpha = 0.0;  // > 0: root noise (self-play)
    double dirichlet_frac = 0.25;
    uint64_t seed = 1;
    int64_t max_nodes = 2000000;  // decision nodes; collect() returns 0 leaves beyond it
    int max_moves = 256;          // moves per position passed to the network; any further ones get the smallest prior
    // Gumbel root (Danihelka et al. 2022; recipe docs/specs/cheqqers_rl_recipe.md section 2). gumbel_m > 0: Gumbel-Top-m
    // candidates at the root + Sequential Halving over gumbel_n simulations; interior nodes select deterministically
    // from the completed Q; the search ends by itself (done()) and gumbel() gives the move and the policy target.
    int gumbel_m = 0;
    int gumbel_n = 64;
    double c_visit = 50.0;
    double c_scale = 1.0;
};

struct MctsStats {
    int64_t sims = 0;        // simulations backed up
    int64_t evals = 0;       // positions evaluated by the network
    int64_t terminal_sims = 0;
    int64_t collisions = 0;  // simulations abandoned on a pending path
    int64_t nodes = 0;       // decision nodes
    int max_depth = 0;       // longest path (decision nodes) of a simulation
    int64_t overflow = 0;    // outcome positions valued by the linear evaluation (buffer full)
};

class Mcts {
   public:
    // raw: the moves allowed at the root (empty: all legal moves); the root keeps the position's legal order.
    Mcts(const Env& root, const std::vector<MoveArr>& raw, const MctsOpts& o);

    // Encodings (x: rows of in_dim floats), moves (mv: rows of max_moves x 3 squares, lens: max_moves per row, 0 =
    // none) of up to max_leaves positions. Returns the count; *mmax = the largest move count among them.
    // cap = rows the buffers hold (>= max_leaves): the outcomes of one move may go past max_leaves.
    int collect(int max_leaves, int cap, float* x, int32_t* mv, int32_t* lens, int* mmax);
    // values[i]: side to move's expected score in [-1, 1]; priors: rows of max_moves (stride), any scale (they are
    // normalised over the position's moves). n must equal the oldest pending collect's count.
    void apply(const float* values, const float* priors, int stride, int n);
    int pending_batches() const { return (int)batches_.size(); }

    struct RootInfo {
        MoveArr move;
        int visits;
        double q;  // white's view; nan if never visited
        double prior;
    };
    std::vector<RootInfo> root_info() const;
    double root_value() const;  // white's view
    double root_vnet() const;   // the network's value of the root, white's view
    int root_turn() const;
    // Gumbel mode: the search has used its budget (or the root has a single candidate)
    bool done() const { return gdone_; }
    struct GumbelOut {
        int chosen = -1;                   // index into the root's moves: the Sequential Halving winner
        std::vector<double> policy;        // softmax(logits + sigma(completed Q)) over the root's moves
        std::vector<double> gumbel;        // the root's Gumbel draws (for tests)
        std::vector<int> candidates;       // the candidates left
    };
    GumbelOut gumbel() const;
    MctsStats stats;
    int in_dim() const { return in_dim_; }
    int max_moves() const { return opts_.max_moves; }

   private:
    struct Edge {
        MoveArr m;
        float prior = 0.f;
        int n = 0;
        int vl = 0;
        double q = 0.0;  // white's view, valid if n > 0
        int chance = -1;
    };
    struct DNode {
        EnvPtr env;
        int8_t turn = WHITE;
        int8_t status = 0;  // 0 new, 1 waiting for the network, 2 evaluated
        bool terminal = false;
        double v_net = 0.0;  // white's view
        double V = 0.0;
        double sumNQ = 0.0;
        int nsum = 0;
        std::vector<Edge> edges;
    };
    struct CNode {
        std::vector<int> child;
        std::vector<double> p;
        std::vector<int> n, vl;
        int waiting = 0;  // children still waiting for the network
        double V = 0.0;
    };
    struct Step {
        int d, a, c, k;  // decision node, edge, chance node, outcome taken below it (-1: the chance node expanded)
    };
    struct Batch {
        std::vector<int> nodes;  // decision nodes evaluated by this batch, in output order
        std::vector<int> sim_off;  // per pending simulation: offset into steps (the last + 1 = steps.size())
        std::vector<Step> steps;
        bool root_eval = false;
    };

    MctsOpts opts_;
    int in_dim_;
    std::vector<DNode> dn_;
    std::vector<CNode> cn_;
    std::vector<Batch> batches_;  // pending, oldest first
    std::mt19937_64 rng_;
    std::vector<Step> path_;
    std::vector<MoveArr> root_raw_;

    int new_dnode(EnvPtr env);
    void set_moves(int d);
    void encode_into(int d, float* x, int32_t* mv, int32_t* lens, int* mmax, int row);
    int select_edge(int d) const;
    double chance_value(const CNode& c) const;
    void backup(const std::vector<Step>& steps, size_t from, size_t to);
    void undo_virtual(const std::vector<Step>& steps, size_t from, size_t to);
    void add_root_noise();

    // Gumbel
    std::vector<double> g_;          // per root edge
    std::vector<int> cand_;          // root candidates still in the halving
    std::vector<int> phase_start_;   // per root edge: its visits when the phase began
    int quota_ = 0;                  // simulations per candidate in this phase
    int per_phase_ = 0;
    bool gdone_ = false;
    void gumbel_init();
    int gumbel_root_pick();
    // completed Q of node d's moves, min-max normalised and scaled: sigma(q_hat) (side to move's view)
    void sigma_completed(int d, std::vector<double>& sigma, std::vector<double>& logit) const;
    int gumbel_interior(int d) const;
};

}  // namespace cq
