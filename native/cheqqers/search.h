// Port of ExpectiCheqqersPlayer (src/search/expecti_cheqqers.py:370): iterative-deepening expectimax alpha-beta
// with Star1 chance pruning, transposition table, killers, history, capture extension, quiescence, quantum_plies.
// Single-threaded. Where parallelism could go later: the root moves of one iteration are independent searches
// given a shared (or per-thread) table; a lazy-SMP split over root moves keeps each thread's node order Python's.
#pragma once

#include <chrono>
#include <deque>
#include <string>
#include <unordered_map>

#include "engine.h"

namespace cq {

class ValueNet;  // value_net.h

// Training data from the search's own leaves (cq_selfplay_leaves): each non-terminal leaf evaluated is kept with
// probability `rate` (its network input, side to move, search ply, whether a capture is pending). The sampler has
// its own generator: the search's moves and nodes are unchanged.
struct LeafSample {
    std::vector<double> x;
    int turn;
    int ply;
    bool capture_pending;
    EnvPtr env;  // the leaf position itself, only if LeafSink::keep_state
};
struct LeafSink {
    double rate = 0.0;
    bool keep_state = false;
    Rng rng;
    std::vector<LeafSample> samples;
};

struct SearchOpts {
    int max_depth = 8;
    double time_limit_ms = 1500.0;
    int max_tt_size = 65536;
    int max_capture_extension = 4;
    bool star1 = true;
    bool quiescence = false;
    int quantum_plies = -1;  // -1 = None
    bool use_cache = true;
    Weights weights;
    const ValueNet* net = nullptr;  // not null: leaf values from this network (value_net.h), not from weights
    LeafSink* leaf_sink = nullptr;  // not null: sample leaves into it
};

struct RootScore {
    MoveArr move;
    double value;
};

struct SearchResult {
    int chosen = -1;  // index into the raw legal list
    int64_t nodes = 0;
    int last_depth = 0;
    bool has_score = false;
    double last_score = 0.0;
    int unexpanded = 0;
    double elapsed_ms = 0.0;
    std::vector<RootScore> root_scores;  // the last finished iteration, best first
    std::string error;  // non-empty: an iteration failed (a bug); chosen is the last finished iteration's move
};

class Searcher {
   public:
    explicit Searcher(const SearchOpts& o) : opts(o) {}
    // get_move(env, available_moves=raw) (expecti_cheqqers.py:447). raw == nullptr: the position's legal moves.
    SearchResult get_move(const Env& env, const std::vector<MoveArr>* raw);

    SearchOpts opts;
    int64_t nodes = 0;
    int unexpanded = 0;

   private:
    struct TTEntry {
        int depth;
        int8_t flag;
        bool has_best;
        double value;
        uint64_t best;
    };
    struct Timeout {};

    // The transposition table: an exact map Key -> TTEntry, as the Python dict (open addressing, linear probing;
    // a key matches only if all its codes are equal). Cleared per get_move by a generation counter.
    class Table {
       public:
        void clear() {
            count_ = 0;
            pool_.clear();
            if (++gen_ == 0) {  // wrapped: really empty the slots
                for (Slot& s : slots_) s.gen = 0;
                gen_ = 1;
            }
        }
        size_t size() const { return count_; }
        const TTEntry* find(const Key& k, uint64_t h) const {
            if (slots_.empty()) return nullptr;
            size_t m = slots_.size() - 1;
            for (size_t i = (size_t)h & m;; i = (i + 1) & m) {
                const Slot& s = slots_[i];
                if (s.gen != gen_) return nullptr;
                if (s.h == h && same(s.kidx, k)) return &s.e;
            }
        }
        void put(const Key& k, uint64_t h, const TTEntry& e) {  // tt[key] = e
            if ((count_ + 1) * 2 > slots_.size()) grow();
            size_t m = slots_.size() - 1;
            for (size_t i = (size_t)h & m;; i = (i + 1) & m) {
                Slot& s = slots_[i];
                if (s.gen != gen_) {
                    s.gen = gen_;
                    s.h = h;
                    s.kidx = (uint32_t)pool_.size();
                    pool_.push_back(k.n);
                    pool_.insert(pool_.end(), k.v, k.v + k.n);
                    s.e = e;
                    ++count_;
                    return;
                }
                if (s.h == h && same(s.kidx, k)) {
                    s.e = e;
                    return;
                }
            }
        }

       private:
        struct Slot {
            uint64_t h;
            uint32_t gen;
            uint32_t kidx;
            TTEntry e;
        };
        std::vector<Slot> slots_;
        std::vector<uint16_t> pool_;  // per entry: n, then the n codes
        size_t count_ = 0;
        uint32_t gen_ = 1;
        bool same(uint32_t kidx, const Key& k) const {
            const uint16_t* p = pool_.data() + kidx;
            return p[0] == k.n && std::memcmp(p + 1, k.v, k.n * sizeof(uint16_t)) == 0;
        }
        void grow() {
            std::vector<Slot> old;
            old.swap(slots_);
            slots_.assign(old.empty() ? 1024 : old.size() * 2, Slot{0, 0, 0, TTEntry{}});
            size_t m = slots_.size() - 1;
            for (const Slot& s : old) {
                if (s.gen != gen_) continue;
                size_t i = (size_t)s.h & m;
                while (slots_[i].gen == gen_) i = (i + 1) & m;
                slots_[i] = s;
            }
        }
    };

    std::chrono::steady_clock::time_point deadline;
    Table tt;
    std::vector<std::vector<uint64_t>> killers;
    std::unordered_map<uint64_t, int> history_table;
    int root_color = WHITE;
    // per-ply buffers: along one line of the search each ply has one chance and one decision frame
    std::deque<std::vector<Outcome>> raw_by_ply, outs_by_ply;
    std::deque<std::vector<MoveArr>> legal_by_ply;
    struct Scored {
        double s;
        MoveArr m;
    };
    std::vector<Scored> order_buf;
    template <class T>
    static std::vector<T>& at_ply(std::deque<std::vector<T>>& d, int ply) {
        while ((int)d.size() <= ply) d.emplace_back();
        return d[ply];
    }

    void check_timeout();
    double decision(Env& env, int depth, double alpha, double beta, int ply, int ext);
    double chance(Env& env, const MoveArr& mv, int depth, double alpha, double beta, int ply, int ext);
    double leaf(Env& env, int ply);
    void order(Env& env, std::vector<MoveArr>& moves, int ply, bool has_tt_best, uint64_t tt_best);
    double score_move(Env& env, const MoveArr& mv, int ply, bool has_tt_best, uint64_t tt_best, const double* chances);
    void record_cutoff(Env& env, const MoveArr& mv, int depth, int ply);
};

uint64_t move_key(const MoveArr& m);

}  // namespace cq
