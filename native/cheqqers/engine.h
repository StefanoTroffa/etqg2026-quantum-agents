// Cheqqers rules engine, a line-by-line port of src/env/cheqqers_official.py (rules of 11 November 2026).
//
// Identity, not approximation, is the contract: every floating-point operation is made in the order the Python
// makes it (dict insertion order, CPython's compensated float sum(), CPython's small-int set iteration order,
// complex products as CPython's _Py_c_prod). Build without -ffast-math and with -ffp-contract=off.
// The Python line a function mirrors is named in its comment as cheqqers_official.py:<line>.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <stdexcept>
#include <vector>

namespace cq {

constexpr int MAXSQ = 50;      // 10x10 board
constexpr int MAXIDS = 16;     // piece ids carried by one square (union of merged pieces)
constexpr int MAXJUMPED = 64;  // ids jumped in one capture sequence
constexpr int32_t NAN_ID = INT32_MIN;  // math.nan piece id (all NaN ids are the same Python object: `in` is true)

enum Color : int8_t { WHITE = 0, BLACK = 1 };
enum GameStateE : int8_t { IN_PROGRESS = 0, WHITE_WON = 1, BLACK_WON = 2, DRAW = 3 };
enum GameType : int8_t { CLASSIC = 0, SUPERPOSITION = 1, ENTANGLEMENT = 2, INTERFERENCE = 3 };

struct QuantumLimit {};                       // RuntimeError("QuantumLimitExceeded ...")
struct Unsupported : std::runtime_error {     // a capacity of this port exceeded: the caller falls back to Python
    using std::runtime_error::runtime_error;
};

struct Cx {
    double re, im;
};
// CPython 3.13 complex arithmetic (Objects/complexobject.c _Py_c_prod / _Py_c_sum); complex*float promotes the float
// to complex(x, 0.0) before 3.14.
static inline Cx cmul(Cx a, Cx b) { return Cx{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
static inline Cx cadd(Cx a, Cx b) { return Cx{a.re + b.re, a.im + b.im}; }
constexpr double INV_SQRT2_JS = 0.70710678;
constexpr double CLEAN_EPSILON = 1e-10;
static inline bool keep_amp(Cx v) { return std::fabs(v.re) >= CLEAN_EPSILON || std::fabs(v.im) >= CLEAN_EPSILON; }

struct Amp {
    uint32_t k;
    Cx v;
};
using AmpVec = std::vector<Amp>;

struct System {  // JointSystem: entity i is bit (n - 1 - i); entity lists may hold a square twice or stale
    std::vector<int8_t> ents;
    AmpVec amps;
    // get_target_probabilities of every entity index j, from one pass over amps (the same additions in the same
    // order as the per-entity pass); valid until ents or amps change (every such site resets it)
    mutable bool sums_valid = false;
    mutable uint32_t or_k = 0, or_notk = 0;  // OR of the keys / of their complements: has-1 / has-0 per bit
    mutable double s0[16], s1[16];
    void compute_sums() const;
    void touch() { sums_valid = false; }
};

struct SqAmps {  // an independent square's amplitudes dict ({0: .., 1: ..} in insertion order, 0..2 entries)
    uint8_t n = 0;
    uint8_t k[2] = {0, 0};
    Cx v[2] = {{0, 0}, {0, 0}};
    void set1(uint8_t key, Cx val) {
        n = 1;
        k[0] = key;
        v[0] = val;
    }
};

struct Square {
    int16_t sys = -1;  // index into Env::systems, -1 = independent
    SqAmps amps;
};

struct Ids {
    uint8_t n = 0;
    int32_t v[MAXJUMPED];
    bool contains(int32_t x) const {
        for (int i = 0; i < n; ++i)
            if (v[i] == x) return true;
        return false;
    }
    bool has_nan() const { return contains(NAN_ID); }
};

struct Props {
    bool present = false;
    int8_t color = WHITE;
    bool crowned = false;
    uint8_t nids = 0;
    int32_t ids[MAXIDS];
};

struct Move {
    int8_t kind = 0;  // 0 classical/take {from a, to b}, 1 split {from a, to1 b, to2 c}, 2 merge {from1 a, from2 b, to c}
    bool take = false;
    int16_t a = 0, b = 0, c = 0;
    int len() const { return kind == 0 ? 2 : 3; }
};
using MoveList = std::vector<Move>;

struct MoveArr {  // tournament array [a, b] or [a, b, c]
    int8_t n = 0;
    int16_t v[3] = {0, 0, 0};
    bool operator==(const MoveArr& o) const {
        if (n != o.n) return false;
        for (int i = 0; i < n; ++i)
            if (v[i] != o.v[i]) return false;
        return true;
    }
};
MoveArr to_array(const Move& m);

// The board hash string of getBoardStateHash (cheqqers_official.py:1567) as a code per square; two keys are equal
// exactly when the two strings are. v[0] = turn, v[1 + i] = square i, (search table only) v[N + 1] = multi-jump + 1.
struct Key {
    uint8_t n = 0;
    uint16_t v[MAXSQ + 2];
    bool operator==(const Key& o) const { return n == o.n && std::memcmp(v, o.v, n * sizeof(uint16_t)) == 0; }
    uint64_t hash() const {
        uint64_t h = 1469598103934665603ULL ^ n;
        for (int i = 0; i < n; ++i) {
            h ^= v[i];
            h *= 1099511628211ULL;
        }
        h ^= h >> 29;
        return h;
    }
};
struct KeyHash {
    size_t operator()(const Key& k) const { return (size_t)k.hash(); }
};

struct HistEntry {
    Key key;
    int32_t count;
};

struct Topo {
    int bs = 0, N = 0, spr = 0;
    int col[MAXSQ], row[MAXSQ];
    int idx_at[10][10];  // [col][row], -1 off the dark squares
    int nray[MAXSQ][4];
    int8_t ray[MAXSQ][4][10];
    uint64_t adj[MAXSQ][3];  // first squares of the rays a piece moves along: [0] white man, [1] black man, [2] king
};
const Topo* get_topo(int board_size);

constexpr int NFEAT = 17;
struct Weights {  // EvalWeights (cheqqers_official.py:120)
    double w[NFEAT] = {1.00, 2.60, 0.20, 0.25, 0.18, 0.22, 0.10, 0.14, 0.12,
                       0.55, 0.45, 0.20, 0.08, 0.0,  0.0,  0.0,  0.0};
    double scale = 25.0;
    double logit_per_man = 0.0;
};

struct Rng {  // only for measurements without a forced outcome (self-play sampling uses get_move_outcomes)
    uint64_t s = 0x9E3779B97F4A7C15ULL;
    double next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return (double)(s >> 11) * (1.0 / 9007199254740992.0);
    }
};

struct SimState;
struct Outcome;
class Env;
// Environments are recycled through a per-thread pool: a child position reuses an old one's vectors.
struct EnvRecycler {
    void operator()(Env* e) const;
};
using EnvPtr = std::unique_ptr<Env, EnvRecycler>;
EnvPtr make_env();  // a pooled Env whose fields the caller sets (clone_into or init)

class Env {
   public:
    const Topo* topo = nullptr;
    int bs = 6, N = 18, start_rows = 2;
    int8_t game_type = ENTANGLEMENT;
    int8_t turn = WHITE;
    int32_t next_piece_id = 1;
    Ids jumped;
    int32_t moves_since_take = 0;
    int16_t mj = -1;  // multi_jump_square, -1 = None
    bool just_crowned = false;
    std::vector<HistEntry> history;  // state_history, insertion-ordered
    int32_t consecutive_queen_moves = 0;
    bool is_calculating_depth = false;
    bool disable_depth_calculation = false;
    bool force_all_moves = false;
    int8_t state = IN_PROGRESS;
    bool has_override = false;
    std::vector<MoveArr> override_moves;
    bool cache_enabled = false;

    Square sq[MAXSQ];
    Props props[MAXSQ];
    std::vector<System> systems;  // pool; systems no square points to are garbage until the next clone

    // per-position caches (cheqqers_official.py:790-796)
    bool probs_valid = false;
    double P0[MAXSQ], P1[MAXSQ];
    bool moves_valid = false;
    bool moves_flags[2] = {false, false};
    MoveList moves_cache;
    MoveList moves_tmp;
    int8_t capture_cache = -1;
    // the captures of each colour from every square, computed once per position (cache_enabled only) and filtered
    // for each caller: find_moves(true, restrict, ...) lists, square by square, the same moves as these scans
    bool takes_valid = false;  // false: both colours' lists are stale
    bool takes_col_valid[2] = {false, false};
    MoveList takes_col[2];
    MoveList takes_all;  // both colours merged by square (find_moves(true, -1, ignore_turn=true))
    bool takes_all_valid = false;
    uint64_t pc_mask[2] = {0, 0};  // squares find_moves expands per colour (scan_prep)
    uint64_t ech_mask = 0;          // squares with P0 > 0.001
    // board_key of this position (cache_enabled only): made by check_win_states, read again by the search's
    // state_key; dropped with the move caches (props or probabilities changed) and checked against turn
    bool bkey_valid = false;
    Key bkey;

    // scratch lists, kept with the object so a pooled Env keeps their capacity
    MoveList b_take, b_std, b_sm, b_tmp, b_split, b_merge, b_cd;
    std::vector<uint32_t> b_seen_s, b_seen_m;

    Env() = default;
    Env(int board_size, int start_rows_, int game_type_);
    void init(int board_size, int start_rows_, int game_type_);
    void reset();
    EnvPtr clone() const;
    void clone_into(Env& out) const;

    // probabilities
    void probs_direct(int i, double& p0, double& p1) const;  // square.get_probabilities() via its own system
    double occ_direct(int i) const;
    double empty_direct(int i) const;
    void square_probs();  // _square_probs / _occ_probs into P0/P1 (cache-aware)
    void invalidate_caches() {
        probs_valid = false;
        moves_valid = false;
        capture_cache = -1;
        takes_valid = false;
        bkey_valid = false;
    }
    void invalidate_moves() {
        moves_valid = false;
        capture_cache = -1;
        takes_valid = false;
        bkey_valid = false;
    }

    // moves
    // first_only: stop at the first move found (only the emptiness of the list is read)
    // only_color >= 0: as if turn were only_color (ignore_turn false)
    void find_moves(bool is_take, int restrict, bool ignore_turn, MoveList& out, bool first_only = false,
                    int only_color = -1);
    bool has_any_move_skip_depth();
    bool piece_moves(int i, bool is_take, bool has_nan_jumped, MoveList& out, bool first_only);
    void find_moves_masked(bool is_take, int color, MoveList& out, bool first_only);
    void scan_prep();
    bool any_plain_move();
    const MoveList& takes_of(int color);  // one colour's captures (cache_enabled only)
    const MoveList& all_takes();          // both colours' (cache_enabled only)
    // find_moves(true, restrict, false, out) from takes_all
    void takes_for_turn(int restrict, MoveList& out);
    bool any_take_for_turn(int restrict);
    bool can_coexist(int i, int j) const;
    void compute_possible_moves(bool skip_depth, bool ignore_turn, bool ignore_forced, MoveList& out);
    const MoveList& possible_moves();  // get_possible_moves() default call (cached when cache_enabled)
    void possible_moves_skip_depth(MoveList& out) { compute_possible_moves(true, false, false, out); }
    bool capture_pending();
    void legal_moves(std::vector<MoveArr>& out);  // get_legal_moves()
    Move resolve(const MoveArr& arr);             // resolve_move_dict(list)
    void find_split_and_merge(const MoveList& classical, MoveList& out);
    void sim_state(SimState& st);
    int calc_max_capture_depth(const SimState& st, const Move& m) const;

    // hashing
    void board_key(Key& k);
    void state_key(Key& k);

    // move application
    int apply_move(const Move& m, const int* forced, int nforced, bool ignore_turn, Rng* rng);
    void move_outcomes(const Move& m, std::vector<Outcome>& out) const;
    bool terminal() const { return state != IN_PROGRESS; }
    int winner() const;  // 1 white, -1 black, 0 draw, 2 in progress

    // evaluation
    void eval_features(int persp, double* f);
    double evaluate(int persp, const Weights& w);

    // internals (public for the C API / tests)
    void involved(const Move& m, int* sqs, int8_t* roles, int& n);
    bool blocks_after(int i, int role) const;
    void finalize(const Move& m, bool failed, bool was_in_mj, bool was_crowned_before, bool was_take, bool ignore_turn);
    int measure_square(int i, int forced, Rng* rng);
    void disentangle();
    void factor_out(int s, int idx, int state);
    void cleanup_props();
    void move_props(int f, int t);
    void classical_override(int f, const int* taken, int ntaken, int t);
    void apply_classical(const Move& m);
    bool classical_set(const int* sqs, const uint8_t* st, int n) const;
    bool apply_take(const Move& m);
    void apply_split(const Move& m);
    void apply_merge(const Move& m);
    void phase_pass();
    void check_win_states();

    // quantum engine
    int create_system(const int* ents, int n);
    int merge_systems(const int* inv, int n);
    void interact_many(const int* ents, int n, int op);
    void conditional_capture(int attacker, int victim, int target);
    void conditional_phase(int s);
    void apply_to_subspace(int s, const int* targets, int m, int op);
    int sys_index_of(int s, int e) const;
    void target_probs(int s, int e, double& p0, double& p1, int& first_key, bool& has0, bool& has1) const;
    int measure_target(int s, int e, int forced, Rng* rng);
    double joint_probability_both_occupied(int s, int a, int b) const;
};

// Stable sort, largest key first: Python's sorted(..., key, reverse=True). Insertion sort: the lists are short.
template <class T, class K>
inline void stable_sort_desc(T* a, size_t n, K key) {
    for (size_t i = 1; i < n; ++i) {
        if (!(key(a[i - 1]) < key(a[i]))) continue;
        T x = std::move(a[i]);
        size_t j = i;
        while (j > 0 && key(a[j - 1]) < key(x)) {
            a[j] = std::move(a[j - 1]);
            --j;
        }
        a[j] = std::move(x);
    }
}

struct Outcome {
    double p;
    EnvPtr env;
};

struct SimState {
    int bs, turn, next_pid, mj;
    Ids jumped;
    Props pm[MAXSQ];
};

// CPython 3.12+ builtin sum() over floats with an int start of 0 (Python/bltinmodule.c, Neumaier compensation)
struct PySum {
    bool any = false;
    double f = 0.0, c = 0.0;
    void add(double x) {
        if (!any) {
            any = true;
            f = 0.0 + x;
            return;
        }
        double t = f + x;
        if (std::fabs(f) >= std::fabs(x))
            c += (f - t) + x;
        else
            c += (x - t) + f;
        f = t;
    }
    double result() const {
        if (!any) return 0.0;
        if (c != 0.0 && std::isfinite(c)) return f + c;
        return f;
    }
};

// CPython set of small non-negative ints: iteration order (Objects/setobject.c, 3.13)
struct PySetSim {
    size_t mask = 7, fill = 0, used = 0;
    int tab[256];
    PySetSim() {
        for (int i = 0; i < 8; ++i) tab[i] = -1;
    }
    static void insert_clean(int* t, size_t m, int key) {
        size_t perturb = (size_t)key, i = (size_t)key & m;
        while (true) {
            if (t[i] < 0) {
                t[i] = key;
                return;
            }
            if (i + 9 <= m) {
                size_t e = i;
                for (int j = 0; j < 9; ++j) {
                    ++e;
                    if (t[e] < 0) {
                        t[e] = key;
                        return;
                    }
                }
            }
            perturb >>= 5;
            i = (i * 5 + 1 + perturb) & m;
        }
    }
    void add(int key) {
        size_t perturb = (size_t)key, i = (size_t)key & mask;
        while (true) {
            size_t e = i;
            int probes = (i + 9 <= mask) ? 9 : 0;
            do {
                if (tab[e] < 0) {
                    tab[e] = key;
                    ++fill;
                    ++used;
                    if (fill * 5 < mask * 3) return;
                    resize(used > 50000 ? used * 2 : used * 4);
                    return;
                }
                if (tab[e] == key) return;
                ++e;
            } while (probes--);
            perturb >>= 5;
            i = (i * 5 + 1 + perturb) & mask;
        }
    }
    void resize(size_t minused) {
        size_t newsize = 8;
        while (newsize <= minused) newsize <<= 1;
        if (newsize > 256) throw Unsupported("set too large");
        int old[256];
        size_t oldmask = mask;
        std::memcpy(old, tab, sizeof(int) * (oldmask + 1));
        for (size_t i = 0; i < newsize; ++i) tab[i] = -1;
        mask = newsize - 1;
        for (size_t i = 0; i <= oldmask; ++i)
            if (old[i] >= 0) insert_clean(tab, mask, old[i]);
    }
};

}  // namespace cq
