// Plain C ABI for ctypes (src/search/native_cheqqers.py). Every entry point catches every C++ exception and reports
// it as a negative return with a message, so a Python caller can always fall back to the Python engine.
//
// Flat state format (int32 array I, double array D), written and read identically by native_cheqqers.py:
//   I: MAGIC, board_size, start_rows, game_type, turn, next_piece_id, moves_since_take, multi_jump (-1 None),
//      just_crowned, consecutive_queen_moves, disable_depth, force_all_moves, state, cache_enabled, N,
//      per square: 0 | 1, color, crowned, nids, ids...      (math.nan id = INT32_MIN)
//      jumped: n, ids...
//      systems (in the order the squares reach them): nsys, per system: nents, ents..., namps, keys...
//      squares: per square: system (-1 independent), namps, keys...
//      history (insertion order): n, per entry: count, ncodes, codes...
//      override legal moves: has, n, per move: len, values...
//   D: the amplitudes (re, im) of the systems, then of the squares, in the order of their keys in I.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "engine.h"
#include "search.h"
#include "value_net.h"

using namespace cq;

#define CQ_API __attribute__((visibility("default")))

namespace {
constexpr int32_t MAGIC = 0x43510001;
constexpr int ABI_VERSION = 2;

void set_err(char* err, int cap, const char* msg) {
    if (err && cap > 0) {
        std::strncpy(err, msg, cap - 1);
        err[cap - 1] = 0;
    }
}

struct Reader {
    const int32_t* I;
    int64_t nI, pi = 0;
    const double* D;
    int64_t nD, pd = 0;
    int32_t i() {
        if (pi >= nI) throw Unsupported("flat state truncated (ints)");
        return I[pi++];
    }
    double d() {
        if (pd >= nD) throw Unsupported("flat state truncated (doubles)");
        return D[pd++];
    }
};

Env* import_flat(Reader& r) {
    if (r.i() != MAGIC) throw Unsupported("not a flat Cheqqers state");
    int bs = r.i();
    std::unique_ptr<Env> e(new Env(bs, 0, CLASSIC));
    e->start_rows = r.i();
    e->game_type = (int8_t)r.i();
    e->turn = (int8_t)r.i();
    e->next_piece_id = r.i();
    e->moves_since_take = r.i();
    e->mj = (int16_t)r.i();
    e->just_crowned = r.i() != 0;
    e->consecutive_queen_moves = r.i();
    e->disable_depth_calculation = r.i() != 0;
    e->force_all_moves = r.i() != 0;
    e->state = (int8_t)r.i();
    e->cache_enabled = r.i() != 0;
    int N = r.i();
    if (N != e->N) throw Unsupported("square count does not match the board size");
    for (int s = 0; s < N; ++s) {
        Props& p = e->props[s];
        p.present = r.i() != 0;
        if (p.present) {
            p.color = (int8_t)r.i();
            p.crowned = r.i() != 0;
            int n = r.i();
            if (n < 0 || n > MAXIDS) throw Unsupported("piece ids");
            p.nids = (uint8_t)n;
            for (int j = 0; j < n; ++j) p.ids[j] = r.i();
        }
    }
    int nj = r.i();
    if (nj < 0 || nj > MAXJUMPED) throw Unsupported("jumped ids");
    e->jumped.n = (uint8_t)nj;
    for (int j = 0; j < nj; ++j) e->jumped.v[j] = r.i();
    int nsys = r.i();
    e->systems.assign(nsys, System());
    for (int s = 0; s < nsys; ++s) {
        System& S = e->systems[s];
        int ne = r.i();
        if (ne < 0 || ne > 15) throw Unsupported("system size");
        for (int j = 0; j < ne; ++j) {
            int x = r.i();
            if (x < 0 || x >= N) throw Unsupported("entity index");
            S.ents.push_back((int8_t)x);
        }
        int na = r.i();
        S.amps.resize(na);
        for (int j = 0; j < na; ++j) S.amps[j].k = (uint32_t)r.i();
    }
    for (int s = 0; s < N; ++s) {
        int sys = r.i();
        if (sys < -1 || sys >= nsys) throw Unsupported("system index");
        e->sq[s].sys = (int16_t)sys;
        int na = r.i();
        if (na < 0 || na > 2) throw Unsupported("square amplitudes");
        e->sq[s].amps.n = (uint8_t)na;
        for (int j = 0; j < na; ++j) {
            int k = r.i();
            if (k != 0 && k != 1) throw Unsupported("square amplitude key");
            e->sq[s].amps.k[j] = (uint8_t)k;
        }
    }
    int nh = r.i();
    e->history.resize(nh);
    for (int h = 0; h < nh; ++h) {
        e->history[h].count = r.i();
        int nc = r.i();
        if (nc != N + 1) throw Unsupported("history key length");
        e->history[h].key.n = (uint8_t)nc;
        for (int j = 0; j < nc; ++j) e->history[h].key.v[j] = (uint16_t)r.i();
    }
    e->has_override = r.i() != 0;
    int no = r.i();
    e->override_moves.resize(no);
    for (int m = 0; m < no; ++m) {
        int len = r.i();
        if (len < 2 || len > 3) throw Unsupported("override move length");
        e->override_moves[m].n = (int8_t)len;
        for (int j = 0; j < len; ++j) e->override_moves[m].v[j] = (int16_t)r.i();
    }
    for (System& S : e->systems)
        for (Amp& a : S.amps) {
            a.v.re = r.d();
            a.v.im = r.d();
        }
    for (int s = 0; s < N; ++s)
        for (int j = 0; j < e->sq[s].amps.n; ++j) {
            e->sq[s].amps.v[j].re = r.d();
            e->sq[s].amps.v[j].im = r.d();
        }
    e->invalidate_caches();
    return e.release();
}

void export_flat(const Env& e, std::vector<int32_t>& I, std::vector<double>& D) {
    I.clear();
    D.clear();
    I.insert(I.end(), {MAGIC, e.bs, e.start_rows, e.game_type, e.turn, e.next_piece_id, e.moves_since_take, e.mj,
                       (int32_t)e.just_crowned, e.consecutive_queen_moves, (int32_t)e.disable_depth_calculation,
                       (int32_t)e.force_all_moves, e.state, (int32_t)e.cache_enabled, e.N});
    for (int s = 0; s < e.N; ++s) {
        const Props& p = e.props[s];
        if (!p.present) {
            I.push_back(0);
            continue;
        }
        I.insert(I.end(), {1, p.color, (int32_t)p.crowned, p.nids});
        for (int j = 0; j < p.nids; ++j) I.push_back(p.ids[j]);
    }
    I.push_back(e.jumped.n);
    for (int j = 0; j < e.jumped.n; ++j) I.push_back(e.jumped.v[j]);
    std::vector<int> remap(e.systems.size(), -1), order;
    for (int s = 0; s < e.N; ++s) {
        int x = e.sq[s].sys;
        if (x >= 0 && remap[x] < 0) {
            remap[x] = (int)order.size();
            order.push_back(x);
        }
    }
    I.push_back((int32_t)order.size());
    for (int x : order) {
        const System& S = e.systems[x];
        I.push_back((int32_t)S.ents.size());
        for (int8_t en : S.ents) I.push_back(en);
        I.push_back((int32_t)S.amps.size());
        for (const Amp& a : S.amps) I.push_back((int32_t)a.k);
    }
    for (int s = 0; s < e.N; ++s) {
        int x = e.sq[s].sys;
        I.push_back(x < 0 ? -1 : remap[x]);
        I.push_back(e.sq[s].amps.n);
        for (int j = 0; j < e.sq[s].amps.n; ++j) I.push_back(e.sq[s].amps.k[j]);
    }
    I.push_back((int32_t)e.history.size());
    for (const HistEntry& h : e.history) {
        I.push_back(h.count);
        I.push_back(h.key.n);
        for (int j = 0; j < h.key.n; ++j) I.push_back(h.key.v[j]);
    }
    I.push_back((int32_t)e.has_override);
    I.push_back((int32_t)e.override_moves.size());
    for (const MoveArr& m : e.override_moves) {
        I.push_back(m.n);
        for (int j = 0; j < m.n; ++j) I.push_back(m.v[j]);
    }
    for (int x : order)
        for (const Amp& a : e.systems[x].amps) {
            D.push_back(a.v.re);
            D.push_back(a.v.im);
        }
    for (int s = 0; s < e.N; ++s)
        for (int j = 0; j < e.sq[s].amps.n; ++j) {
            D.push_back(e.sq[s].amps.v[j].re);
            D.push_back(e.sq[s].amps.v[j].im);
        }
}

MoveArr arr_from(const int32_t* mv, int n) {
    if (n < 2 || n > 3) throw Unsupported("move length");
    MoveArr a;
    a.n = (int8_t)n;
    for (int j = 0; j < n; ++j) {
        if (mv[j] < -32768 || mv[j] > 32767) throw Unsupported("move value");
        a.v[j] = (int16_t)mv[j];
    }
    return a;
}

Weights weights_from(const double* w) {
    Weights W;
    if (w) {
        for (int j = 0; j < NFEAT; ++j) W.w[j] = w[j];
        W.scale = w[NFEAT];
        W.logit_per_man = w[NFEAT + 1];
    }
    return W;
}

}  // namespace

extern "C" {

struct CqOpts {
    int32_t max_depth;
    double time_limit_ms;
    int32_t max_tt_size;
    int32_t max_capture_extension;
    int32_t star1;
    int32_t quiescence;
    int32_t quantum_plies;  // -1 = None
    int32_t use_cache;
    int32_t has_weights;
    double weights[NFEAT + 2];  // features..., scale, logit_per_man
};

struct CqResult {
    int32_t chosen;
    int64_t nodes;
    int32_t last_depth;
    int32_t has_score;
    double last_score;
    int32_t unexpanded;
    double elapsed_ms;
    int32_t n_root;  // root scores written (best first)
    int32_t failed;  // 1: an iteration raised (message in err); chosen is the last finished iteration's best move
};

CQ_API int cq_abi_version() { return ABI_VERSION; }

CQ_API void* cq_env_new(int board_size, int start_rows, int game_type, char* err, int errcap) {
    try {
        return new Env(board_size, start_rows, game_type);
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return nullptr;
}

CQ_API void* cq_env_import(const int32_t* I, int64_t nI, const double* D, int64_t nD, char* err, int errcap) {
    try {
        Reader r{I, nI, 0, D, nD, 0};
        Env* e = import_flat(r);
        if (r.pi != nI || r.pd != nD) {
            delete e;
            throw Unsupported("flat state has trailing data");
        }
        return e;
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return nullptr;
}

CQ_API void cq_env_free(void* h) { delete static_cast<Env*>(h); }

CQ_API void* cq_env_clone(void* h) { return static_cast<Env*>(h)->clone().release(); }

CQ_API void cq_env_set_cache(void* h, int on) {
    Env* e = static_cast<Env*>(h);
    e->cache_enabled = on != 0;
    e->invalidate_caches();
}

// Returns the number of ints; *nD_out the number of doubles. If a buffer is too small nothing is written beyond it
// and the sizes are still returned: call again with larger buffers.
CQ_API int64_t cq_env_export(void* h, int32_t* I, int64_t capI, double* D, int64_t capD, int64_t* nD_out) {
    std::vector<int32_t> vi;
    std::vector<double> vd;
    export_flat(*static_cast<Env*>(h), vi, vd);
    if ((int64_t)vi.size() <= capI) std::memcpy(I, vi.data(), vi.size() * sizeof(int32_t));
    if ((int64_t)vd.size() <= capD) std::memcpy(D, vd.data(), vd.size() * sizeof(double));
    *nD_out = (int64_t)vd.size();
    return (int64_t)vi.size();
}

CQ_API int cq_state(void* h) { return static_cast<Env*>(h)->state; }
CQ_API int cq_turn(void* h) { return static_cast<Env*>(h)->turn; }
CQ_API int cq_multi_jump(void* h) { return static_cast<Env*>(h)->mj; }

// legal moves as rows of 4 ints: len, a, b, c. Returns the count, -1 on error, -(count+2) if cap is too small.
CQ_API int cq_legal_moves(void* h, int32_t* out, int cap, char* err, int errcap) {
    try {
        std::vector<MoveArr> mv;
        static_cast<Env*>(h)->legal_moves(mv);
        if ((int)mv.size() > cap) return -((int)mv.size() + 2);
        for (size_t i = 0; i < mv.size(); ++i) {
            out[4 * i] = mv[i].n;
            for (int j = 0; j < 3; ++j) out[4 * i + 1 + j] = j < mv[i].n ? mv[i].v[j] : 0;
        }
        return (int)mv.size();
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return -1;
}

// get_move_outcomes(move): probabilities and new child handles, most likely first. Returns the count or -1.
CQ_API int cq_move_outcomes(void* h, const int32_t* mv, int n, double* probs, void** children, int cap, char* err,
                     int errcap) {
    try {
        Env* e = static_cast<Env*>(h);
        Move m = e->resolve(arr_from(mv, n));
        std::vector<Outcome> outs;
        e->move_outcomes(m, outs);
        if ((int)outs.size() > cap) throw Unsupported("too many outcomes for the buffer");
        for (size_t i = 0; i < outs.size(); ++i) {
            probs[i] = outs[i].p;
            children[i] = outs[i].env.release();
        }
        return (int)outs.size();
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return -1;
}

// apply_move(move, forced_measurements) in place. Returns the number of measurements or -1.
CQ_API int cq_apply_move(void* h, const int32_t* mv, int n, const int32_t* forced, int nforced, int has_forced,
                  uint64_t seed, char* err, int errcap) {
    try {
        Env* e = static_cast<Env*>(h);
        Move m = e->resolve(arr_from(mv, n));
        Rng rng;
        if (seed) rng.s = seed;
        std::vector<int> f(forced, forced + (has_forced ? nforced : 0));
        return e->apply_move(m, has_forced ? f.data() : nullptr, (int)f.size(), false, &rng);
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return -1;
}

CQ_API int cq_eval_features(void* h, int persp, double* out, char* err, int errcap) {
    try {
        static_cast<Env*>(h)->eval_features(persp, out);
        return 0;
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return -1;
}

CQ_API int cq_evaluate(void* h, int persp, const double* weights, double* out, char* err, int errcap) {
    try {
        *out = static_cast<Env*>(h)->evaluate(persp, weights_from(weights));
        return 0;
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return -1;
}

static SearchOpts opts_from(const CqOpts* o) {
    SearchOpts s;
    s.max_depth = o->max_depth;
    s.time_limit_ms = o->time_limit_ms;
    s.max_tt_size = o->max_tt_size;
    s.max_capture_extension = o->max_capture_extension;
    s.star1 = o->star1 != 0;
    s.quiescence = o->quiescence != 0;
    s.quantum_plies = o->quantum_plies;
    s.use_cache = o->use_cache != 0;
    if (o->has_weights) s.weights = weights_from(o->weights);
    return s;
}

// ExpectiCheqqersPlayer.get_move(env, available_moves=raw). raw rows of 4 ints (len, a, b, c); nraw < 0: the
// position's own legal moves. root_moves (rows of 4) / root_scores receive the last finished iteration's values.
// cq_best_move_net: the same with a value network (cq_net_new) at the leaves; net == NULL is cq_best_move.
CQ_API int cq_best_move_net(void* h, const int32_t* raw, int nraw, const CqOpts* o, const void* net, CqResult* out,
                            int32_t* root_moves, double* root_scores, int cap, char* err, int errcap) {
    try {
        Env* e = static_cast<Env*>(h);
        SearchOpts so = opts_from(o);
        so.net = static_cast<const ValueNet*>(net);
        Searcher s(so);
        std::vector<MoveArr> rawv;
        if (nraw >= 0)
            for (int i = 0; i < nraw; ++i) rawv.push_back(arr_from(raw + 4 * i + 1, raw[4 * i]));
        SearchResult r = s.get_move(*e, nraw >= 0 ? &rawv : nullptr);
        out->chosen = r.chosen;
        out->nodes = r.nodes;
        out->last_depth = r.last_depth;
        out->has_score = r.has_score;
        out->last_score = r.last_score;
        out->unexpanded = r.unexpanded;
        out->elapsed_ms = r.elapsed_ms;
        int n = std::min<int>(cap, (int)r.root_scores.size());
        out->n_root = n;
        out->failed = r.error.empty() ? 0 : 1;
        if (out->failed) set_err(err, errcap, r.error.c_str());
        for (int i = 0; i < n; ++i) {
            const MoveArr& m = r.root_scores[i].move;
            root_moves[4 * i] = m.n;
            for (int j = 0; j < 3; ++j) root_moves[4 * i + 1 + j] = j < m.n ? m.v[j] : 0;
            root_scores[i] = r.root_scores[i].value;
        }
        return 0;
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return -1;
}

CQ_API int cq_best_move(void* h, const int32_t* raw, int nraw, const CqOpts* o, CqResult* out, int32_t* root_moves,
                 double* root_scores, int cap, char* err, int errcap) {
    return cq_best_move_net(h, raw, nraw, o, nullptr, out, root_moves, root_scores, cap, err, errcap);
}

// ---------------------------------------------------------------------------------------------------------------
// Value network (value_net.h). dims[0..nlayers]: input size, then each layer's outputs; params: per layer the
// weights [out][in] row-major then the biases. Returns a handle (cq_net_free) or NULL with err set.
// ---------------------------------------------------------------------------------------------------------------
CQ_API void* cq_net_new(const int32_t* dims, int nlayers, const double* params, int64_t nparams, char* err,
                        int errcap) {
    try {
        std::vector<int> d(dims, dims + nlayers + 1);
        return new ValueNet(d, params, (long long)nparams);
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return nullptr;
}

CQ_API void cq_net_free(void* net) { delete static_cast<ValueNet*>(net); }

CQ_API int cq_net_inputs(const void* net) { return static_cast<const ValueNet*>(net)->inputs(); }

// Residual mode (value_net.h): leaf logit = the weights' linear logit + lambda * the network's output.
CQ_API void cq_net_set_residual(void* net, double lambda) { static_cast<ValueNet*>(net)->set_residual(lambda); }

// Forward pass on given inputs (n rows of inputs() doubles) -> n logits.
CQ_API int cq_net_forward(const void* net, const double* x, int n, double* out, char* err, int errcap) {
    try {
        const ValueNet* v = static_cast<const ValueNet*>(net);
        for (int i = 0; i < n; ++i) out[i] = v->forward(x + (size_t)i * v->inputs());
        return 0;
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return -1;
}

// The network's input for a position (x: inputs() doubles) and its logit, side to move's view.
CQ_API int cq_net_encode(const void* net, void* h, double* x, double* logit, char* err, int errcap) {
    try {
        const ValueNet* v = static_cast<const ValueNet*>(net);
        Env* e = static_cast<Env*>(h);
        v->encode(*e, x);
        *logit = v->forward(x);
        return 0;
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return -1;
}

}  // extern "C"

// Test hooks for the two CPython emulations (tests/native/test_native_cpython_semantics.py).
extern "C" {
// The iteration order of a CPython set after adding `keys` in order; returns its size.
CQ_API int cq_test_set_order(const int32_t* keys, int n, int32_t* out) {
    try {
        PySetSim s;
        for (int i = 0; i < n; ++i) s.add(keys[i]);
        int m = 0;
        for (size_t i = 0; i <= s.mask; ++i)
            if (s.tab[i] >= 0) out[m++] = s.tab[i];
        return m;
    } catch (...) {
        return -1;
    }
}
// sum(values) as CPython 3.12+ computes it for floats.
CQ_API double cq_test_sum(const double* values, int n) {
    PySum s;
    for (int i = 0; i < n; ++i) s.add(values[i]);
    return s.result();
}
}

// ---------------------------------------------------------------------------------------------------------------
// Self-play: one whole game in C++, searched by both sides with the same options, measurement outcomes drawn with
// their probabilities from `seed`. Appends one JSON line per position searched and a last line with the result to
// out_path. start == NULL: the starting position of (board_size, start_rows, quantumness).
// Per position: ply, turn (0 white), multi_jump, features (eval_features for the side to move), presence (each
// square's occupied probability), pieces (0 none, 1 white man, 2 white queen, 3 black man, 4 black queen), moves,
// scores (root value of each move at the last finished iteration, same order; moves searched after a better one
// carry the alpha-beta bound, not the exact value), move (chosen), depth, nodes, ms, outcome_p (probability of the
// outcome that followed) and, with write_state, state = [ints, doubles] (native_cheqqers.import_env reads it).
// Returns the number of positions written (>= 0) or -1.
// ---------------------------------------------------------------------------------------------------------------
namespace {
void json_double(std::ostringstream& o, double x) {
    if (std::isfinite(x)) {
        char b[40];
        std::snprintf(b, sizeof b, "%.17g", x);
        o << b;
    } else {
        o << "null";
    }
}
void json_move(std::ostringstream& o, const MoveArr& m) {
    o << '[';
    for (int j = 0; j < m.n; ++j) o << (j ? "," : "") << m.v[j];
    o << ']';
}
}  // namespace

// cq_selfplay_leaves: cq_selfplay_net that also samples the searches' leaves (search.h LeafSink) with probability
// leaf_rate each and writes them after the position line they were searched from:
// {"leaf": game ply, "turn": side to move at the leaf, "d": search ply, "cp": capture pending, "x": network input}.
// leaf_rate = 0: exactly cq_selfplay_net (the leaf generator is separate from the measurement draws).
extern "C" CQ_API int cq_selfplay_leaves(void* start, int board_size, int start_rows, int quantumness,
                                         const CqOpts* o, const void* net, int max_plies, uint64_t seed,
                                         const char* out_path, int write_state, double leaf_rate, char* err,
                                         int errcap) {
    try {
        EnvPtr env = start ? static_cast<Env*>(start)->clone() : EnvPtr(new Env(board_size, start_rows, quantumness));
        env->cache_enabled = true;
        Rng rng;
        rng.s = seed ? seed : 0x9E3779B97F4A7C15ULL;
        for (int i = 0; i < 4; ++i) rng.next();
        SearchOpts so = opts_from(o);
        so.net = static_cast<const ValueNet*>(net);
        LeafSink sink;
        sink.rate = leaf_rate;
        sink.keep_state = write_state != 0;  // leaves get their flat state too
        sink.rng.s = (seed ^ 0xD1B54A32D192ED03ULL) | 1ULL;
        if (leaf_rate > 0.0) so.leaf_sink = &sink;
        std::ofstream out(out_path, std::ios::app);
        if (!out) throw Unsupported("cannot open the output file");
        int written = 0, ply = 0;
        for (; ply < max_plies && !env->terminal(); ++ply) {
            std::vector<MoveArr> legal;
            env->legal_moves(legal);
            if (legal.empty()) break;
            Searcher s(so);
            sink.samples.clear();
            SearchResult r = s.get_move(*env, &legal);
            if (r.chosen < 0) break;
            const MoveArr chosen = legal[r.chosen];
            std::ostringstream line;
            line << "{\"ply\":" << ply << ",\"turn\":" << (int)env->turn << ",\"multi_jump\":" << env->mj;
            double f[NFEAT];
            env->eval_features(env->turn, f);
            line << ",\"features\":[";
            for (int j = 0; j < NFEAT; ++j) {
                if (j) line << ',';
                json_double(line, f[j]);
            }
            env->square_probs();
            line << "],\"presence\":[";
            for (int i = 0; i < env->N; ++i) {
                if (i) line << ',';
                json_double(line, env->P1[i]);
            }
            line << "],\"pieces\":[";
            for (int i = 0; i < env->N; ++i) {
                const Props& p = env->props[i];
                int code = p.present ? 1 + (p.color == BLACK ? 2 : 0) + (p.crowned ? 1 : 0) : 0;
                line << (i ? "," : "") << code;
            }
            line << "],\"moves\":[";
            for (size_t i = 0; i < legal.size(); ++i) {
                if (i) line << ',';
                json_move(line, legal[i]);
            }
            line << "],\"scores\":[";
            for (size_t i = 0; i < legal.size(); ++i) {
                if (i) line << ',';
                bool found = false;
                for (const RootScore& rs : r.root_scores)
                    if (rs.move == legal[i]) {
                        json_double(line, rs.value);
                        found = true;
                        break;
                    }
                if (!found) line << "null";
            }
            line << "],\"move\":";
            json_move(line, chosen);
            line << ",\"depth\":" << r.last_depth << ",\"nodes\":" << r.nodes << ",\"ms\":";
            json_double(line, r.elapsed_ms);
            std::vector<Outcome> outs;
            env->move_outcomes(env->resolve(chosen), outs);
            double u = rng.next(), acc = 0.0;
            size_t pick = outs.size() - 1;
            for (size_t k = 0; k < outs.size(); ++k) {
                acc += outs[k].p;
                if (u <= acc) {
                    pick = k;
                    break;
                }
            }
            line << ",\"outcome_p\":";
            json_double(line, outs[pick].p);
            if (write_state) {
                std::vector<int32_t> I;
                std::vector<double> D;
                export_flat(*env, I, D);
                line << ",\"state\":[[";
                for (size_t k = 0; k < I.size(); ++k) line << (k ? "," : "") << I[k];
                line << "],[";
                for (size_t k = 0; k < D.size(); ++k) {
                    if (k) line << ',';
                    json_double(line, D[k]);
                }
                line << "]]";
            }
            line << "}\n";
            for (const LeafSample& ls : sink.samples) {
                line << "{\"leaf\":" << ply << ",\"turn\":" << ls.turn << ",\"d\":" << ls.ply
                     << ",\"cp\":" << (ls.capture_pending ? 1 : 0) << ",\"x\":[";
                for (size_t k = 0; k < ls.x.size(); ++k) {
                    char b[32];
                    std::snprintf(b, sizeof b, "%.7g", ls.x[k]);
                    line << (k ? "," : "") << b;
                }
                line << "]";
                if (ls.env) {
                    std::vector<int32_t> I;
                    std::vector<double> D;
                    export_flat(*ls.env, I, D);
                    line << ",\"state\":[[";
                    for (size_t k = 0; k < I.size(); ++k) line << (k ? "," : "") << I[k];
                    line << "],[";
                    for (size_t k = 0; k < D.size(); ++k) {
                        if (k) line << ',';
                        json_double(line, D[k]);
                    }
                    line << "]]";
                }
                line << "}\n";
            }
            out << line.str();
            ++written;
            env = std::move(outs[pick].env);
            env->cache_enabled = true;
        }
        out << "{\"result\":" << (env->terminal() ? env->winner() : 2) << ",\"state\":" << (int)env->state
            << ",\"plies\":" << ply << "}\n";
        return written;
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return -1;
}

extern "C" CQ_API int cq_selfplay_net(void* start, int board_size, int start_rows, int quantumness, const CqOpts* o,
                                      const void* net, int max_plies, uint64_t seed, const char* out_path,
                                      int write_state, char* err, int errcap) {
    return cq_selfplay_leaves(start, board_size, start_rows, quantumness, o, net, max_plies, seed, out_path,
                              write_state, 0.0, err, errcap);
}

// cq_selfplay_net with no network.
extern "C" CQ_API int cq_selfplay(void* start, int board_size, int start_rows, int quantumness, const CqOpts* o,
                                  int max_plies, uint64_t seed, const char* out_path, int write_state, char* err,
                                  int errcap) {
    return cq_selfplay_net(start, board_size, start_rows, quantumness, o, nullptr, max_plies, seed, out_path,
                           write_state, err, errcap);
}

// ---------------------------------------------------------------------------------------------------------------
// MCTS with chance nodes for a network evaluated by the caller (mcts.h; track 33). Additive: nothing above uses it.
// One tree per position searched. collect() fills rows (encodings, moves) for the network, apply() takes its values
// and priors back. The *_many entry points run many trees (many games) on a persistent pool of threads and pack
// their rows into one batch: the tree work runs on the CPU threads, the caller runs one large GPU batch.
// ---------------------------------------------------------------------------------------------------------------
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

#include "mcts.h"

namespace {
struct CqMctsOptsC {
    double c_puct;
    double fpu_reduction;
    double dirichlet_alpha;
    double dirichlet_frac;
    uint64_t seed;
    int64_t max_nodes;
    int32_t max_moves;
    int32_t gumbel_m;  // > 0: Gumbel root (mcts.h)
    int32_t gumbel_n;
    double c_visit;
    double c_scale;
};
struct CqMctsStatsC {
    int64_t sims, evals, terminal_sims, collisions, nodes, overflow;
    int32_t max_depth;
    int32_t root_turn;
    double root_value;  // white's view
    double root_vnet;   // the network's value of the root, white's view
    int32_t done;       // Gumbel: the search has used its budget
};

class Pool {
   public:
    void run(int n, int threads, const std::function<void(int)>& fn) {
        if (threads <= 1 || n <= 1) {
            for (int i = 0; i < n; ++i) fn(i);
            return;
        }
        std::unique_lock<std::mutex> lk(m_);
        while ((int)workers_.size() < threads - 1) workers_.emplace_back([this] { loop(); });
        fn_ = &fn;
        n_ = n;
        next_ = 0;
        active_ = (int)workers_.size();
        ++gen_;
        cv_.notify_all();
        lk.unlock();
        work();
        lk.lock();
        done_cv_.wait(lk, [this] { return active_ == 0; });
        fn_ = nullptr;
    }

   private:
    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    std::vector<std::thread> workers_;
    const std::function<void(int)>* fn_ = nullptr;
    int n_ = 0, active_ = 0;
    uint64_t gen_ = 0;
    std::atomic<int> next_{0};
    void work() {
        for (int i; (i = next_.fetch_add(1)) < n_;) (*fn_)(i);
    }
    void loop() {
        uint64_t seen = 0;
        std::unique_lock<std::mutex> lk(m_);
        for (;;) {
            cv_.wait(lk, [&] { return gen_ != seen; });
            seen = gen_;
            lk.unlock();
            work();
            lk.lock();
            if (--active_ == 0) done_cv_.notify_all();
        }
    }
};
Pool& pool() {
    static Pool* p = new Pool();  // never destroyed: workers live as long as the process
    return *p;
}
std::mutex g_pool_call;  // one parallel call at a time
}  // namespace

extern "C" {

CQ_API int cq_mcts_abi() { return 1; }

CQ_API void* cq_mcts_new(void* h, const int32_t* raw, int nraw, const CqMctsOptsC* o, char* err, int errcap) {
    try {
        MctsOpts mo;
        mo.c_puct = o->c_puct;
        mo.fpu_reduction = o->fpu_reduction;
        mo.dirichlet_alpha = o->dirichlet_alpha;
        mo.dirichlet_frac = o->dirichlet_frac;
        mo.seed = o->seed;
        mo.max_nodes = o->max_nodes;
        mo.max_moves = o->max_moves;
        mo.gumbel_m = o->gumbel_m;
        mo.gumbel_n = o->gumbel_n;
        mo.c_visit = o->c_visit;
        mo.c_scale = o->c_scale;
        std::vector<MoveArr> rawv;
        for (int i = 0; i < nraw; ++i) rawv.push_back(arr_from(raw + 4 * i + 1, raw[4 * i]));
        return new Mcts(*static_cast<Env*>(h), rawv, mo);
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return nullptr;
}

CQ_API void cq_mcts_free(void* t) { delete static_cast<Mcts*>(t); }

CQ_API int cq_mcts_in_dim(void* t) { return static_cast<Mcts*>(t)->in_dim(); }

// Rows written into x (cap x in_dim), mv (cap x max_moves x 3), lens (cap x max_moves); *mmax the largest move
// count. Returns the row count or -1.
CQ_API int cq_mcts_collect(void* t, int max_leaves, int cap, float* x, int32_t* mv, int32_t* lens, int32_t* mmax,
                           char* err, int errcap) {
    try {
        int m = 0;
        int n = static_cast<Mcts*>(t)->collect(max_leaves, cap, x, mv, lens, &m);
        *mmax = m;
        return n;
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return -1;
}

CQ_API int cq_mcts_apply(void* t, const float* values, const float* priors, int stride, int n, char* err,
                         int errcap) {
    try {
        static_cast<Mcts*>(t)->apply(values, priors, stride, n);
        return 0;
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return -1;
}

// Many trees, one batch: tree i writes up to cap_each rows into its own region, then the rows are packed in tree
// order; counts[i] = tree i's rows (-1: it failed, message of the first failure in err). Returns the total rows.
CQ_API int cq_mcts_collect_many(void** trees, int ntrees, int max_leaves_each, int cap_each, float* x, int32_t* mv,
                                int32_t* lens, int32_t* counts, int32_t* mmax, int threads, char* err, int errcap) {
    std::lock_guard<std::mutex> g(g_pool_call);
    if (ntrees <= 0) return 0;
    Mcts* t0 = static_cast<Mcts*>(trees[0]);
    const int D = t0->in_dim();
    int M = -1;
    std::vector<int> mm(ntrees, 0);
    std::vector<std::string> errs(ntrees);
    // max_moves is per tree; all trees of one call must share it (the stride)
    pool().run(ntrees, threads, [&](int i) {
        Mcts* t = static_cast<Mcts*>(trees[i]);
        try {
            int m = 0;
            size_t off = (size_t)i * cap_each;
            counts[i] = t->collect(max_leaves_each, cap_each, x + off * D, mv + off * 3 * (size_t)t->max_moves(),
                                   lens + off * (size_t)t->max_moves(), &m);
            mm[i] = m;
        } catch (const std::exception& ex) {
            counts[i] = -1;
            errs[i] = ex.what();
        } catch (...) {
            counts[i] = -1;
            errs[i] = "unknown error";
        }
    });
    M = t0->max_moves();
    int total = 0, mx = 0;
    for (int i = 0; i < ntrees; ++i) {
        if (counts[i] < 0) {
            set_err(err, errcap, errs[i].c_str());
            continue;
        }
        size_t src = (size_t)i * cap_each;
        if ((size_t)total != src && counts[i] > 0) {
            std::memmove(x + (size_t)total * D, x + src * D, sizeof(float) * (size_t)counts[i] * D);
            std::memmove(mv + (size_t)total * M * 3, mv + src * M * 3, sizeof(int32_t) * (size_t)counts[i] * M * 3);
            std::memmove(lens + (size_t)total * M, lens + src * M, sizeof(int32_t) * (size_t)counts[i] * M);
        }
        total += counts[i];
        mx = std::max(mx, mm[i]);
    }
    *mmax = mx;
    return total;
}

// The network's answers for a cq_mcts_collect_many batch (rows in the same packed order); counts as returned by it
// (trees with count <= 0 are skipped). Returns 0, or -1 if a tree failed (message in err).
CQ_API int cq_mcts_apply_many(void** trees, int ntrees, const int32_t* counts, const float* values,
                              const float* priors, int stride, int threads, char* err, int errcap) {
    std::lock_guard<std::mutex> g(g_pool_call);
    std::vector<size_t> off(ntrees + 1, 0);
    for (int i = 0; i < ntrees; ++i) off[i + 1] = off[i] + (counts[i] > 0 ? counts[i] : 0);
    std::vector<std::string> errs(ntrees);
    pool().run(ntrees, threads, [&](int i) {
        if (counts[i] <= 0) return;
        try {
            static_cast<Mcts*>(trees[i])->apply(values + off[i], priors + off[i] * stride, stride, counts[i]);
        } catch (const std::exception& ex) {
            errs[i] = ex.what();
        } catch (...) {
            errs[i] = "unknown error";
        }
    });
    for (int i = 0; i < ntrees; ++i)
        if (!errs[i].empty()) {
            set_err(err, errcap, errs[i].c_str());
            return -1;
        }
    return 0;
}

// Root moves (rows of 4: len, a, b, c), visits, Q (white's view, NaN unvisited), priors. Returns the move count
// (writes at most cap).
CQ_API int cq_mcts_root(void* t, int32_t* moves, int32_t* visits, double* q, double* prior, int cap,
                        CqMctsStatsC* st) {
    Mcts* m = static_cast<Mcts*>(t);
    auto info = m->root_info();
    int n = std::min<int>(cap, (int)info.size());
    for (int i = 0; i < n; ++i) {
        moves[4 * i] = info[i].move.n;
        for (int j = 0; j < 3; ++j) moves[4 * i + 1 + j] = j < info[i].move.n ? info[i].move.v[j] : 0;
        visits[i] = info[i].visits;
        q[i] = info[i].q;
        prior[i] = info[i].prior;
    }
    if (st) {
        st->sims = m->stats.sims;
        st->evals = m->stats.evals;
        st->terminal_sims = m->stats.terminal_sims;
        st->collisions = m->stats.collisions;
        st->nodes = m->stats.nodes;
        st->overflow = m->stats.overflow;
        st->max_depth = m->stats.max_depth;
        st->root_turn = m->root_turn();
        st->root_value = m->root_value();
        st->root_vnet = m->root_vnet();
        st->done = m->done() ? 1 : 0;
    }
    return (int)info.size();
}

CQ_API int cq_mcts_done(void* t) { return static_cast<Mcts*>(t)->done() ? 1 : 0; }

// Gumbel result: policy[i] = the target softmax(logits + sigma(completed Q)) of root move i, gumbel[i] its Gumbel
// draw, cand[i] = 1 if still a candidate. Returns the chosen move's index (-1 if none); writes at most cap entries.
CQ_API int cq_mcts_gumbel(void* t, double* policy, double* gumbel, int32_t* cand, int cap) {
    Mcts::GumbelOut g = static_cast<Mcts*>(t)->gumbel();
    int n = std::min<int>(cap, (int)g.policy.size());
    for (int i = 0; i < n; ++i) {
        policy[i] = g.policy[i];
        gumbel[i] = i < (int)g.gumbel.size() ? g.gumbel[i] : 0.0;
        cand[i] = 0;
    }
    for (int c : g.candidates)
        if (c < n) cand[c] = 1;
    return g.chosen;
}

}  // extern "C"

// The network input of a position (value_net.h encode_position: 7 per square + features + multi-jump flag, side to
// move's view). x must hold N * 7 + NFEAT + 1 doubles. Returns that count, or -1.
extern "C" CQ_API int cq_encode_position(void* h, double* x, char* err, int errcap) {
    try {
        Env* e = static_cast<Env*>(h);
        encode_position(*e, x);
        return e->N * 7 + NFEAT + 1;
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return -1;
}

// ---------------------------------------------------------------------------------------------------------------
// Batched self-play for a network run by the caller (track 33, RL): N games advance together. cq_sp_encode writes
// every live game's encoding and legal moves (rows packed in game order, game ids in `ids`); the caller picks a move
// per row (any rule) and cq_sp_apply plays them: the outcome is drawn with its probability from each game's own
// generator. Games start from the tournament position after open_lo..open_hi uniformly random plies and end when
// over or at max_plies (result 0). Tree-free: no search here.
// ---------------------------------------------------------------------------------------------------------------
namespace {
struct SpGame {
    EnvPtr env;
    std::vector<MoveArr> legal;
    Rng rng;
    int ply = 0;
    bool done = false;
    int result = 0;  // white's view
};
struct SpBatch {
    std::vector<SpGame> g;
    int max_plies = 300;
    int max_moves = 256;
    std::vector<int> live_rows;  // game index per row of the last encode
};

EnvPtr sp_child(Env& e, const MoveArr& m, Rng& rng) {
    std::vector<Outcome> outs;
    e.move_outcomes(e.resolve(m), outs);
    double u = rng.next(), acc = 0.0;
    size_t pick = outs.size() - 1;
    for (size_t k = 0; k < outs.size(); ++k) {
        acc += outs[k].p;
        if (u <= acc) {
            pick = k;
            break;
        }
    }
    EnvPtr c = std::move(outs[pick].env);
    c->cache_enabled = true;
    return c;
}

void sp_finish(SpGame& s, int max_plies) {
    if (s.env->terminal()) {
        s.done = true;
        int w = s.env->winner();
        s.result = (w == 1 || w == -1) ? w : 0;
    } else if (s.ply >= max_plies) {
        s.done = true;
        s.result = 0;
    }
}
}  // namespace

extern "C" {

CQ_API void* cq_sp_new(int n_games, uint64_t seed, int open_lo, int open_hi, int max_plies, int max_moves,
                       int threads, char* err, int errcap) {
    try {
        std::unique_ptr<SpBatch> b(new SpBatch());
        b->g.resize(n_games);
        b->max_plies = max_plies;
        b->max_moves = max_moves;
        std::vector<std::string> errs(n_games);
        pool().run(n_games, threads, [&](int i) {
            SpGame& s = b->g[i];
            s.rng.s = (seed * 0x9E3779B97F4A7C15ULL) ^ ((uint64_t)(i + 1) * 0xD1B54A32D192ED03ULL);
            if (!s.rng.s) s.rng.s = 1;
            for (int t = 0; t < 4; ++t) s.rng.next();
            try {
                for (int attempt = 0; attempt < 100; ++attempt) {
                    EnvPtr e(new Env(6, 2, ENTANGLEMENT));
                    e->cache_enabled = true;
                    int plies = open_lo + (int)(s.rng.next() * (open_hi - open_lo + 1));
                    if (plies > open_hi) plies = open_hi;
                    bool ok = true;
                    for (int k = 0; k < plies; ++k) {
                        std::vector<MoveArr> legal;
                        e->legal_moves(legal);
                        if (e->terminal() || legal.empty()) {
                            ok = false;
                            break;
                        }
                        size_t j = (size_t)(s.rng.next() * legal.size());
                        if (j >= legal.size()) j = legal.size() - 1;
                        e = sp_child(*e, legal[j], s.rng);
                    }
                    if (ok && !e->terminal()) {
                        s.env = std::move(e);
                        break;
                    }
                }
                if (!s.env) throw Unsupported("no opening found");
            } catch (const std::exception& ex) {
                errs[i] = ex.what();
            }
        });
        for (auto& e : errs)
            if (!e.empty()) throw Unsupported(e);
        return b.release();
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return nullptr;
}

CQ_API void cq_sp_free(void* h) { delete static_cast<SpBatch*>(h); }

// Rows for every live game: x (rows x in_dim floats), mv (rows x max_moves x 3), lens (rows x max_moves), ids (game
// index per row), turn (side to move per row). Games without a legal move end here (result 0). Returns the rows.
CQ_API int cq_sp_encode(void* h, float* x, int32_t* mv, int32_t* lens, int32_t* ids, int32_t* turn, int32_t* mmax,
                        int threads, char* err, int errcap) {
    std::lock_guard<std::mutex> lk(g_pool_call);
    SpBatch* b = static_cast<SpBatch*>(h);
    try {
        // legal moves (parallel), then rows packed in game order
        std::vector<std::string> errs(b->g.size());
        pool().run((int)b->g.size(), threads, [&](int i) {
            SpGame& s = b->g[i];
            if (s.done) return;
            try {
                s.env->legal_moves(s.legal);
                if (s.legal.empty()) {
                    s.done = true;
                    s.result = 0;
                }
            } catch (const std::exception& ex) {
                errs[i] = ex.what();
            }
        });
        for (auto& e : errs)
            if (!e.empty()) throw Unsupported(e);
        b->live_rows.clear();
        for (size_t i = 0; i < b->g.size(); ++i)
            if (!b->g[i].done) b->live_rows.push_back((int)i);
        const int M = b->max_moves;
        const int n = (int)b->live_rows.size();
        if (n == 0) {
            *mmax = 0;
            return 0;
        }
        const int D = b->g[b->live_rows[0]].env->N * 7 + NFEAT + 1;
        std::vector<int> cnt(n, 0);
        pool().run(n, threads, [&](int r) {
            SpGame& s = b->g[b->live_rows[r]];
            std::vector<double> buf(D);
            try {
                encode_position(*s.env, buf.data());
            } catch (const std::exception& ex) {
                errs[b->live_rows[r]] = ex.what();
                return;
            }
            float* xr = x + (size_t)r * D;
            for (int j = 0; j < D; ++j) xr[j] = (float)buf[j];
            int32_t* mr = mv + (size_t)r * M * 3;
            int32_t* lr = lens + (size_t)r * M;
            int c = std::min<int>((int)s.legal.size(), M);
            std::memset(lr, 0, sizeof(int32_t) * M);
            for (int k = 0; k < c; ++k) {
                lr[k] = s.legal[k].n;
                for (int j = 0; j < 3; ++j) mr[3 * k + j] = j < s.legal[k].n ? s.legal[k].v[j] : 0;
            }
            cnt[r] = c;
            ids[r] = b->live_rows[r];
            turn[r] = s.env->turn;
        });
        for (auto& e : errs)
            if (!e.empty()) throw Unsupported(e);
        int mx = 0;
        for (int c : cnt) mx = std::max(mx, c);
        *mmax = mx;
        return n;
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return -1;
}

// Play chosen[r] (index into the row's legal moves) for every row of the last cq_sp_encode. Returns the games still
// live, or -1.
CQ_API int cq_sp_apply(void* h, const int32_t* chosen, int n, int threads, char* err, int errcap) {
    std::lock_guard<std::mutex> lk(g_pool_call);
    SpBatch* b = static_cast<SpBatch*>(h);
    try {
        if (n != (int)b->live_rows.size()) throw Unsupported("cq_sp_apply: rows differ from the last encode");
        std::vector<std::string> errs(n);
        pool().run(n, threads, [&](int r) {
            SpGame& s = b->g[b->live_rows[r]];
            try {
                int k = chosen[r];
                if (k < 0 || k >= (int)s.legal.size()) k = 0;
                s.env = sp_child(*s.env, s.legal[k], s.rng);
                s.ply += 1;
                sp_finish(s, b->max_plies);
            } catch (const std::exception& ex) {
                errs[r] = ex.what();
            }
        });
        for (auto& e : errs)
            if (!e.empty()) throw Unsupported(e);
        int live = 0;
        for (const SpGame& s : b->g) live += s.done ? 0 : 1;
        return live;
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return -1;
}

// results[i] = game i's result (white's view: 1, -1, 0), plies[i] its length. Returns the games still live.
CQ_API int cq_sp_results(void* h, int32_t* results, int32_t* plies) {
    SpBatch* b = static_cast<SpBatch*>(h);
    int live = 0;
    for (size_t i = 0; i < b->g.size(); ++i) {
        results[i] = b->g[i].result;
        plies[i] = b->g[i].ply;
        live += b->g[i].done ? 0 : 1;
    }
    return live;
}

}  // extern "C"

// ---------------------------------------------------------------------------------------------------------------
// Gumbel self-play (recipe section 2): N games, each searched by its own Gumbel MCTS tree (mcts.h, gumbel_m > 0);
// cq_gz_collect / cq_gz_apply batch every live tree's leaves into one network batch (rows packed in game order).
// When a tree has used its budget the game records a sample (root encoding, legal moves, policy target) and plays
// the Sequential Halving winner (outcome drawn with its probability); a fresh tree starts on the next collect.
// Finished games' samples are exported with the game's result (cq_gz_take).
// ---------------------------------------------------------------------------------------------------------------
namespace {
struct GzSample {
    std::vector<float> x;
    std::vector<MoveArr> moves;
    std::vector<float> policy;
    int8_t turn;
};
struct GzGame {
    EnvPtr env;
    std::unique_ptr<Mcts> tree;
    Rng rng;
    int ply = 0;
    int searches = 0;
    bool done = false;
    int result = 0;
    int frozen_side = -1;  // >= 0: that colour's moves are searched with the frozen network (no samples from them)
    int fw_side = -1;      // >= 0: that colour's moves are played by the FW alpha-beta search (no samples)
    std::vector<GzSample> samples;
};
struct GzDone {  // a finished game's samples, waiting for cq_gz_take
    std::vector<GzSample> samples;
    int result = 0;
    int32_t id = 0;
};
struct GzBatch {
    std::vector<GzGame> g;
    MctsOpts mo;
    int max_plies = 300;
    std::vector<int> rows_of;  // rows per game in the last collect
    int64_t moves_played = 0;
    // refill: a finished game restarts from a new random opening while to_start > 0 (keeps the batch full)
    int64_t to_start = 0;
    int open_lo = 2, open_hi = 6;
    int32_t next_id = 0;
    std::vector<int32_t> ids;  // per slot: the game id it is playing
    std::vector<GzDone> done_games;
    std::mutex done_mu;
    SearchOpts fw_opts;  // the FW opponent's search (cq_gz_set_fw)
};

// A new random opening for slot i (its own generator), as cq_sp_new does.
EnvPtr gz_opening(Rng& rng, int lo, int hi) {
    for (int attempt = 0; attempt < 100; ++attempt) {
        EnvPtr e(new Env(6, 2, ENTANGLEMENT));
        e->cache_enabled = true;
        int plies = lo + (int)(rng.next() * (hi - lo + 1));
        if (plies > hi) plies = hi;
        bool ok = true;
        for (int k = 0; k < plies; ++k) {
            std::vector<MoveArr> legal;
            e->legal_moves(legal);
            if (e->terminal() || legal.empty()) {
                ok = false;
                break;
            }
            size_t j = (size_t)(rng.next() * legal.size());
            if (j >= legal.size()) j = legal.size() - 1;
            e = sp_child(*e, legal[j], rng);
        }
        if (ok && !e->terminal()) return e;
    }
    throw Unsupported("no opening found");
}
}  // namespace

extern "C" {

CQ_API void* cq_gz_new(int n_games, uint64_t seed, int open_lo, int open_hi, int max_plies, const CqMctsOptsC* o,
                       int threads, char* err, int errcap) {
    try {
        void* sp = cq_sp_new(n_games, seed, open_lo, open_hi, max_plies, o->max_moves, threads, err, errcap);
        if (!sp) return nullptr;
        std::unique_ptr<SpBatch> s(static_cast<SpBatch*>(sp));
        std::unique_ptr<GzBatch> b(new GzBatch());
        b->g.resize(n_games);
        for (int i = 0; i < n_games; ++i) {
            b->g[i].env = std::move(s->g[i].env);
            b->g[i].rng = s->g[i].rng;
        }
        b->max_plies = max_plies;
        b->mo.c_puct = o->c_puct;
        b->mo.fpu_reduction = o->fpu_reduction;
        b->mo.seed = o->seed;
        b->mo.max_nodes = o->max_nodes;
        b->mo.max_moves = o->max_moves;
        b->mo.gumbel_m = o->gumbel_m > 0 ? o->gumbel_m : 16;
        b->mo.gumbel_n = o->gumbel_n;
        b->mo.c_visit = o->c_visit;
        b->mo.c_scale = o->c_scale;
        b->rows_of.assign(n_games, 0);
        b->open_lo = open_lo;
        b->open_hi = open_hi;
        b->ids.resize(n_games);
        for (int i = 0; i < n_games; ++i) b->ids[i] = b->next_id++;
        return b.release();
    } catch (const std::exception& ex) {
        set_err(err, errcap, ex.what());
    } catch (...) {
        set_err(err, errcap, "unknown error");
    }
    return nullptr;
}

CQ_API void cq_gz_free(void* h) { delete static_cast<GzBatch*>(h); }

// Game i's colour searched by the frozen network (-1: none), set before the first collect.
CQ_API void cq_gz_set_frozen(void* h, const int32_t* side, int n) {
    GzBatch* b = static_cast<GzBatch*>(h);
    for (int i = 0; i < n && i < (int)b->g.size(); ++i) b->g[i].frozen_side = side[i];
}

// Game i's colour played by the FW alpha-beta search (-1: none); FW = the given weights (features..., scale,
// logit_per_man; NULL = default), time_ms and depth cap per move. Set before the first collect.
CQ_API void cq_gz_set_fw(void* h, const int32_t* side, int n, double time_ms, int max_depth, const double* weights) {
    GzBatch* b = static_cast<GzBatch*>(h);
    for (int i = 0; i < n && i < (int)b->g.size(); ++i) b->g[i].fw_side = side[i];
    b->fw_opts = SearchOpts();
    b->fw_opts.time_limit_ms = time_ms;
    b->fw_opts.max_depth = max_depth;
    if (weights) b->fw_opts.weights = weights_from(weights);
}

// Per game: 1 if its current search (and so every row it produced in the last collect) uses the frozen network.
CQ_API void cq_gz_which(void* h, int32_t* which) {
    GzBatch* b = static_cast<GzBatch*>(h);
    for (size_t i = 0; i < b->g.size(); ++i) {
        const GzGame& gm = b->g[i];
        which[i] = (!gm.done && gm.frozen_side >= 0 && gm.env && gm.env->turn == gm.frozen_side) ? 1 : 0;
    }
}

// Rows for every live game's tree (each at most max_leaves_each, buffers cap_each rows per game), packed in game
// order; counts[i] = game i's rows. Returns the total rows, or -1.
CQ_API int cq_gz_collect(void* h, int max_leaves_each, int cap_each, float* x, int32_t* mv, int32_t* lens,
                         int32_t* counts, int32_t* mmax, int threads, char* err, int errcap) {
    std::lock_guard<std::mutex> lk(g_pool_call);
    GzBatch* b = static_cast<GzBatch*>(h);
    const int n = (int)b->g.size();
    const int M = b->mo.max_moves;
    const int D = 6 * 6 / 2 * 7 + NFEAT + 1;
    std::vector<std::string> errs(n);
    std::vector<int> mm(n, 0);
    pool().run(n, threads, [&](int i) {
        GzGame& gm = b->g[i];
        counts[i] = 0;
        if (gm.done) return;
        try {
            // the FW opponent's turns (possibly several: a multi-jump) are played here, without a tree
            while (!gm.done && !gm.tree && gm.fw_side >= 0 && gm.env->turn == gm.fw_side) {
                std::vector<MoveArr> legal;
                gm.env->legal_moves(legal);
                if (legal.empty()) {
                    gm.done = true;
                    gm.result = 0;
                    break;
                }
                int pick = 0;
                if (legal.size() > 1) {
                    Searcher srch(b->fw_opts);
                    SearchResult r = srch.get_move(*gm.env, &legal);
                    pick = r.chosen >= 0 ? r.chosen : 0;
                }
                gm.env = sp_child(*gm.env, legal[pick], gm.rng);
                gm.ply += 1;
                if (gm.env->terminal()) {
                    gm.done = true;
                    int w = gm.env->winner();
                    gm.result = (w == 1 || w == -1) ? w : 0;
                } else if (gm.ply >= b->max_plies) {
                    gm.done = true;
                    gm.result = 0;
                }
            }
            if (gm.done) return;
            if (!gm.tree) {
                MctsOpts mo = b->mo;
                mo.seed = b->mo.seed * 0x9E3779B97F4A7C15ULL + (uint64_t)i * 1000003ULL + (uint64_t)gm.searches;
                gm.tree.reset(new Mcts(*gm.env, std::vector<MoveArr>(), mo));
                gm.searches += 1;
            }
            size_t off = (size_t)i * cap_each;
            int m = 0;
            counts[i] = gm.tree->collect(max_leaves_each, cap_each, x + off * D, mv + off * 3 * (size_t)M,
                                         lens + off * (size_t)M, &m);
            mm[i] = m;
        } catch (const std::exception& ex) {
            errs[i] = ex.what();
            counts[i] = 0;
        }
    });
    int total = 0, mx = 0;
    for (int i = 0; i < n; ++i) {
        if (!errs[i].empty()) {
            set_err(err, errcap, errs[i].c_str());
            return -1;
        }
        size_t src = (size_t)i * cap_each;
        if ((size_t)total != src && counts[i] > 0) {
            std::memmove(x + (size_t)total * D, x + src * D, sizeof(float) * (size_t)counts[i] * D);
            std::memmove(mv + (size_t)total * M * 3, mv + src * M * 3, sizeof(int32_t) * (size_t)counts[i] * M * 3);
            std::memmove(lens + (size_t)total * M, lens + src * M, sizeof(int32_t) * (size_t)counts[i] * M);
        }
        b->rows_of[i] = counts[i];
        total += counts[i];
        mx = std::max(mx, mm[i]);
    }
    *mmax = mx;
    return total;
}

// The network's answers for the last cq_gz_collect (packed rows). Then every tree that has used its budget plays
// its move. Returns the games still live, or -1.
CQ_API int cq_gz_apply(void* h, const float* values, const float* priors, int stride, int threads, char* err,
                       int errcap) {
    std::lock_guard<std::mutex> lk(g_pool_call);
    GzBatch* b = static_cast<GzBatch*>(h);
    const int n = (int)b->g.size();
    std::vector<size_t> off(n + 1, 0);
    for (int i = 0; i < n; ++i) off[i + 1] = off[i] + b->rows_of[i];
    std::vector<std::string> errs(n);
    std::vector<int> played(n, 0);
    pool().run(n, threads, [&](int i) {
        GzGame& gm = b->g[i];
        if (gm.done || !gm.tree) return;
        try {
            if (b->rows_of[i] > 0)
                gm.tree->apply(values + off[i], priors + off[i] * stride, stride, b->rows_of[i]);
            if (!gm.tree->done() || gm.tree->pending_batches() > 0) return;
            Mcts::GumbelOut go = gm.tree->gumbel();
            auto info = gm.tree->root_info();
            if (go.chosen < 0 || info.empty()) {  // no move: the game ends here
                gm.done = true;
                gm.result = 0;
                gm.tree.reset();
                return;
            }
            if (!(gm.frozen_side >= 0 && gm.env->turn == gm.frozen_side)) {  // samples: the current network's moves
                GzSample smp;
                std::vector<double> xd((size_t)gm.env->N * 7 + NFEAT + 1);
                encode_position(*gm.env, xd.data());
                smp.x.assign(xd.begin(), xd.end());
                for (auto& r : info) smp.moves.push_back(r.move);
                smp.policy.assign(go.policy.begin(), go.policy.end());
                smp.turn = gm.env->turn;
                gm.samples.push_back(std::move(smp));
            }
            gm.env = sp_child(*gm.env, info[go.chosen].move, gm.rng);
            gm.tree.reset();
            gm.ply += 1;
            played[i] = 1;
            if (gm.env->terminal()) {
                gm.done = true;
                int w = gm.env->winner();
                gm.result = (w == 1 || w == -1) ? w : 0;
            } else if (gm.ply >= b->max_plies) {
                gm.done = true;
                gm.result = 0;
            }
        } catch (const std::exception& ex) {
            errs[i] = ex.what();
        }
    });
    // finished games: their samples to the done list; the slot restarts while there are games left to start
    for (int i = 0; i < n; ++i) {
        GzGame& gm = b->g[i];
        if (!gm.done || !errs[i].empty()) continue;
        if (!gm.samples.empty() || gm.ply > 0) {
            GzDone d;
            d.samples.swap(gm.samples);
            d.result = gm.result;
            d.id = b->ids[i];
            b->done_games.push_back(std::move(d));
            gm.ply = -1;  // recorded
        }
        if (b->to_start > 0) {
            try {
                gm.env = gz_opening(gm.rng, b->open_lo, b->open_hi);
                gm.tree.reset();
                gm.done = false;
                gm.result = 0;
                gm.ply = 0;
                b->ids[i] = b->next_id++;
                b->to_start -= 1;
            } catch (const std::exception& ex) {
                errs[i] = ex.what();
            }
        }
    }
    int live = 0;
    for (int i = 0; i < n; ++i) {
        if (!errs[i].empty()) {
            set_err(err, errcap, errs[i].c_str());
            return -1;
        }
        b->moves_played += played[i];
        live += b->g[i].done ? 0 : 1;
    }
    return live;
}

// Samples of the finished games not taken yet. First call with x == NULL: *n_samples and *n_moves sizes. Then with
// buffers: x (n_samples x in_dim), moves (n_moves x 4: len, a, b, c), policy (n_moves), mstart (n_samples + 1), z
// (n_samples: the game's result for the side to move at the sample: 1, 0, -1), game (n_samples). Taken games' samples
// are cleared. Returns the number of finished games taken.
CQ_API int cq_gz_take(void* h, float* x, int32_t* moves, float* policy, int64_t* mstart, int8_t* z, int32_t* game,
                      int64_t* n_samples, int64_t* n_moves) {
    GzBatch* b = static_cast<GzBatch*>(h);
    int64_t S = 0, MV = 0;
    for (const GzDone& gm : b->done_games)
        for (const GzSample& s : gm.samples) {
            S += 1;
            MV += (int64_t)s.moves.size();
        }
    *n_samples = S;
    *n_moves = MV;
    if (!x) return 0;
    int64_t si = 0, mi = 0;
    int taken = 0;
    mstart[0] = 0;
    for (GzDone& gm : b->done_games) {
        const int32_t gi = gm.id;
        for (const GzSample& s : gm.samples) {
            std::memcpy(x + si * (int64_t)s.x.size(), s.x.data(), sizeof(float) * s.x.size());
            for (size_t k = 0; k < s.moves.size(); ++k) {
                const MoveArr& m = s.moves[k];
                moves[4 * mi] = m.n;
                for (int j = 0; j < 3; ++j) moves[4 * mi + 1 + j] = j < m.n ? m.v[j] : 0;
                policy[mi] = s.policy[k];
                ++mi;
            }
            int zw = gm.result;
            z[si] = (int8_t)(s.turn == WHITE ? zw : -zw);
            game[si] = gi;
            ++si;
            mstart[si] = mi;
        }
        ++taken;
    }
    b->done_games.clear();
    return taken;
}

// live games, finished games, moves played so far
CQ_API void cq_gz_status(void* h, int32_t* live, int32_t* finished, int64_t* moves_played) {
    GzBatch* b = static_cast<GzBatch*>(h);
    int l = 0;
    for (const GzGame& gm : b->g) l += gm.done ? 0 : 1;
    *live = l;
    *finished = (int32_t)(b->next_id - l - (int32_t)b->to_start * 0);  // games started and no longer live
    *moves_played = b->moves_played;
}

// Games still to start after the current ones (refill): a finished game's slot restarts while this is > 0.
CQ_API void cq_gz_set_refill(void* h, int64_t games) { static_cast<GzBatch*>(h)->to_start = games; }

}  // extern "C"
