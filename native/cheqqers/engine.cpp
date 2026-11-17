// Port of src/env/cheqqers_official.py. See engine.h for the identity contract.
#include "engine.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace cq {

namespace {

constexpr Cx ONE{1.0, 0.0};
constexpr Cx ZERO{0.0, 0.0};
constexpr int DIRS[4][2] = {{-1, 1}, {1, 1}, {-1, -1}, {1, -1}};  // DIRECTIONS (dx, dy)

enum Role : int8_t { SOURCE = 0, LANDING = 1, JUMPED = 2, TRANSIT = 3 };
enum Op : int { OP_MOVE = 0, OP_SPLIT = 1, OP_MERGE = 2 };

// Two lookup tables key -> position in an insertion-ordered vector; every user restores the entries it set to -1.
// A small insertion-ordered dict {state: amp} for the subspace operations (at most 8 keys).
struct Small {
    int n = 0;
    uint32_t k[8];
    Cx v[8];
    int find(uint32_t key) const {
        for (int i = 0; i < n; ++i)
            if (k[i] == key) return i;
        return -1;
    }
    void add(uint32_t key, Cx val) {  // get + v
        int i = find(key);
        if (i < 0) {
            if (n >= 8) throw Unsupported("small dict overflow");
            k[n] = key;
            v[n++] = cadd(ZERO, val);
        } else {
            v[i] = cadd(v[i], val);
        }
    }
    void set(uint32_t key, Cx val) {
        int i = find(key);
        if (i < 0) {
            if (n >= 8) throw Unsupported("small dict overflow");
            k[n] = key;
            v[n++] = val;
        } else {
            v[i] = val;
        }
    }
    void clean() {
        int w = 0;
        for (int r = 0; r < n; ++r)
            if (keep_amp(v[r])) {
                k[w] = k[r];
                v[w++] = v[r];
            }
        n = w;
    }
};

struct OBranch {  // a branch of get_move_outcomes: (prob, env, failed)
    double p;
    EnvPtr env;
    bool failed;
};
struct Scratch {
    int32_t a[32768];
    int32_t b[32768];
    std::vector<OBranch> act, nxt;  // get_move_outcomes is never re-entered on one thread
    AmpVec out, next;               // the joint-system operations: results are swapped in, old storage comes back
    std::vector<uint32_t> gspec;
    std::vector<Small> groups;
    AmpVec t1, t2;  // tensor products and collapses (create_system, merge_systems, measure_target)
    // storage of dropped systems' lists, reused by new systems (a pooled Env drops its surplus systems on clone)
    std::vector<AmpVec> amp_pool;
    std::vector<std::vector<int8_t>> ent_pool;
    Scratch() {
        for (int i = 0; i < 32768; ++i) a[i] = b[i] = -1;
    }
};
// One scratch per thread, reached through an 8-byte initial-exec TLS pointer: a large thread_local in a dlopen'ed
// library would go through __tls_get_addr on every access.
__attribute__((tls_model("initial-exec"))) thread_local Scratch* g_scratch_ptr = nullptr;
inline Scratch& scratch() {
    if (!g_scratch_ptr) g_scratch_ptr = new Scratch();  // one per thread, kept for the thread's life
    return *g_scratch_ptr;
}

constexpr size_t VEC_POOL_MAX = 1 << 14;
inline void recycle_system(System& S) {
    Scratch& sc = scratch();
    if (S.amps.capacity() && sc.amp_pool.size() < VEC_POOL_MAX) {
        sc.amp_pool.emplace_back();
        sc.amp_pool.back().swap(S.amps);
    }
    if (S.ents.capacity() && sc.ent_pool.size() < VEC_POOL_MAX) {
        sc.ent_pool.emplace_back();
        sc.ent_pool.back().swap(S.ents);
    }
}
// a new, empty system at the end of `systems`, with recycled list storage
inline System& new_system(std::vector<System>& systems) {
    systems.emplace_back();
    System& S = systems.back();
    Scratch& sc = scratch();
    if (!sc.amp_pool.empty()) {
        S.amps.swap(sc.amp_pool.back());
        sc.amp_pool.pop_back();
        S.amps.clear();
    }
    if (!sc.ent_pool.empty()) {
        S.ents.swap(sc.ent_pool.back());
        sc.ent_pool.pop_back();
        S.ents.clear();
    }
    return S;
}

inline void check_key(uint32_t k) {
    if (k >= 32768u) throw Unsupported("joint key beyond 15 bits");
}

// d[k] = d.get(k, 0j) + v
inline void acc_add(AmpVec& out, int32_t* pos, uint32_t k, Cx v) {
    check_key(k);
    int32_t p = pos[k];
    if (p < 0) {
        pos[k] = (int32_t)out.size();
        out.push_back(Amp{k, cadd(ZERO, v)});
    } else {
        out[p].v = cadd(out[p].v, v);
    }
}
// d[k] = v
inline void acc_set(AmpVec& out, int32_t* pos, uint32_t k, Cx v) {
    check_key(k);
    int32_t p = pos[k];
    if (p < 0) {
        pos[k] = (int32_t)out.size();
        out.push_back(Amp{k, v});
    } else {
        out[p].v = v;
    }
}
inline void acc_release(const AmpVec& out, int32_t* pos) {
    for (const Amp& a : out) pos[a.k] = -1;
}
// clean_amplitudes (cheqqers_official.py:82)
inline void clean(AmpVec& a) {
    size_t w = 0;
    for (size_t r = 0; r < a.size(); ++r)
        if (keep_amp(a[r].v)) a[w++] = a[r];
    a.resize(w);
}

const Cx R{INV_SQRT2_JS, 0.0};
const Cx NEG_R{-INV_SQRT2_JS, 0.0};

// QuantumOperation.sparse_transition for the three BoardGameHelpers operations (cheqqers_official.py:210, 609-619)
void apply_sparse(int op, const Small& in, Small& out) {
    out.n = 0;
    for (int i = 0; i < in.n; ++i) {
        uint32_t k = in.k[i];
        Cx amp = in.v[i];
        if (op == OP_MOVE) {
            if (k == 2) {
                out.add(1, cmul(amp, ONE));
                continue;
            }
        } else if (op == OP_SPLIT) {
            if (k == 4) {
                out.add(2, cmul(amp, R));
                out.add(1, cmul(amp, R));
                continue;
            }
        } else {  // OP_MERGE
            if (k == 4) {
                out.add(1, cmul(amp, R));
                out.add(6, cmul(amp, R));
                continue;
            }
            if (k == 2) {
                out.add(1, cmul(amp, R));
                out.add(6, cmul(amp, NEG_R));
                continue;
            }
        }
        out.add(k, amp);
    }
    out.clean();
}

struct TopoTable {
    Topo t[11];
    TopoTable() {
        for (int bs = 2; bs <= 10; bs += 2) {
            Topo& T = t[bs];
            T.bs = bs;
            T.N = bs * bs / 2;
            T.spr = bs / 2;
            for (int c = 0; c < 10; ++c)
                for (int r = 0; r < 10; ++r) T.idx_at[c][r] = -1;
            for (int idx = 0; idx < T.N; ++idx) {
                int visual_row = idx / T.spr;
                int row = (bs - 1) - visual_row;
                int cir = idx % T.spr;
                int col = (row % 2 == 0) ? cir * 2 : cir * 2 + 1;
                T.col[idx] = col;
                T.row[idx] = row;
                T.idx_at[col][row] = idx;
            }
            for (int idx = 0; idx < T.N; ++idx)
                for (int d = 0; d < 4; ++d) {
                    int n = 0;
                    int cx = T.col[idx] + DIRS[d][0], cy = T.row[idx] + DIRS[d][1];
                    while (0 <= cx && cx < bs && 0 <= cy && cy < bs) {
                        T.ray[idx][d][n++] = (int8_t)T.idx_at[cx][cy];
                        cx += DIRS[d][0];
                        cy += DIRS[d][1];
                    }
                    T.nray[idx][d] = n;
                }
            for (int idx = 0; idx < T.N; ++idx) {
                T.adj[idx][0] = T.adj[idx][1] = T.adj[idx][2] = 0;
                for (int d = 0; d < 4; ++d) {
                    if (T.nray[idx][d] == 0) continue;
                    uint64_t bit = 1ULL << T.ray[idx][d][0];
                    T.adj[idx][2] |= bit;
                    if (DIRS[d][1] == 1) T.adj[idx][0] |= bit;   // white men move up (fwd +1)
                    if (DIRS[d][1] == -1) T.adj[idx][1] |= bit;  // black men move down (fwd -1)
                }
            }
        }
    }
};

// f"{occ:.3f}" as thousandths, exactly (correct rounding of the binary value, as CPython and glibc both do)
int thousandths(double occ) {
    double t = occ * 1000.0;
    double fl = std::floor(t);
    double fr = t - fl;
    if (std::fabs(fr - 0.5) < 1e-6) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.3f", occ);
        const char* dot = std::strchr(buf, '.');
        int ip = std::atoi(buf);
        int frac = dot ? std::atoi(dot + 1) : 0;
        return ip * 1000 + frac;
    }
    return (int)(fr < 0.5 ? fl : fl + 1.0);
}

inline int pyabs(int x) { return x < 0 ? -x : x; }

}  // namespace

const Topo* get_topo(int board_size) {
    static const TopoTable table;
    if (board_size < 2 || board_size > 10 || board_size % 2 != 0) throw Unsupported("board size");
    return &table.t[board_size];
}

MoveArr to_array(const Move& m) {
    MoveArr a;
    if (m.kind == 0) {
        a.n = 2;
        a.v[0] = m.a;
        a.v[1] = m.b;
    } else {
        a.n = 3;
        a.v[0] = m.a;
        a.v[1] = m.b;
        a.v[2] = m.c;
    }
    return a;
}

// ---------------------------------------------------------------------------------------------------------------
// Construction, reset, clone (cheqqers_official.py:741, 882, 938)
// ---------------------------------------------------------------------------------------------------------------

namespace {
__attribute__((tls_model("initial-exec"))) thread_local std::vector<Env*>* g_pool = nullptr;
constexpr size_t POOL_MAX = 1 << 14;
}  // namespace

void EnvRecycler::operator()(Env* e) const {
    if (!e) return;
    if (!g_pool) g_pool = new std::vector<Env*>();
    if (g_pool->size() < POOL_MAX)
        g_pool->push_back(e);
    else
        delete e;
}

EnvPtr make_env() {
    if (g_pool && !g_pool->empty()) {
        Env* e = g_pool->back();
        g_pool->pop_back();
        return EnvPtr(e);
    }
    return EnvPtr(new Env());
}

Env::Env(int board_size, int start_rows_, int game_type_) { init(board_size, start_rows_, game_type_); }

void Env::init(int board_size, int start_rows_, int game_type_) {
    cache_enabled = false;
    topo = get_topo(board_size);
    bs = board_size;
    N = topo->N;
    start_rows = start_rows_ < 0 ? std::max(1, bs / 2 - 1) : start_rows_;
    game_type = (int8_t)std::max(0, std::min(3, game_type_));
    reset();
}

void Env::reset() {
    invalidate_caches();
    turn = WHITE;
    for (System& S : systems) recycle_system(S);
    systems.clear();
    for (int i = 0; i < N; ++i) {
        sq[i].sys = -1;
        sq[i].amps.set1(0, ONE);
        props[i].present = false;
    }
    next_piece_id = 1;
    if (start_rows > 0) {
        // shift(1) applied to {0: 1}: {1: 1}
        for (int i = 0; i < start_rows * topo->spr; ++i) {
            sq[i].amps.set1(1, ONE);
            props[i].present = true;
            props[i].color = BLACK;
            props[i].crowned = false;
            props[i].nids = 1;
            props[i].ids[0] = next_piece_id++;
        }
        for (int i = N - start_rows * topo->spr; i < N; ++i) {
            sq[i].amps.set1(1, ONE);
            props[i].present = true;
            props[i].color = WHITE;
            props[i].crowned = false;
            props[i].nids = 1;
            props[i].ids[0] = next_piece_id++;
        }
    }
    jumped.n = 0;
    moves_since_take = 0;
    mj = -1;
    just_crowned = false;
    history.clear();
    consecutive_queen_moves = 0;
    is_calculating_depth = false;
    disable_depth_calculation = false;
    force_all_moves = false;
    state = IN_PROGRESS;
    has_override = false;
    override_moves.clear();
}

void Env::clone_into(Env& e) const {
    e.topo = topo;
    e.bs = bs;
    e.N = N;
    e.start_rows = start_rows;
    e.game_type = game_type;
    e.turn = turn;
    e.next_piece_id = next_piece_id;
    e.jumped = jumped;
    e.moves_since_take = moves_since_take;
    e.mj = mj;
    e.just_crowned = just_crowned;
    e.history = history;
    e.consecutive_queen_moves = consecutive_queen_moves;
    e.is_calculating_depth = false;
    e.disable_depth_calculation = disable_depth_calculation;
    e.force_all_moves = force_all_moves;
    e.state = state;
    e.has_override = has_override;
    e.override_moves = override_moves;
    e.cache_enabled = cache_enabled;
    e.probs_valid = false;
    e.moves_valid = false;
    e.capture_cache = -1;
    e.takes_valid = false;
    e.bkey_valid = false;
    for (int i = 0; i < N; ++i) {
        e.sq[i].sys = -1;
        e.sq[i].amps.set1(0, ONE);
        e.props[i] = props[i];
    }
    // systems in the order the squares reach them; every listed entity points to the copy (also a stale one)
    int nsys = (int)systems.size();
    int16_t remap_buf[64];
    std::vector<int16_t> remap_vec;
    int16_t* remap = remap_buf;
    if (nsys > 64) {
        remap_vec.assign(nsys, -1);
        remap = remap_vec.data();
    } else {
        for (int s = 0; s < nsys; ++s) remap[s] = -1;
    }
    int16_t order[MAXSQ];
    int used = 0;
    for (int i = 0; i < N; ++i) {
        int s = sq[i].sys;
        if (s >= 0 && remap[s] < 0) {
            remap[s] = (int16_t)used;
            order[used++] = (int16_t)s;
        }
    }
    for (size_t k = used; k < e.systems.size(); ++k) recycle_system(e.systems[k]);
    e.systems.resize(used);  // the copies reuse the capacity of the systems a pooled Env had
    for (int k = 0; k < used; ++k) {
        const System& S = systems[order[k]];
        System& D = e.systems[k];
        D.ents = S.ents;
        D.amps = S.amps;
        D.sums_valid = S.sums_valid;
        if (S.sums_valid) {
            D.or_k = S.or_k;
            D.or_notk = S.or_notk;
            size_t n = S.ents.size();
            std::memcpy(D.s0, S.s0, n * sizeof(double));
            std::memcpy(D.s1, S.s1, n * sizeof(double));
        }
        for (int8_t ent : S.ents) {
            e.sq[ent].sys = (int16_t)k;
            e.sq[ent].amps.n = 0;
        }
    }
    for (int i = 0; i < N; ++i)
        if (sq[i].sys < 0) e.sq[i].amps = sq[i].amps;
}

EnvPtr Env::clone() const {
    EnvPtr e = make_env();
    clone_into(*e);
    return e;
}

int Env::winner() const {
    if (state == WHITE_WON) return 1;
    if (state == BLACK_WON) return -1;
    if (state == DRAW) return 0;
    return 2;
}

// ---------------------------------------------------------------------------------------------------------------
// Joint systems (cheqqers_official.py:295-603)
// ---------------------------------------------------------------------------------------------------------------

int Env::sys_index_of(int s, int e) const {
    const auto& ents = systems[s].ents;
    for (size_t i = 0; i < ents.size(); ++i)
        if (ents[i] == e) return (int)i;
    throw Unsupported("entity not in its system");
}

// For every entity index j: s1[j] = 0.0, then s1[j] + prob for each amplitude with bit j set, in amps order (the
// Python's `p1 = p1 + prob` from 0.0; its first `0.0 + prob` is the same number); s0[j] likewise for bit j clear.
void System::compute_sums() const {
    int n = (int)ents.size();
    uint32_t ok = 0, onk = 0;
    for (int j = 0; j < n; ++j) s0[j] = s1[j] = 0.0;
    for (const Amp& a : amps) {
        double prob = a.v.re * a.v.re + a.v.im * a.v.im;
        ok |= a.k;
        onk |= ~a.k;
        for (int j = 0; j < n; ++j) {
            if ((a.k >> (n - 1 - j)) & 1)
                s1[j] = s1[j] + prob;
            else
                s0[j] = s0[j] + prob;
        }
    }
    or_k = ok;
    or_notk = onk;
    sums_valid = true;
}

void Env::target_probs(int s, int e, double& p0, double& p1, int& first_key, bool& has0, bool& has1) const {
    const System& S = systems[s];
    int n = (int)S.ents.size();
    int j = sys_index_of(s, e);
    int shift = n - 1 - j;
    if (!S.sums_valid) S.compute_sums();
    p0 = S.s0[j];
    p1 = S.s1[j];
    has0 = (S.or_notk >> shift) & 1;
    has1 = (S.or_k >> shift) & 1;
    first_key = S.amps.empty() ? -1 : (int)((S.amps[0].k >> shift) & 1);
}

void Env::probs_direct(int i, double& p0, double& p1) const {
    if (sq[i].sys >= 0) {
        int fk;
        bool h0, h1;
        target_probs(sq[i].sys, i, p0, p1, fk, h0, h1);
        return;
    }
    p0 = p1 = 0.0;
    const SqAmps& A = sq[i].amps;
    for (int j = 0; j < A.n; ++j) {
        double prob = A.v[j].re * A.v[j].re + A.v[j].im * A.v[j].im;
        if (A.k[j])
            p1 = prob;
        else
            p0 = prob;
    }
}

double Env::occ_direct(int i) const {
    double p0, p1;
    probs_direct(i, p0, p1);
    return p1;
}
double Env::empty_direct(int i) const {
    double p0, p1;
    probs_direct(i, p0, p1);
    return p0;
}

// _square_probs (cheqqers_official.py:1156)
void Env::square_probs() {
    if (!cache_enabled) {
        for (int i = 0; i < N; ++i) probs_direct(i, P0[i], P1[i]);
        return;
    }
    if (probs_valid) return;
    bool set[MAXSQ] = {false};
    char done_buf[256];
    std::vector<char> done_big;
    char* done = done_buf;
    if (systems.size() > sizeof(done_buf)) {
        done_big.assign(systems.size(), 0);
        done = done_big.data();
    } else {
        std::memset(done_buf, 0, systems.size());
    }
    for (int i = 0; i < N; ++i) {
        if (set[i]) continue;
        int s = sq[i].sys;
        if (s < 0) {  // probs_direct(i) of an independent square, inline
            const SqAmps& A = sq[i].amps;
            double p0 = 0.0, p1 = 0.0;
            for (int j = 0; j < A.n; ++j) {
                double prob = A.v[j].re * A.v[j].re + A.v[j].im * A.v[j].im;
                if (A.k[j])
                    p1 = prob;
                else
                    p0 = prob;
            }
            P0[i] = p0;
            P1[i] = p1;
            continue;  // set[i] is not read again for i
        } else if (!done[s]) {
            done[s] = true;
            // all_target_probabilities: each entity's sums in the same order as get_target_probabilities
            const System& S = systems[s];
            int n = (int)S.ents.size();
            if (!S.sums_valid) S.compute_sums();
            for (int j = 0; j < n; ++j) {
                int e = S.ents[j];
                P0[e] = S.s0[j];
                P1[e] = S.s1[j];
                set[e] = true;
            }
        }
        if (!set[i]) {
            probs_direct(i, P0[i], P1[i]);
            set[i] = true;
        }
    }
    probs_valid = true;
}

int Env::create_system(const int* ents, int n) {
    if (n > 15) throw QuantumLimit();
    // _calculate_tensor_product: no cleaning
    Scratch& sc = scratch();
    AmpVec& cur = sc.t1;
    AmpVec& next = sc.t2;
    cur.clear();
    cur.push_back(Amp{0, ONE});
    for (int t = 0; t < n; ++t) {
        next.clear();
        const SqAmps& A = sq[ents[t]].amps;
        for (const Amp& a1 : cur) {
            uint32_t base = a1.k << 1;
            for (int j = 0; j < A.n; ++j) next.push_back(Amp{base | A.k[j], cmul(a1.v, A.v[j])});
        }
        cur.swap(next);
    }
    // the tensor product assigns d[base | k2]: keys are distinct, the push order is the insertion order
    int id = (int)systems.size();
    System& S = new_system(systems);
    S.ents.assign(ents, ents + n);
    S.amps.swap(cur);
    for (int t = 0; t < n; ++t) {
        sq[ents[t]].sys = (int16_t)id;
        sq[ents[t]].amps.n = 0;
    }
    return id;
}

// JointSystem.merge (cheqqers_official.py:328)
int Env::merge_systems(const int* inv, int ninv) {
    int uniq[MAXSQ], nu = 0, stand[MAXSQ], ns = 0;
    for (int i = 0; i < ninv; ++i) {
        int s = sq[inv[i]].sys;
        if (s >= 0) {
            bool seen = false;
            for (int j = 0; j < nu; ++j)
                if (uniq[j] == s) seen = true;
            if (!seen) uniq[nu++] = s;
        } else {
            stand[ns++] = inv[i];
        }
    }
    if (nu == 0) return create_system(inv, ninv);
    if (nu == 1 && ns == 0) return uniq[0];
    size_t total = ns;
    for (int j = 0; j < nu; ++j) total += systems[uniq[j]].ents.size();
    if (total > 15) throw QuantumLimit();
    int8_t merged[16];
    int nm = 0;
    for (int j = 0; j < nu; ++j)
        for (int8_t e : systems[uniq[j]].ents) merged[nm++] = e;
    for (int j = 0; j < ns; ++j) merged[nm++] = (int8_t)stand[j];
    Scratch& sc = scratch();
    AmpVec& cur = sc.t1;
    AmpVec& next = sc.t2;
    cur.clear();
    cur.push_back(Amp{0, ONE});
    for (int j = 0; j < nu; ++j) {
        const System& S = systems[uniq[j]];
        int shift = (int)S.ents.size();
        next.clear();
        next.reserve(cur.size() * S.amps.size());
        for (const Amp& a1 : cur) {
            uint32_t base = a1.k << shift;
            for (const Amp& a2 : S.amps) next.push_back(Amp{base | a2.k, cmul(a1.v, a2.v)});
        }
        clean(next);
        cur.swap(next);
    }
    for (int j = 0; j < ns; ++j) {
        const SqAmps& A = sq[stand[j]].amps;
        next.clear();
        for (const Amp& a1 : cur) {
            uint32_t base = a1.k << 1;
            for (int t = 0; t < A.n; ++t) next.push_back(Amp{base | A.k[t], cmul(a1.v, A.v[t])});
        }
        clean(next);
        cur.swap(next);
    }
    int id = (int)systems.size();
    System& NS = new_system(systems);
    NS.ents.assign(merged, merged + nm);
    NS.amps.swap(cur);
    for (int8_t e : systems[id].ents) {
        sq[e].sys = (int16_t)id;
        sq[e].amps.n = 0;
    }
    return id;
}

// JointSystem.apply_to_subspace (cheqqers_official.py:403)
void Env::apply_to_subspace(int s, const int* targets, int m, int op) {
    System& S = systems[s];
    int n = (int)S.ents.size();
    int shifts[3];
    uint32_t mask = 0;
    for (int j = 0; j < m; ++j) {
        shifts[j] = n - 1 - sys_index_of(s, targets[j]);
        mask |= 1u << shifts[j];
    }
    int32_t* gpos = scratch().b;
    std::vector<uint32_t>& gspec = scratch().gspec;
    std::vector<Small>& groups = scratch().groups;
    gspec.clear();
    groups.clear();
    for (const Amp& a : S.amps) {
        uint32_t spec = a.k & ~mask;
        uint32_t tk = 0;
        for (int j = 0; j < m; ++j) tk = (tk << 1) | ((a.k >> shifts[j]) & 1);
        check_key(spec);
        int32_t g = gpos[spec];
        if (g < 0) {
            g = gpos[spec] = (int32_t)groups.size();
            gspec.push_back(spec);
            groups.emplace_back();
        }
        groups[g].set(tk, a.v);
    }
    for (uint32_t sp : gspec) gpos[sp] = -1;
    S.touch();
    AmpVec& out = scratch().out;
    out.clear();
    int32_t* pos = scratch().a;
    Small res;
    for (size_t g = 0; g < groups.size(); ++g) {
        apply_sparse(op, groups[g], res);
        for (int r = 0; r < res.n; ++r) {
            uint32_t rebuilt = gspec[g];
            for (int t = 0; t < m; ++t)
                if ((res.k[r] >> (m - 1 - t)) & 1) rebuilt |= 1u << shifts[t];
            acc_add(out, pos, rebuilt, res.v[r]);
        }
    }
    acc_release(out, pos);
    clean(out);
    S.amps.swap(out);
}

void Env::interact_many(const int* ents, int n, int op) {
    if (n == 0) return;
    int s = merge_systems(ents, n);
    apply_to_subspace(s, ents, n, op);
}

// BoardGameHelpers.capture -> EntanglementEngine.conditional_interact (cheqqers_official.py:556, 643)
void Env::conditional_capture(int attacker, int victim, int target) {
    int all[3], na = 0;
    auto push = [&](int e) {
        for (int i = 0; i < na; ++i)
            if (all[i] == e) return;
        all[na++] = e;
    };
    push(attacker);
    push(victim);
    push(attacker);
    push(target);
    push(victim);
    int s = merge_systems(all, na);
    System& S = systems[s];
    int n = (int)S.ents.size();
    uint32_t cm = 0, ce = 0;
    int ctrl[2] = {attacker, victim};
    for (int c = 0; c < 2; ++c) {
        int sh = n - 1 - sys_index_of(s, ctrl[c]);
        cm |= 1u << sh;
        ce |= 1u << sh;
    }
    int tsh[3] = {n - 1 - sys_index_of(s, attacker), n - 1 - sys_index_of(s, target), n - 1 - sys_index_of(s, victim)};
    AmpVec& out = scratch().out;
    out.clear();
    int32_t* pos = scratch().a;
    for (const Amp& a : S.amps) {
        if ((a.k & cm) == ce) {
            Small branch, next;
            branch.n = 1;
            branch.k[0] = a.k;
            branch.v[0] = a.v;
            for (int t = 0; t < 3; ++t) {
                int sh = tsh[t];
                next.n = 0;
                for (int b = 0; b < branch.n; ++b) {
                    uint32_t ts = (branch.k[b] >> sh) & 1;
                    // shift(1, 2) on {ts: amp}: {(ts + 1) % 2: amp}, no cleaning
                    uint32_t nt = (ts + 1) % 2;
                    uint32_t base = branch.k[b] & ~(1u << sh);
                    next.add(base | ((nt & 1) << sh), branch.v[b]);
                }
                branch = next;
            }
            for (int b = 0; b < branch.n; ++b) acc_add(out, pos, branch.k[b], branch.v[b]);
        } else {
            acc_add(out, pos, a.k, a.v);
        }
    }
    acc_release(out, pos);
    clean(out);
    systems[s].amps.swap(out);
    systems[s].touch();
}

// _apply_phase_pass for one square: conditional_interact([(sq, 1)], [(sq, phase(pi / 4))]) (cheqqers_official.py:1999)
void Env::conditional_phase(int square) {
    int all[1] = {square};
    int s = merge_systems(all, 1);
    System& S = systems[s];
    int n = (int)S.ents.size();
    int sh = n - 1 - sys_index_of(s, square);
    uint32_t cm = 1u << sh, ce = 1u << sh;
    AmpVec& out = scratch().out;
    out.clear();
    int32_t* pos = scratch().a;
    const double omega = (2.0 * M_PI) / 2;
    const double mult = M_PI / 4.0;
    for (const Amp& a : S.amps) {
        if ((a.k & cm) == ce) {
            uint32_t ts = (a.k >> sh) & 1;
            double angle = omega * (double)ts * mult;
            Cx ph{std::cos(angle), std::sin(angle)};
            Cx na = cmul(a.v, ph);
            uint32_t base = a.k & ~(1u << sh);
            if (keep_amp(na)) acc_add(out, pos, base | ((ts & 1) << sh), cadd(ZERO, na));  // next_branch then new_amps
        } else {
            acc_add(out, pos, a.k, a.v);
        }
    }
    acc_release(out, pos);
    clean(out);
    systems[s].amps.swap(out);
    systems[s].touch();
}

// JointSystem.measure_target (cheqqers_official.py:473)
int Env::measure_target(int s, int e, int forced, Rng* rng) {
    System& S = systems[s];
    int n = (int)S.ents.size();
    int shift = n - 1 - sys_index_of(s, e);
    double p0, p1;
    int fk;
    bool h0, h1;
    target_probs(s, e, p0, p1, fk, h0, h1);
    int selected;
    if (forced >= 0) {
        selected = forced;
    } else {
        static Rng fallback;
        double r = (rng ? rng : &fallback)->next();
        double cum = 0.0;
        selected = -1;
        int order[2] = {fk, fk == 0 ? 1 : 0};
        int nk = (int)h0 + (int)h1;
        for (int i = 0; i < nk; ++i) {
            cum += order[i] ? p1 : p0;
            if (r <= cum) {
                selected = order[i];
                break;
            }
        }
        if (selected == -1) selected = nk ? order[nk - 1] : 0;
    }
    double prob = selected == 1 ? (h1 ? p1 : 0.0) : (selected == 0 ? (h0 ? p0 : 0.0) : 0.0);
    AmpVec& collapsed = scratch().t1;
    collapsed.clear();
    if (prob > 1e-12) {
        double norm = 1.0 / std::sqrt(prob);
        for (const Amp& a : S.amps)
            if ((int)((a.k >> shift) & 1) == selected) collapsed.push_back(Amp{a.k, cmul(a.v, Cx{norm, 0.0})});
    } else {
        double max_prob = -1.0;
        int64_t max_k = -1;
        for (const Amp& a : S.amps) {
            double mag = a.v.re * a.v.re + a.v.im * a.v.im;
            if (mag > max_prob) {
                max_prob = mag;
                max_k = a.k;
            }
        }
        uint32_t nk;
        if (max_k != -1)
            nk = selected == 1 ? ((uint32_t)max_k | (1u << shift)) : ((uint32_t)max_k & ~(1u << shift));
        else
            nk = selected == 1 ? (1u << shift) : 0u;
        collapsed.push_back(Amp{nk, ONE});
    }
    clean(collapsed);
    S.amps.swap(collapsed);
    S.touch();
    return selected;
}

double Env::joint_probability_both_occupied(int s, int a, int b) const {
    const System& S = systems[s];
    int n = (int)S.ents.size();
    uint32_t mask = 0, expected = 0;
    int sa = n - 1 - sys_index_of(s, a), sb = n - 1 - sys_index_of(s, b);
    mask |= 1u << sa;
    expected |= 1u << sa;
    mask |= 1u << sb;
    expected |= 1u << sb;
    double prob = 0.0;
    for (const Amp& x : S.amps)
        if ((x.k & mask) == expected) prob += x.v.re * x.v.re + x.v.im * x.v.im;
    return prob;
}

// ---------------------------------------------------------------------------------------------------------------
// Legal moves (cheqqers_official.py:1030-1441)
// ---------------------------------------------------------------------------------------------------------------

bool Env::can_coexist(int i, int j) const {
    int s = sq[i].sys;
    if (s >= 0 && s == sq[j].sys) return !(joint_probability_both_occupied(s, i, j) < 1e-10);
    return true;
}

// The moves of the piece on square i (find_moves' body for one square), appended to out. true: first_only and found.
inline bool Env::piece_moves(int i, bool is_take, bool has_nan_jumped, MoveList& out, bool first_only) {
    const Topo& T = *topo;
    const Props& pp = props[i];
    int fwd = pp.color == WHITE ? 1 : -1;
    bool back = pp.crowned;
    // the flags of the Python's loop (is_empty, is_occ, is_enemy, is_friendly, has_empty_chance), each read
    // only where its branch needs it
    for (int d = 0; d < 4; ++d) {
        int dy = DIRS[d][1];
        if (!is_take && !back && dy != fwd) continue;
        const int nr = T.nray[i][d];
        const int8_t* ray = T.ray[i][d];
        if (is_take) {
            int found = -1;
            for (int r = 0; r < nr; ++r) {
                int cur = ray[r];
                if (found == -1) {
                    double cocc = P1[cur];
                    const Props& cp = props[cur];
                    if (cocc > 0.0 && cp.present) {
                        if (cp.color == pp.color) break;  // is_friendly
                        bool shares = false;               // is_enemy
                        for (int t = 0; t < cp.nids && !shares; ++t) {
                            int32_t pid = cp.ids[t];
                            if (pid == NAN_ID ? has_nan_jumped : jumped.contains(pid)) shares = true;
                        }
                        if (shares) break;
                        if (!can_coexist(i, cur)) break;
                        found = cur;
                    } else if (cocc < 1.0) {  // is_empty
                        if (!back) break;
                    }
                } else {
                    if (P0[cur] > 0.001) {
                        Move m;
                        m.kind = 0;
                        m.take = true;
                        m.a = (int16_t)i;
                        m.b = (int16_t)cur;
                        out.push_back(m);
                        if (first_only) return true;
                        if (!back) break;
                    } else {
                        break;
                    }
                }
            }
        } else {
            for (int r = 0; r < nr; ++r) {
                int cur = ray[r];
                if (P0[cur] > 0.001) {
                    Move m;
                    m.kind = 0;
                    m.take = false;
                    m.a = (int16_t)i;
                    m.b = (int16_t)cur;
                    out.push_back(m);
                    if (first_only) return true;
                    if (!back) break;
                    const Props& cp = props[cur];
                    if (P1[cur] > 0.0 && cp.present && cp.color != pp.color) break;  // is_enemy
                } else {
                    break;
                }
            }
        }
    }
    return false;
}

void Env::find_moves(bool is_take, int restrict, bool ignore_turn, MoveList& out, bool first_only, int only_color) {
    const int want = only_color >= 0 ? only_color : turn;
    out.clear();
    square_probs();
    int lo = 0, hi = N;
    if (restrict >= 0 || restrict < -1) {
        lo = restrict;
        hi = restrict + 1;
    }
    bool has_nan_jumped = jumped.has_nan();
    for (int i = lo; i < hi; ++i) {
        if (i < 0 || i >= N) continue;
        double occ = P1[i];
        if (occ == 0.0) continue;
        const Props& pp = props[i];
        if (!pp.present) continue;
        if (!ignore_turn && pp.color != want) continue;
        if (piece_moves(i, is_take, has_nan_jumped, out, first_only)) return;
    }
}

// find_moves(is_take, -1, false, out, first_only, color) over the squares of scan_masks(): the same squares in the
// same order, without visiting the others
void Env::find_moves_masked(bool is_take, int color, MoveList& out, bool first_only) {
    scan_prep();
    out.clear();
    bool has_nan_jumped = jumped.has_nan();
    for (uint64_t m = pc_mask[color]; m; m &= m - 1)
        if (piece_moves(__builtin_ctzll(m), is_take, has_nan_jumped, out, first_only)) return;
}

// Per position (cache_enabled): the squares find_moves would expand for each colour (occ != 0, present) and the
// squares with an empty chance (P0 > 0.001). Reset with the capture lists.
void Env::scan_prep() {
    if (takes_valid) return;
    takes_col_valid[0] = takes_col_valid[1] = false;
    takes_all_valid = false;
    square_probs();
    uint64_t pc0 = 0, pc1 = 0, ech = 0;
    for (int i = 0; i < N; ++i) {
        uint64_t bit = 1ULL << i;
        if (P0[i] > 0.001) ech |= bit;
        const Props& pp = props[i];
        if (P1[i] == 0.0 || !pp.present) continue;
        if (pp.color == WHITE)
            pc0 |= bit;
        else
            pc1 |= bit;
    }
    pc_mask[0] = pc0;
    pc_mask[1] = pc1;
    ech_mask = ech;
    takes_valid = true;
}

// find_moves(false, -1, false, out, first_only=true) is non-empty exactly when a piece of the side to move has, in
// one of its directions, a first square with an empty chance (the scan pushes that move and stops)
bool Env::any_plain_move() {
    scan_prep();
    const Topo& T = *topo;
    for (uint64_t m = pc_mask[turn]; m; m &= m - 1) {
        int i = __builtin_ctzll(m);
        const Props& pp = props[i];
        uint64_t adj = T.adj[i][pp.crowned ? 2 : pp.color];
        if (adj & ech_mask) return true;
    }
    return false;
}

void Env::find_split_and_merge(const MoveList& cm, MoveList& out) {
    MoveList& splits = b_split;
    MoveList& merges = b_merge;
    std::vector<uint32_t>& seen_s = b_seen_s;
    std::vector<uint32_t>& seen_m = b_seen_m;
    splits.clear();
    merges.clear();
    seen_s.clear();
    seen_m.clear();
    size_t n = cm.size();
    for (size_t i = 0; i < n; ++i) {
        int fi = cm[i].a, ti = cm[i].b;
        for (size_t j = i + 1; j < n; ++j) {
            if (fi == cm[j].a && ti != cm[j].b) {
                int tj = cm[j].b;
                uint32_t key = ((uint32_t)fi << 16) | ((uint32_t)std::min(ti, tj) << 8) | (uint32_t)std::max(ti, tj);
                if (std::find(seen_s.begin(), seen_s.end(), key) == seen_s.end()) {
                    seen_s.push_back(key);
                    Move m;
                    m.kind = 1;
                    m.a = (int16_t)fi;
                    m.b = (int16_t)ti;
                    m.c = (int16_t)tj;
                    splits.push_back(m);
                }
            }
        }
    }
    square_probs();  // _occ_probs()
    for (size_t i = 0; i < n; ++i) {
        int fi = cm[i].a, ti = cm[i].b;
        for (size_t j = i + 1; j < n; ++j) {
            int fj = cm[j].a;
            if (ti == cm[j].b && fi != fj) {
                if (P1[fi] > 0.99 && P1[fj] > 0.99) continue;
                uint32_t key = ((uint32_t)std::min(fi, fj) << 16) | ((uint32_t)std::max(fi, fj) << 8) | (uint32_t)ti;
                if (std::find(seen_m.begin(), seen_m.end(), key) == seen_m.end()) {
                    seen_m.push_back(key);
                    Move m;
                    m.kind = 2;
                    m.a = (int16_t)fi;
                    m.b = (int16_t)fj;
                    m.c = (int16_t)ti;
                    merges.push_back(m);
                }
            }
        }
    }
    out = splits;
    out.insert(out.end(), merges.begin(), merges.end());
}

void Env::sim_state(SimState& st) {
    st.bs = bs;
    st.turn = turn;
    st.next_pid = next_piece_id;
    st.mj = mj;
    st.jumped = jumped;
    square_probs();  // get_minimal_piece_map -> _occ_probs()
    for (int i = 0; i < N; ++i) {
        if (P1[i] > 0.001 && props[i].present)
            st.pm[i] = props[i];
        else
            st.pm[i].present = false;
    }
}

// calculate_max_capture_depth (cheqqers_official.py:1341)
int Env::calc_max_capture_depth(const SimState& st, const Move& init) const {
    EnvPtr simp = make_env();
    Env& sim = *simp;
    sim.init(st.bs, 0, CLASSIC);
    // The Python's simulation runs uncached; its squares are all classical (no joint system), where the cached
    // and the uncached probabilities are the same numbers.
    sim.cache_enabled = true;
    sim.disable_depth_calculation = true;
    sim.turn = (int8_t)st.turn;
    sim.next_piece_id = st.next_pid;
    sim.jumped = st.jumped;
    sim.mj = (int16_t)st.mj;
    for (int idx = 0; idx < sim.N; ++idx) {
        if (st.pm[idx].present) {
            sim.sq[idx].amps.set1(1, ONE);
            sim.props[idx] = st.pm[idx];
        }
    }
    MoveList& moves = sim.b_cd;
    sim.possible_moves_skip_depth(moves);
    int k = -1;
    for (size_t i = 0; i < moves.size(); ++i)
        if (moves[i].kind == 0 && moves[i].a == init.a && moves[i].b == init.b) {
            k = (int)i;
            break;
        }
    if (k == -1) return 0;
    Move chosen = moves[k];
    sim.apply_move(chosen, nullptr, 0, false, nullptr);
    if (sim.mj < 0) return 1;
    MoveList& next = sim.b_cd;
    sim.possible_moves_skip_depth(next);
    if (next.empty()) return 1;
    SimState ns;
    sim.sim_state(ns);
    int best = 0;
    for (const Move& nm : next) best = std::max(best, calc_max_capture_depth(ns, nm));
    return 1 + best;
}

void Env::compute_possible_moves(bool skip_depth, bool ignore_turn, bool ignore_forced_captures, MoveList& out) {
    bool ignore_forced = ignore_forced_captures || force_all_moves;
    MoveList& take = b_take;
    if (cache_enabled && !ignore_turn)
        takes_for_turn(mj, take);
    else
        find_moves(true, mj, ignore_turn, take);
    auto filter = [&](MoveList& v) {
        if (game_type == CLASSIC) {
            v.erase(std::remove_if(v.begin(), v.end(), [](const Move& m) { return m.kind != 0; }), v.end());
        } else if (game_type == SUPERPOSITION || game_type == ENTANGLEMENT) {
            v.erase(std::remove_if(v.begin(), v.end(), [](const Move& m) { return m.kind == 2; }), v.end());
        }
    };
    MoveList& std_moves = b_std;
    MoveList& sm = b_sm;
    if (ignore_forced) {
        find_moves(false, -1, ignore_turn, std_moves);
        find_split_and_merge(std_moves, sm);
        out = std_moves;
        out.insert(out.end(), sm.begin(), sm.end());
        out.insert(out.end(), take.begin(), take.end());
        if (!ignore_turn) filter(out);
        return;
    }
    if (!take.empty()) {
        if (!skip_depth && !force_all_moves && !is_calculating_depth && !disable_depth_calculation && take.size() > 1) {
            is_calculating_depth = true;
            SimState st;
            sim_state(st);
            std::vector<int> depths(take.size());
            int maxd = INT32_MIN;
            for (size_t i = 0; i < take.size(); ++i) {
                depths[i] = calc_max_capture_depth(st, take[i]);
                maxd = std::max(maxd, depths[i]);
            }
            out.clear();
            for (size_t i = 0; i < take.size(); ++i)
                if (depths[i] == maxd) out.push_back(take[i]);
            is_calculating_depth = false;
            return;
        }
        out = take;
        return;
    }
    if (cache_enabled && !ignore_turn)
        find_moves_masked(false, turn, std_moves, false);
    else
        find_moves(false, -1, ignore_turn, std_moves);
    find_split_and_merge(std_moves, sm);
    out = std_moves;
    out.insert(out.end(), sm.begin(), sm.end());
    filter(out);
}

const MoveList& Env::possible_moves() {
    if (cache_enabled && !is_calculating_depth) {
        if (moves_valid && moves_flags[0] == force_all_moves && moves_flags[1] == disable_depth_calculation)
            return moves_cache;
        compute_possible_moves(false, false, false, moves_cache);
        moves_valid = true;
        moves_flags[0] = force_all_moves;
        moves_flags[1] = disable_depth_calculation;
        return moves_cache;
    }
    // uncached: computed every call
    compute_possible_moves(false, false, false, moves_tmp);
    return moves_tmp;
}

// The list get_possible_moves(skip_depth_calculation=True) is empty exactly when there is no capture (from the
// open sequence's square, if any) and no plain move: splits and merges are made of plain moves, and the filters by
// game type only drop three-square moves.
const MoveList& Env::takes_of(int color) {
    scan_prep();
    if (!takes_col_valid[color]) {
        find_moves_masked(true, color, takes_col[color], false);
        takes_col_valid[color] = true;
    }
    return takes_col[color];
}

// the scan of both colours visits the squares in order and lists each square's moves together: the merge by square
const MoveList& Env::all_takes() {
    const MoveList& w = takes_of(WHITE);
    const MoveList& b = takes_of(BLACK);
    if (!takes_all_valid) {
        takes_all.clear();
        size_t i = 0, j = 0;
        while (i < w.size() || j < b.size()) {
            if (j == b.size() || (i < w.size() && w[i].a < b[j].a))
                takes_all.push_back(w[i++]);
            else
                takes_all.push_back(b[j++]);
        }
        takes_all_valid = true;
    }
    return takes_all;
}

// find_moves(true, restrict, false, ...) visits the same squares with the same rays as the scan of both colours;
// it only skips the squares outside `restrict` and the pieces of the other colour.
void Env::takes_for_turn(int restrict, MoveList& out) {
    const MoveList& all = takes_of(turn);
    out.clear();
    if (restrict < -1) return;
    for (const Move& m : all)
        if (restrict == -1 || m.a == restrict) out.push_back(m);
}

bool Env::any_take_for_turn(int restrict) {
    const MoveList& all = takes_of(turn);
    if (restrict < -1) return false;
    if (restrict == -1) return !all.empty();
    for (const Move& m : all)
        if (m.a == restrict) return true;
    return false;
}

// The two scans only read the position: their order does not change the answer. The plain one usually stops at the
// first piece; the capture scan visits every piece, so it runs only when there is no plain move.
bool Env::has_any_move_skip_depth() {
    MoveList& t = b_tmp;
    if (cache_enabled) return any_plain_move() || any_take_for_turn(mj);
    find_moves(false, -1, false, t, true);
    if (!t.empty()) return true;
    find_moves(true, mj, false, t, true);
    return !t.empty();
}

bool Env::capture_pending() {
    if (cache_enabled) {
        if (capture_cache >= 0) return capture_cache != 0;
        bool pending = any_take_for_turn(mj);
        capture_cache = pending ? 1 : 0;
        return pending;
    }
    MoveList& t = b_tmp;
    find_moves(true, mj, false, t);
    return !t.empty();
}

void Env::legal_moves(std::vector<MoveArr>& out) {
    out.clear();
    if (state != IN_PROGRESS) return;
    if (has_override) {
        out = override_moves;
        return;
    }
    const MoveList& pm = possible_moves();
    out.reserve(pm.size());
    for (const Move& m : pm) out.push_back(to_array(m));
}

// resolve_move_dict for a tournament array (cheqqers_official.py:1471)
Move Env::resolve(const MoveArr& arr) {
    const MoveList& pm = possible_moves();
    for (const Move& m : pm)  // to_array(m) == arr
        if (m.a == arr.v[0] && m.b == arr.v[1] && (m.kind == 0 ? arr.n == 2 : (arr.n == 3 && m.c == arr.v[2]))) return m;
    auto check = [&](int i) {
        if (i < 0 || i >= N) throw Unsupported("move square out of range");
    };
    if (arr.n == 3) {
        for (const Move& m : pm) {
            if (m.kind == 1 && m.a == arr.v[0]) {
                bool eq = (m.b == arr.v[1] && m.c == arr.v[2]) || (m.b == arr.v[2] && m.c == arr.v[1]) ||
                          (m.b == m.c && arr.v[1] == arr.v[2] && m.b == arr.v[1]);
                if (eq) return m;
            }
            if (m.kind == 2 && m.c == arr.v[2]) {
                bool eq = (m.a == arr.v[0] && m.b == arr.v[1]) || (m.a == arr.v[1] && m.b == arr.v[0]) ||
                          (m.a == m.b && arr.v[0] == arr.v[1] && m.a == arr.v[0]);
                if (eq) return m;
            }
        }
        check(arr.v[1]);
        const Props& p1 = props[arr.v[1]];
        double occ1 = occ_direct(arr.v[1]);
        Move m;
        m.a = arr.v[0];
        m.b = arr.v[1];
        m.c = arr.v[2];
        m.kind = (p1.present && occ1 > 0.001 && p1.color == turn) ? 2 : 1;
        return m;
    }
    if (arr.n == 2) {
        check(arr.v[0]);
        check(arr.v[1]);
        int fx = topo->col[arr.v[0]], fy = topo->row[arr.v[0]];
        int tx = topo->col[arr.v[1]], ty = topo->row[arr.v[1]];
        bool is_take = false;
        if (pyabs(tx - fx) >= 2) {
            int dx = tx > fx ? 1 : -1, dy = ty > fy ? 1 : -1;
            for (int step = 1; step < pyabs(tx - fx); ++step) {
                int cx = fx + step * dx, cy = fy + step * dy;
                if (cx < 0 || cx >= bs || cy < 0 || cy >= bs || topo->idx_at[cx][cy] < 0)
                    throw Unsupported("move not on a diagonal");
                if (occ_direct(topo->idx_at[cx][cy]) > 0.001) {
                    is_take = true;
                    break;
                }
            }
        }
        Move m;
        m.kind = 0;
        m.take = is_take;
        m.a = arr.v[0];
        m.b = arr.v[1];
        return m;
    }
    throw Unsupported("move array length");
}

// ---------------------------------------------------------------------------------------------------------------
// Hashes (cheqqers_official.py:1567-1585)
// ---------------------------------------------------------------------------------------------------------------

void Env::board_key(Key& k) {
    if (cache_enabled && bkey_valid && bkey.v[0] == (uint16_t)turn) {
        k.n = bkey.n;
        std::memcpy(k.v, bkey.v, bkey.n * sizeof(uint16_t));
        return;
    }
    square_probs();
    k.n = (uint8_t)(N + 1);
    k.v[0] = (uint16_t)turn;
    for (int i = 0; i < N; ++i) {
        double occ = P1[i];
        if (occ > 0.0001) {
            const Props& p = props[i];
            int colorB = (p.present && p.color == WHITE) ? 0 : 1;
            int crownK = (p.present && p.crowned) ? 1 : 0;
            int oc = occ > 0.999 ? 1000 : thousandths(occ);
            k.v[1 + i] = (uint16_t)(1 + oc * 4 + colorB * 2 + crownK);
        } else {
            k.v[1 + i] = 0;
        }
    }
    if (cache_enabled) {
        bkey.n = k.n;
        std::memcpy(bkey.v, k.v, k.n * sizeof(uint16_t));
        bkey_valid = true;
    }
}

void Env::state_key(Key& k) {
    board_key(k);
    k.v[N + 1] = (uint16_t)(mj + 1);
    k.n = (uint8_t)(N + 2);
}

// ---------------------------------------------------------------------------------------------------------------
// Applying a move (cheqqers_official.py:1525-2060)
// ---------------------------------------------------------------------------------------------------------------

void Env::involved(const Move& m, int* sqs, int8_t* roles, int& n) {
    n = 0;
    if (m.kind == 1) {
        sqs[n] = m.b;
        roles[n++] = LANDING;
        sqs[n] = m.c;
        roles[n++] = LANDING;
    } else if (m.kind == 2) {
        sqs[n] = m.c;
        roles[n++] = LANDING;
    } else {
        int f = m.a, t = m.b;
        sqs[n] = f;
        roles[n++] = SOURCE;
        sqs[n] = t;
        roles[n++] = LANDING;
        int fc = topo->col[f], fr = topo->row[f], tc = topo->col[t], tr = topo->row[t];
        int dx = tc > fc ? 1 : -1, dy = tr > fr ? 1 : -1;
        bool jumped_found = !m.take;
        for (int step = 1; step < pyabs(tc - fc); ++step) {
            int cx = fc + step * dx, cy = fr + step * dy;
            if (cx < 0 || cx >= bs || cy < 0 || cy >= bs || topo->idx_at[cx][cy] < 0)
                throw Unsupported("move not on a diagonal");
            int ci = topo->idx_at[cx][cy];
            if (empty_direct(ci) < 0.999) {
                sqs[n] = ci;
                roles[n++] = jumped_found ? TRANSIT : JUMPED;
                jumped_found = true;
            }
        }
    }
}

bool Env::blocks_after(int i, int role) const {
    double occ = occ_direct(i);
    if ((role == LANDING || role == TRANSIT) && occ >= 0.999) return true;
    if (role == JUMPED && occ <= 0.001) return true;
    return false;
}

int Env::apply_move(const Move& m, const int* forced, int nforced, bool ignore_turn, Rng* rng) {
    has_override = false;
    override_moves.clear();
    bool was_in_mj = mj >= 0;
    int from_before = m.a;  // from_index / from_index1
    bool was_crowned_before = props[from_before].present && props[from_before].crowned;
    bool was_take = m.kind == 0 && m.take;
    int nmeasured = 0, replay_idx = 0;
    just_crowned = false;
    int sqs[16];
    int8_t roles[16];
    int n;
    involved(m, sqs, roles, n);
    bool failed = false;
    for (int j = 0; j < n; ++j) {
        int si = sqs[j];
        int role = roles[j];
        double occ = occ_direct(si);
        bool is_sup = 0.001 < occ && occ < 0.999;
        bool should = game_type == SUPERPOSITION || role == LANDING || role == TRANSIT;
        if (is_sup && should) {
            int f = -1;
            if (forced != nullptr && replay_idx < nforced) {
                f = forced[replay_idx];
                ++replay_idx;
            }
            int result = measure_square(si, f, rng);
            ++nmeasured;
            if ((role == LANDING || role == TRANSIT) && result == 1) {
                failed = true;
                break;
            }
        }
        if (blocks_after(si, role)) {
            failed = true;
            break;
        }
    }
    finalize(m, failed, was_in_mj, was_crowned_before, was_take, ignore_turn);
    return nmeasured;
}

void Env::finalize(const Move& m, bool failed, bool was_in_mj, bool was_crowned_before, bool was_take,
                   bool ignore_turn) {
    invalidate_caches();
    try {
        if (!failed) {
            if (game_type == CLASSIC && m.kind != 0) game_type = SUPERPOSITION;
            if (m.kind == 0 && m.take) {
                if (apply_take(m)) return;
            } else if (m.kind == 1) {
                apply_split(m);
            } else if (m.kind == 2) {
                apply_merge(m);
            } else {
                apply_classical(m);
            }
            phase_pass();
            cleanup_props();
            disentangle();
            if (ignore_turn) {
                mj = -1;
                jumped.n = 0;
            } else if (!was_take) {
                if (was_crowned_before)
                    consecutive_queen_moves += 1;
                else
                    consecutive_queen_moves = 0;
                turn = (int8_t)(1 - turn);
                mj = -1;
                jumped.n = 0;
                moves_since_take += 1;
            } else {
                moves_since_take = 0;
                consecutive_queen_moves = 0;
                history.clear();
                int landing = m.b;
                MoveList& t = b_tmp;
                bool more = false;
                if (!just_crowned) {
                    find_moves(true, landing, false, t);
                    more = !t.empty();
                }
                if (more) {
                    mj = (int16_t)landing;
                } else {
                    turn = (int8_t)(1 - turn);
                    mj = -1;
                    jumped.n = 0;
                    takes_valid = false;  // the capture scan reads the jumped ids
                }
            }
            check_win_states();
        } else {
            turn = (int8_t)(1 - turn);
            mj = -1;
            jumped.n = 0;
            cleanup_props();
            disentangle();
            check_win_states();
        }
    } catch (const QuantumLimit&) {
        takes_valid = false;
    }
    // the move lists are recomputed; takes_all (made by check_win_states for this very position) is kept
    moves_valid = false;
    capture_cache = -1;
}

int Env::measure_square(int i, int forced, Rng* rng) {
    invalidate_caches();
    int res;
    if (sq[i].sys >= 0) {
        res = measure_target(sq[i].sys, i, forced, rng);
    } else if (forced >= 0) {
        sq[i].amps.set1((uint8_t)forced, ONE);
        res = forced;
    } else {
        static Rng fallback;
        double r = (rng ? rng : &fallback)->next();
        double cum = 0.0;
        int last_k = 0;
        res = -1;
        SqAmps& A = sq[i].amps;
        for (int j = 0; j < A.n; ++j) {
            int k = A.k[j];
            last_k = k;
            cum += A.v[j].re * A.v[j].re + A.v[j].im * A.v[j].im;
            if (r <= cum) {
                res = k;
                break;
            }
        }
        if (res < 0) res = last_k;
        A.set1((uint8_t)res, ONE);
    }
    disentangle();
    cleanup_props();
    return res;
}

// disentangle_collapsed_entities (cheqqers_official.py:1776)
// The caches are dropped only if a square changes: a pass that changes nothing leaves the same position, whose
// probabilities and moves are the same numbers and lists.
void Env::disentangle() {
    char processed_buf[256];
    std::vector<char> processed_big;
    char* processed = processed_buf;
    if (systems.size() > sizeof(processed_buf)) {
        processed_big.assign(systems.size(), 0);
        processed = processed_big.data();
    } else {
        std::memset(processed_buf, 0, systems.size());
    }
    for (int i = 0; i < N; ++i) {
        int s = sq[i].sys;
        if (s < 0 || processed[s]) continue;
        processed[s] = true;
        int dent[16], dstate[16], nd = 0;
        bool all_d = true;
        int8_t ents[16];  // a copy: factor_out below edits the system's list
        int nents = (int)systems[s].ents.size();
        std::memcpy(ents, systems[s].ents.data(), nents);
        for (int t = 0; t < nents; ++t) {
            int8_t e = ents[t];
            double p0, p1;
            int fk;
            bool h0, h1;
            target_probs(s, e, p0, p1, fk, h0, h1);
            int d = -1;
            int order[2] = {fk, fk == 0 ? 1 : 0};
            int nk = (int)h0 + (int)h1;
            for (int j = 0; j < nk; ++j) {
                double p = order[j] ? p1 : p0;
                if (p > 0.999) {
                    d = order[j];
                    break;
                }
            }
            if (d >= 0) {
                dent[nd] = e;
                dstate[nd++] = d;
            } else {
                all_d = false;
            }
        }
        if (nd > 0) invalidate_caches();
        if (all_d) {
            for (int j = 0; j < nd; ++j) {
                sq[dent[j]].sys = -1;
                sq[dent[j]].amps.set1((uint8_t)dstate[j], ONE);
            }
        } else {
            for (int j = 0; j < nd; ++j) {
                int e = dent[j];
                sq[e].sys = -1;
                sq[e].amps.set1((uint8_t)dstate[j], ONE);
                const auto& cur = systems[s].ents;
                auto it = std::find(cur.begin(), cur.end(), (int8_t)e);
                if (it != cur.end()) factor_out(s, (int)(it - cur.begin()), dstate[j]);
            }
        }
    }
}

// _factor_out_entity (cheqqers_official.py:1812)
void Env::factor_out(int s, int idx, int st) {
    System& S = systems[s];
    int n = (int)S.ents.size();
    if (n - 1 == 0) return;
    int shift = n - 1 - idx;
    if (n - 1 == 1) {
        int ent = S.ents[idx == 0 ? 1 : 0];
        int r_shift = 1 - shift;
        SqAmps amps;
        amps.n = 0;
        for (const Amp& a : S.amps) {
            if ((int)((a.k >> shift) & 1) == st) {
                uint8_t rs = (uint8_t)((a.k >> r_shift) & 1);
                int j = -1;
                for (int t = 0; t < amps.n; ++t)
                    if (amps.k[t] == rs) j = t;
                if (j < 0) {
                    amps.k[amps.n] = rs;
                    amps.v[amps.n++] = cadd(ZERO, a.v);
                } else {
                    amps.v[j] = cadd(amps.v[j], a.v);
                }
            }
        }
        sq[ent].sys = -1;
        sq[ent].amps = amps;
        return;
    }
    uint32_t low_mask = (1u << shift) - 1;
    AmpVec& out = scratch().out;
    out.clear();
    int32_t* pos = scratch().a;
    for (const Amp& a : S.amps) {
        if ((int)((a.k >> shift) & 1) != st) continue;
        uint32_t high = a.k >> (shift + 1);
        uint32_t low = a.k & low_mask;
        acc_add(out, pos, (high << shift) | low, a.v);
    }
    acc_release(out, pos);
    S.ents.erase(S.ents.begin() + idx);
    S.amps.swap(out);
    S.touch();
    for (int8_t e : S.ents) sq[e].sys = (int16_t)s;
}

// Only a piece actually removed changes the move lists (the probabilities are untouched).
void Env::cleanup_props() {
    square_probs();
    for (int i = 0; i < N; ++i)
        if (P1[i] <= 0.0001 && props[i].present) {
            invalidate_moves();
            props[i].present = false;
        }
}

void Env::move_props(int f, int t) {
    invalidate_moves();
    if (!props[f].present) return;
    Props cur = props[f];
    props[t] = cur;
    int row = topo->row[t];
    if ((cur.color == WHITE && row == bs - 1) || (cur.color == BLACK && row == 0)) {
        if (!cur.crowned) {
            props[t].crowned = true;
            just_crowned = true;
        }
    }
}

void Env::classical_override(int f, const int* taken, int ntaken, int t) {
    invalidate_caches();
    sq[f].amps.set1(0, ONE);
    sq[f].sys = -1;
    for (int j = 0; j < ntaken; ++j) {
        sq[taken[j]].amps.set1(0, ONE);
        sq[taken[j]].sys = -1;
        props[taken[j]].present = false;
    }
    sq[t].amps.set1(1, ONE);
    sq[t].sys = -1;
    move_props(f, t);
    if (f != t) props[f].present = false;
}

void Env::apply_classical(const Move& m) {
    invalidate_caches();
    int f = m.a, t = m.b;
    if (game_type == CLASSIC) {
        classical_override(f, nullptr, 0, t);
        return;
    }
    if (game_type == SUPERPOSITION && occ_direct(f) < 0.999) measure_square(f, -1, nullptr);
    const int pair_sq[2] = {f, t};
    const uint8_t pair_st[2] = {1, 0};
    if (classical_set(pair_sq, pair_st, 2)) {
        // The joint path below would build the system {f, t} with amplitudes {2: 1}, move it to {1: 1} (f empty, t
        // occupied, both exactly 1+0j), and the disentangle pass at the end of the move would then turn both back
        // into independent squares {0: 1} and {1: 1}. In between, every read of f and t gives the same numbers
        // (probabilities 1.0 and 0.0). No other system lists f or t, so nothing else reads or rewrites them.
        sq[f].amps.set1(0, ONE);
        sq[t].amps.set1(1, ONE);
        move_props(f, t);  // occ_direct(t) = 1.0 > 0.0001
        return;
    }
    int ents[2] = {f, t};
    interact_many(ents, 2, OP_MOVE);
    if (occ_direct(t) > 0.0001) move_props(f, t);
}

// Squares of a move that are all classical and outside every joint system: square sqs[j] is independent with the
// amplitudes {st[j]: 1+0j} exactly, and no system reached from a square lists it (also as a stale entry). Only
// SUPERPOSITION and ENTANGLEMENT: the phase pass of INTERFERENCE would act on the joint system the move builds.
bool Env::classical_set(const int* sqs, const uint8_t* st, int n) const {
    if (game_type != SUPERPOSITION && game_type != ENTANGLEMENT) return false;
    uint64_t mine = 0;
    for (int j = 0; j < n; ++j) {
        const Square& S = sq[sqs[j]];
        if (S.sys >= 0) return false;
        if (!(S.amps.n == 1 && S.amps.k[0] == st[j] && S.amps.v[0].re == 1.0 && S.amps.v[0].im == 0.0)) return false;
        mine |= 1ULL << sqs[j];
    }
    uint64_t seen = 0;  // systems already checked (indices < 64; otherwise every one is checked)
    for (int i = 0; i < N; ++i) {
        int s = sq[i].sys;
        if (s < 0) continue;
        if (s < 64) {
            if (seen >> s & 1) continue;
            seen |= 1ULL << s;
        }
        for (int8_t e : systems[s].ents)
            if (mine >> e & 1) return false;
    }
    return true;
}

bool Env::apply_take(const Move& m) {
    invalidate_caches();
    int f = m.a, t = m.b;
    int fc = topo->col[f], fr = topo->row[f], tc = topo->col[t], tr = topo->row[t];
    int dx = tc > fc ? 1 : -1, dy = tr > fr ? 1 : -1;
    int taken[16], nt = 0;
    for (int s = 1; s < pyabs(tc - fc); ++s) {
        int idx = topo->idx_at[fc + s * dx][fr + s * dy];
        if (occ_direct(idx) > 0.001) taken[nt++] = idx;
    }
    if (nt == 0) return false;
    for (int j = 0; j < nt; ++j) {
        const Props& p = props[taken[j]];
        if (p.present) {
            for (int q = 0; q < p.nids; ++q) {
                if (jumped.n >= MAXJUMPED) throw Unsupported("jumped ids");
                jumped.v[jumped.n++] = p.ids[q];
            }
        }
    }
    if (game_type == CLASSIC) {
        classical_override(f, taken, nt, t);
        return false;
    }
    if (game_type == SUPERPOSITION) measure_square(f, -1, nullptr);
    if (nt == 1) {
        const int tri_sq[3] = {f, taken[0], t};
        const uint8_t tri_st[3] = {1, 1, 0};
        if (classical_set(tri_sq, tri_st, 3)) {
            // conditional_capture would build the system {f, v, t} = {6: 1}, flip the three bits under the control
            // f = v = 1 into {1: 1} (f and v empty, t occupied, exactly 1+0j), and the disentangle pass would make
            // the three squares independent again; every read in between gives probabilities 1.0 and 0.0.
            sq[f].amps.set1(0, ONE);
            sq[taken[0]].amps.set1(0, ONE);
            sq[t].amps.set1(1, ONE);
            move_props(f, t);              // occ_direct(t) = 1.0 > 0.0001
            props[f].present = false;      // occ_direct(f) = 0.0 < 0.001
            props[taken[0]].present = false;
            return false;
        }
    }
    for (int j = 0; j < nt; ++j) conditional_capture(f, taken[j], t);
    if (occ_direct(t) > 0.0001) {
        move_props(f, t);
        if (occ_direct(f) < 0.001) props[f].present = false;
        for (int j = 0; j < nt; ++j)
            if (occ_direct(taken[j]) < 0.001) props[taken[j]].present = false;
    }
    return false;
}

void Env::apply_split(const Move& m) {
    invalidate_caches();
    if (game_type == CLASSIC) game_type = SUPERPOSITION;
    int f = m.a, t1 = m.b, t2 = m.c;
    sq[t1].amps.set1(0, ONE);
    sq[t1].sys = -1;
    sq[t2].amps.set1(0, ONE);
    sq[t2].sys = -1;
    int ents[3] = {f, t1, t2};
    interact_many(ents, 3, OP_SPLIT);
    move_props(f, t1);
    move_props(f, t2);
    props[f].present = false;
}

void Env::apply_merge(const Move& m) {
    invalidate_caches();
    int f1 = m.a, f2 = m.b, t = m.c;
    int ents[3] = {f1, f2, t};
    interact_many(ents, 3, OP_MERGE);
    move_props(f1, t);
    Props& p = props[t];
    if (p.present) {
        int32_t merged[2 * MAXIDS];
        int nm = 0;
        auto add_all = [&](const Props& q) {
            if (!q.present) return;
            for (int i = 0; i < q.nids; ++i) {
                int32_t pid = q.ids[i];
                bool seen = false;
                for (int j = 0; j < nm; ++j)
                    if (merged[j] == pid) seen = true;
                if (!seen) merged[nm++] = pid;
            }
        };
        Props q1 = props[f1], q2 = props[f2];
        add_all(q1);
        add_all(q2);
        if (nm > MAXIDS) throw Unsupported("piece ids");
        p.nids = (uint8_t)nm;
        for (int i = 0; i < nm; ++i) p.ids[i] = merged[i];
    }
    props[f1].present = false;
    props[f2].present = false;
}

void Env::phase_pass() {
    invalidate_caches();
    if (game_type == INTERFERENCE)
        for (int i = 0; i < N; ++i) conditional_phase(i);
}

void Env::check_win_states() {
    double w_prob = 0.0, b_prob = 0.0;
    int w_q = 0, b_q = 0;
    square_probs();
    for (int i = 0; i < N; ++i) {
        double p = P1[i];
        const Props& pr = props[i];
        if (pr.present && p > 0.0001) {
            if (pr.color == WHITE) {
                w_prob += p;
                if (pr.crowned) w_q += 1;
            } else {
                b_prob += p;
                if (pr.crowned) b_q += 1;
            }
        }
    }
    if (w_prob < 0.0001 && b_prob < 0.0001) {
        state = DRAW;
        return;
    }
    if (w_prob < 0.0001) {
        state = BLACK_WON;
        return;
    }
    if (b_prob < 0.0001) {
        state = WHITE_WON;
        return;
    }
    Key h;
    board_key(h);
    int count = 0;
    size_t at = history.size();
    for (size_t i = 0; i < history.size(); ++i)
        if (history[i].key == h) {
            at = i;
            break;
        }
    if (at < history.size()) {
        count = history[at].count + 1;
        history[at].count = count;
    } else {
        count = 1;
        history.push_back(HistEntry{h, 1});
    }
    if (history.size() > 500) history.erase(history.begin());
    if (count >= 3) {
        state = DRAW;
        return;
    }
    if (w_q > 0 && b_q > 0 && consecutive_queen_moves >= 50) {
        state = DRAW;
        return;
    }
    if (moves_since_take >= 100) {
        state = DRAW;
        return;
    }
    // len(get_possible_moves(skip_depth_calculation=True)) == 0
    if (!has_any_move_skip_depth()) state = turn == WHITE ? BLACK_WON : WHITE_WON;
}

// get_move_outcomes (cheqqers_official.py:2066)
void Env::move_outcomes(const Move& m, std::vector<Outcome>& out) const {
    out.clear();
    EnvPtr base = clone();
    base->has_override = false;
    base->override_moves.clear();
    bool was_in_mj = base->mj >= 0;
    int from_before = m.a;
    bool was_crowned_before = base->props[from_before].present && base->props[from_before].crowned;
    bool was_take = m.kind == 0 && m.take;
    base->just_crowned = false;
    int sqs[16];
    int8_t roles[16];
    int n;
    base->involved(m, sqs, roles, n);

    using Branch = OBranch;
    std::vector<Branch>& active = scratch().act;
    std::vector<Branch>& next = scratch().nxt;
    active.clear();
    active.push_back(Branch{1.0, std::move(base), false});
    for (int j = 0; j < n; ++j) {
        int si = sqs[j];
        int role = roles[j];
        next.clear();
        for (Branch& b : active) {
            if (b.failed) {
                next.push_back(Branch{b.p, std::move(b.env), true});
                continue;
            }
            Env& be = *b.env;
            double occ = be.occ_direct(si);
            bool is_sup = 0.001 < occ && occ < 0.999;
            bool should = be.game_type == SUPERPOSITION || role == LANDING || role == TRANSIT;
            if (is_sup && should) {
                double p_occ = occ;
                double p_emp = 1.0 - p_occ;
                if (p_emp > 1e-9) {
                    EnvPtr e0;
                    if (p_occ > 1e-9)
                        e0 = be.clone();
                    else
                        e0 = std::move(b.env);
                    e0->measure_square(si, 0, nullptr);
                    bool blk = e0->blocks_after(si, role);
                    next.push_back(Branch{b.p * p_emp, std::move(e0), blk});
                }
                if (p_occ > 1e-9) {
                    EnvPtr e1 = std::move(b.env);
                    e1->measure_square(si, 1, nullptr);
                    bool f = role == LANDING || role == TRANSIT || e1->blocks_after(si, role);
                    next.push_back(Branch{b.p * p_occ, std::move(e1), f});
                }
            } else {
                bool blk = be.blocks_after(si, role);
                next.push_back(Branch{b.p, std::move(b.env), blk});
            }
        }
        active.swap(next);
    }
    PySum total;
    for (Branch& b : active) {
        b.env->finalize(m, b.failed, was_in_mj, was_crowned_before, was_take, false);
        total.add(b.p);
    }
    double tp = total.result();
    for (Branch& b : active) out.push_back(Outcome{tp > 0 ? b.p / tp : b.p, std::move(b.env)});
    active.clear();
    next.clear();
    stable_sort_desc(out.data(), out.size(), [](const Outcome& o) { return o.p; });
}

// ---------------------------------------------------------------------------------------------------------------
// Evaluation (cheqqers_official.py:2668-2823)
// ---------------------------------------------------------------------------------------------------------------

void Env::eval_features(int persp, double* f) {
    const Topo& T = *topo;
    double center = (bs - 1) / 2.0;
    double denom = std::max(1.0, (double)(bs - 1));
    square_probs();
    double occ[MAXSQ];
    for (int i = 0; i < N; ++i) occ[i] = P1[i];
    double white_men = 0.0, black_men = 0.0;
    for (int i = 0; i < N; ++i) {
        double p = occ[i];
        const Props& pr = props[i];
        if (pr.present && p > 0.0001 && !pr.crowned) {
            if (pr.color == WHITE)
                white_men += p;
            else
                black_men += p;
        }
    }
    for (int j = 0; j < NFEAT; ++j) f[j] = 0.0;
    double presence[2] = {0.0, 0.0}, material[2] = {0.0, 0.0}, split[2] = {0.0, 0.0};
    for (int i = 0; i < N; ++i) {
        double p = occ[i];
        const Props& pr = props[i];
        if (!pr.present || p <= 0.0001) continue;
        int color = pr.color;
        bool crowned = pr.crowned;
        int col = T.col[i], row = T.row[i];
        double sign = color == persp ? 1.0 : -1.0;
        if (crowned) {
            f[1] += sign * p;
            if (col == row)
                f[5] += sign * p;
            else if (pyabs(col - row) == 2)
                f[6] += sign * p;
        } else {
            f[0] += sign * p;
            double progress = color == WHITE ? (double)row / denom : (double)(bs - 1 - row) / denom;
            f[2] += sign * p * progress;
            f[3] += sign * p * progress * progress;
            if (color == WHITE && row == 0 && black_men > 0.25)
                f[4] += sign * p;
            else if (color == BLACK && row == bs - 1 && white_men > 0.25)
                f[4] += sign * p;
        }
        double dist_center = std::fabs(col - center) + std::fabs(row - center);
        f[7] += sign * p * (1.0 - dist_center / (2.0 * center + 1e-6));
        if (0.05 < p && p < 0.95) {
            f[8] += sign * p * (1.0 - p);
            split[color] += p * (1.0 - p);
        }
        presence[color] += p;
        material[color] += p * (crowned ? 2.0 : 1.0);
    }
    int other = 1 - persp;
    int mover = turn, waiter = 1 - turn;
    double mover_sign = mover == persp ? 1.0 : -1.0;
    if (mj >= 0) f[9] = mover_sign;
    f[12] = mover_sign;
    MoveList& scan = b_tmp;
    if (!cache_enabled) find_moves(true, -1, true, scan);
    const MoveList& all_takes = cache_enabled ? this->all_takes() : scan;
    double threat[2] = {0.0, 0.0};
    PySetSim victims[2];
    for (const Move& tm : all_takes) {
        int fi = tm.a, ti = tm.b;
        const Props& ap = props[fi];
        if (!ap.present) continue;
        int attacker = ap.color;
        double sw = occ[fi] * (1.0 - occ[ti]);
        if (sw > threat[attacker]) threat[attacker] = sw;
        int fc = T.col[fi], fr = T.row[fi], tc = T.col[ti], tr = T.row[ti];
        int dx = tc > fc ? 1 : -1, dy = tr > fr ? 1 : -1;
        for (int step = 1; step < pyabs(tc - fc); ++step) {
            int between = T.idx_at[fc + step * dx][fr + step * dy];
            if (occ[between] > 0.001) {
                victims[1 - attacker].add(between);
                break;
            }
        }
    }
    f[10] = mover_sign * threat[mover];
    f[11] = -mover_sign * threat[waiter];
    PySum sm, sw;
    for (size_t s = 0; s <= victims[mover].mask; ++s)
        if (victims[mover].tab[s] >= 0) sm.add(occ[victims[mover].tab[s]]);
    for (size_t s = 0; s <= victims[waiter].mask; ++s)
        if (victims[waiter].tab[s] >= 0) sw.add(occ[victims[waiter].tab[s]]);
    f[13] = mover_sign * sm.result();
    f[14] = -mover_sign * sw.result();
    double few_ours = presence[persp] <= 2.0 ? split[persp] : 0.0;
    double few_theirs = presence[other] <= 2.0 ? split[other] : 0.0;
    f[15] = few_ours - few_theirs;
    double total_material = material[persp] + material[other];
    if (total_material > 1e-9) f[16] = (material[persp] - material[other]) / total_material;
}

double Env::evaluate(int persp, const Weights& w) {
    if (state != IN_PROGRESS) {
        if (state == DRAW) return 0.0;
        int won = state == WHITE_WON ? WHITE : BLACK;
        return won == persp ? 1000.0 : -1000.0;
    }
    double f[NFEAT];
    eval_features(persp, f);
    double score = 0.0;
    for (int j = 0; j < NFEAT; ++j) score += w.w[j] * f[j];
    if (w.logit_per_man > 0.0) return 900.0 * std::tanh(0.5 * w.logit_per_man * score);
    double x = score * w.scale;
    double lo = std::min(900.0, x);  // min(900.0, x)
    return std::max(-900.0, lo);     // max(-900.0, ...)
}

}  // namespace cq
