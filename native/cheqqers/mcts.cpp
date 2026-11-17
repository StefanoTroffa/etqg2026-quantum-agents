#include "mcts.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "value_net.h"

namespace cq {

namespace {
const double NaN = std::numeric_limits<double>::quiet_NaN();
inline double sgn(int turn) { return turn == WHITE ? 1.0 : -1.0; }
}  // namespace

Mcts::Mcts(const Env& root, const std::vector<MoveArr>& raw, const MctsOpts& o)
    : opts_(o), in_dim_(root.N * 7 + NFEAT + 1), rng_(o.seed) {
    dn_.reserve(4096);
    cn_.reserve(4096);
    EnvPtr e = root.clone();
    e->cache_enabled = true;
    e->invalidate_caches();
    root_raw_ = raw;
    new_dnode(std::move(e));
}

int Mcts::new_dnode(EnvPtr env) {
    DNode d;
    d.turn = env->turn;
    d.terminal = env->terminal();
    if (d.terminal) {
        int w = env->winner();
        d.v_net = (w == 1) ? 1.0 : (w == -1 ? -1.0 : 0.0);
        d.V = d.v_net;
        d.status = 2;
    }
    d.env = std::move(env);
    dn_.push_back(std::move(d));
    stats.nodes += 1;
    return (int)dn_.size() - 1;
}

// The node's legal moves (the root: only those in root_raw_, in the position's order).
void Mcts::set_moves(int d) {
    DNode& n = dn_[d];
    std::vector<MoveArr> legal;
    n.env->legal_moves(legal);
    if (d == 0 && !root_raw_.empty()) {
        std::vector<MoveArr> keep;
        for (const MoveArr& m : legal)
            if (std::find(root_raw_.begin(), root_raw_.end(), m) != root_raw_.end()) keep.push_back(m);
        legal = keep.empty() ? root_raw_ : keep;
    }
    n.edges.resize(legal.size());
    for (size_t i = 0; i < legal.size(); ++i) n.edges[i].m = legal[i];
    if (legal.empty()) {  // no move but not over (should not happen): a fixed value from the network only
        n.terminal = true;
    }
}

void Mcts::encode_into(int d, float* x, int32_t* mv, int32_t* lens, int* mmax, int row) {
    DNode& n = dn_[d];
    set_moves(d);
    std::vector<double> buf((size_t)in_dim_);
    encode_position(*n.env, buf.data());
    float* xr = x + (size_t)row * in_dim_;
    for (int j = 0; j < in_dim_; ++j) xr[j] = (float)buf[j];
    int M = opts_.max_moves;
    int32_t* mr = mv + (size_t)row * M * 3;
    int32_t* lr = lens + (size_t)row * M;
    int cnt = std::min<int>((int)n.edges.size(), M);
    for (int i = 0; i < M; ++i) {
        lr[i] = 0;
        mr[3 * i] = mr[3 * i + 1] = mr[3 * i + 2] = 0;
    }
    for (int i = 0; i < cnt; ++i) {
        const MoveArr& m = n.edges[i].m;
        lr[i] = m.n;
        for (int j = 0; j < m.n; ++j) mr[3 * i + j] = m.v[j];
    }
    if (cnt > *mmax) *mmax = cnt;
    n.status = 1;
}

int Mcts::select_edge(int d) const {
    const DNode& n = dn_[d];
    double s = sgn(n.turn);
    int ntot = 0;
    for (const Edge& e : n.edges) ntot += e.n + e.vl;
    double sq = std::sqrt((double)std::max(ntot, 1));
    double fpu = s * n.V - opts_.fpu_reduction;
    int best = -1;
    double bv = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n.edges.size(); ++i) {
        const Edge& e = n.edges[i];
        if (e.chance >= 0 && cn_[e.chance].waiting > 0) continue;  // its outcomes are with the network: try others
        int nv = e.n + e.vl;
        double q = nv == 0 ? fpu : (s * e.q * e.n - (double)e.vl) / (double)nv;
        double u = opts_.c_puct * (double)e.prior * sq / (1.0 + nv);
        if (q + u > bv) {
            bv = q + u;
            best = (int)i;
        }
    }
    return best;
}

double Mcts::chance_value(const CNode& c) const {
    double v = 0.0;
    for (size_t k = 0; k < c.child.size(); ++k) v += c.p[k] * dn_[c.child[k]].V;
    return v;
}

// Back the steps [from, to) up, bottom first. The last step's chance node has all its children evaluated.
void Mcts::backup(const std::vector<Step>& steps, size_t from, size_t to) {
    for (size_t i = to; i-- > from;) {
        const Step& st = steps[i];
        CNode& c = cn_[st.c];
        if (st.k >= 0) {
            c.n[st.k] += 1;
            c.vl[st.k] -= 1;
        }
        c.V = chance_value(c);
        DNode& d = dn_[st.d];
        Edge& e = d.edges[st.a];
        double old = e.n > 0 ? e.q : 0.0;
        d.sumNQ += (double)(e.n + 1) * c.V - (double)e.n * old;
        e.n += 1;
        e.vl -= 1;
        e.q = c.V;
        d.nsum += 1;
        d.V = (d.v_net + d.sumNQ) / (1.0 + (double)d.nsum);
    }
    stats.sims += 1;
    stats.max_depth = std::max<int>(stats.max_depth, (int)(to - from));
}

void Mcts::undo_virtual(const std::vector<Step>& steps, size_t from, size_t to) {
    for (size_t i = from; i < to; ++i) {
        const Step& st = steps[i];
        dn_[st.d].edges[st.a].vl -= 1;
        if (st.k >= 0) cn_[st.c].vl[st.k] -= 1;
    }
}

void Mcts::add_root_noise() {
    DNode& r = dn_[0];
    if (opts_.dirichlet_alpha <= 0.0 || r.edges.empty()) return;
    std::gamma_distribution<double> g(opts_.dirichlet_alpha, 1.0);
    std::vector<double> eta(r.edges.size());
    double s = 0.0;
    for (double& v : eta) s += (v = g(rng_));
    if (!(s > 0)) return;
    for (size_t i = 0; i < eta.size(); ++i)
        r.edges[i].prior = (float)((1.0 - opts_.dirichlet_frac) * r.edges[i].prior + opts_.dirichlet_frac * eta[i] / s);
}

int Mcts::collect(int max_leaves, int cap, float* x, int32_t* mv, int32_t* lens, int* mmax) {
    *mmax = 0;
    Batch b;
    int rows = 0;
    if (dn_[0].status == 0) {  // the root first, alone
        encode_into(0, x, mv, lens, mmax, 0);
        b.nodes.push_back(0);
        b.root_eval = true;
        batches_.push_back(std::move(b));
        return 1;
    }
    if (dn_[0].status != 2 || dn_[0].terminal || gdone_) return 0;
    int tries = 0, max_tries = 4 * max_leaves + 16;
    while (rows < max_leaves && tries++ < max_tries) {
        if (stats.nodes >= opts_.max_nodes) break;
        path_.clear();
        int d = 0;
        bool done = false;
        while (!done) {
            DNode& n = dn_[d];
            if (n.terminal) {  // exact value: back up at once
                backup(path_, 0, path_.size());
                stats.terminal_sims += 1;
                done = true;
                break;
            }
            int a;
            if (opts_.gumbel_m > 0 && d == 0) {
                a = gumbel_root_pick();
                if (a == -2) {  // budget used: the search is over
                    done = true;
                    tries = max_tries;
                    break;
                }
                if (a == -1) {  // this phase's simulations are all pending: wait for the network
                    stats.collisions += 1;
                    done = true;
                    tries = max_tries;
                    break;
                }
            } else {
                a = opts_.gumbel_m > 0 ? gumbel_interior(d) : select_edge(d);
            }
            if (a < 0) {  // every move of this node waits for the network: abandon this simulation
                undo_virtual(path_, 0, path_.size());
                stats.collisions += 1;
                done = true;
                break;
            }
            Edge& e = n.edges[a];
            e.vl += 1;
            if (e.chance < 0) {
                // expand: every outcome of the move, the non-terminal ones to the network
                Move m = n.env->resolve(e.m);
                std::vector<Outcome> raw;
                n.env->move_outcomes(m, raw);
                PySum total;
                for (const Outcome& o : raw) total.add(o.p);
                double tp = total.result();
                if (!(tp > 0)) throw Unsupported("outcomes without probability");
                CNode c;
                std::vector<std::pair<double, EnvPtr>> outs;
                for (Outcome& o : raw)
                    if (o.p > 1e-9) outs.emplace_back(o.p / tp, std::move(o.env));
                if (outs.empty())  // every outcome below the filter (their sum is > 0): keep all of them, else the
                    for (Outcome& o : raw)  // chance node would have no child to descend into
                        if (o.p > 0) outs.emplace_back(o.p / tp, std::move(o.env));
                if (outs.empty()) throw Unsupported("a move without outcomes");
                int ci = (int)cn_.size();
                e.chance = ci;  // e is still valid: dn_ has not grown yet
                for (auto& pr : outs) {
                    pr.second->cache_enabled = true;
                    int child = new_dnode(std::move(pr.second));
                    c.child.push_back(child);
                    c.p.push_back(pr.first);
                    c.n.push_back(0);
                    c.vl.push_back(0);
                    if (dn_[child].status == 0) c.waiting += 1;
                }
                cn_.push_back(std::move(c));
                path_.push_back(Step{d, a, ci, -1});
                if (cn_[ci].waiting == 0) {
                    backup(path_, 0, path_.size());
                    stats.terminal_sims += 1;
                } else {
                    size_t off = b.steps.size();
                    b.sim_off.push_back((int)off);
                    b.steps.insert(b.steps.end(), path_.begin(), path_.end());
                    for (int child : cn_[ci].child)
                        if (dn_[child].status == 0) {
                            if (rows < cap) {
                                encode_into(child, x, mv, lens, mmax, rows++);
                                b.nodes.push_back(child);
                            } else {  // no room left in the buffers (a move with very many outcomes): linear value,
                                      // uniform priors, never sent to the network
                                DNode& o = dn_[child];
                                set_moves(child);
                                Weights w;
                                o.v_net = std::max(-1.0, std::min(1.0, o.env->evaluate(WHITE, w) / 1000.0));
                                o.V = o.v_net;
                                o.status = 2;
                                for (Edge& ed : o.edges) ed.prior = (float)(1.0 / o.edges.size());
                                stats.overflow += 1;
                            }
                        }
                }
                done = true;
                break;
            }
            CNode& c = cn_[e.chance];  // not waiting: select_edge skips those
            if (c.child.empty()) throw Unsupported("a chance node without outcomes");
            int k = 0;
            double bk = -1.0;
            for (size_t j = 0; j < c.child.size(); ++j) {
                double s = c.p[j] / (1.0 + c.n[j] + c.vl[j]);
                if (s > bk) {
                    bk = s;
                    k = (int)j;
                }
            }
            c.vl[k] += 1;
            path_.push_back(Step{d, a, e.chance, k});
            d = c.child[k];
        }
    }
    if (rows == 0) return 0;
    batches_.push_back(std::move(b));
    return rows;
}

void Mcts::apply(const float* values, const float* priors, int stride, int n) {
    if (batches_.empty()) throw Unsupported("apply without a pending batch");
    Batch b = std::move(batches_.front());
    batches_.erase(batches_.begin());
    if ((int)b.nodes.size() != n) throw Unsupported("apply: row count differs from the batch");
    for (int i = 0; i < n; ++i) {
        DNode& d = dn_[b.nodes[i]];
        double v = std::max(-1.0, std::min(1.0, (double)values[i]));
        d.v_net = sgn(d.turn) * v;
        d.V = d.v_net;
        d.status = 2;
        if (d.edges.empty()) {
            d.terminal = true;
            continue;
        }
        const float* pr = priors + (size_t)i * stride;
        int cnt = std::min<int>((int)d.edges.size(), opts_.max_moves);
        double s = 0.0, mn = std::numeric_limits<double>::infinity();
        for (int j = 0; j < cnt; ++j) {
            double p = std::isfinite(pr[j]) && pr[j] > 0 ? pr[j] : 0.0;
            s += p;
            mn = std::min(mn, p);
        }
        for (size_t j = 0; j < d.edges.size(); ++j) {
            double p = (int)j < cnt ? (std::isfinite(pr[j]) && pr[j] > 0 ? pr[j] : 0.0) : mn;
            d.edges[j].prior = s > 0 ? (float)(p / s) : (float)(1.0 / d.edges.size());
        }
    }
    if (b.root_eval) {
        if (opts_.gumbel_m > 0)
            gumbel_init();
        else
            add_root_noise();
        return;
    }
    for (size_t i = 0; i < b.sim_off.size(); ++i) {
        size_t from = b.sim_off[i], to = i + 1 < b.sim_off.size() ? b.sim_off[i + 1] : b.steps.size();
        const Step& last = b.steps[to - 1];
        cn_[last.c].waiting = 0;
        backup(b.steps, from, to);
    }
    stats.evals += n;
}

std::vector<Mcts::RootInfo> Mcts::root_info() const {
    std::vector<RootInfo> out;
    for (const Edge& e : dn_[0].edges) out.push_back(RootInfo{e.m, e.n, e.n > 0 ? e.q : NaN, (double)e.prior});
    return out;
}

double Mcts::root_value() const { return dn_[0].V; }
double Mcts::root_vnet() const { return dn_[0].v_net; }

// ── Gumbel root and completed-Q selection ──────────────────────────────────────────────────────────────────────────

void Mcts::sigma_completed(int d, std::vector<double>& sigma, std::vector<double>& logit) const {
    const DNode& n = dn_[d];
    const double s = sgn(n.turn);
    const size_t k = n.edges.size();
    sigma.assign(k, 0.0);
    logit.assign(k, 0.0);
    int sumN = 0, maxN = 0;
    double pi_vis = 0.0, piq_vis = 0.0;
    for (size_t i = 0; i < k; ++i) {
        const Edge& e = n.edges[i];
        logit[i] = std::log(std::max((double)e.prior, 1e-12));
        if (e.n > 0) {
            sumN += e.n;
            maxN = std::max(maxN, e.n);
            pi_vis += e.prior;
            piq_vis += e.prior * s * e.q;
        }
    }
    // mixed value for the unvisited moves (Danihelka et al. 2022, eq. 33)
    const double v_hat = s * n.v_net;
    double v_mix = v_hat;
    if (sumN > 0 && pi_vis > 0) v_mix = (v_hat + (double)sumN * (piq_vis / pi_vis)) / (1.0 + (double)sumN);
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (size_t i = 0; i < k; ++i) {
        const Edge& e = n.edges[i];
        double q = e.n > 0 ? s * e.q : v_mix;
        sigma[i] = q;
        lo = std::min(lo, q);
        hi = std::max(hi, q);
    }
    const double scale = (opts_.c_visit + (double)maxN) * opts_.c_scale;
    const double span = std::max(hi - lo, 1e-8);
    for (size_t i = 0; i < k; ++i) sigma[i] = scale * ((sigma[i] - lo) / span);
}

int Mcts::gumbel_interior(int d) const {
    const DNode& n = dn_[d];
    std::vector<double> sig, lg;
    sigma_completed(d, sig, lg);
    double mx = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < sig.size(); ++i) mx = std::max(mx, lg[i] + sig[i]);
    double z = 0.0;
    std::vector<double> pi(sig.size());
    for (size_t i = 0; i < sig.size(); ++i) z += (pi[i] = std::exp(lg[i] + sig[i] - mx));
    int ntot = 0;
    for (const Edge& e : n.edges) ntot += e.n + e.vl;
    int best = -1;
    double bv = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n.edges.size(); ++i) {
        const Edge& e = n.edges[i];
        if (e.chance >= 0 && cn_[e.chance].waiting > 0) continue;
        double v = pi[i] / z - (double)(e.n + e.vl) / (1.0 + (double)ntot);
        if (v > bv) {
            bv = v;
            best = (int)i;
        }
    }
    return best;
}

void Mcts::gumbel_init() {
    DNode& r = dn_[0];
    const size_t k = r.edges.size();
    g_.assign(k, 0.0);
    std::uniform_real_distribution<double> U(1e-12, 1.0);
    for (size_t i = 0; i < k; ++i) g_[i] = -std::log(-std::log(U(rng_)));
    std::vector<int> idx(k);
    for (size_t i = 0; i < k; ++i) idx[i] = (int)i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        return g_[a] + std::log(std::max((double)r.edges[a].prior, 1e-12)) >
               g_[b] + std::log(std::max((double)r.edges[b].prior, 1e-12));
    });
    const int m = std::min<int>(opts_.gumbel_m, (int)k);
    cand_.assign(idx.begin(), idx.begin() + m);
    phase_start_.assign(k, 0);
    int phases = 0;
    while ((1 << phases) < m) ++phases;
    if (m <= 1 || opts_.gumbel_n <= 0) {
        gdone_ = true;
        return;
    }
    per_phase_ = std::max(1, opts_.gumbel_n / phases);
    quota_ = std::max(1, per_phase_ / (int)cand_.size());
}

// The root move of the next simulation: the candidate with the fewest visits in this phase below its quota (moves
// waiting for the network skipped). -1: the phase's remaining simulations are all pending; -2: the search is over.
int Mcts::gumbel_root_pick() {
    for (;;) {
        DNode& r = dn_[0];
        int best = -1, bv = 1 << 30;
        bool pending = false;
        for (int c : cand_) {
            const Edge& e = r.edges[c];
            int used = e.n - phase_start_[c] + e.vl;
            if (e.vl > 0) pending = true;
            if (used >= quota_) continue;
            if (e.chance >= 0 && cn_[e.chance].waiting > 0) continue;
            if (used < bv) {
                bv = used;
                best = c;
            }
        }
        if (best >= 0) return best;
        bool quota_left = false;
        for (int c : cand_)
            if (r.edges[c].n - phase_start_[c] + r.edges[c].vl < quota_) quota_left = true;
        if (pending || quota_left) return -1;
        // phase over: keep the better half by g + logit + sigma(completed Q)
        if (cand_.size() <= 1) {
            gdone_ = true;
            return -2;
        }
        std::vector<double> sig, lg;
        sigma_completed(0, sig, lg);
        std::stable_sort(cand_.begin(), cand_.end(),
                         [&](int a, int b) { return g_[a] + lg[a] + sig[a] > g_[b] + lg[b] + sig[b]; });
        cand_.resize((cand_.size() + 1) / 2);
        if (cand_.size() <= 1) {
            gdone_ = true;
            return -2;
        }
        for (int c : cand_) phase_start_[c] = r.edges[c].n;
        quota_ = std::max(1, per_phase_ / (int)cand_.size());
    }
}

Mcts::GumbelOut Mcts::gumbel() const {
    GumbelOut out;
    const DNode& r = dn_[0];
    if (r.edges.empty()) return out;
    std::vector<double> sig, lg;
    sigma_completed(0, sig, lg);
    double mx = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < sig.size(); ++i) mx = std::max(mx, lg[i] + sig[i]);
    double z = 0.0;
    out.policy.resize(sig.size());
    for (size_t i = 0; i < sig.size(); ++i) z += (out.policy[i] = std::exp(lg[i] + sig[i] - mx));
    for (double& p : out.policy) p /= z;
    out.gumbel = g_;
    out.candidates = cand_;
    double bv = -std::numeric_limits<double>::infinity();
    const std::vector<int>& pool = cand_.empty() ? std::vector<int>() : cand_;
    for (int c : pool) {
        double v = g_[c] + lg[c] + sig[c];
        if (v > bv) {
            bv = v;
            out.chosen = c;
        }
    }
    if (out.chosen < 0) out.chosen = (int)(std::max_element(out.policy.begin(), out.policy.end()) - out.policy.begin());
    return out;
}
int Mcts::root_turn() const { return dn_[0].turn; }

}  // namespace cq
