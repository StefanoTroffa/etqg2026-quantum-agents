#include "search.h"

#include "value_net.h"

#include <algorithm>
#include <limits>

namespace cq {

namespace {
constexpr double V_MAX = 1000.0;
constexpr int TT_EXACT = 0, TT_LOWERBOUND = 1, TT_UPPERBOUND = 2;
const double INF = std::numeric_limits<double>::infinity();

// _row_col_from_idx (expecti_cheqqers.py:225): (row, col)
inline void row_col(int idx, int bs, int& row, int& col) {
    int spr = bs / 2;
    int visual_row = idx / spr;
    row = (bs - 1) - visual_row;
    int cir = idx % spr;
    col = (row % 2 == 0) ? cir * 2 : cir * 2 + 1;
}
}  // namespace

uint64_t move_key(const MoveArr& m) {
    uint64_t k = (uint64_t)m.n;
    for (int i = 0; i < m.n; ++i) k = (k << 16) | (uint16_t)m.v[i];
    return k;
}

void Searcher::check_timeout() {
    nodes += 1;
    if ((nodes & 31) == 0 && std::chrono::steady_clock::now() >= deadline) throw Timeout();
}

double Searcher::leaf(Env& env, int ply) {
    if (opts.leaf_sink && !env.terminal() && opts.leaf_sink->rng.next() < opts.leaf_sink->rate) {
        LeafSample s;
        s.x.resize((size_t)env.N * 7 + NFEAT + 1);
        encode_position(env, s.x.data());
        s.turn = env.turn;
        s.ply = ply;
        s.capture_pending = env.capture_pending();
        if (opts.leaf_sink->keep_state) s.env = env.clone();
        opts.leaf_sink->samples.push_back(std::move(s));
    }
    double heur = (opts.net && !env.terminal()) ? opts.net->value(env, root_color, opts.weights)
                                                : env.evaluate(root_color, opts.weights);
    if (heur >= V_MAX * 0.99) return V_MAX - (double)ply;
    if (heur <= -V_MAX * 0.99) return -V_MAX + (double)ply;
    double lo = -V_MAX * 0.95, hi = V_MAX * 0.95;
    if (heur < lo) return lo;
    if (heur > hi) return hi;
    return heur;
}

// _score_move_for_ordering (expecti_cheqqers.py:866)
double Searcher::score_move(Env& env, const MoveArr& mv, int ply, bool has_tt_best, uint64_t tt_best,
                            const double* chances) {
    uint64_t key = move_key(mv);
    if (has_tt_best && key == tt_best) return 1000000.0;
    int bs = env.bs;
    auto occ = [&](int sq) -> double { return (sq >= 0 && sq < env.N) ? chances[sq] : 0.0; };
    if (env.capture_pending()) {
        double score = 500000.0;
        if (mv.n == 2) {
            int r1, c1, r2, c2;
            row_col(mv.v[0], bs, r1, c1);
            row_col(mv.v[1], bs, r2, c2);
            int jump_dist = std::abs(r2 - r1);
            double landing_occ = occ(mv.v[1]);
            double source_occ = std::max(0.5, occ(mv.v[0]));
            score += 10000.0 * (1.0 - landing_occ) * source_occ;
            score += 500.0 * (double)jump_dist;
        }
        return score;
    }
    double score = 0.0;
    if (ply >= 0 && ply < (int)killers.size()) {
        const auto& kl = killers[ply];
        if (std::find(kl.begin(), kl.end(), key) != kl.end()) score += 100000.0;
    }
    bool white = env.turn == WHITE;
    int promo_row = white ? (bs - 1) : 0;
    double center = (bs - 1) / 2.0;
    if (mv.n == 2) {
        int r_to, c_to;
        row_col(mv.v[1], bs, r_to, c_to);
        double landing_occ = occ(mv.v[1]);
        if (landing_occ > 0.01) score -= 15000.0 * landing_occ;
        if (r_to == promo_row) score += 80000.0;
        int adv = white ? r_to : (bs - 1 - r_to);
        score += 300.0 * (double)adv - 100.0 * std::fabs(c_to - center);
    } else if (mv.n == 3) {
        int b = mv.v[1], c = mv.v[2];
        double occ_b = occ(b);
        bool is_merge = occ_b > 0.05;
        if (!is_merge) {
            int r1, c1, r2, c2;
            row_col(b, bs, r1, c1);
            row_col(c, bs, r2, c2);
            double occ_to1 = occ(b);
            double occ_to2 = occ(c);
            if (occ_to1 < 0.01 && occ_to2 < 0.01) {
                score += 12000.0;
                if (r1 == promo_row || r2 == promo_row) score += 45000.0;
                score -= 80.0 * (std::fabs(c1 - center) + std::fabs(c2 - center));
            } else {
                score -= 10000.0 * (occ_to1 + occ_to2);
            }
        } else {
            int r_to, c_to;
            row_col(c, bs, r_to, c_to);
            double occ_to = occ(c);
            if (occ_to < 0.01) {
                score += 15000.0;
                if (r_to == promo_row) score += 55000.0;
            }
        }
    }
    auto it = history_table.find(key);
    int h = it == history_table.end() ? 0 : it->second;
    score += (double)std::min(20000, h);
    return score;
}

void Searcher::order(Env& env, std::vector<MoveArr>& moves, int ply, bool has_tt_best, uint64_t tt_best) {
    env.square_probs();  // env.chances: the squares' presences, read once for all the moves
    double chances[MAXSQ];
    for (int i = 0; i < env.N; ++i) chances[i] = env.P1[i];
    std::vector<Scored>& sc = order_buf;
    sc.resize(moves.size());
    for (size_t i = 0; i < moves.size(); ++i) sc[i] = Scored{score_move(env, moves[i], ply, has_tt_best, tt_best, chances), moves[i]};
    stable_sort_desc(sc.data(), sc.size(), [](const Scored& a) { return a.s; });
    for (size_t i = 0; i < moves.size(); ++i) moves[i] = sc[i].m;
}

void Searcher::record_cutoff(Env& env, const MoveArr& mv, int depth, int ply) {
    if (env.capture_pending()) return;
    uint64_t key = move_key(mv);
    if (ply >= (int)killers.size()) killers.resize(ply + 1);
    auto& kl = killers[ply];
    if (std::find(kl.begin(), kl.end(), key) == kl.end()) {
        kl.insert(kl.begin(), key);
        if (kl.size() > 2) kl.pop_back();
    }
    history_table[key] += depth * depth;
}

// _evaluate_chance_node (expecti_cheqqers.py:759) with get_cheqqers_outcomes (:190)
double Searcher::chance(Env& env, const MoveArr& mv, int depth, double alpha, double beta, int ply, int ext) {
    Move m = env.resolve(mv);
    std::vector<Outcome>& raw = at_ply(raw_by_ply, ply);
    env.move_outcomes(m, raw);
    PySum total;
    for (const Outcome& o : raw) total.add(o.p);
    double tp = total.result();
    if (!(tp > 0)) throw Unsupported("outcomes without probability");
    std::vector<Outcome>& outs = at_ply(outs_by_ply, ply);
    outs.clear();
    for (Outcome& o : raw)
        if (o.p > 1e-9) outs.push_back(Outcome{o.p / tp, std::move(o.env)});
    raw.clear();
    stable_sort_desc(outs.data(), outs.size(), [](const Outcome& o) { return o.p; });
    if (outs.size() == 1) return decision(*outs[0].env, depth, alpha, beta, ply, ext);
    double accum = 0.0, rem = 1.0;
    for (Outcome& o : outs) {
        double prob = o.p;
        rem = std::max(0.0, rem - prob);
        double ca, cb;
        if (opts.star1 && prob > 1e-6) {
            if (alpha <= -V_MAX)
                ca = -INF;
            else
                ca = std::max(-V_MAX, (alpha - accum - rem * V_MAX) / prob);
            if (beta >= V_MAX)
                cb = INF;
            else
                cb = std::min(V_MAX, (beta - accum + rem * V_MAX) / prob);
        } else {
            ca = -INF;
            cb = INF;
        }
        if (ca >= cb) {
            double bound = ca >= beta ? ca : cb;
            return accum + prob * bound + rem * (ca >= beta ? V_MAX : -V_MAX);
        }
        double cv = decision(*o.env, depth, ca, cb, ply, ext);
        accum += prob * cv;
    }
    return accum;
}

// _search_decision_node (expecti_cheqqers.py:623)
double Searcher::decision(Env& env, int depth, double alpha, double beta, int ply, int ext) {
    check_timeout();
    if (env.terminal()) return leaf(env, ply);
    bool in_mj = env.mj >= 0;
    bool qply = false;
    if (depth <= 0) {
        if (in_mj && ext > 0) {
            depth = 1;
            ext -= 1;
        } else if (opts.quiescence && ext > 0 && env.capture_pending()) {
            depth = 1;
            ext -= 1;
            qply = true;
        } else {
            return leaf(env, ply);
        }
    }
    Key key;
    env.state_key(key);
    double alpha_orig = alpha, beta_orig = beta;
    int requested = qply ? 0 : depth;
    bool use_tt = opts.quantum_plies < 0 || ply >= opts.quantum_plies;
    bool has_tt_best = false;
    uint64_t tt_best = 0;
    uint64_t key_hash = 0;
    if (use_tt) {
        key_hash = key.hash();
        if (const TTEntry* found = tt.find(key, key_hash)) {
            const TTEntry& e = *found;
            has_tt_best = e.has_best;
            tt_best = e.best;
            if (e.depth >= requested) {
                if (e.flag == TT_EXACT) return e.value;
                if (e.flag == TT_LOWERBOUND)
                    alpha = std::max(alpha, e.value);
                else if (e.flag == TT_UPPERBOUND)
                    beta = std::min(beta, e.value);
                if (alpha >= beta) return e.value;
            }
        }
    }
    std::vector<MoveArr>& legal = at_ply(legal_by_ply, ply);
    env.legal_moves(legal);
    if (legal.empty()) return leaf(env, ply);
    if (opts.quantum_plies >= 0 && ply >= opts.quantum_plies) {
        // keep the two-square moves in order, if there is one (in place: no list per node)
        bool any_two = false;
        for (const MoveArr& m : legal)
            if (m.n == 2) {
                any_two = true;
                break;
            }
        if (any_two) legal.erase(std::remove_if(legal.begin(), legal.end(), [](const MoveArr& m) { return m.n != 2; }), legal.end());
    }
    if (legal.size() == 1 && ext > 0 && !qply && env.capture_pending()) {
        depth += 1;
        ext -= 1;
    }
    order(env, legal, ply, has_tt_best, tt_best);
    bool is_max = env.turn == root_color;
    double best = is_max ? -INF : INF;
    bool has_best = false;
    uint64_t best_key = 0;
    if (is_max) {
        for (const MoveArr& mv : legal) {
            double val = chance(env, mv, depth - 1, alpha, beta, ply + 1, ext);
            if (val > best) {
                best = val;
                best_key = move_key(mv);
                has_best = true;
            }
            if (best > alpha) alpha = best;
            if (alpha >= beta) {
                record_cutoff(env, mv, depth, ply);
                break;
            }
        }
    } else {
        for (const MoveArr& mv : legal) {
            double val = chance(env, mv, depth - 1, alpha, beta, ply + 1, ext);
            if (val < best) {
                best = val;
                best_key = move_key(mv);
                has_best = true;
            }
            if (best < beta) beta = best;
            if (alpha >= beta) {
                record_cutoff(env, mv, depth, ply);
                break;
            }
        }
    }
    int flag;
    if (best <= alpha_orig)
        flag = TT_UPPERBOUND;
    else if (best >= beta_orig)
        flag = TT_LOWERBOUND;
    else
        flag = TT_EXACT;
    if (use_tt && (int64_t)tt.size() < (int64_t)opts.max_tt_size) {
        TTEntry e{requested, (int8_t)flag, has_best, best, best_key};
        tt.put(key, key_hash, e);
    }
    return best;
}

SearchResult Searcher::get_move(const Env& caller, const std::vector<MoveArr>* raw_in) {
    using clock = std::chrono::steady_clock;
    auto start = clock::now();
    deadline = start + std::chrono::duration_cast<clock::duration>(
                           std::chrono::duration<double>((opts.time_limit_ms / 1000.0) * 0.90));
    nodes = 0;
    unexpanded = 0;
    SearchResult res;
    EnvPtr envp = caller.clone();  // the caller's position is left as it was
    Env& env = *envp;
    env.cache_enabled = opts.use_cache ? true : caller.cache_enabled;

    std::vector<MoveArr> raw;
    if (raw_in)
        raw = *raw_in;
    else
        env.legal_moves(raw);
    auto finish = [&]() {
        res.nodes = nodes;
        res.unexpanded = unexpanded;
        res.elapsed_ms = std::chrono::duration<double, std::milli>(clock::now() - start).count();
        return res;
    };
    if (raw.empty()) return finish();
    if (raw.size() == 1) {
        res.chosen = 0;
        return finish();
    }
    root_color = env.turn;
    std::unordered_map<uint64_t, int> key_to_raw;
    for (size_t i = 0; i < raw.size(); ++i) key_to_raw.emplace(move_key(raw[i]), (int)i);
    std::vector<MoveArr> env_moves, search_moves;
    env.legal_moves(env_moves);
    for (const MoveArr& m : env_moves)
        if (key_to_raw.count(move_key(m))) search_moves.push_back(m);
    if (search_moves.empty()) search_moves = raw;

    std::vector<MoveArr> ordered = search_moves;
    order(env, ordered, 0, false, 0);
    uint64_t best_key = move_key(ordered[0]);
    double best_score = -INF;
    tt.clear();
    killers.clear();

    for (int depth = 1; depth <= opts.max_depth; ++depth) {
        if (clock::now() >= deadline) break;
        try {
            double alpha = -INF, beta = INF;
            bool has_iter = false;
            uint64_t iter_key = 0;
            double iter_val = -INF;
            std::vector<RootScore> scores;
            for (const MoveArr& mv : ordered) {
                check_timeout();
                double val = chance(env, mv, depth - 1, alpha, beta, 1, opts.max_capture_extension);
                scores.push_back(RootScore{mv, val});
                if (val > iter_val) {
                    iter_val = val;
                    iter_key = move_key(mv);
                    has_iter = true;
                }
                if (val > alpha) alpha = val;
            }
            if (has_iter) {
                best_key = iter_key;
                best_score = iter_val;
                res.last_depth = depth;
                res.has_score = true;
                res.last_score = iter_val;
            }
            stable_sort_desc(scores.data(), scores.size(), [](const RootScore& a) { return a.value; });
            for (size_t i = 0; i < scores.size(); ++i) ordered[i] = scores[i].move;
            res.root_scores = scores;
            if (best_score >= V_MAX * 0.98) break;
        } catch (const Timeout&) {
            break;
        } catch (const std::exception& ex) {
            res.error = ex.what();
            break;
        } catch (...) {
            res.error = "unknown error in the search";
            break;
        }
    }
    auto it = key_to_raw.find(best_key);
    res.chosen = it == key_to_raw.end() ? 0 : it->second;
    return finish();
}

}  // namespace cq
