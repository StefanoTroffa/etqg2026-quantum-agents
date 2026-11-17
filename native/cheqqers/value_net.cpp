#include "value_net.h"

#include <cmath>
#include <string>

namespace cq {

namespace {
constexpr int MAX_WIDTH = 1024;
}

ValueNet::ValueNet(const std::vector<int>& dims, const double* params, long long nparams) : dims_(dims) {
    if (dims.size() < 2) throw Unsupported("value net: need at least one layer");
    if (dims.back() != 1) throw Unsupported("value net: the last layer must have one output");
    for (int d : dims)
        if (d <= 0 || d > MAX_WIDTH) throw Unsupported("value net: layer width out of range");
    long long need = 0;
    for (size_t k = 1; k < dims.size(); ++k) need += (long long)dims[k] * dims[k - 1] + dims[k];
    if (need != nparams) throw Unsupported("value net: parameter count does not match the layer sizes");
    const double* p = params;
    for (size_t k = 1; k < dims.size(); ++k) {
        int in = dims[k - 1], out = dims[k];
        std::vector<double> wt((size_t)in * out);
        for (int o = 0; o < out; ++o)
            for (int i = 0; i < in; ++i) wt[(size_t)i * out + o] = p[(size_t)o * in + i];
        p += (size_t)in * out;
        wt_.push_back(std::move(wt));
        b_.emplace_back(p, p + out);
        p += out;
    }
}

void ValueNet::encode(Env& env, double* x) const {
    if (dims_[0] != env.N * 7 + NFEAT + 1) throw Unsupported("value net: input size does not match this board");
    encode_position(env, x);
}

void encode_position(Env& env, double* x) {
    int n = env.N;
    double f[NFEAT];
    env.eval_features(env.turn, f);  // as cq_selfplay's "features"
    env.square_probs();              // "presence" = P1
    bool own_white = env.turn == WHITE;
    for (int i = 0; i < n; ++i) {
        double* s = x + 7 * i;
        for (int j = 0; j < 7; ++j) s[j] = 0.0;
        const Props& pr = env.props[i];
        double p = env.P1[i];
        if (!pr.present || p <= 1e-4) {
            s[0] = 1.0;
        } else {
            bool own = (pr.color == WHITE) == own_white;
            s[(own ? 1 : 3) + (pr.crowned ? 1 : 0)] = 1.0;
        }
        s[5] = p;
        s[6] = (1e-4 < p && p < 1 - 1e-4) ? 1.0 : 0.0;
    }
    double* g = x + 7 * n;
    for (int j = 0; j < NFEAT; ++j) g[j] = f[j];
    g[NFEAT] = env.mj >= 0 ? 1.0 : 0.0;
}

double ValueNet::forward(const double* x) const {
    double buf[2][MAX_WIDTH];
    const double* in = x;
    int L = (int)wt_.size();
    for (int k = 0; k < L; ++k) {
        int nin = dims_[k], nout = dims_[k + 1];
        double* h = buf[k & 1];
        const double* b = b_[k].data();
        for (int o = 0; o < nout; ++o) h[o] = b[o];
        const double* w = wt_[k].data();
        for (int i = 0; i < nin; ++i) {
            double xi = in[i];
            if (xi == 0.0) continue;  // one-hot inputs and ReLU zeros: most columns are skipped
            const double* col = w + (size_t)i * nout;
            for (int o = 0; o < nout; ++o) h[o] += xi * col[o];
        }
        if (k + 1 < L)
            for (int o = 0; o < nout; ++o) h[o] = h[o] > 0.0 ? h[o] : 0.0;
        in = h;
    }
    return in[0];
}

double ValueNet::logit(Env& env) const {
    double x[MAX_WIDTH];
    encode(env, x);
    return forward(x);
}

double ValueNet::value(Env& env, int persp, const Weights& w) const {
    double x[MAX_WIDTH];
    encode(env, x);
    double lg;
    if (residual_) {
        const double* g = x + 7 * env.N;  // eval_features(turn)
        double score = 0.0;
        for (int j = 0; j < NFEAT; ++j) score += w.w[j] * g[j];
        lg = w.logit_per_man * score + lambda_ * forward(x);
    } else {
        lg = forward(x);
    }
    double v = 900.0 * std::tanh(0.5 * lg);
    return env.turn == persp ? v : -v;
}

}  // namespace cq
